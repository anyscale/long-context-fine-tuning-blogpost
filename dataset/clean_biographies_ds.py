# This creates the biographies along with the name of the person from the biography and extracts the token length
# It does not create a ready-to-be-trained-on dataset, but merely cleans the data a bit.
# You should probably run this first if you want to reproduce.

import ray
import datasets
import names
from transformers import AutoTokenizer
import unicodedata
import json
import re
import os

MIN_TOKENS = 200
MAX_TOKENS = 300

OUT_PATH = "cleaned_bios_dataset.jsonl"

data_context = ray.data.DataContext.get_current()
data_context.execution_options.verbose_progress = True

runtime_env = {"env_vars": {"HUGGING_FACE_HUB_TOKEN": os.environ["HUGGING_FACE_HUB_TOKEN"]}}
ray.init(runtime_env=runtime_env)

hf_dataset = datasets.load_dataset("wiki_bio")

ds_train = ray.data.from_huggingface(hf_dataset["train"])
ds_val = ray.data.from_huggingface(hf_dataset["val"])
ds_test = ray.data.from_huggingface(hf_dataset["test"])

# Merge all datasets
ds = ds_train.union(ds_val).union(ds_test)

import unicodedata

def reassemble_item(item):
    original_text = item["target_text"]
    original_text = unicodedata.normalize('NFC', original_text)
    original_text = re.sub('(``).+?(\'\' )', '', original_text)
    original_text = original_text.replace("\n", " ")
    original_text = original_text.replace("-lrb- ", "( ").replace(" -rrb-", " )")
    # Add spaces in the beginning and end so that we can be sure to find all first and last names
    better_text = " " + original_text + " "

    # Grab original name and remove additional information "such as occupation, date or place of birth"
    # Original name may sometimes be different from the first appearance of the name in the bio. 
    # Something like "abcdef (born in 123 in blah), better known by her stage name ghijkl, ...."
    # We accept that for now and see how the models do
    original_name = item["input_text"]["context"].replace("\n", "")
    better_name = unicodedata.normalize('NFC', original_name)
    better_name = re.sub('(``).+?(\'\' )', '', better_name)
    better_name = re.sub('( -lrb-).+?( -rrb-)', '', better_name)

    original_partial_names = original_name.split(" ")
    if len(original_partial_names) < 2:
        # This does not fit our "random name" replacement scheme
        return {"name": "", "bio": ""}
    
    # We don't want to replace the original name with a similar name, so create names until we have something somewhat distinct
    found_different_name = False
    while not found_different_name:
        random_name = names.get_full_name().lower()
        random_first_name, random_last_name = random_name.split(" ")
        if (random_first_name not in original_partial_names) and (random_last_name not in original_partial_names):
            found_different_name = True

    # Since all words, commas and dots have speces left and right, we can assume that names will also be spaced
    # Therefore, they can not be contained within each other
    original_partial_names = [" " + n + " " for n in original_partial_names]

    fictional_bio = better_text.replace(original_partial_names[-1], " " + random_last_name + " ")
    # Delete middle names
    for name in original_partial_names[1:-1]:
        fictional_bio = fictional_bio.replace(name, " ")
    fictional_bio = fictional_bio.replace(original_partial_names[0], " " + random_first_name + " ")

    for o_name in original_partial_names:
        # All names should have been replaced
        if o_name in fictional_bio:
            return {"name": "", "bio": ""}
    
    fictional_bio = fictional_bio.strip()
    if not fictional_bio.startswith(random_name):
        return {"name": "", "bio": ""}
    
    return {"name": random_name, "bio": fictional_bio}

items = ds.take(1000)
filtered = []
for item in items:
    datapoint = reassemble_item(item)
    if not datapoint == {"name": "", "bio": ""}:
        filtered.append(datapoint)
print(len(filtered))

class Mapper:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    def __call__(self, row):
        assert "input_text" in row, "Where's the input text? " + str(row)
        row = reassemble_item(row)
        bio_tokens = self.tokenizer(
            row["bio"],
            add_special_tokens=False,
        )
        row["bio_token_len"] = len(bio_tokens["input_ids"])
        return row

final_ds = ds.repartition(100).map(Mapper, compute=ray.data.ActorPoolStrategy(size=100))

df = final_ds.to_pandas()
entries = df.loc[df['bio_token_len'] <= MAX_TOKENS].loc[df['bio_token_len'] > MIN_TOKENS].to_dict("records")

# The number of entries should be of reasonable size to proceed.
# For the biographies dataset, we are looking at approx 30k biographies.
print("Num entries: ", len(entries))

with open(OUT_PATH, "w") as f:
    for entry in entries:
        json.dump(entry, f)
        f.write('\n')