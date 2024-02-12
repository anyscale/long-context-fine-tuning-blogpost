# This assembles the biographies to a training- and evaluation-datasets and some needles for the the benchmark
# We create a range of datasets to benchmark the problem at different context lengths
# This script includes the main steps required to create a synthetic fine-tuning dataset by using JSON mode with the Anyscale Endpoints API

import os
import json
import ray
import json
import numpy as np
import random

from openai import OpenAI
from random import sample
from pathlib import Path

from format import schema_json, get_messages

runtime_env = {"env_vars": {"HUGGING_FACE_HUB_TOKEN": os.environ["HUGGING_FACE_HUB_TOKEN"]}}
ray.init(runtime_env=runtime_env)

AE_API_BASE="https://api.endpoints.anyscale.com/v1"
AE_API_KEY=os.environ["AE_API_KEY"]

# Set this to the number of CPUs you have to spare for ray data to distribute the load
NUM_ACTORS = 190

# Number of needles that we use for benchmarking
TEST_NUM_NEEDLES = 10
# Number of needles that we create a json-representation of so that we can insert them into training haystacks
NUM_TRAIN_NEEDLES = 5_000
NUM_EVAL_NEEDLES = 50

# The number of tokens at the end of each haystack that we want to leave "empty" to leave space for the model to generate tokens etc.
CONTEXT_BUFFER_SIZE = 600

DATASET_BASE_PATH = Path(__file__).parent.resolve()
# If we change the base path to network storage, we should make sure that the path exists
DATASET_BASE_PATH.mkdir(parents=True, exist_ok=True)

# Number of elements in the datasets to be created
TRAINING_DS_SIZE = 5000
EVAL_DS_SIZE = 200

training_context_lens = [4096, 8192, 16384, 32768]
evaluation_context_lens = np.arange(4096, 32768+1, 4096)

# Query to use for mistral to create json labels
QUERY = (
    "{}\n"
    "Given this biography, extract information about {}\n"
    "Try go guess the nationality from the biography. If that is not possible, respond with an 'unknown' nationality."
    "Try to infer the full date of birth and death. You can also represent data that you can't find in the biography with a '0'."
    )

def get_extracted_dict(needle):
    query = QUERY.format(needle['bio'], needle['name'])

    messages = [
            {"role": "system", "content": "You are a helpful assistant that extracts information about a person in json."},
            {"role": "user", "content": query}
        ]

    client = OpenAI(
        base_url=AE_API_BASE,
        api_key=AE_API_KEY,
    )

    chat_completion = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.1",
            response_format={
                "type": "json_object",
                "schema": schema_json,
            },
            messages=messages,
            temperature=0.0,
        )
    
    print(chat_completion)
    
    return json.loads(chat_completion.choices[0].message.content)

def create_datapoint(haystack, needle):
    """Creates a datapoint in our preferred conversational format."""
    # Assemble the messages like we would when prompting the model after fine-tuning
    messages = get_messages(haystack, needle['name'])
    # Add the label for fine-tuning
    messages.append({"role": "assistant", "content": str(needle["json"])})
    datapoint = {"messages" : messages}
    return datapoint

# For each set, each context length gets it's own file
def get_train_path(context_len):
    return DATASET_BASE_PATH / f"train_{context_len}.jsonl"

def get_eval_path(context_len):
    return DATASET_BASE_PATH / f"eval_{context_len}.jsonl"
    
test_haystack_path = DATASET_BASE_PATH / "test_haystack.jsonl"
test_needles_path = DATASET_BASE_PATH / "test_needles.jsonl"

train_needles_path = DATASET_BASE_PATH / "train_needles.jsonl"
eval_needles_path = DATASET_BASE_PATH / "eval_needles.jsonl"

# Create eval and test set

ds = ray.data.read_json(DATASET_BASE_PATH / "cleaned_bios_dataset.jsonl")
df = ds.to_pandas()

# Grab Test data to build a haystack from
cumsum = df.bio_token_len.cumsum()
nearest_index = (cumsum - max(training_context_lens)).abs().idxmin() + 1
test_haystack_df = df[:nearest_index]
with open(test_haystack_path, "w") as f:
    for index, row in test_haystack_df.iterrows():
        _dict = row.to_dict()
        f.write(json.dumps(_dict) + "\n")

df = df[nearest_index:]
test_haystack_needles = df[:TEST_NUM_NEEDLES]

test_needles_list = []

# Create test needles
if os.path.exists(test_needles_path):
    with open(test_needles_path, "r") as f:
        test_needles_list = [json.loads(l) for l in f.readlines()]

for index, row in test_haystack_needles.iterrows():
    if index > len(test_needles_list):
        _needle_dict = row.to_dict()
        _needle_dict["json"] = get_extracted_dict(_needle_dict)
        test_needles_list.append(_needle_dict)

with open(test_needles_path, "w") as f:
    for _needle_dict in test_needles_list:
        f.write(json.dumps(_needle_dict) + "\n")

df = df[TEST_NUM_NEEDLES:]

# Reserve 20% for creating the eval dataset
split_idx = int(len(df) * 0.8)
train_df, eval_df = df[:split_idx], df[split_idx:]

# Create train dataset
train_needles_df = train_df[:NUM_TRAIN_NEEDLES]
train_df = train_df[NUM_TRAIN_NEEDLES:]

train_needles_list = []

if os.path.exists(train_needles_path):
    with open(train_needles_path, "r") as f:
        train_needles_list = [json.loads(l) for l in f.readlines()]

first_idx = train_needles_df.index[0]
for idx, row in train_needles_df.iterrows():
    if len(train_needles_list) > idx - first_idx:
        continue
    row = row.to_dict()
    try:
        row["json"] = get_extracted_dict(row)
    except Exception as e:
        print(e)
        continue
    train_needles_list.append(row)

    with open(train_needles_path, "a+") as f:
        f.write(json.dumps(row) + "\n")

# Create eval dataset
eval_needles_df = eval_df[:NUM_EVAL_NEEDLES]
eval_df = eval_df[NUM_EVAL_NEEDLES:]

eval_needles_list = []
if os.path.exists(eval_needles_path):
    with open(eval_needles_path, "r") as f:
        eval_needles_list = [json.loads(l) for l in f.readlines()]

first_idx = eval_needles_df.index[0]
for idx, row in eval_needles_df.iterrows():
    if len(eval_needles_list) > idx - first_idx:
        continue
    row = row.to_dict()
    try:
        row["json"] = get_extracted_dict(row)
    except Exception as e:
        print(e)
        continue
    eval_needles_list.append(row)

    with open(eval_needles_path, "a+") as f:
        f.write(json.dumps(row) + "\n")


@ray.remote
def create_dataset(_df, context_len_ds_counter, train):
    dataset_by_len = {_len: [] for _len in context_len_ds_counter.keys()}
    haystack_by_len = {_len: [] for _len in context_len_ds_counter.keys()}
    haystack_by_len_token_counters = {_len: 0 for _len in context_len_ds_counter.keys()}

    num_tokens_per_ds = 0
    while sum(list(context_len_ds_counter.values())) > 0:
        for _len, counter in haystack_by_len_token_counters.items():
            # Sample one element at a time
            hay = _df.sample(n=1).iloc[0].to_dict()
            needle = sample(train_needles_list, 1)[0]
            # Extend existing haystacks and their needles
            bio_token_len = int(hay["bio_token_len"])
            needle_len = int(needle["bio_token_len"])
            num_tokens_per_ds += bio_token_len
            
            if context_len_ds_counter[_len] <= 0:
                continue
                
            counter += bio_token_len
            # Increase counter by length
            haystack_by_len_token_counters[_len] = counter
            # If length is now larger than target context, prepare haystack and start over for this context length
            max_context_len = _len - CONTEXT_BUFFER_SIZE
            total_context_length = counter + needle_len
            if total_context_length // max_context_len > 0:
                # Case where haystack is finished and we want to complete it
                haystack = haystack_by_len[_len]

                haystack.append(hay["bio"])
                haystack_by_len_token_counters[_len] = 0
                
                # Insert needle bio somewhere
                haystack.insert(random.randint(0, len(haystack)), needle["bio"])
                
                haystack = "".join(haystack)
                
                overhead = total_context_length % (max_context_len - 1) # Sub one to not add empty haystack with overhead=0
                haystack = haystack[:-overhead] 
                datapoint = create_datapoint(haystack=haystack, needle=needle)
                # If we train, we also use shorter samples
                if train:
                    for __len in evaluation_context_lens:
                        if __len > _len and context_len_ds_counter[_len] > 0:
                            dataset_by_len[_len].append(datapoint)
                
                dataset_by_len[_len].append(datapoint)
                
                # Reset haystack
                haystack_by_len[_len] = []
                context_len_ds_counter[_len] -= 1
            else:
                # Case where haystack can simply be extended and we have another needle
                haystack_by_len[_len].append(hay["bio"])
    
    return dataset_by_len

def write(dataset_by_len, get_path, stop_at):
    for _len, _ds in dataset_by_len.items():
        with open(get_path(_len), "a+") as f:
            for elem in _ds[:stop_at]:
                f.write(json.dumps(elem) + "\n")

print("Creating datasets, this can take a while...")

# Create evaluation dataset
context_len_ds_counter = {_len: EVAL_DS_SIZE/NUM_ACTORS for _len in evaluation_context_lens}

dataset_by_lens_futures = [create_dataset.remote(eval_df, context_len_ds_counter, False) for i in range(NUM_ACTORS)]
dataset_by_lens = [ray.get(f) for f in dataset_by_lens_futures]
for dataset_by_len in dataset_by_lens:
    write(dataset_by_len, get_eval_path, int(EVAL_DS_SIZE/NUM_ACTORS))

# Create training dataset
context_len_ds_counter = {_len: int(TRAINING_DS_SIZE/NUM_ACTORS) for _len in training_context_lens}

dataset_by_lens_futures = [create_dataset.remote(train_df, context_len_ds_counter, True) for i in range(NUM_ACTORS)]
dataset_by_lens = [ray.get(f) for f in dataset_by_lens_futures]

for dataset_by_len in dataset_by_lens:
    write(dataset_by_len, get_train_path, int(TRAINING_DS_SIZE/NUM_ACTORS))