import os
import numpy as np
import json
import ast
import re
from pathlib import Path
import pandas as pd
from openai import OpenAI

from dataset.format import Results


MAX_RESPONSE_TOKENS = 300

RAY_LLM_API_BASE="http://localhost:8000/v1"
AE_API_BASE="https://api.endpoints.anyscale.com/v1"
AE_API_KEY=os.environ["AE_API_KEY"]
OPENAI_API_BASE="https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]

# Adapt this to wherever you created your dataset
TEST_HAYSTACK_BASE_PATH = Path("dataset")
test_haystack_path = TEST_HAYSTACK_BASE_PATH / "test_haystack.jsonl"
test_needles_path = TEST_HAYSTACK_BASE_PATH / "test_needles.jsonl"


def get_test_result_file(model_to_test, results_version):
    filename = "results/" + model_to_test + "/results" + str(results_version) + ".json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    return filename

def get_test_plot_file(model_to_test, results_version):
    filename = "results/" + model_to_test + "/results" + str(results_version) + ".png"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    return filename

def create_results_overview(model_to_test, results_version):
    results_file = get_test_result_file(model_to_test, results_version)
    results = pd.read_json(results_file)

    def unpack(row):
        row = pd.DataFrame(row)
        response = row[:][1]
        score = row[:][2]
        evaluation = row[:][3]
        return pd.concat({'Score': score, 'response': response, 'evaluation': evaluation}, axis=0)

    to_plot = results.evaluation_detail.apply(unpack)

    filename = "results/" + model_to_test + "/results_overview.html"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w") as f:
        to_plot.to_html(f)


def create_context(context_len):
    """Create a context of a given length.
    
    This is the haystack along with the needles to be injected later.
    We also keep track of and return the token boundaries so that we can insert the needles at the boundaries between biographies.
    """
    context = ""
    num_tokens = 0
    # Just use a single needle for now
    needles = []
    with open(test_needles_path, 'r') as f:
        for needle in f.readlines():
            needles.append(json.loads(needle))
    
    token_bio_boundaries = [0]
    with open(test_haystack_path, 'r') as f:
        for entry in f.readlines():
            entry = json.loads(entry)
            context += entry["bio"]
            num_tokens += entry["bio_token_len"]
            token_bio_boundaries += [num_tokens]

            if num_tokens >= context_len:
                break

    return context, np.array(token_bio_boundaries), needles

def ask_ae(messages, model_to_test):
    client = OpenAI(
        base_url=AE_API_BASE,
        api_key=AE_API_KEY,
    )
    breakpoint()
    chat_completion = client.chat.completions.create(
        model=model_to_test,
        max_tokens=MAX_RESPONSE_TOKENS,
        messages=messages,
        temperature=0.0,
    )
    breakpoint()
    answer = chat_completion.choices[0].message.content
    
    return answer

def ask_mixtral(messages):
    schema_json = Results.schema_json()

    client = OpenAI(
        base_url=AE_API_BASE,
        api_key=AE_API_KEY,
    )

    chat_completion = client.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        max_tokens=MAX_RESPONSE_TOKENS,
        messages=messages,
        temperature=0.0,
        response_format={
            "type": "json_object",
            "schema": schema_json,
        }
    )
    answer = chat_completion.choices[0].message.content

    try:
        json.loads(answer)
    except:
        breakpoint()
    
    # Mixtral tends to introduce leading zeros to integer numbers here, which breaks json loading.
    answer = re.sub(r'\b0+(\d+)', r'\1', str(answer))
    
    try:
        answer = str(json.loads(answer))
    except json.decoder.JSONDecodeError as e:
        print(f"{answer=}")
        raise e
    
    return answer

def ask_openai(messages, model):
    schema_json = Results.schema_json()
    messages[0]["content"] += f" Obey the following json format: {schema_json}. If a date is now known, set month, day and year to 0."

    client = OpenAI(
        api_key=OPENAI_API_KEY,
    )
   
    chat_completion = client.chat.completions.create(
        model=model,
        seed=1337,
        max_tokens=MAX_RESPONSE_TOKENS,
        messages=messages,
        temperature=0.0,
    )
    try:
        response = chat_completion.choices[0].message.content
        response = response.replace("false", "False")
        response = response.replace("true", "True")
        _dict = ast.literal_eval(response)
        if "properties" in _dict:
            answer = str(_dict["properties"])
        else:
            answer = str(_dict)
        return answer, chat_completion.system_fingerprint
    except Exception as e:
        # We can't force GPT to output JSON, but it should do so reliably.
        print(chat_completion)
        raise e
