"""
This file is the entrypoint to benchmarking with our biography dataset.
It contains the main configurable settings as well as the glue code to run the benchmarking.
"""

from transformers import AutoTokenizer
from dataset.format import get_messages
from flatten_dict import flatten
import json
import ray
import numpy as np

from utils import get_test_result_file, ask_ae, ask_openai, ask_mixtral, create_context, create_results_overview
from plot_haystack import plot

# We keep track of token lengths with the llama 2 tokenizer. We specifically don't use other tokenizers so that all models get the same prompts.
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

results_version = 0
model_to_test = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# model_to_test = "meta-llama/Llama-2-13b-chat-hf" 
# model_to_test = "gpt-3.5-turbo-1106"
# model_to_test = "AE:<Your Anyscale Endpoints fine-tuned model ID here>"

DEPTH_STEPS = 10
NUM_NEEDLES = 10

# This is where we configure how we want to run this test in steps of our minimum context length (4k)
context_lengths = np.arange(4096, 16384+1, 4096) # 8192, 16384, 4096, 32768, 65536, 131072

# This will product a list of document depths to place your random statement (needle) at.
# Suggestion: Try out different distributions (like a sigmoid) to test non-evenly space intervals
document_depth_percents = np.round(np.linspace(0, 100, num=DEPTH_STEPS, endpoint=True)).astype(int)


env_vars = {
        "HF_HOME": "/mnt/local_storage/.cache/huggingface",
        # Put you HF token here!
        "HUGGING_FACE_HUB_TOKEN": "",
    }

ray.init(
    runtime_env={
        "env_vars": env_vars,
    }
)


# Insert a needle
def insert_needle(needle, context, token_bio_boundaries, depth_percent, context_length, tokenizer):
    # We should no add special tokens here, they will already be added later on.
    tokens_needle = tokenizer(needle["bio"], add_special_tokens=False)["input_ids"]
    tokens_context = tokenizer(context, add_special_tokens=False)["input_ids"]

    # Reducing the context length by 500 buffer. This is to account for system message, the user question, and response.
    context_length = context_length - 500

    # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
    if len(tokens_context) + len(tokens_needle) > context_length:
        tokens_context = tokens_context[:context_length - len(tokens_needle)]

    if depth_percent == 100:
        # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
        tokens_new_context = tokens_context + tokens_needle
    else:
        # Go get the position (in terms of tokens) to insert your needle
        insertion_point = int(len(tokens_context) * (depth_percent / 100))

        # Then we iteration backwards until we find the first period
        boundary_idx = np.abs(token_bio_boundaries - insertion_point).argmin()
        insertion_point_boundary = token_bio_boundaries[boundary_idx]

        # tokens_new_context represents the tokens before the needle
        tokens_new_context = tokens_context[:insertion_point_boundary + 1] + tokens_needle + tokens_context[insertion_point_boundary + 1:]

    # Convert back to a string and return it
    new_context = tokenizer.decode(tokens_new_context)
    
    return new_context

# Go through the whole insertion process once
def generate_haystacks_with_needles(context_length, depth_percent):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")

    # Get your Paul Graham files loaded into a string
    context, token_bio_boundaries, needles = create_context(context_length)

    haystack_with_needles = []
    for needle in needles:
        # Insert your random statement according to your depth percent
        haystack_with_needles.append((insert_needle(needle, context, token_bio_boundaries, depth_percent, context_length=context_length, tokenizer=tokenizer), needle))

    return haystack_with_needles


def evaluate_json_response(response, needle):
    try:
        dict_response = eval(response)
    except Exception as e:
        print(e)
        return 0, ""
    golden_dict = needle["json"]
    set_dict_response = set(flatten(dict_response).items())
    set_golden_dict = set(flatten(golden_dict).items())
    difference = set_dict_response ^ set_golden_dict
    # Return the scrore as the percentage of overlapping values from the model response and golden dict
    return (len(set_dict_response) - len(difference)/2)/len(set_dict_response), "Difference " + str(difference)

def result_exists(results, context_length, depth_percent, version, model):
    """
    Checks to see if a result has already been evaluated or not
    """
    conditions_met = []
    for result in results:
        context_length_met = result['context_length'] == context_length
        depth_percent_met = result['depth_percent'] == depth_percent
        version_met = result.get('version', 1) == version
        model_met = result['model'] == model
        conditions_met.append(context_length_met and depth_percent_met and version_met)
    return any(conditions_met)

def test(model_to_test_description, results_version, model=None):
    # Run through each iteration of context_lengths and depths
    openai_fingerprint = -1
    experiment_file_name = get_test_result_file(model_to_test_description, results_version)

    for context_length in context_lengths:
        for depth_percent in document_depth_percents:
            # Load results from file. 
            try:
                with open(experiment_file_name, 'r') as f:
                    results = json.load(f)
            except FileNotFoundError:
                results = []
                pass

            # Checks to see if you've already checked a length/percent/version.
            # This helps if the program stops running and you want to restart later
            if result_exists(results, context_length, depth_percent, results_version, model_to_test_description):
                continue

            # Go generate the required length context and place your needle statement in
            haystacks_with_needles = generate_haystacks_with_needles(context_length, depth_percent)
            scores = []
            # Text the first NUM_NEEDLES haystacks and needles we have created
            evaluation_detail = []
            for context, needle in haystacks_with_needles[:NUM_NEEDLES]:

                messages = get_messages(context, needle["name"])
                if model_to_test_description.startswith("AE:"):
                    response = ask_ae(messages, model_to_test_description[3:])
                elif "gpt" in model_to_test_description:
                    response, openai_fingerprint = ask_openai(messages, model_to_test_description)
                elif model_to_test_description == "mistralai/Mixtral-8x7B-Instruct-v0.1":
                    response = ask_mixtral(messages)
                else:
                    response = ask_ray_llm(messages, model_to_test_description)

                # Compare the reponse to the actual needle you placed
                score, evaluation = evaluate_json_response(response, needle)
                evaluation_detail.append((messages, response, score, evaluation))
                if score is not None:
                    scores.append(score)
            
            score = np.mean(scores)

            results.append({
                # 'context' : context, # Uncomment this line if you'd like to save the context the model was asked to retrieve from. Warning: This will become very large.
                'model' : model_to_test_description,
                'context_length' : int(context_length),
                'depth_percent' : int(depth_percent),
                'version' : results_version,
                'evaluation_detail' : evaluation_detail,
                'score' : score,
                'fingerprint': openai_fingerprint,
            })

            print (f"Result #: {len(results)}/{len(context_lengths) * len(document_depth_percents)}")
            print (f"Context: {context_length} tokens")
            print (f"Depth: {depth_percent}%")
            print (f"Score: {score}")

            # Save results to a JSON file each run
            with open(experiment_file_name, 'w') as f:
                json.dump(results, f)

test(model_to_test, results_version)
create_results_overview(model_to_test, results_version)
plot(model_to_test, results_version, context_lengths=context_lengths, depth_steps=DEPTH_STEPS, document_depth_percents=document_depth_percents)