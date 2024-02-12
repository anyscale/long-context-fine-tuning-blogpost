"""
This file contains the utility needed to plot the aggregated results of the "Needle In A Haystack" benchmarking.
The aggregated results compare the accuracy of multiple models over different context lengths.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import get_test_result_file
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
import matplotlib.ticker as mtick
import seaborn as sns

MAX_EVALUATION_CONTEXT_LEN = 16384


MODELS_TO_PLOT = [
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Llama-2-13b-chat-hf-16k-context",
    "gpt-3.5-turbo-1106",
]

MODELS_TO_PLOT_NAME_MAP = {
    "meta-llama/Llama-2-13b-chat-hf-16k-context": "Llama 2 13b (16k)",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "Mixtral (not finetuned)",
    "gpt-3.5-turbo-1106": "GPT 3.5 Turbo (not finetuned)",
}


def plot_aggregated():
    results_map = {}
    for model_to_test in MODELS_TO_PLOT:
        results_file = get_test_result_file(model_to_test, 0)
        with open(results_file, 'r') as f:
            results = json.load(f)
        model = MODELS_TO_PLOT_NAME_MAP[model_to_test]
        results_map[model] = results
    
    data = {}
    for model_id, results in results_map.items():
        context_len_scores = {}
        for result in results:
            context_length = result["context_length"]
            score = result["score"]
            if not context_length in context_len_scores:
                context_len_scores[context_length] = []
            context_len_scores[context_length].append(score)
        data[model_id] = context_len_scores
        
    # Calculate averages
    for model_id, context_len_scores in data.items():
        for context_len, scores in context_len_scores.items():
            data[model_id][context_len] = np.mean(scores) * 100
    
    df = pd.DataFrame.from_dict(data)
    
    sns.set(style="white", palette=sns.color_palette("hls", len(MODELS_TO_PLOT)))
    sns.lineplot(data=df, markers=["o"] * len(MODELS_TO_PLOT))
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # This part of the styling has to be fitted per plot to the number of models
    # Start of per plot styling
    lines = ax.get_lines()
    styles = {
        'opensource1': {'dashes': [], 'marker': 's'},
        'opensource2': {'dashes': [], 'marker': 's'},
        'gpt1': {'dashes': [1], 'marker': 'o'},
        'gpt2': {'dashes': [1], 'marker': 'o'},
    }

    for line, (category, style) in zip(lines, styles.items()):
        line.set_dashes(style['dashes'])
        line.set_marker(style['marker'])
    # End of per plot styling

    plt.title(f'Model accuracy over context lengths', fontsize=16)
    plt.legend(title="Model (fine-tuning context length)", fontsize=13)
    plt.xlabel('Context length', fontsize=13)
    plt.ylabel('Average accuracy', fontsize=13)
    plt.xticks(np.arange(4096, MAX_EVALUATION_CONTEXT_LEN + 1, 4096))
    plt.ylim(0, 110)
    plt.yticks(np.arange(0, 101, 20))
    
    plt.tight_layout()

    plt.savefig("results/aggregated_results.png", format='png')

if __name__ == "__main__":
    plot_aggregated()
