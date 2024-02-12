"""
This file contains the utility needed to plot the results. The original version of this code can be found here: https://github.com/gkamradt/LLMTest_NeedleInAHaystack/.
The output is styled to look like the ones from the original benchmark from Greg Kamrady (https://twitter.com/GregKamradt/status/1727018183608193393), 
with some modifications such as adding mean values.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import get_test_result_file, get_test_plot_file
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import json

model_to_test_map = {
    "meta-llama/Llama-2-13b-chat-hf": "Llama 2 13b (fine-tuned)",
    "gpt-3.5-turbo-1106": "GPT 3.5 Turbo (not fine-tuned)",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "Mixtral (not finetuned)"
}


def plot(model_to_test, result_version, context_lengths, depth_steps, document_depth_percents):
    # Main plot
    test_result_file = get_test_result_file(model_to_test, result_version)

    data = []

    with open(test_result_file, 'r') as f:
        results = json.load(f)

    scores = []
    for result in results:
        depth_percent = result["depth_percent"]
        score = result["score"]
        context_length = result["context_length"]
        data.append({
            "Needle Depth":  np.where(document_depth_percents == depth_percent)[0][0] / 10,
            "Context Length": context_length,
            "Score": score
        })
        scores.append(score)

    df = pd.DataFrame(data)

    pivot_table = pd.pivot_table(df, values='Score', index=['Needle Depth', 'Context Length'], aggfunc='mean').reset_index() # This will aggregate
    pivot_table = pivot_table.pivot(index="Needle Depth", columns="Context Length", values="Score") # This will turn into a proper pivot
    pivot_table.iloc[:5, :5]

    for idx in pivot_table.keys().to_list():
        mean = pivot_table[idx].mean()
        pivot_table = pivot_table.rename(columns={idx: str(idx) + f"\n(mean={np.round(mean, 2)})"})

    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    plt.figure(figsize=(17.5, 8))
    heatmap = sns.heatmap(
        pivot_table,
        vmin=0.0,
        vmax=1.0,
        fmt="g",
        cmap=cmap,
    )

    cbar = heatmap.collections[0].colorbar
    # Set the label size
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(label=f'Score (mean={np.round(np.mean(scores), 2)})', fontsize=18)

    # More aesthetics
    plt.title(f'Retrieval Accuracy over Context Length with {model_to_test_map[model_to_test]}', fontsize=22)  # Adds a title
    plt.xlabel('Context Length', fontsize=18)  # X-axis label
    plt.ylabel('Needle Depth', fontsize=18)  # Y-axis label
    plt.xticks(rotation=0)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.tight_layout()  # Fits everything neatly into the figure area

    # Show the plot
    plt.savefig(get_test_plot_file(model_to_test, result_version), format='png')
