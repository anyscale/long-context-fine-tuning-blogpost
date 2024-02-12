# Needle in a Haystack - Biographies Benchmark

This repo is for benchmarking LLM's ability to extract small bits of information from long context.
We adapted the benchmark from Greg Kamradt's original Needle in a Haystack Benchmark to our preferences.

[Original benchmark](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)
[Original tweet](https://twitter.com/GregKamradt/status/1722386725635580292)

Aside from the benchmarking code, we also create a dataset to fine-tune for the task at hand.

## The original benchmark

In the original "Needle in a Haystack" benchmark, we extact a small bit of information, called "the needle", from a large context.
The large context, called "the haystack", are concatenated esseys by Paul Graham.
The following text ("needle") is inserted at varying positions into these esseys varying positions: "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.".
Note that the information from the esseys and the needle are not related much. Therefore, it might be easier for a model to single out information about the needle in the haystack.
The model is then given the haystack with the needle and asked "What is the best thing to do in San Francisco?" and _not_ to "give information outside the document or repeat your findings".
Note that eating a sandwich and sitting in Dolores Park on a sunny day is understood to be a good answer to the posed question based on general knowledge outside the context. We therefore expect models trained on large chunks of publicly available data to be preconditioned to output such information.

## Notable changes to the original benchmark

This benchmark makes the following changes:

* Based on the [wiki_bio](https://huggingface.co/datasets/wiki_bio) dataset.
    * We randomize all names so that model's can not rely on previously learnt information.
    * While the needle is a random biography, the haystack is a concatenation of equally random biographies.
        * The intuition behind the design choice is that information in the haystack should be similar to the needle so that the benchmark gives us a better understanding of how well-suited the model is to index large chunks of similar data.
* Model must extract multiple small bits of information:
    * Date of birth
    * Date of death
    * Nationality
    * Whether or not the person in question is/was a sportsperson
    * Whether or not the person in question is/was a politician
* Structure model's outputs as json or dictionaries
    * This breaks down the complexity of evaluation and makes it more reliable.
    * It also reduces the cost of the benchmark.

# Usage

You can use this repo by going through the following steps:
1. Create synthetic datasets with the tools under data/
2. Maybe fine-tune
3. Benchmark and plot with bio_haystack_benchmark.py