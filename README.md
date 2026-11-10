# Revealing the Biases of Large Language Models by Inferring Prior Distributions

This repo contains the research code accompanying our paper on reconstructing latent prior distributions from LLM response probabilities. We formulate the reconstruction as a Bayesian inverse problem and use MAP estimation with Laplace approximation to quantify uncertainty.

Experiments investigate preferences over shift-cipher rotations and social biased on ambiguous examples from the Bias Benchmark for Question Answering (BBQ).

## Repository Contents

The main results from the paper come from `BBQExperiments`. `BBQExperiments/prepare_data.py` prepares ambiguous BBQ examples and maps the answer choices to the stereotype, anti-stereotype, and unknown categories. `BBQExperiments/score_models.py` computes the model response probabilities. `BBQExperiments/fit_priors.py` implements the algorithms from our paper to fit priors for each model and bias category along with uncertainty estimates.
