# Neurosymbolic-Reasoning

## Overview 
This project combines LLMs with a symbolic engine to build a reliable reasoning system.
Specifically, this project builds system to determine whether a `conclusion` is determined as True/False/Uncertain under the given `premises` with logic reasoning. The premises and conclusion are all written in English.

## Method
We evaluated 2 approaches:
1. Direct approach: One-shot prompting the models to perform reasoning and predict the label.
2. First-Order-Logic (FOL) related: Fine-tuning/One-shot prompting the models to translate the premises and conclusion into FOL language, then using Prover9 (a logic prover) to determine the label based on the FOL translation.

We tested the 2 approaches on 3 different models: 
- mT5-base
- DeepSeek-distill-Qwen-7B
- Gemini-2.5-Flash

This source code contains code for implementing and evaluating these approaches on these models.

## Project structure
```
NEUROSYMBOLIC-REASONING/
├── dataset/
│   ├── train/              # Training sets
│   ├── valid/              # Custom validation set
│   └── test/               # Custom test set
├── gemini/
│   ├── results/            # Inference results
│   │   └── *.json
│   ├── inference.py        # Main file for calling Gemini API for inference 
│   ├── test_data.json      # File containing testing data                 
│   ├── predictions.json    # File containing Gemini's output
│   ├── utils.ipynb         # Notebook for re-formatting test data and compute testing accuracy
│   └── .env                # File containing Gemini API key
├── prover/
│   ├── prover.py           # Code for using Prover9 to infer labels from premises and conclusion in FOL language
│   └── mock_data.json      # Data that needs predicting labels using prover
├── analysis/
│   ├── analysis.ipynb      # Notebook for analyzing the testing result
│   └── predictions.json    # The testing predictions result
├── notebooks/
│   ├── 
│   └── 
├── requirements.txt        # Neccessary dependencies
└── README.md
```

## Kaggle notebooks
