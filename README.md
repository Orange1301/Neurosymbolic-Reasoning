# Neurosymbolic-Reasoning

## Overview 
This project combines LLMs with a symbolic engine to build a reliable reasoning system.
Specifically, this project builds system to determine whether a `conclusion` is determined as True/False/Uncertain under the given `premises` with logic reasoning. The premises and conclusion are all written in English.

## Method
We evaluated 2 approaches:
1. Direct approach: One-shot prompting the models to perform reasoning and predict the label.
2. First-Order-Logic (FOL) related: Fine-tuning/One-shot prompting the models to translate the premises and conclusion into FOL language, then using Prover9 (a logic prover) to determine the label based on the FOL translation. In this approach, we have 2 sub-approaches:
- FOL-Base: Just simply train the model on a small FOLIO dataset.
- FOL-Loop: Continue training the model on other large datasets, using our training loop pipeline.

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
├── prover/
│   ├── prover.py           # Code for using Prover9 to infer labels from premises and conclusion in FOL language
│   └── mock_data.json      # Data that needs predicting labels using prover
├── models/
│   ├── mt5/
│   │   ├── mt5_direct.ipynb    # Notebook for using mT5 with Direct approach               
│   │   └── mt5_fol_base.ipynb  # Notebook for training and testing mT5 with FOL-Base approach
│   ├── qwen7b/
│   │   ├── qwen_direct.ipynb           # Notebook for using DeepSeek-distill-Qwen-7B with Direct approach      
│   │   ├── qwen_fol_base_train.ipynb   # Notebook for training DeepSeek-distill-Qwen-7B with FOL-Base approach 
│   │   ├── qwen_fol_base_predict.ipynb # Notebook for using DeepSeek-distill-Qwen-7B to infer with FOL-related approach        
│   │   └── qwen_train_pipeline.ipynb   # Notebook performing training loop for DeepSeek-distill-Qwen-7B
│   └── gemini/
│       ├── results/            # Inference results
│       │   └── *.json
│       ├── inference.py        # Main file for calling Gemini API for inference 
│       ├── test_data.json      # File containing testing data                 
│       ├── predictions.json    # File containing Gemini's output
│       ├── utils.ipynb         # Notebook for re-formatting test data and compute testing accuracy
│       └── .env                # File containing Gemini API key
├── analysis/
│   ├── analysis.ipynb      # Notebook for analyzing the testing result
│   └── predictions.json    # The testing predictions result
├── requirements.txt        # Neccessary dependencies
└── README.md
```

Note that the model notebooks are designed to run on Kaggle.

## Kaggle notebooks
We have run the model notebooks mentioned above on Kaggle, with 
1. mT5-base:
- Testing the Direct approach (`mT5_direct.ipynb`): https://www.kaggle.com/code/trnlnhtho/mt5-basic1
- Training and Testing the FOL-Base approach (`mT5_fol_base`): https://www.kaggle.com/code/trnlnhtho/mt5-basic2

2. DeepSeek-distill-Qwen-7B:
- Testing the Direct approach (`qwen_direct.ipynb`): https://www.kaggle.com/code/ductri0981/natural-label
- Training with the FOL-Base approach (`qwen_fol_base_train.ipynb`): https://www.kaggle.com/code/ductri0981/sft-fol
- Training pipeline (`qwen_train_pipeline.ipynb`): https://www.kaggle.com/code/minhkhiphan/new-pipeline
- Testing with the FOL-related approach (`qwen_fol_base_predict.ipynb`): https://www.kaggle.com/code/ductri0981/natural-fol-label

## Authors
- [Huynh Gia Bao](https://github.com/baohg153)
- [Nguyen Ngoc Canh](https://github.com/Orange1301)
- [Phan Huynh Minh Khoi](https://github.com/phmkhoi)
- [Nguyen Duc Tri](https://github.com/TriNguyen1208)
- [Tran Ly Nhat Hao](https://github.com/tranlynhathao)