import os
from model.SFT import FOLModel

if __name__ == "__main__":
    os.environ["HF_TOKEN"] = "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"
    fol_model = FOLModel()
    fol_model.load_finetune_model("/kaggle/input/models/ductri0981/fol-model/transformers/default/1")