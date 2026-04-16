from pipeline import Pipeline
import os

if __name__ == "__main__":
    os.environ["HF_TOKEN"] = "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"
    pipeline = Pipeline()
    pipeline.train()