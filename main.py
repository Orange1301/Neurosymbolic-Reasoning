from pipeline import Pipeline
from model.SFT import FOLModel
if __name__ == "__main__":
    model = FOLModel()
    while True:
        prompt = input("Enter your logical question: ")
        if prompt.lower() == "exit":
            break
        print(model.predict(prompt, num_outputs=1))