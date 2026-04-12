from model.SFT import FOLModel
from filter.DataFilter import DataFilter
from filter.filter2 import filter2
from engine.engine import Engine
import json

class Pipeline:
    def __init__(self):
        self.fol_model = FOLModel()
        self.filter_1 = DataFilter()
        self.engine = Engine()
    
    def get_data(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        return dataset
    
    def format_predict_dataset(self, dataset):
        dataset_formated_predicted = []

        for data in dataset:
            premises = data["nat_premises"]
            conclusion = data["nat_conclusion"]

            obj = {}
            obj["story_id"] = data["story_id"]
            obj["natural"] = premises + conclusion
            fol_list_predictions = self.fol_model.predict(obj["natural"])
            obj["predictions"] = [
                {"fol": fol, "label": None}
                for fol in fol_list_predictions
            ]
            obj["label"] = data["label"]

            dataset_formated_predicted.append(obj)

        return dataset_formated_predicted

    def format_finetune_dataset(self, dataset_formated_predicted):        
        dataset_formated_predicted = self.filter_1.filter_list(dataset_formated_predicted)
        dataset_fine_tune = []

        for fol in dataset_formated_predicted:
            fol = self.engine.check_conclusion(fol)
            result_filter2 = filter2(fol['predictions'], fol["label"])
            sentences = [s.strip() for s in fol["natural"].split('.') if s.strip()]
            if result_filter2 is not None:
                positive_fol, _ = result_filter2
                dataset = [
                    {
                        "story_id": fol["story_id"],
                        "nat_premises": '.'.join(sentences[:-1]),
                        "nat_conclusion": sentences[-1],
                        "fol_premises": '\n'.join(data['fol'].split('\n')[:-1]),
                        "fol_conclusion": data['fol'].split('\n')[-1],
                    }
                    for data in positive_fol
                ]
                dataset_fine_tune.extend(dataset)
        return dataset_fine_tune
    
    def train(self, dataset_train_path="example.json", dataset_valid_path="example_valid.json", num_loops=1):
        dataset_train = self.get_data(dataset_train_path)
        dataset_valid = self.get_data(dataset_valid_path)

        for _ in range(num_loops):
            dataset_train_formated_predicted = self.format_predict_dataset(dataset_train)
            dataset_valid_formated_predicted = self.format_predict_dataset(dataset_valid)

            dataset_train = self.format_finetune_dataset(dataset_train_formated_predicted)
            dataset_valid = self.format_finetune_dataset(dataset_valid_formated_predicted)

            self.fol_model.train(dataset_train, dataset_valid)