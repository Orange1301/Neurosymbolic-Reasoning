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
        '''
        With the input dataset, model FOL will predict the result (num_outputs FOL)
        
        Parameters
        ----------
        - dataset : list[dict]
            dataset contains natural language
            
        Returns
        -------
        - list[dict]
            dataset with predicted FOL
        Example: 
        ------- 
        - input:
            [
                {
                    "story_id": "reclor_train_0",
                    "nat_premises": "In rheumatoid arthritis, the body's immune system misfunctions by attacking healthy cells in the joints causing the release of a hormone that in turn causes pain and swelling. This hormone is normally activated only in reaction to injury or infection. A new arthritis medication will contain a protein that inhibits the functioning of the hormone that causes pain and swelling in the joints.",
                    "fol_premises": null,
                    "nat_conclusion": "A patient treated with the new medication for rheumatoid arthritis could sustain a joint injury without becoming aware of it.",
                    "fol_conclusion": null,
                    "label": "True"
                },
                ....
            ]
        - output:
            [
                {
                    "story_id": "reclor_train_0",
                    "natural": "In rheumatoid arthritis, the body's immune system misfunctions by attacking healthy cells in the joints causing the release of a hormone that in turn causes pain and swelling. This hormone is normally activated only in reaction to injury or infection. A new arthritis medication will contain a protein that inhibits the functioning of the hormone that causes pain and swelling in the joints.A patient treated with the new medication for rheumatoid arthritis could sustain a joint injury without becoming aware of it.",
                    "predictions": [
                        {
                            "fol": "∀x ∀y (In(x, rheumatoidArthritis) ∧ Attack(x, y) ∧ In(y, immuneSystem) ∧ ¬HealthyCell(y) ∧ In(y, joints) ∧ In(y, body) ∧ Causes(y, releaseOfHormone) ∧ Causes(y, pain) ∧ Causes(y, swelling))\n∀x (ReleaseOfHormone(x) ↔ (OnlyInResponseTo(x, injury) ∨ OnlyInResponseTo(x, infection)))\n∀x (In(x, newMedication) ∧ Contains(x, protein) ∧ Inhibits(x, functioningOf, [",
                            "label": None
                        },
                        {
                            "fol": "∀x ∀y (Arthritis(x) ∧ InRheumatoid(x) ∧ Misfunctioning(x, y) ∧ Attack(y, healthyCells) ∧ In(y, joints) ∧ Release(y, painHormone) ∧ (Pain(x) ∨ Swelling(x)))\n∀x (Hormone(y) ∧ Inhibits(y, painHormone) ∧ ActivatedOnlyInReactionTo(y, injury) ∨ ActivatedOnlyInReactionTo(y, infection))\n∀x (Arthritis(x) ∧ In(x, newMedication) ∧ InhibitsProtein(y) ∧ InhibitsProtein(y, hormone) ∧ InhibitsProtein(y, painHormone) ∧ InhibitsProtein(y, joints))\n∀x (Arthritis(x) ∧ In(x, newMedication) ∧ InhibitsProtein(y) ∧ InhibitsProtein(y, painHormone) ∧ InhibitsProtein(y, joints) ∧ Injurious(y) ∧ In(y, joints) ∧ (¬Pain(x) ∧ ¬Swelling(x)))",
                            "label": None
                        },
                        {
                            "fol": "∀x ∀y ∀z ((In(x, y) ∧ (y = rheumatoidArthritis) ∧ (Body(x) ∧ In(x, y) ∧ (¬Healthy(z) ∧ In(z, x) ∧ In(z, y) ∧ In(z, joints)))) → (ReleaseOf(x, hormone) ∧ (Pain(x) ∧ Swelling(x))))\n∀x ((ReleaseOf(x, hormone) ∧ (In(x, injury) ∨ In(x, infection))) → Activated(x))\n∀x ((In(x, medication) ∧ (New(x, medication) ∧ (In(x, newArthritisMedication)) ∧ (Inhibits(x, protein) ∧ Inhibits(x, inhibitsHormone)) ∧ (Inhibits(x, hormone) ∧ Inhibits(x, causesPainAndSwellingInTheJoints))))\n∀x ((In(x, patient) ∧ (In(x, newMedication) ∧ (In(x, treatedFor) ∧ (In(x, rheumatoidArthritis)))) ∧ (In(x, injury) ∧ In(x, without))) → (AwareOf(x)))",
                            "label": None
                        },
                        {
                            "fol": "∀x ∀y ((Body(x) ∧ IsIn(rheumatoidArthritis(x)) ∧ Misfunction(x, immuneSystem) ∧ Attack(x, healthyCells) ∧ In(x, joints)) → In(y, hormone) ∧ Caused(y, pain) ∧ Caused(y, swelling))\n∀x (In(x, hormone) → (Activated(x) ↔ (Injured(x) ∨ Infected(x))))\n∀x ∀y ((Body(x) ∧ In(x, newArthritisMedication) ∧ Inhibits(x, hormone) ∧ In(y, joint) ∧ In(y, injury)) → ¬AwareOf(y))\n∀x ((Body(x) ∧ In(x, treatedWithNewMedication) ∧ In(x, rheumatoidArthritis)) → (In(y, joint) ∧ Injury(y) → ¬AwareOf(y)))",
                            "label": None
                        },
                        {
                            "fol": "∀x ∀y (Body(x) ∧ In(x, immuneSystem) ∧ Misfunction(x) ∧ Attack(x, healthyCellsInTheJoints) ∧ Release(x, theHormone) ∧ Causing(x, pain) ∧ Causing(x, swelling))\n∀x ∀y (Body(x) ∧ In(x, immuneSystem) ∧ Misfunction(x) ∧ Attack(x, healthyCellsInTheJoints) ∧ Release(x, theHormone) ∧ ActivatedOnlyInReactionTo(x, injury) ∧ ActivatedOnlyInReactionTo(x, infection))\n∀x ∀y (Body(x) ∧ Contains(x, theNewMedication) ∧ Inhibits(x, theHormone) ∧ Causing(y, pain) ∧ Causing(y, swelling) ∧ In(y, theJoints))\n∀x (Body(x) ∧ In(x, thePatient) ∧ In(x, theNewMedication) ∧ Inhibits(x, theHormone) ∧ Attack(x, healthyCellsInTheJoints) ∧ Attack(x, healthyCellsInTheJoints) ∧ JointInjury(x) ∧ Not(x, awarenessOf(x, theInjury)))",
                            "label": None
                        },
                        {
                            "fol": "∀x ∀y (Body(x) ∧ Attack(x, y) ∧ (¬(x=y) ∧ Human(x) ∧ (¬(x=y) ∧ Human(y))) ∧ (¬(x=y) ∧ Joint(x) ∧ (¬(x=y) ∧ Joint(y))) ∧ Misfunction(x, immuneSystem) ∧ Inhibit(x, healthyCells) ∧ (¬(x=y) ∧ Cause(x, releaseOfHormone) ∧ (¬(x=y) ∧ Cause(x, pain)) ∧ (¬(x=y) ∧ Cause(x, swelling)))",
                            "label": None
                        },
                        {
                            "fol": "∀x (In(x, rheumatoidArthritis) → ∃y ∃z (In(x, y) ∧ In(y, z) ∧ Attack(z, x) ∧ In(x, z) ∧ ∃y (In(y, joints) ∧ Attack(x, y)) ∧ In(hormone, y) ∧ In(hormone, z) ∧ Cause(hormone, pain) ∧ Cause(hormone, swelling)))\n∀x (In(x, hormone) ∧ In(hormone, pain) ∧ In(hormone, swelling) → ActivatedInResponseTo(x, injury) ∨ ActivatedInResponseTo(x, infection))\n∀x (In(x, newArthritisMedication) ∧ In(hormone, x) ∧ In(hormone, x) → Inhibits(hormone, x))\n∀x (In(x, treatedWith) ∧ In(newArthritisMedication, x) ∧ In(rheumatoidArthritis, x) → ∃y (In(y, jointInjury) ∧ In(y, x) ∧ ∃y (In(y, pain) ∧ In(y, x)) ∨ ∃y (In(y, swelling) ∧ In(y, x)) → ¬AwareOf(y)))",
                            "label": None
                        },
                        {
                            "∀x ∀y ((Body(x, immuneSystem) ∧ Misfunction(x) ∧ Attack(x, y) ∧ (¬Healthy(y) ∨ Joint)) → ((Release(x) ∧ Cause(x, pain) ∧ Cause(x, swelling))))\n∀x ((Hormone(x) ∧ ActivatedOnly(x, injury) ∧ ActivatedOnly(x, infection)) → ¬(Release(x)))\n∀x (Medication(x) ∧ New(x) ∧ Arthritis(x) ∧ Contains(x, protein) → InhibitFunction(x, hormone))\n∀x ((Patient(x) ∧ TreatedWith(x, newMedication) ∧ Arthritis(x)) → InhibitFunction(hormone, x) ∨ ((¬Injury(y) ∨ Joint) → (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∧ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∧ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (",
                            "label": None
                        },
                        {
                            "fol": "∀x (In(x, rheumatoidArthritis) → (¬Function(x, immuneSystem) ∨ Function(x, healthyCells) ∨ Attack(x, healthyCells) ∨ CausedBy(x, releaseOfHormone) ∨ Cause(x, pain) ∨ Cause(x, swelling)))\n∀x (In(x, activatedOnlyInResponseToInjuryOrInfection) ↔ Inhibits(x, releaseOfHormone))\n∀x (In(x, newMedication) → Inhibits(x, protein))\n∀x ((In(x, treatedWithNewMedication) ∧ In(x, rheumatoidArthritis)) → (¬Function(x, jointInjury) ∨ ¬Function(x, injuredInJoints)))",
                            "label": None
                        },
                        {
                            "fol": "∀x ∀y (RheumatoidArthritis(x) ∧ BodyImmuneSystemMisfunctions(x) ∧ AttackHealthyCells(x) ∧ In(y) ∧ CausePain(y) ∧ CauseSwelling(y) ∧ In(y, j) ∧ In(x, j) ∧ CausePain(y) ∧ CauseSwelling(y))\n∀x ∀y (Hormone(y) ∧ NormallyActivatedOnlyInResponseTo(y, injury) ∨ (y, infection))\n∀x ∀y (ArthritisDrug(y) ∧ ContainsProtein(x, y) ∧ Inhibits(x, hormone) ∧ Inhibits(hormone, functioning) ∧ CausePain(y) ∧ CauseSwelling(y))\n∀x (RheumatoidArthritis(x) ∧ TreatWith(x, newMedication) ∧ In(y) ∧ CausePain(y) ∧ CauseSwelling(y) ∧ In(y, j) ∧ In(x, j) ∧ ¬AwareOf(y)"
                            "label": None
                        }
                    ]
                    "label": "True"
                },
                ....
            ]
        '''
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
        '''
        With the input of result that is predicted from FOL model.
        
        Parameters
        ----------
        - dataset_formated_predicted : list[dict]
            dataset contains natural language
            
        Returns
        -------
        - list[dict]
            dataset with predicted FOL
        Example: 
        ------- 
        - input:
            [
                {
                    "story_id": "reclor_train_0",
                    "natural": "In rheumatoid arthritis, the body's immune system misfunctions by attacking healthy cells in the joints causing the release of a hormone that in turn causes pain and swelling. This hormone is normally activated only in reaction to injury or infection. A new arthritis medication will contain a protein that inhibits the functioning of the hormone that causes pain and swelling in the joints.A patient treated with the new medication for rheumatoid arthritis could sustain a joint injury without becoming aware of it.",
                    "predictions": [
                        {
                            "fol": "∀x ∀y (In(x, rheumatoidArthritis) ∧ Attack(x, y) ∧ In(y, immuneSystem) ∧ ¬HealthyCell(y) ∧ In(y, joints) ∧ In(y, body) ∧ Causes(y, releaseOfHormone) ∧ Causes(y, pain) ∧ Causes(y, swelling))\n∀x (ReleaseOfHormone(x) ↔ (OnlyInResponseTo(x, injury) ∨ OnlyInResponseTo(x, infection)))\n∀x (In(x, newMedication) ∧ Contains(x, protein) ∧ Inhibits(x, functioningOf, [",
                            "label": None
                        },
                        {
                            "fol": "∀x ∀y (Arthritis(x) ∧ InRheumatoid(x) ∧ Misfunctioning(x, y) ∧ Attack(y, healthyCells) ∧ In(y, joints) ∧ Release(y, painHormone) ∧ (Pain(x) ∨ Swelling(x)))\n∀x (Hormone(y) ∧ Inhibits(y, painHormone) ∧ ActivatedOnlyInReactionTo(y, injury) ∨ ActivatedOnlyInReactionTo(y, infection))\n∀x (Arthritis(x) ∧ In(x, newMedication) ∧ InhibitsProtein(y) ∧ InhibitsProtein(y, hormone) ∧ InhibitsProtein(y, painHormone) ∧ InhibitsProtein(y, joints))\n∀x (Arthritis(x) ∧ In(x, newMedication) ∧ InhibitsProtein(y) ∧ InhibitsProtein(y, painHormone) ∧ InhibitsProtein(y, joints) ∧ Injurious(y) ∧ In(y, joints) ∧ (¬Pain(x) ∧ ¬Swelling(x)))",
                            "label": None
                        },
                        {
                            "fol": "∀x ∀y ∀z ((In(x, y) ∧ (y = rheumatoidArthritis) ∧ (Body(x) ∧ In(x, y) ∧ (¬Healthy(z) ∧ In(z, x) ∧ In(z, y) ∧ In(z, joints)))) → (ReleaseOf(x, hormone) ∧ (Pain(x) ∧ Swelling(x))))\n∀x ((ReleaseOf(x, hormone) ∧ (In(x, injury) ∨ In(x, infection))) → Activated(x))\n∀x ((In(x, medication) ∧ (New(x, medication) ∧ (In(x, newArthritisMedication)) ∧ (Inhibits(x, protein) ∧ Inhibits(x, inhibitsHormone)) ∧ (Inhibits(x, hormone) ∧ Inhibits(x, causesPainAndSwellingInTheJoints))))\n∀x ((In(x, patient) ∧ (In(x, newMedication) ∧ (In(x, treatedFor) ∧ (In(x, rheumatoidArthritis)))) ∧ (In(x, injury) ∧ In(x, without))) → (AwareOf(x)))",
                            "label": None
                        },
                        {
                            "fol": "∀x ∀y ((Body(x) ∧ IsIn(rheumatoidArthritis(x)) ∧ Misfunction(x, immuneSystem) ∧ Attack(x, healthyCells) ∧ In(x, joints)) → In(y, hormone) ∧ Caused(y, pain) ∧ Caused(y, swelling))\n∀x (In(x, hormone) → (Activated(x) ↔ (Injured(x) ∨ Infected(x))))\n∀x ∀y ((Body(x) ∧ In(x, newArthritisMedication) ∧ Inhibits(x, hormone) ∧ In(y, joint) ∧ In(y, injury)) → ¬AwareOf(y))\n∀x ((Body(x) ∧ In(x, treatedWithNewMedication) ∧ In(x, rheumatoidArthritis)) → (In(y, joint) ∧ Injury(y) → ¬AwareOf(y)))",
                            "label": None
                        },
                        {
                            "fol": "∀x ∀y (Body(x) ∧ In(x, immuneSystem) ∧ Misfunction(x) ∧ Attack(x, healthyCellsInTheJoints) ∧ Release(x, theHormone) ∧ Causing(x, pain) ∧ Causing(x, swelling))\n∀x ∀y (Body(x) ∧ In(x, immuneSystem) ∧ Misfunction(x) ∧ Attack(x, healthyCellsInTheJoints) ∧ Release(x, theHormone) ∧ ActivatedOnlyInReactionTo(x, injury) ∧ ActivatedOnlyInReactionTo(x, infection))\n∀x ∀y (Body(x) ∧ Contains(x, theNewMedication) ∧ Inhibits(x, theHormone) ∧ Causing(y, pain) ∧ Causing(y, swelling) ∧ In(y, theJoints))\n∀x (Body(x) ∧ In(x, thePatient) ∧ In(x, theNewMedication) ∧ Inhibits(x, theHormone) ∧ Attack(x, healthyCellsInTheJoints) ∧ Attack(x, healthyCellsInTheJoints) ∧ JointInjury(x) ∧ Not(x, awarenessOf(x, theInjury)))",
                            "label": None
                        },
                        {
                            "fol": "∀x ∀y (Body(x) ∧ Attack(x, y) ∧ (¬(x=y) ∧ Human(x) ∧ (¬(x=y) ∧ Human(y))) ∧ (¬(x=y) ∧ Joint(x) ∧ (¬(x=y) ∧ Joint(y))) ∧ Misfunction(x, immuneSystem) ∧ Inhibit(x, healthyCells) ∧ (¬(x=y) ∧ Cause(x, releaseOfHormone) ∧ (¬(x=y) ∧ Cause(x, pain)) ∧ (¬(x=y) ∧ Cause(x, swelling)))",
                            "label": None
                        },
                        {
                            "fol": "∀x (In(x, rheumatoidArthritis) → ∃y ∃z (In(x, y) ∧ In(y, z) ∧ Attack(z, x) ∧ In(x, z) ∧ ∃y (In(y, joints) ∧ Attack(x, y)) ∧ In(hormone, y) ∧ In(hormone, z) ∧ Cause(hormone, pain) ∧ Cause(hormone, swelling)))\n∀x (In(x, hormone) ∧ In(hormone, pain) ∧ In(hormone, swelling) → ActivatedInResponseTo(x, injury) ∨ ActivatedInResponseTo(x, infection))\n∀x (In(x, newArthritisMedication) ∧ In(hormone, x) ∧ In(hormone, x) → Inhibits(hormone, x))\n∀x (In(x, treatedWith) ∧ In(newArthritisMedication, x) ∧ In(rheumatoidArthritis, x) → ∃y (In(y, jointInjury) ∧ In(y, x) ∧ ∃y (In(y, pain) ∧ In(y, x)) ∨ ∃y (In(y, swelling) ∧ In(y, x)) → ¬AwareOf(y)))",
                            "label": None
                        },
                        {
                            "∀x ∀y ((Body(x, immuneSystem) ∧ Misfunction(x) ∧ Attack(x, y) ∧ (¬Healthy(y) ∨ Joint)) → ((Release(x) ∧ Cause(x, pain) ∧ Cause(x, swelling))))\n∀x ((Hormone(x) ∧ ActivatedOnly(x, injury) ∧ ActivatedOnly(x, infection)) → ¬(Release(x)))\n∀x (Medication(x) ∧ New(x) ∧ Arthritis(x) ∧ Contains(x, protein) → InhibitFunction(x, hormone))\n∀x ((Patient(x) ∧ TreatedWith(x, newMedication) ∧ Arthritis(x)) → InhibitFunction(hormone, x) ∨ ((¬Injury(y) ∨ Joint) → (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∧ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∧ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (¬AwareOf(x, y) ∨ (",
                            "label": None
                        },
                        {
                            "fol": "∀x (In(x, rheumatoidArthritis) → (¬Function(x, immuneSystem) ∨ Function(x, healthyCells) ∨ Attack(x, healthyCells) ∨ CausedBy(x, releaseOfHormone) ∨ Cause(x, pain) ∨ Cause(x, swelling)))\n∀x (In(x, activatedOnlyInResponseToInjuryOrInfection) ↔ Inhibits(x, releaseOfHormone))\n∀x (In(x, newMedication) → Inhibits(x, protein))\n∀x ((In(x, treatedWithNewMedication) ∧ In(x, rheumatoidArthritis)) → (¬Function(x, jointInjury) ∨ ¬Function(x, injuredInJoints)))",
                            "label": None
                        },
                        {
                            "fol": "∀x ∀y (RheumatoidArthritis(x) ∧ BodyImmuneSystemMisfunctions(x) ∧ AttackHealthyCells(x) ∧ In(y) ∧ CausePain(y) ∧ CauseSwelling(y) ∧ In(y, j) ∧ In(x, j) ∧ CausePain(y) ∧ CauseSwelling(y))\n∀x ∀y (Hormone(y) ∧ NormallyActivatedOnlyInResponseTo(y, injury) ∨ (y, infection))\n∀x ∀y (ArthritisDrug(y) ∧ ContainsProtein(x, y) ∧ Inhibits(x, hormone) ∧ Inhibits(hormone, functioning) ∧ CausePain(y) ∧ CauseSwelling(y))\n∀x (RheumatoidArthritis(x) ∧ TreatWith(x, newMedication) ∧ In(y) ∧ CausePain(y) ∧ CauseSwelling(y) ∧ In(y, j) ∧ In(x, j) ∧ ¬AwareOf(y)"
                            "label": None
                        }
                    ]
                    "label": "True"
                },
                ....
            ]
        - output:
            [
                {
                    "story_id": "reclor_train_0",
                    "nat_premises": "In rheumatoid arthritis, the body's immune system misfunctions by attacking healthy cells in the joints causing the release of a hormone that in turn causes pain and swelling. This hormone is normally activated only in reaction to injury or infection. A new arthritis medication will contain a protein that inhibits the functioning of the hormone that causes pain and swelling in the joints.",
                    "fol_premises":"∀x ∀y (RheumatoidArthritis(x) ∧ BodyImmuneSystemMisfunctions(x) ∧ AttackHealthyCells(x) ∧ In(y) ∧ CausePain(y) ∧ CauseSwelling(y) ∧ In(y, j) ∧ In(x, j) ∧ CausePain(y) ∧ CauseSwelling(y))\n∀x ∀y (Hormone(y) ∧ NormallyActivatedOnlyInResponseTo(y, injury) ∨ (y, infection))\n∀x ∀y (ArthritisDrug(y) ∧ ContainsProtein(x, y) ∧ Inhibits(x, hormone) ∧ Inhibits(hormone, functioning) ∧ CausePain(y) ∧ CauseSwelling(y))"
                    "nat_conclusion": "A patient treated with the new medication for rheumatoid arthritis could sustain a joint injury without becoming aware of it." 
                    "fol_conclusion": "∀x (RheumatoidArthritis(x) ∧ TreatWith(x, newMedication) ∧ In(y) ∧ CausePain(y) ∧ CauseSwelling(y) ∧ In(y, j) ∧ In(x, j) ∧ ¬AwareOf(y)",
                    "label": "True"
                },
                ....
            ]
        '''
        dataset_fine_tune = []

        for fol in dataset_formated_predicted:
            fol = self.filter_1.filter(fol)
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
                        "label": fol["label"]
                    }
                    for data in positive_fol
                ]
                dataset_fine_tune.extend(dataset)
        return dataset_fine_tune
    
    def train(self, dataset_train_path="example.json", dataset_valid_path="example_valid.json", num_loops=1):
        self.fol_model.load_finetune_model("/kaggle/input/models/ductri0981/fol-model/transformers/default/1")
        
        dataset_train = self.get_data(dataset_train_path)
        print("Dataset train: ",dataset_train)

        dataset_valid = self.get_data(dataset_valid_path)
        print("Dataset train: ",dataset_train)
        for _ in range(num_loops):
            dataset_train_formated_predicted = self.format_predict_dataset(dataset_train)

            print("Dataset_train_format_predict: ", dataset_train_formated_predicted)
            dataset_train = self.format_finetune_dataset(dataset_train_formated_predicted)

            print("Dataset_train_format_finetune: ", dataset_train)
            self.fol_model.train(dataset_train, dataset_valid)