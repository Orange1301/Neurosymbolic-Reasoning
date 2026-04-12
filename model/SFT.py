import torch
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, PeftModel
from torch.nn import CrossEntropyLoss

class FOLModel:
    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        output_dir: str = "./fol_model",
        max_length: int = 768,
        use_lora: bool = True,
        use_qlora: bool = False,
        full_lora: bool = True,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_length = max_length
        self.use_lora = use_lora
        self.use_qlora = use_qlora
        self.full_lora = full_lora
        #Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        #Load base model.
        print("Loading model...")
        if self.use_qlora: 
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        
            # Load model with QLoRA
            print("Loading model with QLoRA...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map={"": 0},
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16,
                device_map={"": 0},
                trust_remote_code=True
            )
            
        if self.use_lora:
            self._apply_lora()
            
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.use_cache = False

    def load_finetune_model(self, path_adapter: str):
        self.model.load_adapter(path_adapter, adapter_name="default")
        print("Load adapter successfully")
        self.model.config.use_cache = False
        
    def _apply_lora(self):
        print("Applying LoRA...")
        target_modules=["q_proj", "v_proj"]
        if self.full_lora:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, config)
        self.model.print_trainable_parameters()

    def _prompt_template(self, data_example):
        premises_list = [
            s.strip() for s in data_example['nat_premises'].split('.') if s.strip()
        ]
        formatted_premises = "\n".join(
            [f"{i+1}. {s}." for i, s in enumerate(premises_list)]
        )
        formatted_all = formatted_premises + f"\n{len(premises_list)+1}. {data_example['nat_conclusion']}"
        prompt = f"""
### Instruction:
    Convert the following text into First-Order Logic (FOL).
    Identify all entities and the relationships between them.
    Resolve any coreferences before generating the logical form.
    Write each FOL formula on a separate line.

### Input:
{formatted_all}

### Output:
{data_example['fol_premises']}\n{data_example['fol_conclusion']}
"""
        return {"text": prompt}

    def _tokenize(self, data):
        prompt = data['text']
        tokenized = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        labels = tokenized["input_ids"].copy()
        # Only predict next token after "output prompt"
        output_start = prompt.index("### Output:")
        output_tokens = self.tokenizer(prompt[:output_start])["input_ids"]
        #Label from start to output tokens => mask with -100.
        labels[:len(output_tokens)] = [-100] * len(output_tokens)
        tokenized["labels"] = labels
        return tokenized
    
    def train(
        self,
        dataset_train,
        dataset_valid,
        epochs: int = 64,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        gradient_accumulation_steps=8 
    ):        
        print("Loading dataset...")
        dataset = {
            "train": Dataset.from_list(dataset_train),
            "valid": Dataset.from_list(dataset_valid)
        }

        dataset = dataset.map(self._prompt_template)
        dataset = dataset.map(self._tokenize)

        dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
    
            eval_strategy="epoch",
            save_strategy="epoch",
            
            logging_steps=10,
            report_to="none",

            bf16=True,                           # Thay fp16 bằng bf16 cho H100
            tf32=True,                           # Kích hoạt TensorFloat-32 để tăng tốc
            dataloader_num_workers=4,

            load_best_model_at_end=True,
            metric_for_best_model="loss", #
            greater_is_better=False, #Loss càng thấp thì càng tốt
            save_total_limit=2
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["valid"],
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=10,      
                    early_stopping_threshold=0.0 
                )
            ]
        )

        trainer.train()
        trainer.save_model(self.output_dir)
        print("Training complete.")

    def predict(self, sentence: str, max_new_tokens: int = 512, num_outputs=10):
        """
        Generate FOL from a natural language sentence.
        """

        formatted = [
            s.strip() for s in sentence.split('.') if s.strip()
        ]
        formatted = "\n".join(
            [f"{i+1}. {s}." for i, s in enumerate(formatted)]
        )    
        prompt = f"""
You are a precise and mechanical translator from natural language to First-Order Logic (FOL).

Strict Rules (must follow exactly):
1. Count the exact number of sentences in the input (based on periods, question marks, exclamation marks).
2. Generate exactly that number of FOL formulas — one formula per sentence, in the same order.
3. Do not generate more or fewer formulas.
4. Do not infer or add any facts not explicitly stated.
5. Resolve coreferences appropriately within each sentence.
6. Output each FOL formula on its own separate line.
7. Output NOTHING else: no explanations, no comments, no markdown, no backticks, no "Here is the output", no extra text before the first formula or after the last one.

The output must start immediately with the first FOL formula and end immediately with the last FOL formula.

### Example:

Input:
1. All people who regularly drink coffee are dependent on caffeine.
2. People regularly drink coffee, or they don't want to be addicted to caffeine, or both.

Output:
∀x (DrinkRegularly(x, coffee) → IsDependentOn(x, caffeine))
∀x (DrinkRegularly(x, coffee) ∨ ¬WantToBeAddictedTo(x, caffeine))

Now apply the rules strictly to the following input:

### Input:
{formatted}

### Output:
"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.model.device)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.92,
                top_k=50,

                num_return_sequences=num_outputs,
                pad_token_id=self.tokenizer.eos_token_id
            ).to(self.model.device)

        results = []
        for output in outputs:
            full_text = self.tokenizer.decode(output, skip_special_tokens=True)
    
            if "### Output:" in full_text:
                result = full_text.split("### Output:")[-1].strip()
            else:
                result = full_text[len(prompt):].strip()
    
            results.append(result)
        
        return results