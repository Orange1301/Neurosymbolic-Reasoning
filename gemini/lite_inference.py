from google import genai
from google.genai import types
import json
import time
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv('API_KEY')
client = genai.Client(api_key=API_KEY)

def evaluate_logic_batch(batch_samples):
    """
    Sử dụng gemini-2.5-flash-lite với Chain-of-Thought.
    """
    prompt = f"""You are an expert in formal logic.
Task: Analyze each problem using Chain-of-Thought reasoning, then provide a final label.

Label rules:
- "True": Conclusion is definitely true.
- "False": Conclusion is definitely false.
- "Uncertain": Uncertain/Not enough information.

Input data:
{json.dumps(batch_samples, indent=2, ensure_ascii=False)}

CRITICAL REQUIREMENT:
Return a JSON object where each key is the problem 'id'.
The value for each key must be another object containing:
1. "reasoning": A brief step-by-step logical deduction.
2. "label": Only 'True', 'False', or 'Uncertain'.

Example format:
{{
  "id_01": {{
    "reasoning": "Premise says All A are B. X is A. Therefore X must be B.",
    "label": "True"
  }}
}}
"""
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
                max_output_tokens=8192
            )
        )
        
        # full_result = json.loads(response.text)
        full_result = json.loads(response.text, strict=False)

        final_labels = {k: v['label'] for k, v in full_result.items() if 'label' in v}
        return final_labels
        
    except Exception as e:
        print(f"Lỗi khi gọi API: {e}")
        return None

def save_results_to_file(results_dict, filename="predictions.json"):
    data = {}
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
            if content: data = json.loads(content)

    data.update(results_dict)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def run_large_scale_test(all_samples, batch_size=80, output_file="predictions.json"):
    total_samples = len(all_samples)
    print(f"Total: {total_samples} - Mode: Flash-Lite + CoT")

    for i in range(0, total_samples, batch_size):
        current_batch = all_samples[i : i + batch_size]
        print(f"--- Processing batch {(i // batch_size) + 1} ({i} -> {min(i + batch_size, total_samples)}) ---")
        
        results = evaluate_logic_batch(current_batch)
        
        if results:
            save_results_to_file(results, output_file)
            print(f"Saved {(i // batch_size) + 1}")
        else:
            print(f"Skipped {(i // batch_size) + 1} due to error.")
            break

        time.sleep(10)

if __name__ == "__main__":
    with open('test_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    for d in data:
        d.pop("label", None)

    run_large_scale_test(data, batch_size=90)