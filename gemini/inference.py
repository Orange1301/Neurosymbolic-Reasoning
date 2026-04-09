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
    Sử dụng gemini-2.5-flash để inference trên batch các samples.
    """
    prompt = f"""You are an expert in logic and reasoning. Analyze the logical problems provided in the input batch.
For each problem, determine whether the 'conclusion' logically follows from the 'premises'.

Label rules:
- If the conclusion is definitely true based on the premises: "True"
- If the conclusion is definitely false based on the premises: "False"
- If there is not enough information to definitively determine whether it is true or false: "Uncertain"

---
EXAMPLE (1-shot):

Input:
[
  {{
    "id": "example_01",
    "premises": "All birds have feathers. A penguin is a bird.",
    "conclusion": "A penguin has feathers."
  }},
  {{
    "id": "example_02",
    "premises": "If the sun is out, Alice goes for a walk. Alice is going for a walk.",
    "conclusion": "The sun is out."
  }}
]

Output:
{{
  "example_01": "True",
  "example_02": "Uncertain"
}}
---

Now, process the following batch of problems.

Input data:
{json.dumps(batch_samples, indent=2, ensure_ascii=False)}

CRITICAL REQUIREMENT:
Return ONLY a single valid JSON object (dictionary) where the keys are the 'id's of the problems and the values are the predicted labels ('T', 'F', or 'U').
Do not include any explanations, markdown blocks, or extra text.
"""
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )
        result_dict = json.loads(response.text)
        return result_dict
    except Exception as e:
        print(f"Lỗi khi gọi API: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"Raw response: {response.text}")
        return None

import json
import os

def save_results_to_file(results_dict, filename="predictions.json"):
    """Ghi tiếp (cập nhật) kết quả vào file JSON một cách an toàn."""
    data = {}

    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
            if content:
                data = json.loads(content)

    data.update(results_dict)

    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error when writting file: {e}")

def run_large_scale_test(all_samples, batch_size=150, output_file="predictions.json"):
    """Chia nhỏ tập test thành từng batch, gọi API và lưu file liên tục."""
    total_samples = len(all_samples)
    print(f"Total samples: {total_samples} - Batch size={batch_size}...")

    for i in range(0, total_samples, batch_size):
        cur_batch_size = min(batch_size, total_samples - i)
        current_batch = all_samples[i : i + cur_batch_size]
        batch_index = (i // batch_size) + 1
        
        print(f"--- Processing batch {batch_index} ({i} -> {min(i + batch_size, total_samples)}) ---")
        
        results = evaluate_logic_batch(current_batch)
        
        if results:
            save_results_to_file(results, output_file)
            print(f"Completed batch {batch_index}! Saved in {output_file}.")
        else:
            print(f"Error in batch {batch_index}!!!")

        time.sleep(5)

    print("\n>>> ALL DONE!")

# Main -----------------
with open('test_data.json', 'r', encoding='utf=8') as f:
    data = json.load(f)

for d in data:
    d.pop("label", None)  
run_large_scale_test(data, 100)
