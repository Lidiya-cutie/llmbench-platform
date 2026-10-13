import requests, json, os, sys, time, re
from concurrent.futures import ThreadPoolExecutor, as_completed

OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "")
PROXY = {"https": "http://127.0.0.1:10811"}
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-chat"
MAX_WORKERS = 8

headers = {
    "Authorization": f"Bearer {OPENROUTER_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost",
    "X-Title": "llmbench",
}

def ask_model(question, choices, max_retries=4):
    letters = "ABCDE"
    choices_text = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": f"{question}\n\n{choices_text}\n\nThe answer is"}
        ],
        "max_tokens": 32,
        "temperature": 0,
        "provider": {"allow_fallbacks": False},
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(BASE_URL, headers=headers, json=payload, proxies=PROXY, timeout=120)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            elif r.status_code == 429:
                time.sleep(3 * (attempt + 1))
            else:
                time.sleep(2)
        except Exception:
            time.sleep(2)
    return "ERROR"

# Smoke
test = ask_model("What is 2+2?", ["3", "4", "5", "6"])
print(f"Smoke: '{test}'", flush=True)
if "4" not in test:
    print("SMOKE FAILED", flush=True)
    sys.exit(1)
print("Smoke OK", flush=True)

from datasets import load_dataset
ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
print(f"Loaded {len(ds)} questions", flush=True)

def extract_answer(text):
    m = re.search(r'\b([A-E])\b', text.upper())
    return m.group(1) if m else "?"

def process_item(item):
    answer_idx = item["answer_index"]
    correct_letter = "ABCDE"[answer_idx]
    resp = ask_model(item["question"], item["options"])
    return extract_answer(resp) == correct_letter

correct = 0
total = 0
start = time.time()
BATCH = 100

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    pending = {}
    for i, item in enumerate(ds):
        pending[i] = pool.submit(process_item, item)
        total += 1
        if len(pending) >= BATCH:
            for idx, f in list(pending.items()):
                if f.done():
                    if f.result():
                        correct += 1
                    del pending[idx]
            
            done = total - len(pending)
            elapsed = time.time() - start
            pct = correct / max(done, 1) * 100
            print(f"[{done}/{len(ds)}] acc={pct:.1f}% elapsed={elapsed:.0f}s", flush=True)

    for f in as_completed(pending.values()):
        if f.result():
            correct += 1

acc = correct / total * 100
print(f"\nFINAL: {correct}/{total} = {acc:.1f}%", flush=True)

out = "/opt/llmbench/out/lmeval_deepseek-v4-flash_queue/mmlu_pro"
os.makedirs(out, exist_ok=True)
json.dump({"model": "deepseek-v4-flash", "task": "mmlu_pro", "acc": acc, "correct": correct, "total": total},
          open(f"{out}/results_mmlu_pro_openrouter.json", "w"), indent=2)
print("Saved.", flush=True)
