import requests, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "")
PROXY = {"https": "http://127.0.0.1:10811"}
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-chat"
MAX_WORKERS = 4

headers = {
    "Authorization": f"Bearer {OPENROUTER_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost",
    "X-Title": "llmbench",
}

def ask_model(question, choices_text, max_retries=3):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Answer the multiple choice question. Reply with only the letter (A, B, C, D, or E) of the correct answer."},
            {"role": "user", "content": f"{question}\n\n{choices_text}\n\nWhat is the correct answer? Reply with only the letter."}
        ],
        "max_tokens": 16,
        "temperature": 0,
        "provider": {"allow_fallbacks": False},
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(BASE_URL, headers=headers, json=payload, proxies=PROXY, timeout=120)
            if r.status_code == 200:
                d = r.json()
                return d["choices"][0]["message"]["content"].strip()
            elif r.status_code == 429:
                time.sleep(5 * (attempt + 1))
            else:
                time.sleep(2)
        except Exception:
            time.sleep(2)
    return "ERROR"

print("DeepSeek MMLU-Pro via OpenRouter (requests + HTTP proxy)", flush=True)

# Quick test
test = ask_model("What is 2+2?", "A) 3\nB) 4\nC) 5\nD) 6")
print(f"Smoke test: {test}", flush=True)

if "ERROR" in test or "4" not in test:
    print("SMOKE FAILED — aborting", flush=True)
    sys.exit(1)

print("Smoke OK — loading MMLU-Pro dataset...", flush=True)

# Load MMLU-Pro test set
try:
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    print(f"Loaded {len(ds)} questions", flush=True)
except Exception as e:
    print(f"Failed to load dataset: {e}", flush=True)
    sys.exit(1)

import re
def extract_answer(text):
    # Find letter A-E in response
    m = re.search(r'\b([A-E])\b', text.upper())
    return m.group(1) if m else "?"

correct = 0
total = 0
start = time.time()

# Process in batches with thread pool
def process_item(item):
    q = item["question"]
    choices = item["options"]
    answer_idx = item["answer_index"]
    correct_letter = chr(65 + answer_idx)
    
    choices_text = "\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(choices))
    resp = ask_model(q, choices_text)
    pred = extract_answer(resp)
    return pred == correct_letter, pred, correct_letter

BATCH = 50
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    futures = {}
    for i, item in enumerate(ds):
        f = pool.submit(process_item, item)
        futures[f] = i
        total += 1
        
        if len(futures) >= BATCH:
            for f in as_completed(futures):
                ok, pred, gold = f.result()
                if ok:
                    correct += 1
            elapsed = time.time() - start
            pct = correct / total * 100
            rate = total / elapsed
            eta = (len(ds) - total) / rate if rate > 0 else 0
            print(f"[{total}/{len(ds)}] acc={pct:.1f}% rate={rate:.1f}q/s ETA={eta/60:.0f}min", flush=True)
            futures = {}

    # Remaining
    for f in as_completed(futures):
        ok, pred, gold = f.result()
        if ok:
            correct += 1

elapsed = time.time() - start
acc = correct / total * 100
print(f"\nFINAL: {correct}/{total} = {acc:.1f}% ({elapsed/60:.1f} min)", flush=True)

# Save result
out = "/opt/llm-data/llmbench/out/lmeval_deepseek-v4-flash_queue/mmlu_pro"
os.makedirs(out, exist_ok=True)
result = {"model": "deepseek-v4-flash", "task": "mmlu_pro", "acc": acc, "correct": correct, "total": total}
json.dump(result, open(f"{out}/results_mmlu_pro_openrouter.json", "w"), indent=2)
print(f"Saved to {out}/results_mmlu_pro_openrouter.json", flush=True)
