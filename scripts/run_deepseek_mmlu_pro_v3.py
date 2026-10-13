import requests, json, os, sys, time, re
from concurrent.futures import ThreadPoolExecutor, as_completed

KEY = os.environ.get("OPENROUTER_KEY", "")
PROXY = {"https": "http://127.0.0.1:10811"}
URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-chat"
WORKERS = 8

HDRS = {
    "Authorization": f"Bearer {KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost",
    "X-Title": "llmbench",
}

def ask(question, options, retries=4):
    letters = "ABCDEFGHIJ"
    choices = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(options))
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"{question}\n\n{choices}\n\nThe answer is"}],
        "max_tokens": 32,
        "temperature": 0,
        "provider": {"allow_fallbacks": False},
    }
    for attempt in range(retries):
        try:
            r = requests.post(URL, headers=HDRS, json=payload, proxies=PROXY, timeout=120)
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
t = ask("What is 2+2?", ["3", "4", "5", "6"])
print(f"Smoke: '{t}'", flush=True)
if "4" not in t:
    print("SMOKE FAIL", flush=True)
    sys.exit(1)
print("Smoke OK — loading dataset...", flush=True)

from datasets import load_dataset
ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
print(f"Loaded {len(ds)} questions", flush=True)

def extract(text):
    m = re.search(r'\b([A-J])\b', text.upper())
    return m.group(1) if m else "?"

def process(item):
    idx = int(item["answer_index"])
    gold = "ABCDEFGHIJ"[idx]
    resp = ask(item["question"], item["options"])
    return extract(resp) == gold

correct = 0
total = 0
start = time.time()

with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures = {}
    for i, item in enumerate(ds):
        futures[i] = pool.submit(process, item)
        total += 1
        if len(futures) >= 100:
            done_now = []
            for k, f in list(futures.items()):
                if f.done():
                    done_now.append(k)
                    if f.result():
                        correct += 1
                    del futures[k]
            if done_now:
                elapsed = time.time() - start
                processed = total - len(futures)
                pct = correct / max(processed, 1) * 100
                print(f"[{processed}/{len(ds)}] acc={pct:.1f}% t={elapsed:.0f}s", flush=True)

    for f in as_completed(futures.values()):
        if f.result():
            correct += 1

acc = correct / total * 100
elapsed = time.time() - start
print(f"\nFINAL: {correct}/{total} = {acc:.1f}% ({elapsed/60:.1f} min)", flush=True)

out = "/opt/llmbench/out/lmeval_deepseek-v4-flash_queue/mmlu_pro"
os.makedirs(out, exist_ok=True)
json.dump({"model": "deepseek-v4-flash", "task": "mmlu_pro", "acc": acc, "correct": correct, "total": total},
          open(f"{out}/results_mmlu_pro_openrouter.json", "w"), indent=2)
print("Saved.", flush=True)
