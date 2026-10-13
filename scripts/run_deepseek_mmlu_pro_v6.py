import requests, json, os, sys, time, re
from concurrent.futures import ThreadPoolExecutor, as_completed

KEY = os.environ.get("OPENROUTER_KEY", "")
PROXY = {"https": "http://127.0.0.1:10811"}
URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-chat"
WORKERS = 4

HDRS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

def ask(question, options, retries=3):
    letters = "ABCDEFGHIJ"
    choices = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(options))
    body = json.dumps({"model": MODEL,
               "messages": [{"role": "user", "content": f"{question}\n\n{choices}\n\nThe answer is"}],
               "max_tokens": 16, "temperature": 0}).encode("utf-8")
    for a in range(retries):
        try:
            r = requests.post(URL, headers=HDRS, data=body, proxies=PROXY, timeout=60)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            elif r.status_code == 429:
                time.sleep(5*(a+1))
            else:
                time.sleep(3)
        except Exception as e:
            time.sleep(3)
    return "ERR"

t = ask("What is 2+2?", ["3","4","5","6"])
print(f"Smoke: '{t}'", flush=True)
if "4" not in t:
    print("SMOKE FAIL", flush=True)
    sys.exit(1)
print("Smoke OK", flush=True)

from datasets import load_dataset
ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
print(f"Total: {len(ds)}", flush=True)

def extract(text):
    m = re.search(r'\b([A-J])\b', text.upper())
    return m.group(1) if m else "?"

def process(i):
    item = ds[i]
    idx = int(item["answer_index"])
    gold = "ABCDEFGHIJ"[idx]
    resp = ask(item["question"], item["options"])
    return extract(resp) == gold

correct = 0
done = 0
start = time.time()

with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futs = {pool.submit(process, i): i for i in range(len(ds))}
    for f in as_completed(futs):
        done += 1
        if f.result():
            correct += 1
        if done % 50 == 0:
            el = time.time() - start
            print(f"[{done}/{len(ds)}] acc={correct/done*100:.1f}% t={el:.0f}s rate={done/el:.1f}q/s", flush=True)

acc = correct/done*100
el = time.time() - start
print(f"\nFINAL: {correct}/{done} = {acc:.1f}% ({el/60:.1f} min)", flush=True)
outdir = "/opt/llmbench/out/lmeval_deepseek-v4-flash_queue/mmlu_pro"
os.makedirs(outdir, exist_ok=True)
json.dump({"model":"deepseek-v4-flash","task":"mmlu_pro","acc":acc,"correct":correct,"total":done},
          open(f"{outdir}/results_mmlu_pro_openrouter.json","w"), indent=2)
print("Saved.", flush=True)
