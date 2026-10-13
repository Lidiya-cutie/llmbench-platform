import json, sys, os
from pathlib import Path

OUT = Path("/opt/llm-data/llmbench/out")

card_path = OUT / "card_deepseek-v4-flash.json"
card = json.loads(card_path.read_text())
bak = card_path.with_suffix(f".json.bak-step1")
bak.write_text(card_path.read_text())
print(f"Backup: {bak.name}")

results_file = OUT / "lmeval_deepseek-v4-flash_queue/gsm8k/deepseek-v4-flash-dspark/results_2026-07-28T08-49-02.030731.json"
data = json.loads(results_file.read_text())
gsm = data["results"]["gsm8k"]
val = gsm.get("exact_match,flexible-extract", gsm.get("exact_match,strict-match", 0.0))
card.setdefault("quality", {})
if not card["quality"].get("gsm8k"):
    card["quality"]["gsm8k"] = round(val, 6)
    print(f"  gsm8k = {val:.6f} (flexible-extract)")
else:
    print(f"  gsm8k already present: {card['quality']['gsm8k']}")

card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2))
print(f"Updated: {card_path.name}")

print("\n--- Regenerating dashboard ---")
sys.path.insert(0, str(Path("/opt/llm-data/llmbench/scripts")))
try:
    from generate_dashboard_v2 import DashboardGenerator
    gen = DashboardGenerator()
    gen.run()
    print("Dashboard v2 regenerated.")
except Exception as e:
    print(f"Dashboard v2 failed ({e}), trying subprocess...")
    os.system("cd /opt/llm-data/llmbench && python3 scripts/generate_dashboard_v2.py")
