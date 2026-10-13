import json, csv, argparse
from pathlib import Path

OUT = Path("/opt/llmbench/out")

def load_cards():
    cards = {}
    for f in sorted(OUT.glob("card_*.json")):
        try:
            d = json.loads(f.read_text())
            key = f.stem.replace("card_", "")
            cards[key] = d
        except: pass
    return cards

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["csv","json","excel"], default="csv")
    a = ap.parse_args()
    cards = load_cards()
    if not cards:
        print("No cards found"); return
    rows = []
    for key, c in cards.items():
        row = {"model": key}
        row.update(c.get("quality", {}))
        perf = c.get("performance", {})
        row["tps"] = perf.get("tokens_per_second")
        row["ttft"] = perf.get("time_to_first_token")
        row["perf_ok"] = perf.get("ok_count", 0)
        row["perf_n"] = perf.get("total_count", 0)
        rows.append(row)
    cols = sorted({k for r in rows for k in r})
    if a.format == "json":
        out = OUT / "llmbench_export.json"
        out.write_text(json.dumps(rows, indent=2))
        print(f"JSON: {out}")
    elif a.format == "excel":
        try:
            import openpyxl
        except ImportError:
            print("openpyxl not installed, using csv")
            a.format = "csv"
    if a.format in ("csv", "excel"):
        out = OUT / "llmbench_export.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"CSV: {out} ({len(rows)} rows)")
        if a.format == "excel":
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.append(cols)
            for r in rows:
                ws.append([r.get(c, "") for c in cols])
            xl = OUT / "llmbench_export.xlsx"
            wb.save(xl)
            print(f"XLSX: {xl}")
    print(f"\n--- Summary ({len(rows)} models) ---")
    for r in rows:
        q = ", ".join(f"{k}={v}" for k,v in r.items() if k not in ("model","tps","ttft","perf_ok","perf_n") and v is not None)
        print(f"  {r['model']}: {q or 'no quality'}")

if __name__ == "__main__":
    main()
