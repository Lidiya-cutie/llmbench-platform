import json, subprocess, argparse, time, os
from pathlib import Path
from datetime import datetime

OUT = Path("/opt/llmbench/out")
MODELS_YAML = Path("/opt/llmbench/models.yaml")

KNOWN_ENDPOINTS = [
    {"name":"vllm-vl","url":"http://localhost:8077/v1/models","type":"openai"},
    {"name":"triton-gemma","url":"http://localhost:8001/v2/models/gemma-3-27b","type":"triton"},
    {"name":"gemma-proxy","url":"http://localhost:19892/v1/models","type":"openai"},
    {"name":"clearml-serving","url":"http://localhost:9095/serve","type":"http"},
    {"name":"open-webui","url":"http://localhost:3001","type":"http"},
    {"name":"jupyter","url":"http://localhost:8888","type":"http"},
]

def gpu_status():
    try:
        out = subprocess.check_output(["nvidia-smi","--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu","--format=csv,noheader,nounits"], timeout=10).decode()
        gpus = []
        for line in out.strip().split(chr(10)):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpus.append({"id":int(parts[0]),"name":parts[1],"mem_used":int(parts[2]),"mem_total":int(parts[3]),"util":int(parts[4]),"temp":int(parts[5])})
        return gpus
    except Exception as e:
        return [{"error":str(e)}]

def check_endpoint(ep):
    import requests
    try:
        r = requests.get(ep["url"], timeout=5)
        return {"name":ep["name"],"status":"up" if r.status_code < 500 else "error","code":r.status_code}
    except Exception as e:
        return {"name":ep["name"],"status":"down","error":str(e)[:80]}

def active_evals():
    procs = []
    try:
        out = subprocess.check_output(["ps","aux"], timeout=10).decode()
        for line in out.split(chr(10)):
            if "lm_eval" in line or "evaluate_model" in line:
                parts = line.split(None, 10)
                if len(parts) > 10:
                    procs.append({"pid":parts[1],"cmd":parts[10][:120]})
    except: pass
    return procs

def card_completeness():
    cards = {}
    for f in sorted(OUT.glob("card_*.json")):
        try:
            d = json.loads(f.read_text())
            key = f.stem.replace("card_","")
            q = d.get("quality",{})
            p = d.get("performance",{})
            cards[key] = {
                "quality_tasks": list(q.keys()) if q else [],
                "perf_ok": p.get("ok_count",0),
                "perf_total": p.get("total_count",0),
            }
        except: pass
    return cards

def generate_html(data):
    rows = []
    rows.append("<html><head><title>LLBench Status</title>")
    rows.append("<style>body{font-family:monospace;margin:20px} table{border-collapse:collapse} td,th{border:1px solid #999;padding:4px 8px} .up{color:green} .down{color:red} .error{color:orange}</style>")
    rows.append("<meta http-equiv=refresh content=30></head><body>")
    rows.append(f"<h2>LLBench Status - {data["timestamp"]}</h2>")
    rows.append("<h3>GPU</h3><table><tr><th>ID</th><th>Name</th><th>Mem</th><th>Util%</th><th>Temp</th></tr>")
    for g in data.get("gpus",[]):
        if "error" in g: rows.append(f"<tr><td colspan=5 class=error>{g["error"]}</td></tr>")
        else: rows.append(f"<tr><td>{g["id"]}</td><td>{g["name"]}</td><td>{g["mem_used"]}/{g["mem_total"]} MB</td><td>{g["util"]}</td><td>{g["temp"]}C</td></tr>")
    rows.append("</table>")
    rows.append("<h3>Endpoints</h3><table><tr><th>Name</th><th>Status</th><th>Code/Error</th></tr>")
    for e in data.get("endpoints",[]):
        cls = f"{e.get("status","")}"
        extra = e.get("code","") or e.get("error","")
        rows.append(f"<tr><td>{e["name"]}</td><td class={cls}>{e["status"]}</td><td>{extra}</td></tr>")
    rows.append("</table>")
    rows.append("<h3>Active Evals</h3><table><tr><th>PID</th><th>Command</th></tr>")
    for p in data.get("evals",[]):
        rows.append(f"<tr><td>{p["pid"]}</td><td>{p["cmd"]}</td></tr>")
    if not data.get("evals"): rows.append("<tr><td colspan=2>None</td></tr>")
    rows.append("</table>")
    rows.append("<h3>Card Completeness</h3><table><tr><th>Model</th><th>Quality</th><th>Perf ok/total</th></tr>")
    for k,v in data.get("cards",{}).items():
        rows.append(f"<tr><td>{k}</td><td>{", ".join(v["quality_tasks"]) if v["quality_tasks"] else "-"}</td><td>{v["perf_ok"]}/{v["perf_total"]}</td></tr>")
    rows.append("</table></body></html>")
    return chr(10).join(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gpus": gpu_status(),
        "endpoints": [check_endpoint(e) for e in KNOWN_ENDPOINTS],
        "evals": active_evals(),
        "cards": card_completeness(),
    }
    if a.json or not a.html:
        print(json.dumps(data, indent=2, default=str))
        jf = OUT / "system_report.json"
        jf.write_text(json.dumps(data, indent=2, default=str))
    if a.html:
        hf = OUT / "system_status_v2.html"
        hf.write_text(generate_html(data))
        print(f"HTML: {hf}")

if __name__ == "__main__":
    main()