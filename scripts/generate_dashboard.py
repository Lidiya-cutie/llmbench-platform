"""
Генератор дашборда llmbench.

Читает card_*.json из out/, генерирует единый HTML-файл с встроенными данными.
Масштабируемо: новая модель = новый card_*.json → перегенерация.

Использование:
    python scripts/generate_dashboard.py --out-dir out --dest out/dashboard.html

После генерации открыть dashboard.html в браузере или отдать через nginx/python -m http.server.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def normalize_quality(v) -> float:
    """Привести метрику к шкале 0-100."""
    if not isinstance(v, (int, float)):
        return 0.0
    if 0 < v <= 1:
        return round(v * 100, 1)
    return round(v, 1)


def extract_model_data(card: dict) -> dict:
    """Извлечь данные модели из карточки для дашборда."""
    q = card.get("quality", {})
    p = card.get("perf", {})
    hw = card.get("hardware", {})
    cost = card.get("cost", {})
    micro = card.get("micro", {})

    gsm8k = normalize_quality(q.get("gsm8k", 0))
    minerva = normalize_quality(q.get("minerva_math", 0))
    mmlu_pro = normalize_quality(q.get("mmlu_pro", 0))
    humaneval = normalize_quality(q.get("humaneval", 0))

    gen_tok = p.get("gen_tok_s_mean", 0) or 0
    ttft = p.get("ttft_s_mean", 0) or 0
    lat_p50 = p.get("latency_p50", 0) or 0
    vram = hw.get("peak_vram_mb", 0) or hw.get("avg_vram_mb", 0) or 0
    power = hw.get("avg_power_w", 0) or 0
    util = hw.get("avg_gpu_util", 0) or 0

    usd_1m = cost.get("usd_per_1m_tokens", 0) or cost.get("cost_per_1m_tokens", 0) or 0
    usd_h = cost.get("usd_per_hour", 0) or cost.get("cost_per_hour", 0) or 0

    math_acc = normalize_quality(micro.get("math_acc", 0))
    json_valid = micro.get("json_valid", 0) or 0
    tool_ok = micro.get("tool_call_ok", 0) or 0
    if isinstance(json_valid, (int, float)) and 0 < json_valid <= 1:
        json_valid = round(json_valid * 100, 1)
    if isinstance(tool_ok, (int, float)) and 0 < tool_ok <= 1:
        tool_ok = round(tool_ok * 100, 1)

    # Normalized axes (0-100)
    cost_eff = round(max(0, min(100, 100 - usd_1m * 80)), 1) if usd_1m > 0 else 0
    speed = round(min(100, gen_tok), 1) if gen_tok > 0 else 0
    ctx_max = card.get("context_max", 8192)
    context_score = round(min(100, max(0, (ctx_max - 4096) / (32768 - 4096) * 100)), 1)
    has_quality = any(v > 0 for v in [gsm8k, minerva, mmlu_pro])

    return {
        "name": card.get("name", ""),
        "maker": f"{card.get('engine', '')} · {card.get('quant_format', '')}",
        "ev": "v" if has_quality else "e",
        "reasoning": mmlu_pro,
        "coding": humaneval,
        "math": max(minerva, gsm8k),
        "speed": speed,
        "cost": cost_eff,
        "context": context_score,
        "safety": 0,
        "multiling": 0,
        "raw": {
            "gsm8k": gsm8k, "minerva": minerva,
            "mmlu_pro": mmlu_pro, "humaneval": humaneval,
            "gen_tok": round(gen_tok, 1), "ttft": round(ttft, 3),
            "latency": round(lat_p50, 2), "vram": round(vram, 0),
            "power": round(power, 1), "util": round(util, 1),
            "usd_1m": round(usd_1m, 3), "usd_h": round(usd_h, 3),
            "math_acc": math_acc, "json_valid": round(json_valid, 1),
            "tool_ok": round(tool_ok, 1),
        },
        "engine": card.get("engine", ""),
        "quant": card.get("quant_format", ""),
        "ctx_max": ctx_max,
        "modalities": card.get("modalities", ""),
        "conclusion": (card.get("conclusion", "") or "")[:300],
    }


def load_all_cards(out_dir: str) -> list[dict]:
    """Загрузить все карточки."""
    models = []
    for p in sorted(Path(out_dir).glob("card_*.json")):
        if ".bak." in p.name:
            continue
        try:
            with open(p, encoding="utf-8") as f:
                card = json.load(f)
            models.append(extract_model_data(card))
        except Exception as e:
            print(f"WARNING: skip {p.name}: {e}")
    return models


def generate_html(models: list[dict], title: str = "LLM Bench Console") -> str:
    """Сгенерировать HTML с встроенными данными."""
    data_json = json.dumps(models, ensure_ascii=False, indent=2)
    template = _HTML_TEMPLATE
    return template.replace("__DATA__", data_json).replace("__TITLE__", title).replace("__COUNT__", str(len(models)))


_HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet"/>
<style>
:root{--ink:#0E1117;--panel:#161B24;--panel2:#1D2430;--line:#29313F;--text:#E7EAF0;--muted:#8B95A7;--faint:#5B6577;--gold:#F2B035;--teal:#35D0B8;--violet:#9B8CFA;--red:#F2685C;--green:#57C97E;--shadow:0 1px 0 rgba(255,255,255,.03),0 12px 30px -18px rgba(0,0,0,.7)}
*{box-sizing:border-box}html,body{margin:0;padding:0}
body{background:radial-gradient(1200px 600px at 85% -10%,rgba(242,176,53,.06),transparent 60%),radial-gradient(900px 500px at 0% 0%,rgba(53,208,184,.05),transparent 55%),var(--ink);color:var(--text);font-family:"Inter",system-ui,sans-serif;-webkit-font-smoothing:antialiased;line-height:1.5;padding:28px clamp(14px,4vw,44px) 60px}
.wrap{max-width:1240px;margin:0 auto}
header{display:flex;flex-wrap:wrap;align-items:flex-end;gap:18px;justify-content:space-between;margin-bottom:26px}
.brand h1{font-family:"Space Grotesk",sans-serif;font-weight:700;font-size:clamp(26px,4vw,40px);letter-spacing:-.02em;margin:0;line-height:1}
.brand h1 .dot{color:var(--gold)}
.brand p{margin:8px 0 0;color:var(--muted);max-width:52ch;font-size:14px}
.live-badge{display:inline-flex;align-items:center;gap:8px;font-family:"JetBrains Mono",monospace;font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:var(--green);background:rgba(87,201,126,.08);border:1px solid rgba(87,201,126,.28);border-radius:999px;padding:7px 13px}
.live-badge::before{content:"";width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 10px var(--green);animation:pulse 2s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.eyebrow{font-family:"JetBrains Mono",monospace;font-size:11px;text-transform:uppercase;letter-spacing:.16em;color:var(--faint);margin:0 0 12px}
.usecases{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:26px}
.chip{cursor:pointer;font-family:"Space Grotesk",sans-serif;font-weight:500;font-size:13.5px;color:var(--muted);background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:9px 15px;transition:.16s ease;display:flex;align-items:center;gap:8px}
.chip:hover{color:var(--text);border-color:#3a4454}
.chip .g{font-size:15px;filter:grayscale(.2)}
.chip.active{color:var(--ink);background:var(--gold);border-color:var(--gold);font-weight:600}
.chip.active .g{filter:none}
.grid{display:grid;grid-template-columns:1.55fr 1fr;gap:20px;align-items:start}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow)}
.card-h{display:flex;align-items:center;justify-content:space-between;padding:16px 20px 14px;border-bottom:1px solid var(--line)}
.card-h h2{font-family:"Space Grotesk",sans-serif;font-size:15px;font-weight:600;margin:0}
.card-h .hint{font-size:11px;color:var(--faint);font-family:"JetBrains Mono",monospace}
table{width:100%;border-collapse:collapse;font-size:13px}
thead th{text-align:right;padding:11px 12px;color:var(--muted);font-weight:500;font-family:"JetBrains Mono",monospace;font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;cursor:pointer;user-select:none;white-space:nowrap;border-bottom:1px solid var(--line)}
thead th:first-child{text-align:left}
thead th:hover{color:var(--text)}
thead th.sorted{color:var(--gold)}
thead th .arw{font-size:9px;margin-left:3px}
tbody tr{cursor:pointer;transition:background .12s}
tbody tr:hover{background:var(--panel2)}
tbody tr.sel{background:rgba(242,176,53,.07);box-shadow:inset 2px 0 0 var(--gold)}
tbody td{padding:11px 12px;text-align:right;border-bottom:1px solid rgba(41,49,63,.5);font-family:"JetBrains Mono",monospace}
tbody td:first-child{text-align:left;font-family:"Inter",sans-serif}
.rank{color:var(--faint);font-size:12px;margin-right:9px}
.mname{font-weight:600;font-size:13.5px}
.mmaker{display:block;color:var(--faint);font-size:11px;margin-top:1px}
.score-pill{display:inline-block;min-width:42px;padding:3px 8px;border-radius:7px;font-weight:700;font-size:13px;color:var(--ink)}
.ev{font-size:9.5px;text-transform:uppercase;letter-spacing:.05em;padding:2px 6px;border-radius:5px;border:1px solid}
.ev.v{color:var(--green);border-color:rgba(87,201,126,.35);background:rgba(87,201,126,.08)}
.ev.e{color:var(--muted);border-color:var(--line)}
.cell-na{color:var(--faint);opacity:.4}
.detail-body{padding:20px}
.gauge-wrap{display:flex;align-items:center;gap:18px;margin-bottom:20px}
.gauge{position:relative;flex:0 0 auto}
.gauge .val{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.gauge .val b{font-family:"Space Grotesk",sans-serif;font-size:30px;font-weight:700;line-height:1}
.gauge .val span{font-size:10px;color:var(--faint);font-family:"JetBrains Mono",monospace;text-transform:uppercase;letter-spacing:.1em;margin-top:2px}
.detail-name{font-family:"Space Grotesk",sans-serif;font-size:20px;font-weight:700;margin:0}
.detail-maker{color:var(--muted);font-size:13px;margin:2px 0 10px}
.verdict{font-size:12.5px;color:var(--muted);line-height:1.55}
.verdict b{color:var(--teal)}
.bars{margin-top:18px;display:flex;flex-direction:column;gap:9px}
.bar-row{display:grid;grid-template-columns:96px 1fr 34px;align-items:center;gap:10px;font-size:12px}
.bar-row .lab{color:var(--muted);font-family:"JetBrains Mono",monospace;font-size:10.5px;text-transform:uppercase}
.track{height:7px;background:var(--panel2);border-radius:4px;overflow:hidden}
.fill{height:100%;border-radius:4px;transition:width .5s cubic-bezier(.2,.7,.2,1)}
.bar-row .n{text-align:right;font-family:"JetBrains Mono",monospace;font-weight:500}
.wt{font-size:9px;color:var(--faint)}
.raw-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:16px;padding-top:16px;border-top:1px solid var(--line)}
.raw-cell{background:var(--panel2);border-radius:10px;padding:12px;text-align:center}
.raw-cell .lbl{font-size:10px;color:var(--faint);font-family:"JetBrains Mono",monospace;text-transform:uppercase;margin-bottom:4px}
.raw-cell .num{font-family:"Space Grotesk",sans-serif;font-size:18px;font-weight:600}
.raw-cell .unit{font-size:11px;color:var(--muted);margin-left:2px}
.compare-card{margin-top:20px}
.cmp-controls{display:flex;gap:12px;flex-wrap:wrap;padding:16px 20px;border-bottom:1px solid var(--line)}
.sel-group{display:flex;align-items:center;gap:8px}
.swatch{width:11px;height:11px;border-radius:3px}
select{background:var(--panel2);color:var(--text);border:1px solid var(--line);border-radius:9px;padding:8px 11px;font-family:"Space Grotesk",sans-serif;font-size:13px;font-weight:500;cursor:pointer;outline:none}
select:focus-visible{border-color:var(--gold)}
.cmp-body{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:20px}
@media(max-width:760px){.cmp-body{grid-template-columns:1fr}}
#radar{width:100%;height:auto}
.delta-tbl{width:100%;font-size:12.5px}
.delta-tbl td{padding:8px 6px;border-bottom:1px solid rgba(41,49,63,.5);font-family:"JetBrains Mono",monospace}
.delta-tbl td:first-child{color:var(--muted);font-family:"Inter",sans-serif;text-transform:capitalize}
.delta-tbl .a{color:var(--teal);text-align:center;font-weight:600}
.delta-tbl .b{color:var(--violet);text-align:center;font-weight:600}
.delta-tbl .d{text-align:right;font-size:11px}
.up{color:var(--green)}.down{color:var(--red)}
.note{margin-top:26px;padding:18px 20px;border:1px dashed var(--line);border-radius:14px;background:rgba(29,36,48,.4);font-size:12.5px;color:var(--muted);line-height:1.6}
.note b{color:var(--text)}
.note code{font-family:"JetBrains Mono",monospace;color:var(--gold);background:rgba(242,176,53,.08);padding:1px 6px;border-radius:5px}
</style>
</head>
<body>
<div class="wrap">
<header>
<div class="brand">
<h1>LLM Bench Console<span class="dot">.</span></h1>
<p>Сравнительный анализ локальных LLM на RTX 4090 — композитный score, топ-K по метрикам, радар A vs B и сырые замеры. Данные из card_*.json (llmbench).</p>
</div>
<span class="live-badge" id="badge">live · __COUNT__ моделей</span>
</header>
<p class="eyebrow">Движок рекомендаций · выберите задачу</p>
<div class="usecases" id="usecases"></div>
<div class="grid">
<div class="card"><div class="card-h"><h2>Лидерборд</h2><span class="hint">сортировка по клику · детали по строке</span></div><div style="overflow-x:auto"><table id="board"><thead></thead><tbody></tbody></table></div></div>
<div class="card"><div class="card-h"><h2>Композитный score</h2><span class="hint" id="detail-rank"></span></div><div class="detail-body" id="detail"></div></div>
</div>
<div class="card compare-card"><div class="card-h"><h2>Сравнение двух моделей</h2><span class="hint">радар + дельта</span></div>
<div class="cmp-controls"><div class="sel-group"><span class="swatch" style="background:var(--teal)"></span><select id="selA"></select></div><div class="sel-group"><span class="swatch" style="background:var(--violet)"></span><select id="selB"></select></div></div>
<div class="cmp-body"><svg id="radar" viewBox="0 0 360 320"></svg><table class="delta-tbl"><tbody id="delta"></tbody></table></div>
</div>
<div class="note"><b>Источник:</b> <code>card_*.json</code> из <code>out/</code>. Score динамический по весам задачи. Метрики: gsm8k (exact_match), minerva_math (math_verify), mmlu_pro (custom-extract), humaneval (pass@1). Для добавления модели — поместить card_*.json и перегенерировать: <code>python scripts/generate_dashboard.py</code></div>
</div>
<script>
const AXES=[{key:"reasoning",label:"Reasoning"},{key:"coding",label:"Coding"},{key:"math",label:"Math"},{key:"speed",label:"Speed"},{key:"cost",label:"Cost-eff"},{key:"context",label:"Context"},{key:"safety",label:"Safety"},{key:"multiling",label:"Multiling"}];
const MODELS=__DATA__;
const USECASES={balanced:{g:"⚖️",label:"Сбалансированно",w:{reasoning:1,coding:1,math:1,speed:1,cost:1,context:1,safety:1,multiling:1}},coding:{g:"⌨️",label:"Кодинг",w:{reasoning:1.2,coding:3,math:1,speed:1.3,cost:.8,context:1.4,safety:.6,multiling:.4}},reasoning:{g:"🧠",label:"Рассуждения",w:{reasoning:3,coding:.8,math:2,speed:.5,cost:.6,context:1,safety:1,multiling:.6}},math:{g:"🔢",label:"Математика",w:{reasoning:2,coding:1,math:3,speed:.7,cost:.7,context:.8,safety:.6,multiling:.4}},budget:{g:"💸",label:"Бюджет",w:{reasoning:.8,coding:.8,math:.6,speed:1.4,cost:3.2,context:.8,safety:.6,multiling:.6}},speed:{g:"⚡",label:"Скорость",w:{reasoning:.7,coding:.8,math:.6,speed:3,cost:1.2,context:.8,safety:.5,multiling:.4}},longctx:{g:"📚",label:"Контекст",w:{reasoning:1.2,coding:.9,math:.7,speed:.7,cost:.9,context:3.2,safety:.8,multiling:.8}}};
let state={usecase:"balanced",sortKey:"score",sortDir:-1,selected:0};
const cf=v=>v>=90?"var(--green)":v>=75?"var(--teal)":v>=50?"var(--gold)":v>=25?"#E8894B":v>0?"var(--red)":"var(--faint)";
const $=s=>document.querySelector(s);
const wt=()=>USECASES[state.usecase].w;
const comp=m=>{const w=wt();let s=0,ws=0;for(const a of AXES){const x=w[a.key]??1;s+=(m[a.key]||0)*x;ws+=x}return s/ws};
const enrich=()=>MODELS.map((m,i)=>({...m,_i:i,score:comp(m)}));
function ru(){const e=$("#usecases");e.innerHTML="";for(const[k,u]of Object.entries(USECASES)){const c=document.createElement("div");c.className="chip"+(state.usecase===k?" active":"");c.innerHTML=`<span class="g">${u.g}</span>${u.label}`;c.onclick=()=>{state.usecase=k;ra()};e.appendChild(c)}}
function rb(){const rows=enrich().sort((a,b)=>(a[state.sortKey]-b[state.sortKey])*state.sortDir);const th=$("#board thead"),tb=$("#board tbody");const cols=[{key:"name",label:"Модель"},{key:"score",label:"Score"},...AXES.map(a=>({key:a.key,label:a.label})),{key:"ev",label:"Evid"}];
th.innerHTML="<tr>"+cols.map(c=>{const s=state.sortKey===c.key;return`<th data-k="${c.key}" class="${s?'sorted':''}">${c.label}<span class="arw">${s?(state.sortDir<0?"▼":"▲"):""}</span></th>`}).join("")+"</tr>";
th.querySelectorAll("th").forEach(t=>{t.onclick=()=>{const k=t.dataset.k;if(state.sortKey===k)state.sortDir*=-1;else{state.sortKey=k;state.sortDir=k==="name"?1:-1}rb()}});
tb.innerHTML=rows.map((m,i)=>{const ac=AXES.map(a=>{const v=m[a.key]||0;return`<td class="${v>0?"cell-dim":"cell-na"}">${v>0?v:"—"}</td>`}).join("");const ev=m.ev==="v"?'<span class="ev v">verif</span>':'<span class="ev e">est</span>';
return`<tr data-i="${m._i}" class="${m._i===state.selected?'sel':''}"><td><span class="rank">${String(i+1).padStart(2,'0')}</span><span class="mname">${m.name}</span><span class="mmaker">${m.maker}</span></td><td><span class="score-pill" style="background:${cf(m.score)}">${m.score.toFixed(0)}</span></td>${ac}<td>${ev}</td></tr>`}).join("");
tb.querySelectorAll("tr").forEach(t=>{t.onclick=()=>{state.selected=+t.dataset.i;rb();rd()}})}
function rd(){const m=enrich().find(x=>x._i===state.selected);const rk=enrich().sort((a,b)=>b.score-a.score);const r=rk.findIndex(x=>x._i===m._i)+1;
$("#detail-rank").textContent=`#${r} под задачу`;const R=46,C=2*Math.PI*R,pct=m.score/100;
const kn=AXES.filter(a=>(m[a.key]||0)>0);const st=kn.sort((a,b)=>m[b.key]-m[a.key]).slice(0,2).map(a=>a.label.toLowerCase());const wk=kn.sort((a,b)=>m[a.key]-m[b.key])[0];const wl=wk?wk.label.toLowerCase():"нет данных";
const bars=AXES.map(a=>{const w=wt()[a.key]??1;const v=m[a.key]||0;return`<div class="bar-row"><span class="lab">${a.label}</span><div class="track"><div class="fill" style="width:${v}%;background:${cf(v)}"></div></div><span class="n">${v>0?v.toFixed(1):"—"}<span class="wt"> ×${w}</span></span></div>`}).join("");
const r=m.raw||{};const rc=[{l:"Gen t/s",v:r.gen_tok||0,u:""},{l:"TTFT",v:r.ttft||0,u:"s"},{l:"Latency",v:r.latency||0,u:"s"},{l:"VRAM",v:r.vram||0,u:"MB"},{l:"Power",v:r.power||0,u:"W"},{l:"GPU util",v:r.util||0,u:"%"},{l:"$/1M tok",v:r.usd_1m||0,u:""},{l:"$/hour",v:r.usd_h||0,u:""},{l:"Math acc",v:r.math_acc||0,u:"%"}].map(c=>`<div class="raw-cell"><div class="lbl">${c.l}</div><div class="num">${c.v>0?c.v:"—"}<span class="unit">${c.v>0?c.u:""}</span></div></div>`).join("");
$("#detail").innerHTML=`<div class="gauge-wrap"><div class="gauge"><svg width="112" height="112" viewBox="0 0 112 112"><circle cx="56" cy="56" r="${R}" fill="none" stroke="var(--panel2)" stroke-width="9"/><circle cx="56" cy="56" r="${R}" fill="none" stroke="${cf(m.score)}" stroke-width="9" stroke-linecap="round" stroke-dasharray="${C}" stroke-dashoffset="${C*(1-pct)}" transform="rotate(-90 56 56)" style="transition:stroke-dashoffset .6s cubic-bezier(.2,.7,.2,1)"/></svg><div class="val"><b>${m.score.toFixed(0)}</b><span>/ 100</span></div></div><div><p class="detail-name">${m.name}</p><p class="detail-maker">${m.maker} · ${m.ev==="v"?"проверенные замеры":"оценка"}</p><p class="verdict">${st.length>0?`Под задачу «${USECASES[state.usecase].label}» лидирует в <b>${st.join(", ")}</b>. Слабое место — <b style="color:var(--red)">${wl}</b>`:"Данные качества отсутствуют — требуется прогон lm-eval"}</p></div></div><div class="bars">${bars}</div><div class="raw-grid">${rc}</div>`}
function fs(){const a=$("#selA"),b=$("#selB");a.innerHTML=b.innerHTML=MODELS.map((m,i)=>`<option value="${i}">${m.name}</option>`).join("");const rk=enrich().sort((x,y)=>y.score-x.score);a.value=rk[0]._i;b.value=rk.length>1?rk[1]._i:0;a.onchange=b.onchange=rcmp}
function rp(cx,cy,r,v,i,n){const a=-Math.PI/2+i*2*Math.PI/n;const rr=r*(v/100);return[cx+rr*Math.cos(a),cy+rr*Math.sin(a)]}
function rcmp(){const A=MODELS[+$("#selA").value],B=MODELS[+$("#selB").value];const svg=$("#radar"),cx=180,cy=155,r=110,n=AXES.length;let g="";
for(const ring of[.25,.5,.75,1]){g+=`<polygon points="${AXES.map((a,i)=>rp(cx,cy,r*ring,100,i,n).join(",")).join(" ")}" fill="none" stroke="var(--line)" stroke-width="1"/>`}
AXES.forEach((a,i)=>{const[x,y]=rp(cx,cy,r,100,i,n);g+=`<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="var(--line)" stroke-width="1"/>`;const[lx,ly]=rp(cx,cy,r+18,100,i,n);const an=Math.abs(lx-cx)<6?"middle":lx>cx?"start":"end";g+=`<text x="${lx}" y="${ly+3}" text-anchor="${an}" fill="var(--muted)" font-family="JetBrains Mono" font-size="9.5">${a.label}</text>`});
const poly=(m,c)=>{const pts=AXES.map((a,i)=>rp(cx,cy,r,m[a.key]||0,i,n).join(",")).join(" ");let s=`<polygon points="${pts}" fill="${c}" fill-opacity=".14" stroke="${c}" stroke-width="2"/>`;AXES.forEach((a,i)=>{const[x,y]=rp(cx,cy,r,m[a.key]||0,i,n);s+=`<circle cx="${x}" cy="${y}" r="2.6" fill="${c}"/>`});return s};
g+=poly(A,"#35D0B8")+poly(B,"#9B8CFA");svg.innerHTML=g;
$("#delta").innerHTML=`<tr><td></td><td class="a">${A.name}</td><td class="b">${B.name}</td><td class="d">Δ</td></tr>`+AXES.map(a=>{const av=A[a.key]||0,bv=B[a.key]||0,d=av-bv;const cls=d>0?"up":d<0?"down":"";return`<tr><td>${a.label}</td><td class="a">${av>0?av:"—"}</td><td class="b">${bv>0?bv:"—"}</td><td class="d ${cls}">${av>0||bv>0?(d>0?"+":"")+d:""}</td></tr>`}).join("")}
function ra(){ru();rb();rd()}
ra();fs();rcmp();
</script>
</body>
</html>'''


def main():
    parser = argparse.ArgumentParser(description="Генератор дашборда llmbench")
    parser.add_argument("--out-dir", default="out", help="Директория с card_*.json")
    parser.add_argument("--dest", default="out/dashboard.html", help="Куда писать HTML")
    parser.add_argument("--title", default="LLM Bench Console")
    args = parser.parse_args()

    models = load_all_cards(args.out_dir)
    if not models:
        print(f"ERROR: карточки не найдены в {args.out_dir}")
        return 1

    html = generate_html(models, args.title)
    Path(args.dest).parent.mkdir(parents=True, exist_ok=True)
    with open(args.dest, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"OK: {len(models)} моделей → {args.dest} ({len(html)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
