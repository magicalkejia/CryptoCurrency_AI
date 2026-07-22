"""
app/server.py — WEB front-end (entry point) for the multi-agent crypto quant system.

Run:
    # 1) Results dashboard fed by a REAL experiment output dir (recommended for demo):
    EXP_DIR=data_storage/experiments/20260627T074907Z python app/server.py
    # 2) Live agent demo on REAL data (loads processed parquet):
    AGENT_DATA=real python app/server.py
    # 3) Plain (synthetic live agent, results tab still reads EXP_DIR if set):
    python app/server.py
    # then open http://127.0.0.1:8000

Two panels:
  * 实验结果仪表盘 — reads the ACTUAL files your HPC run produced
    (incremental_ladder.csv, real_returns_table.json, decision_vs_signal.json,
    decision_stack_metrics.json, frozen_config.json, console_log.txt) and renders
    the thesis highlights: the incremental proof ladder, the real-returns table,
    the decision-stack-vs-signal comparison, and the governance gates (PIT/ECE/PBO).
  * 实时 Agent 决策 — runs ONE decision through the actual agent graph and lights
    up the path the agents took, including the Risk agent's veto. AGENT_DATA=real
    runs it on real data.

The dashboard only READS frozen experiment outputs; it never re-runs the experiment,
so the demo is reproducible and matches exactly what you report.
"""
from __future__ import annotations
import os, sys, csv, glob, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flask import Flask, jsonify, request, render_template_string

app = Flask(__name__)
_STATE = {}

EXP_DIR = os.environ.get("EXP_DIR", "")
AGENT_DATA = os.environ.get("AGENT_DATA", "synthetic")
SYMBOLS_REAL = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
                "DOGE/USDT", "LTC/USDT", "LINK/USDT", "TRX/USDT", "ADA/USDT"]
AGENTS = [
    ("DataAgent", "Fetch features + PIT/data-quality checks", "data"),
    ("SignalResearchAgent", "Structured model + PatchTST forecasts", "signal"),
    ("NarrativeAgent", "LLM events/narrative -> structured factors", "narrative"),
    ("FusionAgent", "Regime + multi-modal fusion + meta model + calibration", "fusion"),
    ("RiskAgent", "Vol target / sizing / breaker — holds supreme veto", "risk"),
    ("ExecutionAgent", "Order plan + fills (paper trading)", "execution"),
    ("ReviewAgent", "Review / audit summary", "review"),
]

def _find_exp_dir():
    if EXP_DIR and os.path.isdir(EXP_DIR):
        return EXP_DIR
    cands = [c for c in sorted(glob.glob("data_storage/experiments/*"), reverse=True) if os.path.isdir(c)]
    return cands[0] if cands else ""

def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _read_ladder_csv(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return None

def _parse_governance(console_path):
    g = {}
    if not os.path.exists(console_path):
        return g
    try:
        txt = open(console_path, encoding="utf-8").read()
    except Exception:
        return g
    m = re.search(r"future_function_checks_passed\s*=\s*(\w+).*?violations=(\d+)", txt, re.S)
    if m: g["pit_passed"] = (m.group(1) == "True"); g["pit_violations"] = int(m.group(2))
    m = re.search(r"ece_calibrated[\":=\s]+([0-9.]+)", txt)
    if m: g["ece_calibrated"] = float(m.group(1))
    m = re.search(r"PBO\s*=\s*([0-9.]+)\s+over\s+(\d+)\s+combinations", txt)
    if m: g["pbo"] = float(m.group(1)); g["pbo_combos"] = int(m.group(2))
    m = re.search(r"beats TSMOM baseline.*?Sharpe\s*([+\-0-9.]+)\s*vs TSMOM\s*([+\-0-9.]+)", txt)
    if m: g["best_sharpe"] = float(m.group(1)); g["tsmom_sharpe"] = float(m.group(2))
    m = re.search(r"event diagnostic\]\s*cross-sectional IC t=([+\-0-9.]+)\s*vs.*?t=([+\-0-9.]+)", txt)
    if m: g["event_xs_t"] = float(m.group(1)); g["event_cm_t"] = float(m.group(2))
    m = re.search(r"event RISK->vol diagnostic\].*?IC=([+\-0-9.]+)\s+NW t=([+\-0-9.]+)", txt)
    if m: g["event_vol_ic"] = float(m.group(1)); g["event_vol_t"] = float(m.group(2))
    return g

def _load_results(exp_dir):
    if not exp_dir:
        return {"ok": False, "msg": "Experiment output directory not found. Set the EXP_DIR environment variable to a data_storage/experiments/<timestamp> directory."}
    return {"ok": True, "exp_dir": exp_dir,
            "real_returns": _read_json(os.path.join(exp_dir, "real_returns_table.json")),
            "ladder": _read_ladder_csv(os.path.join(exp_dir, "incremental_ladder.csv")),
            "decision_vs_signal": _read_json(os.path.join(exp_dir, "decision_vs_signal.json")),
            "decision_stack": _read_json(os.path.join(exp_dir, "decision_stack_metrics.json")),
            "directional_stack": _read_json(os.path.join(exp_dir, "directional_stack_step7.json")),
            "holdout": _read_json(os.path.join(exp_dir, "holdout_real_returns_table.json")),
            "frozen_config": _read_json(os.path.join(exp_dir, "frozen_config.json")),
            "governance": _parse_governance(os.path.join(exp_dir, "console_log.txt"))}

def _supports_real():
    try:
        import inspect
        from run_agent import build_graph
        return "real" in inspect.signature(build_graph).parameters
    except Exception:
        return False

def get_graph():
    if "graph" not in _STATE:
        import time as _t
        from run_agent import build_graph
        from crypto.live.oms import PaperBroker
        use_real = (AGENT_DATA == "real")
        _t0 = _t.time()
        print(f"[server] building agent graph (real={use_real}) ... this can take a while on first call", flush=True)
        try:
            if use_real and _supports_real():
                print("[server]   loading real market dataset + fitting model bundle ...", flush=True)
                g, cm, feats, fcols, fcfg = build_graph(symbols=tuple(SYMBOLS_REAL), real=True)
                _STATE["data_mode"] = "real"
            else:
                g, cm, feats, fcols, fcfg = build_graph(symbols=tuple(SYMBOLS_REAL))
                _STATE["data_mode"] = "synthetic"
        except Exception as e:
            import traceback; traceback.print_exc()
            g, cm, feats, fcols, fcfg = build_graph(symbols=tuple(SYMBOLS_REAL))
            _STATE["data_mode"] = "synthetic (real load failed)"
            print("[server] real load failed, fell back to synthetic:", e, flush=True)
        _STATE.update(graph=g, close_map=cm, feats=feats, fcols=fcols, fcfg=fcfg,
                      broker=PaperBroker(max_slippage_bps=3))
        nsym = len(feats["symbol"].unique()) if feats is not None and "symbol" in getattr(feats, "columns", []) else 0
        print(f"[server] agent graph ready in {_t.time()-_t0:.1f}s  mode={_STATE['data_mode']}  symbols={nsym}", flush=True)
    return _STATE


PAGE = r"""
<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Multi-Modal Agent Crypto Quant · Demo Console</title>
<style>
 :root{--bg:#0f1117;--panel:#161a23;--line:#232838;--muted:#8b93a7;--fg:#e6e6e6;
       --green:#36d399;--red:#f87272;--amber:#fbbd23;--blue:#60a5fa;--dim:#3a4151}
 *{box-sizing:border-box}
 body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;background:var(--bg);color:var(--fg)}
 header{padding:14px 22px;background:var(--panel);border-bottom:1px solid var(--line);display:flex;align-items:center;gap:16px}
 h1{font-size:17px;margin:0}.muted{color:var(--muted);font-size:13px}
 .tabs{display:flex;gap:8px;margin-left:auto}
 .tab{padding:7px 14px;border:1px solid var(--line);border-radius:8px;cursor:pointer;font-size:13px;color:var(--muted);background:#12151d}
 .tab.active{color:var(--fg);border-color:var(--blue);background:#172033}
 main{padding:20px 22px;max-width:1180px;margin:0 auto}
 .view{display:none}.view.active{display:block}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-bottom:16px}
 .card h2{font-size:14px;margin:0 0 12px;color:#cfd6e6;letter-spacing:.3px}
 .grid{display:grid;gap:14px}.g4{grid-template-columns:repeat(4,1fr)}.g3{grid-template-columns:repeat(3,1fr)}
 .kpi{background:#12151d;border:1px solid var(--line);border-radius:10px;padding:12px 14px;transition:all .3s ease}
 .kpi .v{font-size:22px;font-weight:600}.kpi .l{font-size:12px;color:var(--muted);margin-top:2px}
 .kpi.alert{background:#2a1410;border-color:var(--red);box-shadow:0 0 14px -4px var(--red)}
 .kpi.alert .v{color:var(--red)}.kpi.alert .l{color:#f0a59f}
 .kpi.warn{background:#2a2410;border-color:var(--amber);box-shadow:0 0 14px -4px var(--amber)}
 .kpi.warn .v{color:var(--amber)}.kpi.warn .l{color:#e8d08a}
 .pos{color:var(--green)}.neg{color:var(--red)}
 table{width:100%;border-collapse:collapse;font-size:13px}
 th,td{text-align:right;padding:7px 9px;border-bottom:1px solid var(--line)}
 th:first-child,td:first-child{text-align:left}
 th{color:var(--muted);font-weight:500}
 tr.hi td{background:#172033}
 .pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px}
 .pill.ok{background:#10301f;color:var(--green)}.pill.bad{background:#3a1414;color:var(--red)}
 .pill.warn{background:#332600;color:var(--amber)}
 select,button{font-size:13px;padding:8px 12px;border-radius:8px;border:1px solid var(--line);background:#12151d;color:var(--fg)}
 button{cursor:pointer;background:#1c2740;border-color:#2b3a5e}button:hover{background:#223052}
 .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
 .pipe{display:flex;flex-direction:column;gap:8px}
 .node{display:flex;gap:10px;align-items:center;padding:10px 12px;border-radius:10px;border:1px solid var(--line);background:#12151d;transition:border-color .35s ease, box-shadow .35s ease, background .35s ease}
 .node.ran{border-color:var(--green);box-shadow:0 0 0 1px var(--green) inset, 0 0 12px -2px var(--green)}
 .node.ran .dot{background:var(--green)}
 .node.veto{border-color:var(--red);box-shadow:0 0 0 1px var(--red) inset, 0 0 14px -2px var(--red)}
 .node.veto .dot{background:var(--red)}
 .node.warn{border-color:var(--amber);box-shadow:0 0 0 1px var(--amber) inset, 0 0 12px -2px var(--amber)}
 .node.warn .dot{background:var(--amber)}
 .node .dot{width:10px;height:10px;border-radius:50%;background:var(--dim);flex:0 0 auto;transition:background .35s ease}
 .node .nm{font-weight:600;font-size:13px}.node .rl{color:var(--muted);font-size:12px}
 .small{font-size:12px;color:var(--muted)}
 pre{background:#0b0e14;border:1px solid var(--line);border-radius:10px;padding:12px;overflow:auto;font-size:12px;color:#c7d0e0}
 code{color:var(--amber)} a{color:var(--blue)} ol{padding-left:18px}
</style></head><body>
<header>
  <h1>Multi-Modal Agent Crypto Quant <span class="muted">· Demo Console</span></h1>
  <div class="tabs">
    <div class="tab active" data-v="results">Results Dashboard</div>
    <div class="tab" data-v="live">Live Agent Decision</div>
    <div class="tab" data-v="arch">Agent Architecture</div>
  </div>
</header>
<main>
<section class="view active" id="v-results"><div id="results-root"><div class="card muted">Loading…</div></div></section>
<section class="view" id="v-live">
  <div class="card">
    <h2>Live Agent Decision (data source: <span id="dmode" class="muted">…</span>)</h2>
    <div class="row">
      <select id="sym"></select>
      <label class="small">Breaker injection (what-if):</label>
      <select id="cb"><option value="auto">Real breaker (from data)</option><option value="0" selected>L0 normal</option><option value="1">L1 warn</option>
        <option value="2">L2 delever</option><option value="3">L3 halt (veto)</option></select>
      <button onclick="decide()">Run one decision</button>
      <button onclick="resetPos()">Reset positions</button>
    </div>
    <p class="small">After you click, the pipeline below lights up along the <b style="color:var(--blue)">path this decision actually took</b>: <span style="color:var(--green)">green = executed</span>, <span style="color:var(--red)">red = risk veto / stop</span>, grey = not reached.</p>
    <p class="small">Demoing a real breaker (synthetic mode): set injection to "Real breaker (from data)" and pick <code>SOL/USDT</code>→L1 warn (>20%), <code>DOGE/USDT</code>→L2 delever (>25%), <code>XRP/USDT</code>→L3 halt/veto (>30%); all other coins are normal.</p>
  </div>
  <div class="grid g3">
    <div class="card" style="grid-column:span 1"><h2>Agent Pipeline</h2><div class="pipe" id="pipe"></div></div>
    <div class="card" style="grid-column:span 2"><h2>This Decision</h2>
      <div class="grid g4" id="dkpi"></div><pre id="djson" style="margin-top:12px">—</pre></div>
  </div>
</section>
<section class="view" id="v-arch">
  <div class="card">
    <h2>Agent architecture — execution order, inputs and outputs</h2>
    <div class="row" style="margin-bottom:12px">
      <button id="archPlay">▶ Play</button>
      <button id="archPrev">‹ Prev</button>
      <button id="archNext">Next ›</button>
    </div>
    <div style="display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap">
      <div id="archPipe" style="width:250px;flex:0 0 auto"></div>
      <div id="archDetail" style="flex:1;min-width:300px"></div>
    </div>
  </div>
</section>
</main>
<script>
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
$$(".tab").forEach(t=>t.onclick=()=>{$$(".tab").forEach(x=>x.classList.remove("active"));t.classList.add("active");
  $$(".view").forEach(v=>v.classList.remove("active"));$("#v-"+t.dataset.v).classList.add("active");
  if(t.dataset.v==="live"){initLive();}});
const fmtPct=x=>(x==null||isNaN(x))?"—":(x*100).toFixed(1)+"%";
const fmtN=x=>(x==null||isNaN(x))?"—":(+x).toFixed(2);
const cls=x=>x>=0?"pos":"neg";
const STRAT={
 "Step0_TSMOM (benchmark)":"Momentum benchmark (TSMOM)",
 "Step5_fusion (pure signal)":"ML multi-modal signal (pure, no risk stack)",
 "Step7 + circuit-breaker (deliverable+risk)":"Main deliverable + circuit breaker",
 "Step7 + directional stack (full risk)":"Main deliverable + full directional risk stack",
 "Step7_tsmom_fusion (MAIN deliverable)":"MAIN deliverable \u00b7 ML + momentum fusion",
 "decision_stack (risk+port+CB)":"ML signal + neutral risk stack (with breaker)",
 "decision_stack(risk+port+CB)":"ML signal + neutral risk stack (with breaker)",
 "decision_stack_noCB (risk+port)":"ML signal + neutral risk stack (no breaker)",
 "Step6_meta_gate (signal+gate)":"ML signal + meta-gate (harmful \u2014 kept honest)",
 "Step5 + neutral decision stack":"ML signal + neutral risk stack"};
const LSTEP={
 "Step0_baseline_tsmom":"Step0 \u00b7 TSMOM baseline",
 "Step1_market":"Step1 \u00b7 market",
 "Step1b_+xsmom":"Step1b \u00b7 market+xsmom",
 "Step2_+onchain":"Step2 \u00b7 market+onchain (diagnostic branch)",
 "Step3_+narrative":"Step3 \u00b7 market+narrative",
 "Step3b_+event":"Step3b \u00b7 market+narrative+event",
 "Step4_+patchtst":"Step4 \u00b7 market+narrative+event+patchtst",
 "Step5_fusion":"Step5 \u00b7 skill-weighted fusion",
 "Step6_meta_gate":"Step6 \u00b7 fusion+meta-gate (branch)",
 "Step7_tsmom_fusion":"Step7 \u00b7 fusion+TSMOM (MAIN)",
 "Step8_onchain_overlay":"Step8 \u00b7 on-chain overlay (branch)",
 "Step3c_event_overlay":"Step3c \u00b7 event overlay (branch)",
 "Step3d_event_risk_gate":"Step3d \u00b7 news-risk gate (branch)"};
const dName=k=>STRAT[k]||k, lName=k=>LSTEP[k]||k;
const DEV_SPAN="Feb 2021 \u2013 16 May 2025", HO_SPAN="16 May 2025 \u2013 12 Jun 2026";
async function loadResults(){
  const r=await fetch("/api/results").then(r=>r.json());const root=$("#results-root");
  if(!r.ok){root.innerHTML=`<div class="card"><h2>Results</h2><p class="muted">${r.msg}</p></div>`;return;}
  let h="";const rr=r.real_returns||{};const k=rr["Step7_tsmom_fusion (MAIN deliverable)"];
  if(k){h+=`<div class="card"><h2>Main Deliverable (ML + Momentum Fusion) \u00b7 Development Period: ${DEV_SPAN}</h2><div class="grid g4">
      <div class="kpi"><div class="v ${cls(k.ann_return)}">${fmtPct(k.ann_return)}</div><div class="l">Annual return</div></div>
      <div class="kpi"><div class="v ${cls(k.cum_return)}">${fmtPct(k.cum_return)}</div><div class="l">Cumulative return</div></div>
      <div class="kpi"><div class="v">${fmtPct(k.max_drawdown)}</div><div class="l">Max drawdown</div></div>
      <div class="kpi"><div class="v">${fmtN(k.sharpe)}</div><div class="l">Sharpe</div></div>
    </div><p class="small">Experiment dir: <code>${r.exp_dir}</code></p></div>`;}
  if(Object.keys(rr).length){h+=`<div class="card"><h2>Real-Returns Comparison \u00b7 development period, net of costs</h2><table><thead><tr>
      <th>Strategy</th><th>Annual</th><th>Cumulative</th><th>Max DD</th><th>Sharpe</th></tr></thead><tbody>`;
    for(const [name,s] of Object.entries(rr)){const hi=name.includes("MAIN")?"hi":"";
      h+=`<tr class="${hi}"><td>${dName(name)}</td><td class="${cls(s.ann_return)}">${fmtPct(s.ann_return)}</td>
        <td class="${cls(s.cum_return)}">${fmtPct(s.cum_return)}</td><td>${fmtPct(s.max_drawdown)}</td><td>${fmtN(s.sharpe)}</td></tr>`;}
    h+=`</tbody></table></div>`;}
  if(r.ladder&&r.ladder.length){h+=`<div class="card"><h2>Incremental Proof Ladder \u2014 does each modality add a significant increment?</h2>
      <p class="small">Main-chain rows are cumulative (each adds one modality to the previous main-chain row); branch rows are diagnostics off the chain. incr_NW_t &gt; 2 marks a significant increment \u2014 honestly, most alternative-data modalities show none, a finding rather than a defect.</p>
      <table><thead><tr><th>Stage</th><th>Sharpe</th><th>DSR</th><th>incr_t (vs prev main-chain)</th><th>incr_t (vs TSMOM)</th></tr></thead><tbody>`;
    for(const row of r.ladder){const step=row.step||row[""]||Object.values(row)[0]||"";
      const it=row.incr_NW_t,ib=row.incr_NW_t_base;const sig=(Math.abs(+it)>2)?"hi":"";
      h+=`<tr class="${sig}"><td>${lName(step)}</td><td>${fmtN(row.sharpe_ann)}</td><td>${fmtN(row.deflated_sharpe)}</td>
        <td>${fmtN(it)}</td><td>${fmtN(ib)}</td></tr>`;}
    h+=`</tbody></table></div>`;}
  const dv=r.decision_vs_signal&&r.decision_vs_signal.metrics;
  if(dv){h+=`<div class="card"><h2>Risk Stack vs Pure Signal \u2014 what risk control buys: drawdown</h2>
      <table><thead><tr><th>Strategy</th><th>Sharpe</th><th>Annual</th><th>Max DD</th></tr></thead><tbody>`;
    for(const [name,s] of Object.entries(dv)){h+=`<tr><td>${dName(name)}</td><td>${fmtN(s.sharpe)}</td>
        <td class="${cls(s.ann_return)}">${fmtPct(s.ann_return)}</td><td>${fmtPct(s.max_drawdown)}</td></tr>`;}
    h+=`</tbody></table></div>`;}
  const ds=r.decision_stack||{}, dir=(r.directional_stack&&r.directional_stack.metrics)||null;
  if(dir){
    const n_sh=ds.strategy_sharpe, n_ar=ds.strategy_annual_return, n_dd=ds.strategy_max_drawdown;
    const d_sh=dir.strategy_sharpe, d_ar=dir.strategy_annual_return, d_dd=dir.strategy_max_drawdown;
    const di=(r.directional_stack.info)||{};
    h+=`<div class="card"><h2>Two Strategy Paradigms \u00d7 Matched Risk Control</h2>
      <table><thead><tr><th>Pipeline</th><th>Signal</th><th>Risk paradigm</th><th>Sharpe</th><th>Annual</th><th>Max DD</th></tr></thead><tbody>
      <tr><td>Neutral book + neutral risk stack</td><td>Cross-sectional coin selection</td><td>Correlation haircut \u00b7 cluster cap \u00b7 breaker</td>
        <td>${fmtN(n_sh)}</td><td class="${cls(n_ar)}">${fmtPct(n_ar)}</td><td>${fmtPct(n_dd)}</td></tr>
      <tr class="hi"><td>Directional book + directional risk stack</td><td>ML + momentum (directional)</td><td>Net-exposure cap \u00b7 vol target \u00b7 breaker</td>
        <td>${fmtN(d_sh)}</td><td class="${cls(d_ar)}">${fmtPct(d_ar)}</td><td>${fmtPct(d_dd)}</td></tr>
      </tbody></table>
      <p class="small">The neutral book uses neutral risk control (long/short hedged, correlation-neutralized); the directional book uses directional risk control (net-exposure cap, portfolio vol target). The directional stack keeps avg |net exposure| = <code>${fmtN(di.avg_abs_net_exposure)}</code> (cap ${fmtN(di.net_exposure_cap)}) \u2014 exactly where the momentum alpha lives; a neutral overlay would cancel it. Both pipelines are production-ready.</p></div>`;
  }
  const g=r.governance||{};
  const ho=r.holdout||null;
  if(ho){
    const order=["Step5_fusion (pure signal)","Step7_tsmom_fusion (MAIN deliverable)",
      "Step5 + neutral decision stack","Step7 + directional stack (full risk)","Step0_TSMOM (benchmark)"];
    h+=`<div class="card" style="border:2px solid var(--blue)"><h2>Out-of-Sample Final Test \u00b7 frozen config \u00b7 ${HO_SPAN}</h2>
      <p class="small">A once-only confirmatory test on untouched data, same conventions as development \u2014 the part that actually proves deliverable quality.</p>
      <table><thead><tr><th>Strategy</th><th>Annual</th><th>Cumulative</th><th>Vol</th><th>Max DD</th><th>Sharpe</th><th>Sortino</th><th>Calmar</th><th>Turnover</th><th>Win rate</th></tr></thead><tbody>`;
    for(const nm of order){ const s=ho[nm]; if(!s) continue;
      const hi=nm.includes("MAIN")?"hi":"";
      h+=`<tr class="${hi}"><td>${dName(nm)}</td><td class="${cls(s.ann_return)}">${fmtPct(s.ann_return)}</td>
        <td class="${cls(s.cum_return)}">${fmtPct(s.cum_return)}</td><td>${fmtPct(s.ann_vol)}</td>
        <td>${fmtPct(s.max_drawdown)}</td><td>${fmtN(s.sharpe)}</td><td>${fmtN(s.sortino)}</td>
        <td>${fmtN(s.calmar)}</td><td>${s.ann_turnover!=null?Math.round(s.ann_turnover):'\u2014'}</td>
        <td>${fmtPct(s.win_rate)}</td></tr>`;}
    h+=`</tbody></table></div>`;
  }
  if(Object.keys(g).length){
    const pit=g.pit_passed?`<span class="pill ok">PASS \u00b7 0 leaks</span>`:`<span class="pill bad">FAILED</span>`;
    const ece=g.ece_calibrated!=null?`<span class="pill ${g.ece_calibrated<0.05?'ok':'warn'}">ECE ${g.ece_calibrated.toFixed(4)}</span>`:"\u2014";
    const pbo=g.pbo!=null?`<span class="pill ${g.pbo<0.5?'ok':'warn'}">PBO ${g.pbo.toFixed(3)}</span>`:"\u2014";
    h+=`<div class="card"><h2>Three Governance Gates \u00b7 result credibility</h2><div class="grid g3">
      <div class="kpi"><div class="v">${pit}</div><div class="l">Point-in-time correctness (no look-ahead)</div></div>
      <div class="kpi"><div class="v">${ece}</div><div class="l">Probability calibration error (&lt;0.05 good)</div></div>
      <div class="kpi"><div class="v">${pbo}</div><div class="l">Probability of backtest overfitting (&lt;0.5 good)</div></div></div>`;
    if(g.event_cm_t!=null||g.event_vol_t!=null){h+=`<p class="small" style="margin-top:12px">Honest news/event tests:
      cross-sectional t=<code>${fmtN(g.event_xs_t)}</code>, common-mode t=<code>${fmtN(g.event_cm_t)}</code>, volatility-forecast t=<code>${fmtN(g.event_vol_t)}</code>
      \u2014 the signal concentrates in the common-mode (market-level) dimension but is not tradeable as an increment; four deployments were tested.</p>`;}
    h+=`</div>`;}
  root.innerHTML=h||`<div class="card muted">Experiment directory found, but no displayable files were parsed.</div>`;
}
let _liveReady=false, _liveLoading=false;
const FALLBACK_SYMS=["BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT","DOGE/USDT","LTC/USDT","LINK/USDT","TRX/USDT","ADA/USDT"];
function fillSymbols(syms){$("#sym").innerHTML=(syms||[]).map(s=>`<option>${s}</option>`).join("");}
async function initLive(){
  if(_liveReady||_liveLoading) return;
  _liveLoading=true;
  console.log("[live] initLive: fetching /api/agents ...");
  $("#dmode").textContent="Building the model graph… (first call is slow — loading real data; see the backend console for progress)";
  // Pre-fill the dropdown immediately so the UI is usable even while the graph builds.
  fillSymbols(FALLBACK_SYMS);
  const ctrl=new AbortController();
  const timer=setTimeout(()=>ctrl.abort(), 360000); // 6-min ceiling (real build can take ~300s)
  try{
    const res=await fetch("/api/agents",{signal:ctrl.signal});
    clearTimeout(timer);
    console.log("[live] /api/agents status",res.status);
    if(!res.ok){throw new Error("HTTP "+res.status);}
    const a=await res.json();
    console.log("[live] /api/agents payload",a);
    if(a.error){throw new Error(a.error);}
    $("#dmode").textContent=a.data_mode||"…";
    const syms=(a.symbols&&a.symbols.length)?a.symbols:FALLBACK_SYMS;
    fillSymbols(syms);
    renderPipe(a.agents||[],null);
    _liveReady=true;
    console.log("[live] ready, symbols:",syms.length);
  }catch(e){
    clearTimeout(timer);
    console.error("[live] initLive failed:",e);
    const msg=(e.name==="AbortError")?"Timed out (>6 min). The real-data/model build is too slow — check the backend console.":("Load failed: "+e.message);
    $("#dmode").innerHTML=`<span style="color:var(--red)">${msg}</span> <a href="#" onclick="_liveReady=false;_liveLoading=false;initLive();return false;">retry</a>`;
    // keep the fallback dropdown so the user can still try a decision
    renderPipe([{agent:"(graph unavailable)",role:"Backend graph build failed; the dropdown is pre-filled with the default 10 symbols",category:"data"}],null);
  }finally{ _liveLoading=false; }
}
function pipeNodeHTML(a,st){
  return `<div class="node ${st}" data-cat="${a.category}"><span class="dot"></span>
    <div><div class="nm">${a.agent}</div><div class="rl">${a.role}</div></div></div>`;
}
function renderPipe(agents,path){
  $("#pipe").innerHTML=agents.map(a=>{let st="";
    if(path){if(path.veto===a.category)st="veto";else if(path.ran&&path.ran.includes(a.category))st="ran";}
    return pipeNodeHTML(a,st);}).join("");
}
function sleep(ms){return new Promise(r=>setTimeout(r,ms));}
async function animatePipe(agents,path,onRisk){
  $("#pipe").innerHTML=agents.map(a=>pipeNodeHTML(a,"")).join("");
  await sleep(150);
  for(const a of agents){
    const el=$(`#pipe .node[data-cat="${a.category}"]`);
    if(!el){ if(a.category==="risk"&&onRisk){onRisk();} continue; }
    if(path.veto===a.category){
      el.classList.add("veto");
      if(onRisk) onRisk();        // reveal decision the moment Risk vetoes
      await sleep(520); break;
    } else if(path.ran&&path.ran.includes(a.category)){
      // RiskAgent shows amber when circuit breaker is at L1/L2 (delever, not halt)
      if(a.category==="risk"&&path.cbWarn){ el.classList.add("warn"); }
      else { el.classList.add("ran"); }
      if(a.category==="risk"&&onRisk) onRisk();  // reveal decision once Risk passes
    }
    await sleep(440);
  }
}
function cbClass(lvl){ lvl=+lvl||0; return lvl>=3?"alert":(lvl>=1?"warn":""); }
function riskClass(rl){ return (rl==="blocked")?"alert":(rl==="medium"?"":""); }
async function decide(){
  if(!_liveReady){await initLive(); if(!_liveReady){alert("Agent graph not ready — check the backend log.");return;}}
  const sym=$("#sym").value,cbRaw=$("#cb").value,cb=(cbRaw==="auto"?0:+cbRaw),cbAuto=(cbRaw==="auto");
  if(!sym){alert("Pick a symbol first");return;}
  // clear right panel while the pipeline runs, so it doesn't pre-empt the animation
  $("#djson").textContent="Deciding…";
  $("#dkpi").innerHTML="";
  let r;
  try{
    r=await fetch("/api/decide",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({symbol:sym,cb_level:cbRaw})}).then(r=>r.json());
  }catch(e){$("#djson").textContent="Request failed: "+e.message;return;}
  if(r.error){alert(r.error);$("#djson").textContent=r.error;return;}const d=r.decision||{};
  const ran=(r.audit_log||[]).map(x=>x.category);
  const cbReal=+(d.circuit_breaker_level||0);
  const veto=((d.risk_level==="blocked")||cb>=3||cbReal>=3)?"risk":null;
  const cbWarn=(cbReal===1||cbReal===2||cb===1||cb===2);
  const a=await fetch("/api/agents").then(r=>r.json());
  // build the structured-output renderer, but only fire it when the pipeline
  // reaches the RiskAgent node (so the right panel reveals in sync with the path).
  const reveal=()=>{
    const cbc=cbClass(d.circuit_breaker_level), rkc=riskClass(d.risk_level);
    const kpis=[
      ["Direction",d.primary_direction||d.action||"—",""],
      ["Target position",d.target_position!=null?(+d.target_position).toFixed(3):"—",""],
      ["Breaker level","L"+(+d.circuit_breaker_level||0),cbc],
      ["Risk level",d.risk_level||"—",rkc],
      ["Realized drawdown",r.real_drawdown!=null?(r.real_drawdown*100).toFixed(1)+"%":"—",cbc]
    ];
    $("#dkpi").innerHTML=kpis.map(([l,v,c])=>`<div class="kpi ${c}"><div class="v" style="font-size:16px">${v}</div><div class="l">${l}</div></div>`).join("");
    $("#djson").textContent=JSON.stringify(d,null,2);
  };
  await animatePipe(a.agents,{ran,veto,cbWarn},reveal);
  reveal();  // safety: ensure it's shown even if risk node was absent
}
async function resetPos(){await fetch("/api/reset_positions",{method:"POST"});alert("Positions reset");}
/* ---------------- Agent architecture tab (presentation animation) -------- */
(function(){
  var A=[
    {name:"Data agent",skills:"get_feature_row \u00b7 check_data_quality",
     input:[["2 fields","symbol + decision time"]],
     output:[["45 features","1 point-in-time feature row:<br>\u2022 27 market \u2014 multi-horizon returns, volatility, ATR, range, RSI, momentum, funding<br>\u2022 6 on-chain \u2014 z-scores + growth rates<br>\u2022 12 PatchTST \u2014 4 horizon forecasts + 8-dim embedding"],
             ["1 score","data_quality_score in [0, 1] \u2014 1 \u2212 missing/45, forced to 0 on any PIT violation"]],
     ex:[["call","get_feature_row('DOGE/USDT', 2026-06-12 04:01)"],
         ["row excerpt","ret_4h \u22121.2% \u00b7 vol_24h 3.1% \u00b7 RSI 34 \u00b7 funding \u22120.010% \u00b7 onchain_z \u22120.8 \u00b7 patchtst_forecast_4h +0.0004"],
         ["quality","45/45 features present, availability_ts \u2264 decision time \u2192 data_quality_score = 1.0"]]},
    {name:"Signal research agent",skills:"signal_infer",
     input:[["45 features","the feature row from the data agent + the feature column list"]],
     output:[["4 channels","base_model_pred: the PatchTST forecasts for 4h, 12h, 24h, 3d"],
             ["1 entry","audit-log record (category: signal)"]],
     ex:[["surfaces","patchtst_forecast_4h +0.0004 \u00b7 12h +0.0007 \u00b7 24h +0.0007 \u00b7 3d +0.0024"],
         ["audit","signal_infer logged: category=signal, ok=true, 0.4 ms"]]},
    {name:"Narrative agent",skills:"narrative_infer",
     input:[["factors","this symbol's news/event factors at the decision time \u2014 produced upstream by CryptoBERT (sentiment over 34,284 articles) and DeepSeek LLM (63,007 structured events)"]],
     output:[["4 fields","sentiment \u00b7 severity \u00b7 event type \u00b7 stub flag \u2014 structured factors only, never a trade direction"]],
     ex:[["factors","sentiment \u22120.2 \u00b7 severity low \u00b7 event_type none \u00b7 narrative_stub = true (offline mode)"],
         ["meaning","no fresh DOGE news at this timestamp \u2192 the narrative alpha stays neutral"]]},
    {name:"Fusion agent",skills:"detect_regime \u00b7 fusion_infer \u00b7 compute_confidence",
     input:[["45 features","the feature row"],
            ["narrative","the narrative agent's factors"],
            ["1 bundle","the frozen LightGBM two-stage bundle, trained offline, config hash locked"]],
     proc:[["LightGBM","this is where LightGBM actually runs \u2014 stage 1: 3-class triple-barrier classifier on the 45 features \u2192 market alpha = p(up) \u2212 p(down); stage 2: meta-label + Platt calibration (ECE 0.0056) \u2192 the calibrated trade probability used for sizing"],
           ["fusion","the market alpha is skill-weight fused with the narrative/event alphas (weights earned from data: market 0.86 \u00b7 narrative 0.08 \u00b7 event 0.06) + regime detection"]],
     output:[["5 fields","combined_alpha \u00b7 primary_direction (long / short / flat) \u00b7 meta_trade_prob_calibrated \u00b7 regime \u00b7 confidence"]],
     ex:[["bundle.predict","p(down)=0.62 \u00b7 p(flat)=0.10 \u00b7 p(up)=0.28 \u2192 market alpha = 0.28 \u2212 0.62 = \u22120.34"],
         ["fuse + calibrate","combined_alpha \u22120.99 (z-scored) \u00b7 meta_trade_prob 0.92 \u00b7 regime trending_down \u00b7 confidence 0.78"],
         ["read","strongly bearish with high conviction \u2192 propose SHORT"]]},
    {name:"Risk agent",skills:"compute_circuit_breaker \u00b7 risk_size_and_gate",veto:1,
     input:[["5 fields","all fusion outputs"],
            ["from row","ATR + volatility for sizing"],
            ["1 value","trailing drawdown (or an injected what-if breaker level)"]],
     output:[["7 fields","target_position (signed: + long / \u2212 short) \u00b7 risk_level \u00b7 circuit_breaker_level (L0\u2013L3) \u00b7 risk_approved \u00b7 stop_loss \u00b7 take_profit \u00b7 vol_target_scalar"],
             ["in place","L2 halves the position, L3 zeroes it \u2014 the reduced position flows forward, control never loops back"]],
     ex:[["sizing","conviction \u00d7 vol target \u2192 \u22120.25 (a short at the 25% per-symbol cap)"],
         ["breaker","trailing 90-day drawdown 17.6% > 15% \u2192 L2 delever \u2192 position \u00d7 0.5 = \u22120.125"],
         ["verdict","approved \u00b7 risk_level medium \u00b7 target_position = \u22120.125 (a 12.5% short)"]]},
    {name:"Execution agent",skills:"execute_paper",
     input:[["3 values","target_position + current holding + reference price"]],
     output:[["2 fields","execution_status (filled / not_submitted) \u00b7 filled_price \u2014 the order is the delta only (target \u2212 holding), costed at fee 4bp + slippage 3bp per side"]],
     ex:[["netting","target \u22120.125 \u2212 current 0 \u2192 sell order for 12.5% of the book"],
         ["fill","execution_status = filled \u00b7 costs deducted: fee 4bp + slippage 3bp"]]},
    {name:"Review agent",skills:"review_decision",
     input:[["full state","the complete decision state accumulated across all six agents"]],
     output:[["1 record","the decision record"],
             ["1 audit id","symbol + time + model + data + code + config hash + environment hash \u2014 every number is reproducible from this id alone"]],
     ex:[["record","action=short \u00b7 target=\u22120.125 \u00b7 cb=L2 \u00b7 reason='drawdown>15% (L2 delever)'"],
         ["audit id","DOGEUSDT__2026061204__model=demo__data=demo__code=dev__config=03b05360dcde__env=\u2026"]]}
  ];
  var pipeEl=document.getElementById("archPipe");
  if(!pipeEl) return;
  A.forEach(function(a,i){
    var n=document.createElement("div");
    n.id="archN"+i;
    n.style.cssText="border:1px solid var(--line);border-radius:10px;padding:9px 12px;cursor:pointer;transition:all .3s;background:#12151d;";
    n.innerHTML='<div style="display:flex;align-items:center;gap:8px"><span id="archNum'+i+'" style="font-size:12px;font-weight:600;color:var(--muted);min-width:14px">'+(i+1)+'</span><span style="font-size:13px;font-weight:600">'+a.name+'</span>'+(a.veto?'<span class="pill bad" style="margin-left:auto">veto</span>':'')+'</div>'+
      '<div style="font-size:11px;color:var(--muted);margin-top:2px;font-family:ui-monospace,Consolas,monospace">'+a.skills+'</div>';
    n.onclick=function(){if(archPlaying)archStop();archRender(i);};
    pipeEl.appendChild(n);
    if(i<A.length-1){
      var ar=document.createElement("div");
      ar.style.cssText="text-align:center;color:var(--dim);font-size:13px;line-height:1;padding:3px 0";
      ar.textContent="\u2193";
      pipeEl.appendChild(ar);
    }
  });
  function archRows(title,items,accent){
    var h='<div style="font-size:13px;font-weight:600;color:'+(accent?'var(--blue)':'var(--muted)')+';margin:0 0 6px">'+title+'</div>';
    items.forEach(function(it){
      h+='<div style="display:grid;grid-template-columns:88px minmax(0,1fr);gap:10px;padding:8px 10px;border-radius:8px;margin-bottom:5px;background:'+(accent?'#172033':'#12151d')+';border:1px solid '+(accent?'#2b3a5e':'var(--line)')+'">'+
        '<div style="font-size:12px;font-weight:600;color:'+(accent?'var(--blue)':'var(--fg)')+'">'+it[0]+'</div>'+
        '<div style="font-size:13px;line-height:1.55;color:'+(accent?'#c7d6f2':'var(--fg)')+'">'+it[1]+'</div></div>';
    });
    return '<div style="margin-bottom:14px">'+h+'</div>';
  }
  function archEx(items){
    var h='<div style="font-size:13px;font-weight:600;color:var(--amber);margin:0 0 6px">Worked example \u00b7 DOGE/USDT, one demo decision (breaker L2)</div>';
    items.forEach(function(it){
      h+='<div style="display:grid;grid-template-columns:98px minmax(0,1fr);gap:10px;padding:8px 10px;border-radius:8px;margin-bottom:5px;background:#1b1710;border:1px solid #3a2f14">'+
        '<div style="font-size:12px;font-weight:600;color:var(--amber)">'+it[0]+'</div>'+
        '<div style="font-size:12.5px;line-height:1.6;color:#e8d9a8;font-family:ui-monospace,Consolas,monospace">'+it[1]+'</div></div>';
    });
    return '<div style="margin-bottom:14px">'+h+'</div>';
  }
  var archCur=0,archPlaying=false,archTimer=null;
  function archRender(i){
    archCur=i;
    A.forEach(function(a,k){
      var n=document.getElementById("archN"+k),on=(k===i);
      n.style.borderColor=on?"var(--blue)":"var(--line)";
      n.style.background=on?"#172033":"#12151d";
      n.style.boxShadow=on?"0 0 10px -3px var(--blue)":"none";
      n.style.opacity=(k<=i)?"1":"0.5";
      document.getElementById("archNum"+k).style.color=on?"var(--blue)":"var(--muted)";
    });
    var a=A[i];
    document.getElementById("archDetail").innerHTML=
      '<div style="font-size:16px;font-weight:600;margin:0 0 12px">'+(i+1)+' \u00b7 '+a.name+'</div>'+
      archRows("Input",a.input)+(a.proc?archRows("Processing",a.proc):'')+archRows("Output",a.output,1)+(a.ex?archEx(a.ex):'');
  }
  function archStop(){archPlaying=false;if(archTimer)clearInterval(archTimer);
    document.getElementById("archPlay").textContent=archCur>=A.length-1?"\u21ba Replay":"\u25b6 Play";}
  function archPlayFn(){
    if(archPlaying){archStop();return;}
    archPlaying=true;if(archCur>=A.length-1)archRender(0);
    document.getElementById("archPlay").textContent="\u275a\u275a Pause";
    archTimer=setInterval(function(){if(archCur<A.length-1)archRender(archCur+1);else archStop();},5000);
  }
  document.getElementById("archPlay").onclick=archPlayFn;
  document.getElementById("archNext").onclick=function(){if(archPlaying)archStop();if(archCur<A.length-1)archRender(archCur+1);};
  document.getElementById("archPrev").onclick=function(){if(archPlaying)archStop();if(archCur>0)archRender(archCur-1);};
  archRender(0);
})();
loadResults();
</script></body></html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)

@app.route("/api/results")
def api_results():
    return jsonify(_load_results(_find_exp_dir()))

@app.route("/api/agents")
def api_agents():
    print("[server] /api/agents requested (will build graph if first time)", flush=True)
    try:
        s = get_graph()
        syms = list(s["feats"]["symbol"].unique()) if s.get("feats") is not None else SYMBOLS_REAL
        return jsonify({"agents": [{"agent": a, "role": d, "category": c} for a, d, c in AGENTS],
                        "symbols": syms, "data_mode": _STATE.get("data_mode", "synthetic")})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": f"build_graph failed: {e}",
                        "agents": [{"agent": a, "role": d, "category": c} for a, d, c in AGENTS],
                        "symbols": []}), 200

@app.route("/api/decide", methods=["POST"])
def api_decide():
    from run_agent import run_one
    try:
        s = get_graph()
        body = request.get_json(force=True)
        sym = body.get("symbol")
        if not sym:
            return jsonify({"error": "no symbol provided"}), 200
        cb_raw = body.get("cb_level", 0)
        auto_cb = (str(cb_raw) == "auto")
        cb_level = 0 if auto_cb else int(cb_raw)
        out = run_one(s["graph"], s["close_map"], s["feats"], s["fcols"], s["fcfg"],
                      sym, cb_level=cb_level, auto_cb=auto_cb, broker=s["broker"])
        if out is None:
            avail = sorted(s["feats"]["symbol"].unique().tolist()) if s.get("feats") is not None else []
            return jsonify({"error": f"No decision data for {sym}. Available: {', '.join(avail) or '(none)'}"}), 200
        return jsonify(out)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": f"Decision failed: {e}"}), 200

@app.route("/api/reset_positions", methods=["POST"])
def api_reset():
    from crypto.live.oms import PaperBroker
    _STATE["broker"] = PaperBroker(max_slippage_bps=3)
    return jsonify({"ok": True})

if __name__ == "__main__":
    print(f"[server] experiment dir : {_find_exp_dir() or '(none — set EXP_DIR)'}")
    print(f"[server] live agent data: {AGENT_DATA}")
    print("[server] open http://127.0.0.1:8000")
    app.run(host="127.0.0.1", port=8000, debug=False, threaded=True)
