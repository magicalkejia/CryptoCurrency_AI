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
    ("DataAgent", "拉取特征 + PIT/数据质量检查", "data"),
    ("SignalResearchAgent", "结构化模型 + PatchTST 预测", "signal"),
    ("NarrativeAgent", "LLM 事件/叙事 -> 结构化因子", "narrative"),
    ("FusionAgent", "Regime + 多模态融合 + 元模型 + 校准", "fusion"),
    ("RiskAgent", "波动率目标/仓位/熔断 —— 拥有最高否决权", "risk"),
    ("ExecutionAgent", "订单计划 + 成交(纸交易)", "execution"),
    ("ReviewAgent", "复盘/审计摘要", "review"),
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
        return {"ok": False, "msg": "未找到实验输出目录。请用 EXP_DIR 环境变量指向 data_storage/experiments/<时间戳> 目录。"}
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
<title>多模态 Agent 加密量化 · 演示控制台</title>
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
  <h1>多模态 Agent 加密量化 <span class="muted">· 演示控制台</span></h1>
  <div class="tabs">
    <div class="tab active" data-v="results">实验结果仪表盘</div>
    <div class="tab" data-v="live">实时 Agent 决策</div>
  </div>
</header>
<main>
<section class="view active" id="v-results"><div id="results-root"><div class="card muted">加载中…</div></div></section>
<section class="view" id="v-live">
  <div class="card">
    <h2>实时 Agent 决策（数据源：<span id="dmode" class="muted">…</span>）</h2>
    <div class="row">
      <select id="sym"></select>
      <label class="small">熔断注入(what-if)：</label>
      <select id="cb"><option value="auto">真实熔断(按数据)</option><option value="0" selected>L0 正常</option><option value="1">L1 警告</option>
        <option value="2">L2 降仓</option><option value="3">L3 暂停(触发否决)</option></select>
      <button onclick="decide()">运行一次决策</button>
      <button onclick="resetPos()">重置持仓</button>
    </div>
    <p class="small">点击后，下方 Agent 流水线会按本次决策<b style="color:var(--blue)">实际经过的路径</b>依次点亮：<span style="color:var(--green)">绿=执行</span>，<span style="color:var(--red)">红=Risk 否决/停止</span>，灰=未到达。</p>
    <p class="small">演示真实熔断（合成数据模式）：熔断注入选「真实熔断(按数据)」，标的选 <code>SOL/USDT</code>→L1 警告、<code>DOGE/USDT</code>→L2 降仓、<code>XRP/USDT</code>→L3 暂停否决；其余币为正常。</p>
  </div>
  <div class="grid g3">
    <div class="card" style="grid-column:span 1"><h2>Agent 流水线</h2><div class="pipe" id="pipe"></div></div>
    <div class="card" style="grid-column:span 2"><h2>本次决策</h2>
      <div class="grid g4" id="dkpi"></div><pre id="djson" style="margin-top:12px">—</pre></div>
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
async function loadResults(){
  const r=await fetch("/api/results").then(r=>r.json());const root=$("#results-root");
  if(!r.ok){root.innerHTML=`<div class="card"><h2>实验结果</h2><p class="muted">${r.msg}</p></div>`;return;}
  let h="";const rr=r.real_returns||{};const k=rr["Step7_tsmom_fusion (MAIN deliverable)"];
  if(k){h+=`<div class="card"><h2>主交付 Step7（ML+TSMOM 融合）· 开发期真实表现</h2><div class="grid g4">
      <div class="kpi"><div class="v ${cls(k.ann_return)}">${fmtPct(k.ann_return)}</div><div class="l">年化收益</div></div>
      <div class="kpi"><div class="v ${cls(k.cum_return)}">${fmtPct(k.cum_return)}</div><div class="l">累计收益</div></div>
      <div class="kpi"><div class="v">${fmtPct(k.max_drawdown)}</div><div class="l">最大回撤</div></div>
      <div class="kpi"><div class="v">${fmtN(k.sharpe)}</div><div class="l">Sharpe</div></div>
    </div><p class="small">实验目录：<code>${r.exp_dir}</code></p></div>`;}
  if(Object.keys(rr).length){h+=`<div class="card"><h2>真实收益率对照表</h2><table><thead><tr>
      <th>策略</th><th>年化</th><th>累计</th><th>最大回撤</th><th>Sharpe</th></tr></thead><tbody>`;
    for(const [name,s] of Object.entries(rr)){const hi=name.includes("MAIN")?"hi":"";
      h+=`<tr class="${hi}"><td>${name}</td><td class="${cls(s.ann_return)}">${fmtPct(s.ann_return)}</td>
        <td class="${cls(s.cum_return)}">${fmtPct(s.cum_return)}</td><td>${fmtPct(s.max_drawdown)}</td><td>${fmtN(s.sharpe)}</td></tr>`;}
    h+=`</tbody></table></div>`;}
  if(r.ladder&&r.ladder.length){h+=`<div class="card"><h2>增量证明阶梯（每个模态是否带来显著增量）</h2>
      <p class="small">incr_NW_t &gt; 2 视为显著增量；诚实地，多数另类模态在本框架下无显著增量——这是结论而非缺陷。</p>
      <table><thead><tr><th>步骤</th><th>Sharpe</th><th>DSR</th><th>incr_t(vs上一步)</th><th>incr_t(vs TSMOM)</th></tr></thead><tbody>`;
    for(const row of r.ladder){const step=row.step||row[""]||Object.values(row)[0]||"";
      const it=row.incr_NW_t,ib=row.incr_NW_t_base;const sig=(Math.abs(+it)>2)?"hi":"";
      h+=`<tr class="${sig}"><td>${step}</td><td>${fmtN(row.sharpe_ann)}</td><td>${fmtN(row.deflated_sharpe)}</td>
        <td>${fmtN(it)}</td><td>${fmtN(ib)}</td></tr>`;}
    h+=`</tbody></table></div>`;}
  const dv=r.decision_vs_signal&&r.decision_vs_signal.metrics;
  if(dv){h+=`<div class="card"><h2>决策栈 vs 纯信号（风控的价值：回撤控制）</h2>
      <table><thead><tr><th>策略</th><th>Sharpe</th><th>年化</th><th>最大回撤</th></tr></thead><tbody>`;
    for(const [name,s] of Object.entries(dv)){h+=`<tr><td>${name}</td><td>${fmtN(s.sharpe)}</td>
        <td class="${cls(s.ann_return)}">${fmtPct(s.ann_return)}</td><td>${fmtPct(s.max_drawdown)}</td></tr>`;}
    h+=`</tbody></table></div>`;}
  // two paradigms x matched risk: Step5 neutral stack vs Step7 directional stack
  const ds=r.decision_stack||{}, dir=(r.directional_stack&&r.directional_stack.metrics)||null;
  if(dir){
    const n_sh=ds.strategy_sharpe, n_ar=ds.strategy_annual_return, n_dd=ds.strategy_max_drawdown;
    const d_sh=dir.strategy_sharpe, d_ar=dir.strategy_annual_return, d_dd=dir.strategy_max_drawdown;
    const di=(r.directional_stack.info)||{};
    h+=`<div class="card"><h2>两种策略范式 × 各自适配的风控（完整链路对照）</h2>
      <table><thead><tr><th>链路</th><th>信号</th><th>风控范式</th><th>Sharpe</th><th>年化</th><th>最大回撤</th></tr></thead><tbody>
      <tr><td>Step5 + 中性决策栈</td><td>横截面中性(选币)</td><td>相关性haircut/簇上限/熔断</td>
        <td>${fmtN(n_sh)}</td><td class="${cls(n_ar)}">${fmtPct(n_ar)}</td><td>${fmtPct(n_dd)}</td></tr>
      <tr class="hi"><td>Step7 + 方向性决策栈</td><td>ML+TSMOM(方向)</td><td>净暴露上限/总波动目标/熔断</td>
        <td>${fmtN(d_sh)}</td><td class="${cls(d_ar)}">${fmtPct(d_ar)}</td><td>${fmtPct(d_dd)}</td></tr>
      </tbody></table>
      <p class="small">中性 book 用中性风控（多空对冲、相关性中性化）；方向 book 用方向风控（净暴露/总波动目标）。
        Step7 方向栈保留净暴露 avg|net|=<code>${fmtN(di.avg_abs_net_exposure)}</code>（上限 ${fmtN(di.net_exposure_cap)}），
        这正是 TSMOM 的方向性 alpha 来源——若误用中性 overlay 会把它中和掉。两条链路都可实盘。</p></div>`;
  }
  const g=r.governance||{};
  // holdout sample-out full table (the confirmatory result — the report's centerpiece)
  const ho=r.holdout||null;
  if(ho){
    const order=["Step5_fusion (pure signal)","Step7_tsmom_fusion (MAIN deliverable)",
      "Step5 + neutral decision stack","Step7 + directional stack (full risk)","Step0_TSMOM (benchmark)"];
    h+=`<div class="card" style="border:2px solid var(--blue)"><h2>样本外最终检验 Holdout-A（冻结配置，与 dev 同口径）</h2>
      <p class="small">这是消耗一次的确认性检验——真正证明交付物质量的部分。</p>
      <table><thead><tr><th>策略</th><th>年化</th><th>累计</th><th>波动</th><th>最大回撤</th><th>Sharpe</th><th>Sortino</th><th>Calmar</th><th>换手</th><th>胜率</th></tr></thead><tbody>`;
    for(const nm of order){ const s=ho[nm]; if(!s) continue;
      const hi=nm.includes("MAIN")?"hi":"";
      h+=`<tr class="${hi}"><td>${nm}</td><td class="${cls(s.ann_return)}">${fmtPct(s.ann_return)}</td>
        <td class="${cls(s.cum_return)}">${fmtPct(s.cum_return)}</td><td>${fmtPct(s.ann_vol)}</td>
        <td>${fmtPct(s.max_drawdown)}</td><td>${fmtN(s.sharpe)}</td><td>${fmtN(s.sortino)}</td>
        <td>${fmtN(s.calmar)}</td><td>${s.ann_turnover!=null?Math.round(s.ann_turnover):'—'}</td>
        <td>${fmtPct(s.win_rate)}</td></tr>`;}
    h+=`</tbody></table></div>`;
  }
  if(Object.keys(g).length){
    const pit=g.pit_passed?`<span class="pill ok">通过 · 0 泄露</span>`:`<span class="pill bad">未通过</span>`;
    const ece=g.ece_calibrated!=null?`<span class="pill ${g.ece_calibrated<0.05?'ok':'warn'}">ECE ${g.ece_calibrated.toFixed(4)}</span>`:"—";
    const pbo=g.pbo!=null?`<span class="pill ${g.pbo<0.5?'ok':'warn'}">PBO ${g.pbo.toFixed(3)}</span>`:"—";
    h+=`<div class="card"><h2>治理三闸门（结果可信度）</h2><div class="grid g3">
      <div class="kpi"><div class="v">${pit}</div><div class="l">PIT 时点正确性（无未来泄露）</div></div>
      <div class="kpi"><div class="v">${ece}</div><div class="l">概率校准误差（&lt;0.05 良好）</div></div>
      <div class="kpi"><div class="v">${pbo}</div><div class="l">回测过拟合概率（&lt;0.5 良好）</div></div></div>`;
    if(g.event_cm_t!=null||g.event_vol_t!=null){h+=`<p class="small" style="margin-top:12px">新闻/事件因子诚实检验：
      横截面 t=<code>${fmtN(g.event_xs_t)}</code>、共模 t=<code>${fmtN(g.event_cm_t)}</code>、波动预测 t=<code>${fmtN(g.event_vol_t)}</code>
      —— 信号集中在共模(市场级)维度但不足以转化为可交易增量，四种部署方式均经检验。</p>`;}
    h+=`</div>`;}
  root.innerHTML=h||`<div class="card muted">实验目录已找到，但未解析到可展示文件。</div>`;
}
let _liveReady=false, _liveLoading=false;
const FALLBACK_SYMS=["BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT","DOGE/USDT","LTC/USDT","LINK/USDT","TRX/USDT","ADA/USDT"];
function fillSymbols(syms){$("#sym").innerHTML=(syms||[]).map(s=>`<option>${s}</option>`).join("");}
async function initLive(){
  if(_liveReady||_liveLoading) return;
  _liveLoading=true;
  console.log("[live] initLive: fetching /api/agents ...");
  $("#dmode").textContent="加载模型图中…（首次较慢，正在载入真实数据，请看后端控制台进度）";
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
    const msg=(e.name==="AbortError")?"加载超时（>6分钟）。真实数据/模型构建过慢，请看后端控制台。":("加载失败："+e.message);
    $("#dmode").innerHTML=`<span style="color:var(--red)">${msg}</span> <a href="#" onclick="_liveReady=false;_liveLoading=false;initLive();return false;">点此重试</a>`;
    // keep the fallback dropdown so the user can still try a decision
    renderPipe([{agent:"(graph unavailable)",role:"后端构图失败，下拉框已用默认10币占位",category:"data"}],null);
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
  if(!_liveReady){await initLive(); if(!_liveReady){alert("Agent 图尚未就绪，请查看后端日志。");return;}}
  const sym=$("#sym").value,cbRaw=$("#cb").value,cb=(cbRaw==="auto"?0:+cbRaw),cbAuto=(cbRaw==="auto");
  if(!sym){alert("请先选择交易标的");return;}
  // clear right panel while the pipeline runs, so it doesn't pre-empt the animation
  $("#djson").textContent="决策中…";
  $("#dkpi").innerHTML="";
  let r;
  try{
    r=await fetch("/api/decide",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({symbol:sym,cb_level:cbRaw})}).then(r=>r.json());
  }catch(e){$("#djson").textContent="请求失败："+e.message;return;}
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
      ["方向",d.primary_direction||d.action||"—",""],
      ["目标仓位",d.target_position!=null?(+d.target_position).toFixed(3):"—",""],
      ["熔断级别","L"+(+d.circuit_breaker_level||0),cbc],
      ["风险级别",d.risk_level||"—",rkc],
      ["实测回撤",r.real_drawdown!=null?(r.real_drawdown*100).toFixed(1)+"%":"—",cbc]
    ];
    $("#dkpi").innerHTML=kpis.map(([l,v,c])=>`<div class="kpi ${c}"><div class="v" style="font-size:16px">${v}</div><div class="l">${l}</div></div>`).join("");
    $("#djson").textContent=JSON.stringify(d,null,2);
  };
  await animatePipe(a.agents,{ran,veto,cbWarn},reveal);
  reveal();  // safety: ensure it's shown even if risk node was absent
}
async function resetPos(){await fetch("/api/reset_positions",{method:"POST"});alert("持仓已重置");}
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
            return jsonify({"error": f"该标的无可用决策数据：{sym}。可用标的：{', '.join(avail) or '(无)'}"}), 200
        return jsonify(out)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": f"决策执行异常：{e}"}), 200

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
