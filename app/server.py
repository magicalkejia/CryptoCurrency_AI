"""
app/server.py — WEB front-end (entry point) for the Agent + Skills system.

Run:
    python app/server.py            # then open http://127.0.0.1:8000

A thin Flask shell over the SAME agent API used by run_agent.py.

UI semantics: the Agent list and pipeline nodes are *status indicators*, NOT
clickable buttons. Controls = symbol dropdown, simulated circuit-breaker
dropdown, and the run buttons. After a decision the pipeline lights up to show
the ACTUAL path the agents took (green=ran, red=stopped/vetoed, dim=not reached).

A single persistent PaperBroker accumulates the running position across "运行一次
决策" clicks (so 当前持仓 is meaningful); "最近 N 次" uses throwaway read-only
brokers so it never pollutes the live position.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request, render_template_string

from run_agent import build_graph, run_one
from crypto.skills.registry import REGISTRY
from crypto.live.oms import PaperBroker

app = Flask(__name__)
_STATE = {}

AGENTS = [
    ("DataAgent", "拉取特征 + PIT/数据质量检查", "data"),
    ("SignalResearchAgent", "暴露基模型/PatchTST 预测", "signal"),
    ("NarrativeAgent", "LLM 事件/叙事 → 结构化因子(可插拔)", "narrative"),
    ("FusionAgent", "Regime + 融合 + 元模型 + 校准 + 置信度", "fusion"),
    ("RiskAgent", "波动率目标/仓位/熔断 —— 拥有最高否决权", "risk"),
    ("ExecutionAgent", "订单计划 + 成交(纸交易/实盘)", "execution"),
    ("ReviewAgent", "复盘/审计摘要/再训练触发", "review"),
]

# 熔断级别说明(实盘由 Risk Agent 计算;此处为演示用手动注入 what-if)
CB_LEVELS = [
    (0, "0 正常 — 不限制"),
    (1, "1 警告 — 仅告警"),
    (2, "2 降仓 — 目标仓位减半"),
    (3, "3 暂停 — 停开仓(触发 Risk 否决)"),
]

PAGE = """
<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>Crypto · 多模态 Agent 量化控制台</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;background:#0f1117;color:#e6e6e6}
 header{padding:16px 24px;background:#161a23;border-bottom:1px solid #232838}
 h1{font-size:18px;margin:0}.muted{color:#8b93a7;font-size:13px}
 .wrap{display:grid;grid-template-columns:300px 1fr;min-height:calc(100vh - 58px)}
 aside{background:#12151d;border-right:1px solid #232838;padding:16px}
 main{padding:18px 24px}
 .pipe{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:6px 0 6px}
 .node{background:#1b2030;border:1px solid #2c3346;border-radius:8px;padding:6px 10px;font-size:12px;transition:.2s}
 .node.ran{border-color:#3a8f5e;background:#16271d;color:#9ff0bf}
 .node.stopped{border-color:#c0533a;background:#2a1714;color:#ffb59c}
 .node.dim{opacity:.4}
 .veto{font-size:10px;color:#ffb59c;border:1px solid #5a3a34;border-radius:8px;padding:1px 5px;margin-left:4px}
 .arrow{color:#5b647d}
 .legend{font-size:11px;color:#8b93a7;margin:4px 0 14px}
 .dot{display:inline-block;width:9px;height:9px;border-radius:2px;margin:0 3px 0 10px;vertical-align:middle}
 .card{background:#12151d;border:1px solid #232838;border-radius:10px;padding:14px;margin-bottom:14px}
 label{font-size:13px;color:#aab1c4}select,button{font-size:14px;padding:7px 10px;border-radius:8px;border:1px solid #2c3346;background:#1b2030;color:#e6e6e6}
 button{background:#2d6cdf;border-color:#2d6cdf;cursor:pointer}button:hover{background:#3b78ee}
 button.alt{background:#1b2030;border-color:#2c3346}
 pre{background:#0b0d13;border:1px solid #232838;border-radius:8px;padding:12px;overflow:auto;font-size:12px;line-height:1.5;max-height:320px}
 table{width:100%;border-collapse:collapse;font-size:12px}td,th{border-bottom:1px solid #232838;padding:5px 8px;text-align:left}
 .skill-cat{font-size:12px;color:#8b93a7;margin-top:10px}.skill{font-size:12px;background:#1b2030;border:1px solid #2c3346;border-radius:6px;padding:2px 7px;display:inline-block;margin:2px}
 .badge{font-size:11px;padding:2px 7px;border-radius:10px;background:#23304a}
 .ok{color:#7fd1a3}.no{color:#ff9c9c}
 .big{font-size:22px;font-weight:700}
 .kv{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:8px}
 .kv div{background:#0b0d13;border:1px solid #232838;border-radius:8px;padding:8px}
 .kv .lbl{font-size:11px;color:#8b93a7}.kv .val{font-size:15px;margin-top:2px}
 .bar{height:8px;background:#1b2030;border-radius:5px;overflow:hidden;margin-top:6px}
 .bar>i{display:block;height:100%;background:#2d6cdf}
 .long{color:#7fd1a3}.short{color:#ff9c9c}.flat{color:#c7cad4}
 .hint{font-size:11px;color:#8b93a7;margin-top:4px}
</style></head><body>
<header><h1>Crypto · 多模态 Agent 量化控制台</h1>
<div class="muted">7-Agent 状态机 · Skills 可审计调用 · 编排后端: <span class="badge">{{backend}}</span> · 仅演示(合成数据,数字无意义)</div></header>
<div class="wrap">
 <aside>
  <div class="card"><b>Agents</b>
   <div class="muted" style="margin:4px 0 8px">下方是 7 个智能体(状态指示,非按钮)</div>
   {% for name,desc,cat in agents %}<div style="margin:8px 0"><div class="node {{cat}}">{{name}}{% if cat=='risk' %}<span class="veto">否决权</span>{% endif %}</div>
   <div class="muted">{{desc}}</div></div>{% endfor %}
  </div>
  <div class="card"><b>Skills 注册表</b>
   {% for cat, skills in catalog.items() %}<div class="skill-cat">{{cat}}</div>
   {% for s in skills %}<span class="skill">{{s}}</span>{% endfor %}{% endfor %}
  </div>
 </aside>
 <main>
  <div class="card">
   <b>决策流水线</b>(下方节点是<u>执行状态指示</u>,不是可点按钮)
   <div class="pipe" id="pipe">
    <span class="node dim" id="n-data">Data</span><span class="arrow">→</span>
    <span class="node dim" id="n-gate">质量闸门</span><span class="arrow">→</span>
    <span class="node dim" id="n-signal">Signal</span><span class="arrow">→</span>
    <span class="node dim" id="n-narrative">Narrative</span><span class="arrow">→</span>
    <span class="node dim" id="n-fusion">Fusion</span><span class="arrow">→</span>
    <span class="node dim" id="n-risk">Risk<span class="veto">否决权</span></span><span class="arrow">→</span>
    <span class="node dim" id="n-exec">Execution</span><span class="arrow">→</span>
    <span class="node dim" id="n-review">Review</span>
   </div>
   <div class="legend"><span class="dot" style="background:#3a8f5e"></span>已执行
     <span class="dot" style="background:#c0533a"></span>停止/否决
     <span class="dot" style="background:#2c3346"></span>未到达
     <span style="margin-left:12px">红色「否决权」= Risk 是最高权限,可一票否决(并非"选中")</span></div>
  </div>

  <div class="card">
   <label>标的</label>
   <select id="symbol">{% for s in symbols %}<option>{{s}}</option>{% endfor %}</select>
   &nbsp;<label>模拟熔断级别</label>
   <select id="cb">{% for lv,txt in cb_levels %}<option value="{{lv}}">{{txt}}</option>{% endfor %}</select>
   &nbsp;<button onclick="decide()">运行一次决策</button>
   &nbsp;<button class="alt" onclick="recent()">最近 8 次</button>
   &nbsp;<button class="alt" onclick="resetPos()">重置持仓</button>
   <div class="hint">「模拟熔断级别」是演示用的 what-if 注入:实盘中此值由 Risk Agent 依据回撤/亏损/连接状态自动计算,并非用户选择。这里手动设它,是为了演示 Risk Agent 在不同熔断态下的否决行为(选 3 → Risk 一票否决 → no_trade)。</div>
  </div>

  <div class="card"><b>决策摘要</b>
   <div id="summary"><span class="muted">点击「运行一次决策」</span></div>
  </div>

  <div class="card"><b>决策记录</b>(每次「运行一次决策」会追加一行;「最近 8 次」批量载入)
   <table id="recent"><thead><tr><th>时间</th><th>动作</th><th>方向</th><th>regime</th><th>α</th><th>meta概率</th><th>目标仓位</th><th>当前持仓</th></tr></thead><tbody></tbody></table>
  </div>

  <div class="card"><b>审计日志(Skill 调用顺序)</b>
   <table id="audit"><thead><tr><th>#</th><th>skill</th><th>类别</th><th>耗时(ms)</th><th>状态</th></tr></thead><tbody></tbody></table>
  </div>

  <div class="card"><b>原始结构化决策 (v6 §1.4)</b><pre id="decision">—</pre></div>
 </main>
</div>
<script>
function setNode(id,s){const e=document.getElementById(id);e.classList.remove('ran','stopped','dim');e.classList.add(s);}
function highlight(d){
 const sk=d.audit_log.map(a=>a.skill), has=s=>sk.includes(s);
 ['data','gate','signal','narrative','fusion','risk','exec','review'].forEach(i=>setNode('n-'+i,'dim'));
 if(has('get_feature_row'))setNode('n-data','ran');
 const gatePassed=has('narrative_infer');
 if(has('check_data_quality'))setNode('n-gate', gatePassed?'ran':'stopped');
 if(gatePassed){setNode('n-signal','ran');setNode('n-narrative','ran');}
 if(has('fusion_infer'))setNode('n-fusion','ran');
 if(has('risk_size_and_gate')){const vetoed=!has('execute_paper')&&d.decision.action==='no_trade';setNode('n-risk',vetoed?'stopped':'ran');}
 if(has('execute_paper'))setNode('n-exec','ran');
 if(has('review_decision'))setNode('n-review','ran');
}
function fmt(x){return x==null?'—':x;}
function summary(dec,curpos){
 const c={long:'long',short:'short'}[dec.action]||'flat';
 const conf=Math.round((dec.confidence||0)*100);
 document.getElementById('summary').innerHTML=
  `<div class="big ${c}">${dec.action.toUpperCase()}</div>
   <div class="kv">
     <div><div class="lbl">方向</div><div class="val ${c}">${dec.primary_direction}</div></div>
     <div><div class="lbl">市场状态 regime</div><div class="val">${dec.regime}</div></div>
     <div><div class="lbl">combined_alpha(模型内部)</div><div class="val">${fmt(dec.combined_alpha)}</div></div>
     <div><div class="lbl">校准后 meta 概率</div><div class="val">${fmt(dec.meta_trade_prob_calibrated)}</div></div>
     <div><div class="lbl">vol_target_scalar(波动率缩放)</div><div class="val">${fmt(dec.vol_target_scalar)}</div></div>
     <div><div class="lbl">目标仓位</div><div class="val">${fmt(dec.target_position)}</div></div>
     <div><div class="lbl">当前持仓(累计)</div><div class="val">${fmt(curpos)}</div></div>
     <div><div class="lbl">止损 / 止盈</div><div class="val">${fmt(dec.stop_loss)} / ${fmt(dec.take_profit)}</div></div>
   </div>
   <div style="margin-top:10px"><span class="lbl muted">置信度 ${conf}%</span><div class="bar"><i style="width:${conf}%"></i></div></div>
   <div style="margin-top:10px" class="muted">原因: ${dec.reason}</div>
   <div class="muted" style="margin-top:4px">熔断级别 ${dec.circuit_breaker_level} · 数据质量 ${dec.data_quality_score} · 叙事LLM ${dec.narrative_stub?'离线stub':'已接入'}</div>`;
}
function addRow(dec,curpos,top){
 const tb=document.querySelector('#recent tbody');
 const c={long:'long',short:'short'}[dec.primary_direction]||'flat';
 const tr=document.createElement('tr');
 tr.innerHTML=`<td>${dec.decision_time}</td><td class="${c}">${dec.action}</td><td>${dec.primary_direction}</td><td>${dec.regime}</td><td>${fmt(dec.combined_alpha)}</td><td>${fmt(dec.meta_trade_prob_calibrated)}</td><td>${fmt(dec.target_position)}</td><td>${fmt(curpos)}</td>`;
 if(top&&tb.firstChild)tb.insertBefore(tr,tb.firstChild);else tb.appendChild(tr);
}
async function decide(){
 const symbol=document.getElementById('symbol').value, cb=+document.getElementById('cb').value;
 const r=await fetch('/api/decide',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({symbol,cb_level:cb})});
 const d=await r.json();
 summary(d.decision,d.current_position); highlight(d);
 document.getElementById('decision').textContent=JSON.stringify(d.decision,null,2);
 const tb=document.querySelector('#audit tbody');tb.innerHTML='';
 d.audit_log.forEach((a,i)=>tb.innerHTML+=`<tr><td>${i+1}</td><td>${a.skill}</td><td>${a.category}</td><td>${a.ms}</td><td class="${a.ok?'ok':'no'}">${a.ok?'OK':'FAIL'}</td></tr>`);
 addRow(d.decision,d.current_position,true);   // 每次决策追加到记录顶部
}
async function recent(){
 const symbol=document.getElementById('symbol').value, cb=+document.getElementById('cb').value;
 const r=await fetch('/api/recent',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({symbol,n:8,cb_level:cb})});
 const rows=await r.json();
 document.querySelector('#recent tbody').innerHTML='';
 rows.forEach(x=>addRow(x.decision,x.current_position,false));
}
async function resetPos(){await fetch('/api/reset_positions',{method:'POST'});document.querySelector('#recent tbody').innerHTML='';}
</script></body></html>
"""


def get_graph():
    if "graph" not in _STATE:
        g, cm, feats, fcols, fcfg = build_graph()
        _STATE.update(graph=g, close_map=cm, feats=feats, fcols=fcols, fcfg=fcfg,
                      broker=PaperBroker(max_slippage_bps=3))
    return _STATE


@app.route("/")
def index():
    s = get_graph()
    catalog = {c: REGISTRY.list(c) for c in
               ["data", "narrative", "fusion", "risk", "execution", "review"]}
    return render_template_string(
        PAGE, backend=s["graph"].backend, agents=AGENTS, catalog=catalog,
        cb_levels=CB_LEVELS, symbols=sorted(s["feats"]["symbol"].unique()))


@app.route("/api/agents")
def api_agents():
    return jsonify([{"agent": a, "role": d, "category": c} for a, d, c in AGENTS])


@app.route("/api/skills")
def api_skills():
    return jsonify({c: REGISTRY.list(c) for c in
                    ["data", "narrative", "fusion", "risk", "execution", "review"]})


@app.route("/api/decide", methods=["POST"])
def api_decide():
    s = get_graph()
    body = request.get_json(force=True)
    # persistent broker -> 当前持仓 累计
    out = run_one(s["graph"], s["close_map"], s["feats"], s["fcols"], s["fcfg"],
                  symbol=body.get("symbol", "BTC/USDT"), cb_level=int(body.get("cb_level", 0)),
                  broker=s["broker"])
    if out is None:
        return jsonify({"error": "no data for symbol"}), 404
    return jsonify(out)


@app.route("/api/recent", methods=["POST"])
def api_recent():
    s = get_graph()
    body = request.get_json(force=True)
    symbol, n, cb = body.get("symbol", "BTC/USDT"), int(body.get("n", 8)), int(body.get("cb_level", 0))
    sub = s["feats"][s["feats"]["symbol"] == symbol].dropna(subset=s["fcols"]).tail(n)
    rows = []
    for _, row in sub.iterrows():
        # throwaway broker -> read-only, doesn't pollute live position
        out = run_one(s["graph"], s["close_map"], s["feats"], s["fcols"], s["fcfg"],
                      symbol=symbol, decision_time=row["decision_time"], cb_level=cb, broker=None)
        if out:
            rows.append({"decision": out["decision"], "current_position": out["current_position"]})
    return jsonify(rows)


@app.route("/api/reset_positions", methods=["POST"])
def api_reset():
    _STATE["broker"] = PaperBroker(max_slippage_bps=3)
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("building agent graph (one-time, synthetic, 4 symbols)...")
    get_graph()
    print("serving on http://127.0.0.1:8000")
    app.run(host="127.0.0.1", port=8000, debug=False)
