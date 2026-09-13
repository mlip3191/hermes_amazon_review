#!/usr/bin/env python3
"""Build the self-contained report.html from report_data.json (any sample size)."""
import json

D = json.load(open("report_data.json"))
rows = D["rows"]
stats = D["stats"]
emotion_stats = D.get("emotion_stats")
rating_stats = D.get("rating_stats")
EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]

# Words/markers that signal a complaint or reservation in the review.
COMPLAINT = ["mistake", "deduction", "complaint", "note", "without the fees",
             "realized i could", "worse", "issue", "problem", "no more",
             "did not", "wasn't", "was not", "failed", "missing"]
# Understated / hedged qualifiers that read as "less than enthusiastic".
HEDGE = ["what can you say", "nice to have", "a little", "good value",
         "little thank-you", "rec'd a promo", "saves a little", "over time"]


def assign_class(row):
    lt = (row["text"] + " " + row["reason"]).lower()
    low = lt.lower()
    wrong = row.get("correct") is False
    if not wrong:
        return None, None
    if row["pred"] > row["true"]:
        return "Upward misread", (
            "The review reads positive on its face, but the true rating is lower. "
            "Ambiguous, overstated, or sarcastic phrasing (e.g. 'but no more', "
            "'wish I knew about it') flatters the text toward a higher star than "
            "the reviewer actually gave.")
    # under-prediction
    if any(w in low for w in COMPLAINT):
        return "Complaint / regret dominates", (
            "A concrete complaint or regret ('note wasn't attached', 'hit button "
            "by mistake', 'realized there were cheaper options') overrides the "
            "overall sentiment and drags the score a full star below the review's "
            "net feeling.")
    if any(w in low for w in HEDGE):
        return "Hedged / lukewarm phrasing", (
            "Understated qualifiers ('nice to have', 'saves a little', 'what can "
            "you say') are read as 'satisfied but not effusive', so a 5-star "
            "review is dialed back to 4 even though the reviewer clearly liked it.")
    return "Generic praise held back", (
        "Terse, superlative-free praise ('Good Product. Good Product') is judged "
        "'lacks enthusiasm' and capped at 4. Amazon reviewers hand out 5s for "
        "one-line 'good product' reviews far more readily than the model expects, "
        "so the model is calibrated stricter than the actual rating distribution.")


for r in rows:
    r["class"], r["inference"] = assign_class(r)

# Aggregate classes over the wrong rows.
agg = {}
for r in rows:
    if r["class"]:
        agg.setdefault(r["class"], []).append(r)
class_rows = [
    {
        "name": name,
        "count": len(items),
        "example": items[0]["text"][:72],
        "inference": items[0]["inference"],
        "pred_vs_true": "/".join(sorted({f"{i['pred']}\u2192{i['true']}" for i in items})),
    }
    for name, items in agg.items()
]

dist = {}
for r in rows:
    if r["pred"] is not None:
        dist[r["pred"]] = dist.get(r["pred"], 0) + 1
payload = {
    "stats": stats,
    "rows": rows,
    "classes": class_rows,
    "distribution": [{"star": k, "count": v} for k, v in sorted(dist.items())],
    "emotion_stats": emotion_stats,
    "rating_stats": rating_stats,
}

# ---- HTML (same theme; only the classification source changed) ----
html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review Scoring Report</title>
<style>
  :root{
    --bg:#f1efea;--card:#faf9f6;--ink:#2e2c28;--muted:#8a867d;--hair:#dcd8cf;
    --accent:#6b6f50;--accent-soft:#e4e3d6;--good:#7b8570;--good-soft:#e6e9e2;
    --bad:#a97a6b;--bad-soft:#f0e4e0;
    --shadow:0 1px 2px rgba(0,0,0,.05),0 6px 18px rgba(0,0,0,.04);
    --radius:10px;
    --font:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--font);line-height:1.5;-webkit-font-smoothing:antialiased;}
  .wrap{max-width:960px;margin:0 auto;padding:40px 22px 80px;}
  header{padding:8px 0 4px}
  .kicker{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
  h1{font-size:30px;font-weight:650;margin:6px 0 4px;letter-spacing:-.01em}
  .sub{color:var(--muted);font-size:14px;margin:0}
  .rule{height:1px;background:var(--hair);margin:26px 0}
  section{margin:34px 0}
  h2{font-size:15px;letter-spacing:.06em;text-transform:uppercase;color:var(--accent);margin:0 0 14px;font-weight:600}
  h2 span{color:var(--muted);font-weight:400;text-transform:none;letter-spacing:0}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
  .card{background:var(--card);border:1px solid var(--hair);border-radius:var(--radius);box-shadow:var(--shadow);padding:16px 18px}
  .card b{font-size:28px;font-weight:650;display:block;line-height:1.1}
  .card b small{font-size:15px;font-weight:500;color:var(--muted)}
  .card span{font-size:12px;color:var(--muted)}
  .bars{display:flex;gap:6px;align-items:flex-end;height:120px;margin:24px 0 22px}
  .bar{flex:1;background:var(--accent);border-radius:4px 4px 0 0;position:relative;min-height:3px}
  .bar small{position:absolute;top:-18px;left:0;right:0;text-align:center;color:var(--muted);font-size:11px}
  .bar fig{position:absolute;top:100%;left:50%;margin:6px 0 0;transform:translateX(-50%);
    font-size:9px;color:var(--muted);text-align:center;white-space:nowrap}
  .filterbar{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-end;margin:0 0 16px}
  .fgroup{display:flex;flex-direction:column;gap:4px}
  .fgroup label{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:600}
  .filterbar select{padding:7px 10px;border:1px solid var(--hair);border-radius:6px;background:var(--card);
    color:var(--ink);font-size:13px;font-family:var(--font);cursor:pointer}
  .filterbar select:focus{outline:2px solid var(--accent-soft);border-color:var(--accent)}
  .filterbar .countnote{margin-left:auto;align-self:flex-end;font-size:12px;color:var(--muted);padding-bottom:4px}
  .clear{background:none;border:none;color:var(--accent);cursor:pointer;font-size:12px;text-decoration:underline;
    font-family:var(--font);padding:0;align-self:flex-end;margin-bottom:4px}
  .clear:hover{color:var(--ink)}
  .table-wrapper{display:flex;flex-direction:column;gap:10px}
  .table-toggle{background:var(--hair);border:1px solid var(--hair);color:var(--ink);border-radius:6px;padding:8px 12px;font-size:12px;font-weight:600;cursor:pointer;transition:all 0.2s}
  .table-toggle:hover{background:var(--accent-soft);border-color:var(--accent)}
  .table-toggle.expanded{background:var(--accent-soft);border-color:var(--accent);color:var(--accent)}
  .table-content.hidden{display:none}
  table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--hair);border-radius:var(--radius);overflow:hidden;box-shadow:var(--shadow)}
  table.hidden{display:none}
  th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--hair);vertical-align:top;font-size:13.5px}
  th{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);background:var(--accent-soft);font-weight:600}
  tr:last-child td{border-bottom:none}
  tbody tr:hover{background:#f6f4ef}
  .num{font-variant-numeric:tabular-nums;white-space:nowrap}
  .pill{display:inline-block;padding:1px 8px;border-radius:20px;font-size:11px;font-weight:600;letter-spacing:.03em}
  .pill.ok{background:var(--good-soft);color:var(--good)}
  .pill.bad{background:var(--bad-soft);color:var(--bad)}
  .pill.emo{background:var(--accent-soft);color:var(--accent)}
  .emo-bars-wrap{display:flex;gap:24px;flex-wrap:wrap;margin:14px 0 4px}
  .emo-bars-wrap>div{flex:1;min-width:260px}
  .emo-bars-wrap h4{margin:0 0 6px;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:600}
  .emo-bars-wrap .bars{height:100px}
  .overview-grid{display:flex;gap:32px;flex-wrap:wrap}
  .overview-grid>div{flex:1;min-width:300px}
  .overview-grid h3{margin:0 0 10px;font-size:13px;letter-spacing:.04em;color:var(--accent);font-weight:650}
  .overview-grid h3 span{color:var(--muted);font-weight:400;font-size:11px;margin-left:6px;letter-spacing:0}
  .overview-grid .cards{grid-template-columns:repeat(auto-fit,minmax(100px,1fr));gap:8px}
  .overview-grid .card{padding:10px 12px}
  .overview-grid .card b{font-size:19px}
  .overview-grid .card span{font-size:10.5px}
  .overview-grid .bars{height:64px;margin:20px 0 16px}
  .overview-grid .emo-bars-wrap{gap:14px;margin:10px 0 2px}
  .overview-grid .emo-bars-wrap>div{min-width:140px}
  .overview-grid .emo-bars-wrap .bars{height:64px}
  .stars{color:var(--accent);letter-spacing:.05em;font-weight:600}
  .review{color:var(--ink);max-width:330px}
  .review small{color:var(--muted)}
  .reason{color:var(--muted);font-size:12.5px;max-width:250px}
  .cls{background:var(--card);border:1px solid var(--hair);border-radius:var(--radius);padding:16px 18px;margin-bottom:12px;box-shadow:var(--shadow)}
  .cls h3{margin:0 0 2px;font-size:15px;font-weight:650}
  .cls .meta{font-size:12px;color:var(--muted);margin-bottom:10px}
  .cls p{margin:8px 0 0;font-size:13.5px;color:var(--ink)}
  .cls .tag{display:inline-block;background:var(--accent-soft);color:var(--accent);border-radius:20px;padding:1px 9px;font-size:11px;font-weight:600;margin-left:8px}
  .conf-diag{background:var(--good-soft);color:var(--good);font-weight:600}
  .conf-off{background:rgba(169,122,107,0.3)}
  .accuracy-card{background:var(--card);border:2px solid var(--hair);border-radius:6px;padding:8px 10px;text-align:center;font-size:11px}
  .accuracy-card.good{border-color:var(--good)}
  .accuracy-card.bad{border-color:var(--bad)}
  .accuracy-card b{display:block;font-size:16px;margin:4px 0 2px}
  .accuracy-card span{color:var(--muted);display:block;font-size:9px}
  footer{margin-top:46px;color:var(--muted);font-size:12px;border-top:1px solid var(--hair);padding-top:16px}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="kicker">HW1 · Amazon Gift Cards · Phase 3</div>
    <h1>Review Scoring Report</h1>
    <p class="sub">Star rating (predicted 1&#8211;5) from title + text only, checked against the reviewer's true score.</p>
  </header>
  <div class="rule"></div>
  <section>
    <h2>Headline numbers</h2>
    <div class="cards" id="cards"></div>
  </section>
  <div class="rule"></div>
  <section>
    <h2>Primary emotion <span>— LLM take vs. NRC lexicon take, compared</span></h2>
    <div class="cards" id="emo-cards"></div>
    <div class="emo-bars-wrap" id="emo-bars"></div>
  </section>
  <div class="rule"></div>
  <section>
    <h2>Star rating distribution <span>— actual vs. predicted</span></h2>
    <div class="emo-bars-wrap" id="rating-bars"></div>
    <div id="per-class-accuracy" style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:12px"></div>
  </section>
  <div class="rule"></div>
  <section>
    <h2>View details</h2>
    <div style="display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap">
      <button class="table-toggle expanded" id="tab-emo" data-table="emo">Primary emotion</button>
      <button class="table-toggle collapsed" id="tab-ratings" data-table="ratings">Star ratings</button>
      <button class="table-toggle collapsed" id="tab-confusion" data-table="confusion-table">Confusion matrix</button>
      <button class="table-toggle collapsed" id="tab-classes" data-table="classes">Failure classes</button>
    </div>
    <div id="emo" class="table-content">
      <div class="filterbar">
        <div class="fgroup"><label>Agree</label>
          <select id="f-emo-agree"><option value="">All</option><option>agree</option><option>differ</option></select></div>
        <div class="fgroup"><label>LLM emotion</label>
          <select id="f-emo-llm"><option value="">All</option></select></div>
        <div class="fgroup"><label>Lexicon emotion</label>
          <select id="f-emo-lex"><option value="">All</option></select></div>
        <div class="countnote" id="emo-count"></div>
        <button class="clear" id="emo-clearf">clear filters</button>
      </div>
      <table>
        <thead><tr><th>#</th><th>Review</th><th>LLM emotion</th><th>Lexicon emotion</th><th>Agree</th></tr></thead>
        <tbody id="emo-rows"></tbody>
      </table>
    </div>
    <div id="ratings" class="table-content hidden">
      <div class="filterbar">
        <div class="fgroup"><label>Result</label>
          <select id="f-result"><option value="">All</option><option>wrong</option><option>correct</option></select></div>
        <div class="fgroup"><label>Actual</label>
          <select id="f-actual"><option value="">All</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></div>
        <div class="fgroup"><label>Predicted</label>
          <select id="f-pred"><option value="">All</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></div>
        <div class="countnote" id="count"></div>
        <button class="clear" id="clearf">clear filters</button>
      </div>
      <table>
        <thead><tr><th>#</th><th>Review</th><th>Predicted</th><th>Actual</th><th>Result</th><th>Model's reason</th></tr></thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
    <div id="classes" class="table-content hidden"></div>
    <div id="confusion-table" class="table-content hidden">
      <table style="font-size:12px">
        <thead id="confusion-header"></thead>
        <tbody id="confusion-body"></tbody>
      </table>
    </div>
  </section>
  <footer>
    Scoring used title + text only; the true rating was applied only after scoring and was never written to the model input.
    Model: claude-haiku-4-5-20251001 via Anthropic's OpenAI-compatible endpoint.
  </footer>
</div>
<script>
const DATA = __DATA__;
(function(){const s=DATA.stats;const cards=[
  ["Exact matches",s.pct_exact+"%",s.exact+"/"+s.n+" predicted == actual"],
  ["Within ±1 star",s.pct_within1+"%","off by one or resolved"],
  ["Mean abs. error",s.mae.toFixed(2)+" <small>★</small>","average distance from true"],
  ["Reviewed",s.n,"reviews scored from title + text"]];
let h="";for(const[t,v,d]of cards)h+=`<div class="card"><b>${v}</b><span>${t} — ${d}</span></div>`;document.getElementById('cards').innerHTML=h;})();
(function(){const d=DATA.distribution;const max=Math.max(...d.map(x=>x.count),1);let h="";for(const x of d)h+=`<div class="bar" style="height:${(x.count/max)*100}%" title="${x.count} review(s)"><small>${x.count}</small><fig>${x.star}★</fig></div>`;document.getElementById('bars').innerHTML=h;})();
(function(){const r=DATA.rows;const rowsEl=document.getElementById('rows');
function rowHtml(x,i){const ok=x.correct;const badge=ok?`<span class="pill ok">correct</span>`:`<span class="pill bad">wrong</span>`;const title=(x.title||"").split(" ").slice(0,6).join(" ");return `<tr><td class="num">${i+1}</td><td class="review">${esc(x.text.slice(0,70))}${x.text.length>70?"…":""}<br><small>${esc(title)}</small></td><td class="num stars">${x.pred}★</td><td class="num">${x.true}★</td><td>${badge}</td><td class="reason">${esc(x.reason)}</td></tr>`;}
function render(){
  const fRes=document.getElementById('f-result').value;
  const fAct=document.getElementById('f-actual').value;
  const fPred=document.getElementById('f-pred').value;
  let shown=0,html="";
  r.forEach((x,i)=>{if(x.pred==null)return;
    if(fRes && (fRes==='correct')!==x.correct)return;
    if(fAct && String(x.true)!==fAct)return;
    if(fPred && String(x.pred)!==fPred)return;
    shown++;html+=rowHtml(x,i);});
  rowsEl.innerHTML=html;
  document.getElementById('count').textContent=shown===r.filter(x=>x.pred!=null).length
    ? `showing all ${shown}`
    : `showing ${shown} of ${r.filter(x=>x.pred!=null).length}`;
}
render();
['f-result','f-actual','f-pred'].forEach(id=>document.getElementById(id).addEventListener('change',render));
document.getElementById('clearf').addEventListener('click',()=>{['f-result','f-actual','f-pred'].forEach(id=>document.getElementById(id).value='');render();});
})();
(function(){const c=DATA.classes;if(!c||!c.length){document.getElementById('classes').innerHTML='<p style="color:var(--muted)">No wrong answers to classify.</p>';return;}let h="";for(const x of c){h+=`<div class="cls"><h3>${esc(x.name)}<span class="tag">${x.count}× · ${esc(x.pred_vs_true)}</span></h3><div class="meta">e.g. ${esc(x.example)}…</div><p>${esc(x.inference)}</p></div>`;}document.getElementById('classes').innerHTML=h;})();
(function(){
  const es=DATA.emotion_stats;
  const section=document.getElementById('emo-cards').closest('section');
  if(!es){section.style.display='none';return;}
  const EMOTIONS=["anger","anticipation","disgust","fear","joy","sadness","surprise","trust"];
  const cards=[
    ["Emotion agreement",es.pct_agree+"%",es.agree+"/"+es.n+" LLM and lexicon picked the same emotion"],
    ["Rows compared",es.n,"reviews with both an LLM and lexicon emotion"]];
  let h="";for(const[t,v,d]of cards)h+=`<div class="card"><b>${v}</b><span>${t} — ${d}</span></div>`;
  document.getElementById('emo-cards').innerHTML=h;

  function barsHtml(dist,label){
    const max=Math.max(...EMOTIONS.map(e=>dist[e]||0),1);
    let b="";for(const e of EMOTIONS){const v=dist[e]||0;b+=`<div class="bar" style="height:${(v/max)*100}%" title="${v} review(s)"><small>${v}</small><fig>${e}</fig></div>`;}
    return `<div><h4>${label}</h4><div class="bars">${b}</div></div>`;
  }
  document.getElementById('emo-bars').innerHTML =
    barsHtml(es.llm_distribution,'LLM emotion') + barsHtml(es.lexicon_distribution,'Lexicon emotion');

  const llmSel=document.getElementById('f-emo-llm'), lexSel=document.getElementById('f-emo-lex');
  for(const e of EMOTIONS){
    llmSel.insertAdjacentHTML('beforeend',`<option>${e}</option>`);
    lexSel.insertAdjacentHTML('beforeend',`<option>${e}</option>`);
  }

  const r=DATA.rows.filter(x=>x.llm_emotion&&x.lexicon_emotion);
  const rowsEl=document.getElementById('emo-rows');
  function rowHtml(x,i){
    const agreeBadge=x.emotion_agree?`<span class="pill ok">agree</span>`:`<span class="pill bad">differ</span>`;
    return `<tr><td class="num">${i+1}</td><td class="review">${esc(x.text.slice(0,70))}${x.text.length>70?"…":""}</td>`+
      `<td><span class="pill emo">${esc(x.llm_emotion)}</span></td>`+
      `<td><span class="pill emo">${esc(x.lexicon_emotion)}</span></td>`+
      `<td>${agreeBadge}</td></tr>`;
  }
  function render(){
    const fAgree=document.getElementById('f-emo-agree').value;
    const fLlm=llmSel.value, fLex=lexSel.value;
    let shown=0,html="";
    r.forEach((x,i)=>{
      if(fAgree && (fAgree==='agree')!==!!x.emotion_agree)return;
      if(fLlm && x.llm_emotion!==fLlm)return;
      if(fLex && x.lexicon_emotion!==fLex)return;
      shown++;html+=rowHtml(x,i);
    });
    rowsEl.innerHTML=html;
    document.getElementById('emo-count').textContent=shown===r.length?`showing all ${shown}`:`showing ${shown} of ${r.length}`;
  }
  render();
  ['f-emo-agree','f-emo-llm','f-emo-lex'].forEach(id=>document.getElementById(id).addEventListener('change',render));
  document.getElementById('emo-clearf').addEventListener('click',()=>{['f-emo-agree','f-emo-llm','f-emo-lex'].forEach(id=>document.getElementById(id).value='');render();});
})();
function barsHtml(categories,dist,label,fmt){const max=Math.max(...categories.map(c=>dist[c]||0),1);let b="";for(const c of categories){const v=dist[c]||0;b+=`<div class="bar" style="height:${(v/max)*100}%" title="${v} review(s)"><small>${v}</small><fig>${fmt?fmt(c):c}</fig></div>`;}return `<div><h4>${label}</h4><div class="bars">${b}</div></div>`;}
(function(){const rs=DATA.rating_stats;if(!rs){return;}
document.getElementById('rating-bars').innerHTML=barsHtml([1,2,3,4,5],rs.true_distribution,'Actual',c=>c+'★')+barsHtml([1,2,3,4,5],rs.pred_distribution,'Predicted',c=>c+'★');
let html="";for(const c of[1,2,3,4,5]){const pc=rs.per_class[c];const pct=pc.pct_correct;const isGood=pct>=70;html+=`<div class="accuracy-card ${isGood?'good':'bad'}"><b>${pct}%</b><span>${pc.correct}/${pc.n}</span></div>`;}
document.getElementById('per-class-accuracy').innerHTML=html;
let confHtml="<tr><th style='width:50px'>True\\Pred</th>";for(let p=1;p<=5;p++)confHtml+=`<th style='text-align:center'>${p}★</th>`;confHtml+="</tr>";
const maxOff=Math.max(...[].concat(...[1,2,3,4,5].map(t=>[1,2,3,4,5].filter(p=>p!==t).map(p=>rs.confusion[t][p]))),1);
for(let t=1;t<=5;t++){confHtml+=`<tr><th>${t}★</th>`;for(let p=1;p<=5;p++){const cnt=rs.confusion[t][p];const isD=t===p;const opacity=!isD?Math.min(cnt/maxOff,1)*0.5:0;confHtml+=`<td class="${isD?'conf-diag':'conf-off'}" style="${!isD?`opacity:${opacity+0.3}`:''};text-align:center;padding:4px;border:1px solid var(--hair)"><b>${cnt}</b></td>`;}confHtml+="</tr>";}
document.getElementById('confusion-header').innerHTML=confHtml.split('<tr>')[1].split('</tr>')[0];
document.getElementById('confusion-body').innerHTML=confHtml.split('</tr>').slice(1,-1).map(r=>'<tr>'+r+'</tr>').join('');
})();
['tab-emo','tab-ratings','tab-classes','tab-confusion'].forEach(id=>{const btn=document.getElementById(id);if(!btn)return;btn.addEventListener('click',()=>{const tableId=btn.dataset.table;document.querySelectorAll('.table-content').forEach(el=>el.classList.add('hidden'));document.querySelectorAll('.table-toggle').forEach(el=>el.classList.remove('expanded'));document.getElementById(tableId).classList.remove('hidden');btn.classList.add('expanded');});});
function esc(s){return(s||"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
</script>
</body>
</html>
"""

html = html.replace("__DATA__", json.dumps(payload))
open("report.html", "w").write(html)
wrong = sum(1 for r in rows if r["correct"] is False)
print(f"Wrote report.html ({len(html)} bytes) | {len(rows)} rows, {wrong} wrong")
print("Classes:", [f"{c['name']}={c['count']}" for c in class_rows])
