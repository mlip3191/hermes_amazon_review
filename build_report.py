#!/usr/bin/env python3
"""Build the self-contained report.html from report_data.json."""
import json

D = json.load(open("report_data.json"))
rows = D["rows"]
stats = D["stats"]

# Assign a failure class to each wrong row (by index) + a human inference.
CLASSES = {
    11: ("Hedged framing", "The understated 'What can you say about a gift card?' tone reads as ambivalence, so the model hedges to 4; the actual reviewer was fully satisfied (5)."),
    17: ("Complaint outweighs praise", "A concrete, repeated failing ('the note wasn't attached') dominates the model's read and drags a loved gift card to 4, even though the reviewer praised it elsewhere."),
    20: ("Generic praise underweighted", "Repetitive, superlative-free text ('Good Product. Good Product') is judged 'satisfied but not effusive' -> 4; Amazon reviewers routinely give bare 'good product' a 5."),
    21: ("Generic praise underweighted", "Same as above: brevity and absence of enthusiasm are read as below-perfect, while the true rating is 5."),
    22: ("Generic praise underweighted", "Same as above: the model withholds the top score for terse, un-adjectival praise; actual rating is 5."),
    23: ("Generic praise underweighted", "Same as above: minimal text without superlatives is scored 4 by the model, but the reviewer awarded 5."),
}

for i, r in enumerate(rows):
    if i in CLASSES:
        r["class"], r["inference"] = CLASSES[i]
    else:
        r["class"], r["inference"] = None, None

matched = [r for r in rows if r.get("class")]
classes_agg = {}
for r in matched:
    classes_agg.setdefault(r["class"], []).append(r)

class_rows = [
    {
        "name": name,
        "count": len(items),
        "example": items[0]["text"][:70],
        "inference": items[0]["inference"],
        "pred": "/".join(str(i["pred"]) for i in items),
    }
    for name, items in classes_agg.items()
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
}

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review Scoring Report</title>
<style>
  /* THEME — edit these variables to restyle the whole page */
  :root{
    --bg:#f1efea;            /* page background */
    --card:#faf9f6;          /* card / table surface */
    --ink:#2e2c28;           /* primary text */
    --muted:#8a867d;         /* secondary text */
    --hair:#dcd8cf;          /* borders */
    --accent:#6b6f50;        /* dull olive accent */
    --accent-soft:#e4e3d6;   /* accent tint */
    --good:#7b8570;          /* correct */
    --good-soft:#e6e9e2;
    --bad:#a97a6b;           /* incorrect (muted terracotta) */
    --bad-soft:#f0e4e0;
    --shadow:0 1px 2px rgba(0,0,0,.05),0 6px 18px rgba(0,0,0,.04);
    --radius:10px;
    --font:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font-family:var(--font);line-height:1.5;-webkit-font-smoothing:antialiased;}
  .wrap{max-width:920px;margin:0 auto;padding:40px 22px 80px;}
  header{padding:8px 0 4px}
  .kicker{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
  h1{font-size:30px;font-weight:650;margin:6px 0 4px;letter-spacing:-.01em}
  .sub{color:var(--muted);font-size:14px;margin:0}
  .rule{height:1px;background:var(--hair);margin:26px 0}

  section{margin:34px 0}
  h2{font-size:15px;letter-spacing:.06em;text-transform:uppercase;color:var(--accent);
     margin:0 0 14px;font-weight:600}
  h2 span{color:var(--muted);font-weight:400;text-transform:none;letter-spacing:0}

  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
  .card{background:var(--card);border:1px solid var(--hair);border-radius:var(--radius);
        box-shadow:var(--shadow);padding:16px 18px}
  .card b{font-size:28px;font-weight:650;display:block;line-height:1.1}
  .card b small{font-size:15px;font-weight:500;color:var(--muted)}
  .card span{font-size:12px;color:var(--muted)}

  .bars{display:flex;gap:6px;align-items:flex-end;height:120px;margin-top:6px}
  .bar{flex:1;background:var(--accent);border-radius:4px 4px 0 0;position:relative;
       min-height:3px;transition:background .15s}
  .bar small{position:absolute;top:-18px;left:0;right:0;text-align:center;color:var(--muted);font-size:11px}
  .bar fig{font-size:9px;color:var(--muted);text-align:center;margin-top:6px}

  table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--hair);
        border-radius:var(--radius);overflow:hidden;box-shadow:var(--shadow)}
  th,td{text-align:left;padding:10px 12px;border-bottom:1px solid var(--hair);vertical-align:top;
        font-size:13.5px}
  th{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);
     background:var(--accent-soft);font-weight:600}
  tr:last-child td{border-bottom:none}
  tbody tr:hover{background:#f6f4ef}
  .num{font-variant-numeric:tabular-nums;white-space:nowrap}
  .pill{display:inline-block;padding:1px 8px;border-radius:20px;font-size:11px;font-weight:600;
        letter-spacing:.03em}
  .pill.ok{background:var(--good-soft);color:var(--good)}
  .pill.bad{background:var(--bad-soft);color:var(--bad)}
  .stars{color:var(--accent);letter-spacing:.05em;font-weight:600}
  .review{color:var(--ink);max-width:340px}
  .review small{color:var(--muted)}
  .reason{color:var(--muted);font-size:12.5px;max-width:260px}

  .cls{background:var(--card);border:1px solid var(--hair);border-radius:var(--radius);
       padding:16px 18px;margin-bottom:12px;box-shadow:var(--shadow)}
  .cls h3{margin:0 0 2px;font-size:15px;font-weight:650}
  .cls .meta{font-size:12px;color:var(--muted);margin-bottom:10px}
  .cls p{margin:8px 0 0;font-size:13.5px;color:var(--ink)}
  .cls .tag{display:inline-block;background:var(--accent-soft);color:var(--accent);
            border-radius:20px;padding:1px 9px;font-size:11px;font-weight:600;margin-left:8px}

  footer{margin-top:46px;color:var(--muted);font-size:12px;border-top:1px solid var(--hair);
         padding-top:16px}
</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="kicker">HW1 · Amazon Gift Cards · Phase 3</div>
    <h1>Review Scoring Report</h1>
    <p class="sub">Star rating (predicted 1–5) from title + text only, checked against the reviewer's true score.</p>
  </header>

  <div class="rule"></div>

  <section>
    <h2>Headline numbers</h2>
    <div class="cards" id="cards"></div>
  </section>

  <section>
    <h2>Predicted rating distribution <span>— how each review was scored</span></h2>
    <div class="bars" id="bars"></div>
  </section>

  <div class="rule"></div>

  <section>
    <h2>Answer-by-answer · right &amp; wrong</h2>
    <table>
      <thead>
        <tr><th>#</th><th>Review</th><th>Predicted</th><th>Actual</th><th>Result</th><th>Model's reason</th></tr>
      </thead>
      <tbody id="rows"></tbody>
    </table>
  </section>

  <div class="rule"></div>

  <section>
    <h2>Why it got it wrong <span>— grouped failure classes</span></h2>
    <div id="classes"></div>
  </section>

  <footer>
    Scoring used title + text only; the true rating was applied only after scoring to measure accuracy
    and was never written to the output. Model: claude-haiku-4-5-20251001 via Anthropic's OpenAI-compatible endpoint.
  </footer>

</div>
<script>
const DATA = __DATA__;

/* Headline cards */
(function(){
  const s=DATA.stats;
  const cards=[
    ["Exact matches", s.pct_exact+"%", s.exact+"/"+s.n+" predicted == actual"],
    ["Within ±1 star", s.pct_within1+"%", "off by one or resolved"],
    ["Mean abs. error", s.mae.toFixed(2)+" <small>★</small>", "average distance from true"],
    ["Reviewed", s.n, "reviews scored from title + text"],
  ];
  let h="";
  for(const [t,v,d] of cards) h+=`<div class="card"><b>${v}</b><span>${t} — ${d}</span></div>`;
  document.getElementById('cards').innerHTML=h;
})();

/* Distribution bars */
(function(){
  const d=DATA.distribution; const max=Math.max(...d.map(x=>x.count),1);
  let h="";
  for(const x of d) h+=`<div class="bar" style="height:${(x.count/max)*100}%" title="${x.count} review(s)"><small>${x.count}</small><fig>${x.star}★</fig></div>`;
  document.getElementById('bars').innerHTML=h;
})();

/* Rows table */
(function(){
  const r=DATA.rows; let h="";
  r.forEach((x,i)=>{
    if(x.pred==null) return;
    const ok=x.correct;
    const badge=ok?`<span class="pill ok">correct</span>`:`<span class="pill bad">wrong</span>`;
    const title=(x.title||"").split(" ").slice(0,6).join(" ");
    h+=`<tr>
      <td class="num">${i+1}</td>
      <td class="review">${esc(x.text.slice(0,70))}${x.text.length>70?"…":""}<br><small>${esc(title)}</small></td>
      <td class="num stars">${x.pred}★</td>
      <td class="num">${x.true}★</td>
      <td>${badge}</td>
      <td class="reason">${esc(x.reason)}</td>
    </tr>`;
  });
  document.getElementById('rows').innerHTML=h;
})();

/* Classes */
(function(){
  const c=DATA.classes; if(!c||!c.length){document.getElementById('classes').innerHTML='<p style="color:var(--muted)">No wrong answers to classify.</p>';return;}
  let h="";
  for(const x of c){
    h+=`<div class="cls"><h3>${esc(x.name)}<span class="tag">${x.count}× · &#8594;${esc(x.pred)}</span></h3>
      <div class="meta">e.g. ${esc(x.example)}…</div>
      <p>${esc(x.inference)}</p></div>`;
  }
  document.getElementById('classes').innerHTML=h;
})();

function esc(s){return (s||"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
</script>
</body>
</html>
"""

html = html.replace("__DATA__", json.dumps(payload))
open("report.html", "w").write(html)
print("Wrote report.html  ", len(html), "bytes")
print("rows:", len(rows), "| wrong:", sum(1 for r in rows if r["correct"] is False))
