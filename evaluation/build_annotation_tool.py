"""
Build annotation_tool.html: a self-contained web page for labeling
eval_sample.csv in the browser (no install, works offline).

The 1,200 chunks are split into 3 batches of 400 in file order
(the file is already shuffled across star ratings).

Usage (from the repository root):

    uv run python evaluation/build_annotation_tool.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
N_BATCHES = 3


def main() -> None:
    with open(HERE / "eval_sample.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    batch_size = -(-len(rows) // N_BATCHES)
    items = [
        {
            "id": r["chunk_id"],
            "batch": i // batch_size + 1,
            "product": f"{r['brand']} {r['model']}".strip(),
            "text": r["text"],
        }
        for i, r in enumerate(rows)
    ]

    # "</" would end the <script> block early if a review contains "</script>".
    data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    batch_options = "".join(
        f'<option value="{b}">Batch {b}</option>' for b in range(1, N_BATCHES + 1)
    )
    html = TEMPLATE.replace("__DATA__", data).replace("__BATCH_OPTIONS__", batch_options)

    out = HERE / "annotation_tool.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {len(items)} chunks in {N_BATCHES} batches to {out}")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AI6127 Eval Set Annotation</title>
<style>
  body { margin: 0; font-family: system-ui, sans-serif; color: #222; }
  #setup { max-width: 520px; margin: 80px auto; line-height: 1.6; }
  #setup input, #setup select, #setup button { font-size: 16px; padding: 6px 10px; margin: 4px 0; }
  #app { display: flex; height: 100vh; }
  main { flex: 1; padding: 20px 32px; overflow: auto; }
  aside { width: 400px; padding: 16px 20px; background: #f5f5f5; border-left: 1px solid #ddd;
          overflow: auto; font-size: 14px; line-height: 1.5; }
  .top { display: flex; justify-content: space-between; color: #555; font-size: 14px; }
  .track { height: 6px; background: #e3e3e3; border-radius: 3px; margin: 8px 0 18px; }
  #bar { height: 6px; background: #3a7d44; border-radius: 3px; width: 0; }
  #product { font-weight: 600; margin-bottom: 6px; }
  #text { font-size: 18px; line-height: 1.6; white-space: pre-wrap; background: #fff;
          border: 1px solid #ccc; border-radius: 6px; padding: 16px; min-height: 140px; }
  .labels { margin: 18px 0 10px; }
  .labels button { font-size: 16px; padding: 10px 14px; margin: 4px 6px 4px 0; cursor: pointer;
                   border: 2px solid #aaa; border-radius: 6px; background: #fff; }
  .labels button.on { border-color: #222; background: #ffe7a0; font-weight: 600; }
  .nav button { font-size: 14px; padding: 6px 12px; margin-right: 6px; cursor: pointer; }
  #status { color: #3a7d44; margin-top: 10px; min-height: 1.4em; }
  #chunk-id { color: #999; font-size: 12px; margin-top: 6px; }
  aside h3 { margin: 14px 0 4px; }
  aside .ex { color: #555; font-style: italic; }
  kbd { border: 1px solid #aaa; border-radius: 3px; padding: 0 4px; background: #fff; font-size: 12px; }
</style>
</head>
<body>

<div id="setup">
  <h2>AI6127 evaluation set: annotation</h2>
  <p>Label the sentiment of smartphone review chunks. Your progress is saved in this
     browser automatically, so you can close the page and come back later
     (same browser, same name and batch).</p>
  <p>Your name<br><input id="name" placeholder="e.g. Jay"></p>
  <p>Your batch (ask the coordinator)<br>
     <select id="batch"><option value="">Choose...</option>__BATCH_OPTIONS__</select></p>
  <p><button onclick="start()">Start</button> <span id="setup-msg" style="color:#b00"></span></p>
</div>

<div id="app" hidden>
  <main>
    <div class="top"><span id="who"></span><span id="progress"></span></div>
    <div class="track"><div id="bar"></div></div>
    <div id="product"></div>
    <div id="text"></div>
    <div id="chunk-id"></div>
    <div class="labels" id="labels"></div>
    <div class="nav">
      <button onclick="go(-1)">&larr; Previous</button>
      <button onclick="go(1)">Next &rarr;</button>
      <button onclick="nextUnlabeled()">Jump to next unlabeled</button>
      <button onclick="download()"><b>Download my labels (CSV)</b></button>
    </div>
    <div id="status"></div>
  </main>

  <aside>
    <h2 style="margin-top:0">How to label</h2>
    <p>Read the whole chunk and pick <b>one</b> label for it. Press the number key
       (or click); the next chunk opens automatically. <kbd>&larr;</kbd> <kbd>&rarr;</kbd>
       move back and forth, so you can fix a label any time.</p>

    <h3><kbd>1</kbd> Positive</h3>
    Only positive opinions (about the phone, its features, the seller/service or the carrier).
    <div class="ex">"Battery easily lasts two days. Love it."</div>

    <h3><kbd>2</kbd> Negative</h3>
    Only negative opinions.
    <div class="ex">"Screen cracked after a week. Do not buy."</div>

    <h3><kbd>3</kbd> Mixed</h3>
    Both a clear positive and a clear negative opinion.
    <div class="ex">"Small screen somewhat limiting but great for travel."</div>

    <h3><kbd>4</kbd> Neutral (no opinion)</h3>
    No opinion at all: only facts, questions, wishes, or flat words like
    "moderate", "nothing special", "not an issue".
    <div class="ex">"Bought it for my mother last month."</div>
    <div class="ex">"I would like at least a 4 hr battery life."</div>

    <h3><kbd>5</kbd> Unusable</h3>
    Not English, gibberish, or not about a phone at all. Use rarely.

    <h3>Rules</h3>
    <ul style="padding-left:18px">
      <li><b>Facts + one opinion &rarr; the opinion wins.</b> "Got it Monday. Works great." is Positive.</li>
      <li>A clear complaint or praise counts even without opinion words:
          "Died after two months." is Negative.</li>
      <li>A wish alone is not an opinion: "I wish it had more storage." is Neutral.</li>
      <li>Comparisons: "Much better than my old LG" is Positive about this phone.</li>
      <li>Sarcasm: label what the reviewer means. "Great, it died in 2 days." is Negative.</li>
      <li>Judge the text only. The star rating is hidden on purpose.</li>
      <li>Do not discuss labels with other annotators until everyone is done.</li>
      <li>Unsure? Pick your best guess and move on (about 15&ndash;30 s per chunk).</li>
    </ul>

    <h3>When you finish</h3>
    Click <b>Download my labels (CSV)</b> and send the file to the coordinator.

    <p style="color:#888;font-size:12px;margin-top:18px">Label definitions adapted from the
       SemEval-2014 Task 4 ABSA annotation guidelines (sections 3.1, 3.2, 5).</p>
  </aside>
</div>

<script>
const ITEMS = __DATA__;
const LABELS = [
  { key: "1", value: "positive", name: "Positive" },
  { key: "2", value: "negative", name: "Negative" },
  { key: "3", value: "mixed", name: "Mixed" },
  { key: "4", value: "neutral", name: "Neutral (no opinion)" },
  { key: "5", value: "unusable", name: "Unusable" },
];
const $ = id => document.getElementById(id);
let items = [], labels = {}, pos = 0, annotator = "", batch = 0, storeKey = "";

$("labels").innerHTML = LABELS.map(l =>
  `<button data-value="${l.value}" onclick="setLabel('${l.value}')"><kbd>${l.key}</kbd> ${l.name}</button>`
).join("");

const last = JSON.parse(localStorage.getItem("eval-annotation:last") || "null");
if (last) { $("name").value = last.annotator; $("batch").value = last.batch; }

function start() {
  annotator = $("name").value.trim();
  batch = Number($("batch").value);
  if (!annotator || !batch) { $("setup-msg").textContent = "Enter your name and choose a batch."; return; }
  items = ITEMS.filter(it => it.batch === batch);
  storeKey = `eval-annotation:${batch}:${annotator.toLowerCase()}`;
  labels = JSON.parse(localStorage.getItem(storeKey) || "{}");
  localStorage.setItem("eval-annotation:last", JSON.stringify({ annotator, batch }));
  const first = items.findIndex(it => !labels[it.id]);
  pos = first === -1 ? 0 : first;
  $("who").textContent = `${annotator} · batch ${batch}`;
  $("setup").hidden = true;
  $("app").hidden = false;
  render();
}

function render() {
  const it = items[pos];
  const done = items.filter(x => labels[x.id]).length;
  $("product").textContent = it.product || "(unknown model)";
  $("text").textContent = it.text;
  $("chunk-id").textContent = it.id;
  $("progress").textContent = `Chunk ${pos + 1} / ${items.length} · labeled ${done}`;
  $("bar").style.width = `${100 * done / items.length}%`;
  document.querySelectorAll("#labels button").forEach(b =>
    b.classList.toggle("on", b.dataset.value === labels[it.id]));
}

function setLabel(value) {
  labels[items[pos].id] = value;
  localStorage.setItem(storeKey, JSON.stringify(labels));
  $("status").textContent = "";
  if (pos < items.length - 1) pos++;
  else $("status").textContent = "Last chunk. Check for unlabeled ones, then download.";
  render();
}

function go(step) { pos = Math.min(items.length - 1, Math.max(0, pos + step)); render(); }

function nextUnlabeled() {
  const i = items.findIndex(it => !labels[it.id]);
  if (i === -1) $("status").textContent = "All chunks are labeled.";
  else { pos = i; render(); }
}

function toCsv() {
  const rows = [["chunk_id", "batch", "annotator", "label", "subjectivity", "polarity"]];
  for (const it of items) {
    const l = labels[it.id] || "";
    const subjectivity = l === "neutral" ? "neutral" : (l && l !== "unusable") ? "opinionated" : "";
    const polarity = ["positive", "negative", "mixed"].includes(l) ? l : "";
    rows.push([it.id, it.batch, annotator, l, subjectivity, polarity]);
  }
  return rows.map(r => r.map(v => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\r\n");
}

function download() {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob(["﻿" + toCsv()], { type: "text/csv" }));
  a.download = `labels_batch${batch}_${annotator.replace(/\W+/g, "_")}.csv`;
  a.click();
  const missing = items.filter(it => !labels[it.id]).length;
  $("status").textContent = missing
    ? `Downloaded. ${missing} chunks are still unlabeled.`
    : "Downloaded. All chunks labeled, thank you!";
}

document.addEventListener("keydown", e => {
  if ($("app").hidden || e.ctrlKey || e.metaKey || e.altKey) return;
  const l = LABELS.find(x => x.key === e.key);
  if (l) setLabel(l.value);
  else if (e.key === "ArrowRight") go(1);
  else if (e.key === "ArrowLeft") go(-1);
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
