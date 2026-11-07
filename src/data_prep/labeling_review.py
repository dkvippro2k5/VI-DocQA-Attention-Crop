"""Soạn trang duyệt nhãn (HTML, mở thẳng trên máy) và file Excel dự phòng; câu đáng ngờ lên đầu.

Ưu tiên (số nhỏ duyệt trước; một câu có nhiều lý do thì lấy mức nhỏ nhất):
  1  model trả answerable=false, hoặc các worker bất đồng ở batch hiệu chuẩn
  2  câu hỏi có dạng liệt kê / nhiều ý (dễ bị cắt còn một ý do quy tắc "ngắn nhất")
  3  lệch nhãn Qwen3-VL (F1 < 0,5)
  4  đáp án không khớp text trang tham khảo
  5  còn lại

Phạm vi: bằng chứng nằm trong MỘT trang (trang page_index của câu hỏi). Câu cần trang khác hoặc nhiều trang thì
"bỏ câu" kèm lý do, để đếm được giả định một trang sai ở bao nhiêu câu.

Trang HTML: kéo chuột trên ảnh để vẽ khung bằng chứng (một hoặc nhiều khung, lưu theo pixel ảnh gốc);
câu chưa duyệt được vẽ sẵn khung gợi ý (màu xanh) bằng cách dò đáp án trong tọa độ từ của trang (text PDF, hoặc
OCR nếu không có PDF; xem page_words.py); bấm vào một khung để xóa nó; G = dò lại theo đáp án đang sửa
(sửa đáp án khi chưa vẽ tay thì tự dò lại). Kết quả ghi bbox_source để đếm bao nhiêu khung giữ nguyên gợi ý.
Nhiều khung có hai nghĩa, ghi ở bbox_mode: "all" = mỗi khung là một phần đáp án (crop phải phủ tất cả),
"any" = các khung là chỗ thay thế cùng một đáp án lặp lại (crop phủ một chỗ là đủ); phím A bật/tắt "any".
Phím 1 = đúng, 2 = sửa, 3 = bỏ câu (cần chọn lý do), X = xóa khung cuối, F = vừa cả trang, +/- (hoặc Ctrl + lăn chuột, hoặc chụm/mở hai ngón trên touchpad;
chuột trên ảnh) = phóng/thu, Z = bật/tắt 170%, L = gắn/bỏ nhãn
"đáp án dài" (đánh giá riêng), mũi tên = chuyển câu,
Ctrl+Enter = lưu bản sửa và sang câu tiếp. "Đúng"/"Sửa" cần ít nhất một khung.
Tiến độ tự lưu trong trình duyệt (localStorage); "Xuất kết quả" tải về review_decisions_<vòng>.json để gộp.

Chạy từ gốc repo:
    python -X utf8 -m src.data_prep.labeling_review              # vòng 1
    python -X utf8 -m src.data_prep.labeling_review --round r2   # vòng 2
Ra: data/processed/visr_bench_vi/review/review_<vòng>.html (+ .xlsx), đã gitignore. Mỗi vòng có khóa localStorage và
file xuất riêng (review_decisions_<vòng>.json) nên duyệt vòng sau không đè kết quả vòng trước.
Vòng 2 không có đáp án Qwen để so; ưu tiên 1 thêm đáp án chỉ là dấu gạch (ô trống trong bảng).
"""

import json
import os
import re
import sys
from pathlib import Path

from src.data_prep.label_visr_answers import IN_DIR, collect_questions
from src.data_prep.page_words import page_words
from src.eval.score import f1, normalize

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "visr_bench_vi"
ROUNDS = {  # vòng -> (nhãn sau validate, thư mục worker)
    "v1": (DATA / "qa_vi_visr_labels_external.jsonl", DATA / "label_tasks_multi"),
    "r2": (DATA / "qa_vi_visr_labels_external_r2.jsonl", DATA / "label_tasks_r2"),
}
QWEN = DATA / "qa_vi_visr_answers.jsonl"
OUT_DIR = DATA / "review"
MULTI_ITEM = re.compile(r"liệt kê|những|bao gồm|\bgồm\b|^các .*(gì|nào)", re.IGNORECASE)
DOC_NAME = {"0004e630": "GIGABYTE", "01d7b45b": "An Bằng", "1aa33928": "C++"}
PRIO_NAME = {1: "không trả lời được / worker bất đồng", 2: "nghi câu nhiều ý", 3: "lệch Qwen",
             4: "không khớp text trang", 5: "còn lại"}
DROP_REASONS = ["đáp án ở trang khác", "cần nhiều trang", "trang không có đáp án", "câu hỏi mơ hồ hoặc sai", "khác"]


def load(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_rows(rnd):
    labels_path, workers = ROUNDS[rnd]
    labels = load(labels_path)
    qwen = {r["qid"]: r["answer"] for r in load(QWEN)} if QWEN.exists() else {}
    page_text = {q["qid"]: q["page_text"] for q in collect_questions()}
    anchor = {}
    for f in sorted(workers.glob("worker_*/outputs/a00_anchor.jsonl")):
        for r in load(f):
            anchor.setdefault(r["qid"], {})[r["agent"]] = r["answer"] if r["answerable"] else "(không trả lời được)"

    rows = []
    for r in labels:
        reasons, prio = [], 5
        q_ans = qwen.get(r["qid"])
        f1q = f1(r["answer"], q_ans) if (q_ans and r["answerable"]) else None
        ag = anchor.get(r["qid"])
        if not r["answerable"]:
            reasons.append("model nói không trả lời được"); prio = min(prio, 1)
        if r["answerable"] and not re.search(r"\w", r["answer"]):
            reasons.append("đáp án không có chữ/số"); prio = min(prio, 1)
        if ag and len({normalize(v) for v in ag.values()}) > 1:
            reasons.append("worker bất đồng"); prio = min(prio, 1)
        if MULTI_ITEM.search(r["question"].strip()):
            reasons.append("nghi câu nhiều ý"); prio = min(prio, 2)
        if f1q is not None and f1q < 0.5:
            reasons.append("lệch Qwen"); prio = min(prio, 3)
        if "không_có_trong_text_trang" in r.get("flags", []):
            reasons.append("không khớp text trang"); prio = min(prio, 4)
        doc = r["document_id"][:8]
        img = (IN_DIR / r["image"]).resolve()
        rows.append({
            "prio": prio, "reasons": reasons, "qid": r["qid"], "doc": DOC_NAME.get(doc, doc), "page": r["page_id"],
            "question": r["question"], "answer": r["answer"], "answerable": r["answerable"],
            "evidence_type": r["evidence_type"], "qwen": q_ans, "f1q": None if f1q is None else round(f1q, 2),
            "workers": ag or {}, "img_abs": img, "img": Path(os.path.relpath(img, OUT_DIR)).as_posix(),
            "page_text": page_text.get(r["qid"], ""), "document_id": r["document_id"], "image_rel": r["image"],
        })
    rows.sort(key=lambda x: (x["prio"], x["doc"], x["page"], x["qid"]))
    return rows


HTML = r"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Duyệt nhãn VisR-Bench VI (/*ROUND*/)</title>
<style>
:root { --bg:#f6f7f9; --card:#fff; --ink:#1d2330; --muted:#5f6b7a; --line:#d9dee5; --accent:#2557d6;
        --ok:#1f8a4c; --fix:#b7791f; --drop:#c0392b; --box:#e0262f; }
* { box-sizing:border-box; }
body { margin:0; font:15px/1.5 system-ui, "Segoe UI", sans-serif; background:var(--bg); color:var(--ink); }
header { display:flex; gap:12px; align-items:center; padding:8px 14px; background:var(--card); border-bottom:1px solid var(--line); flex-wrap:wrap; }
header b { font-size:16px; }
.prog { flex:1; min-width:160px; height:8px; background:#e6e9ee; border-radius:4px; overflow:hidden; }
.prog > div { height:100%; background:var(--ok); width:0; }
button, select { font:inherit; padding:5px 10px; border:1px solid var(--line); background:#fff; border-radius:6px; cursor:pointer; }
button:hover { border-color:var(--accent); }
main { display:grid; grid-template-columns: minmax(0,1.15fr) minmax(0,1fr); gap:12px; padding:12px; height:calc(100vh - 54px); }
.imgbox { background:var(--card); border:1px solid var(--line); border-radius:8px; overflow:auto; }
.stage { position:relative; width:100%; margin:0 auto; cursor:crosshair; user-select:none; }
.zoomLbl { min-width:64px; text-align:center; color:var(--muted); font-size:13px; }
.stage img { display:block; width:100%; height:auto; pointer-events:none; }
.bx { position:absolute; border:2px solid var(--box); background:rgba(224,38,47,.10); pointer-events:none; }
.bx.tmp { border-style:dashed; }
.bx.sg { border-color:#2557d6; background:rgba(37,87,214,.10); } .bx.sg span { background:#2557d6; }
.bx span { position:absolute; top:-18px; left:-2px; background:var(--box); color:#fff; font-size:11px; padding:0 4px; border-radius:3px; }
.side { overflow:auto; display:flex; flex-direction:column; gap:10px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 12px; }
.meta { color:var(--muted); font-size:13px; }
.q { font-size:18px; font-weight:600; margin:4px 0; }
.tag { display:inline-block; font-size:12px; padding:1px 8px; border-radius:10px; background:#eef2fb; color:#24407f; margin:2px 4px 2px 0; }
.p1 { background:#fde8e6; color:#8e2b20; } .p2 { background:#fdf1dc; color:#7a4e0e; }
label { font-size:13px; color:var(--muted); display:block; margin-bottom:3px; }
textarea, input[type=text], .side select { width:100%; font:inherit; padding:6px 8px; border:1px solid var(--line); border-radius:6px; }
textarea { min-height:70px; resize:vertical; }
.other { white-space:pre-wrap; margin-bottom:2px; } .src { color:var(--muted); font-size:12px; margin-top:8px; }
.src:first-child { margin-top:0; }
.btns { display:flex; gap:8px; flex-wrap:wrap; }
.btns button { flex:1; min-width:90px; padding:9px; font-weight:600; }
.b-ok { border-color:var(--ok); color:var(--ok); } .b-fix { border-color:var(--fix); color:var(--fix); } .b-drop { border-color:var(--drop); color:var(--drop); }
.boxbar { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.state { font-weight:600; } .s-đúng { color:var(--ok); } .s-sửa { color:var(--fix); } .s-bỏ { color:var(--drop); } .s-thiếu { color:var(--drop); }
.msg { color:var(--drop); font-size:13px; min-height:18px; }
details pre { white-space:pre-wrap; font-size:13px; max-height:260px; overflow:auto; background:#fafbfc; padding:8px; border-radius:6px; }
.keys { font-size:12px; color:var(--muted); }
@media (max-width: 900px) { main { grid-template-columns:1fr; height:auto; } }
</style></head><body>
<header>
  <b>Duyệt nhãn</b><span id="pos" class="meta"></span>
  <div class="prog"><div id="bar"></div></div><span id="done" class="meta"></span>
  <select id="filter"><option value="all">Mọi mức ưu tiên</option></select>
  <button id="nextTodo" title="Câu chưa duyệt xong kế tiếp">Câu chưa xong →</button>
  <button id="export">Xuất kết quả</button>
  <button id="importBtn">Nạp kết quả</button><input id="importFile" type="file" accept=".json" hidden>
</header>
<main>
  <div class="imgbox" id="imgbox"><div class="stage" id="stage"><img id="img" alt="Ảnh trang" draggable="false"></div></div>
  <div class="side">
    <div class="card">
      <div class="meta" id="where"></div>
      <div class="q" id="question"></div>
      <div id="tags"></div>
    </div>
    <div class="card">
      <div class="boxbar"><span>Khung bằng chứng: <b id="nbox">0</b></span>
        <button id="undoBox">Xóa khung cuối (X)</button><button id="clearBox">Xóa hết</button><button id="zoomPage" title="Thu ảnh để thấy cả trang (F)">Cả trang (F)</button><button id="zoomOut" title="Thu nhỏ (-)">−</button><span class="zoomLbl" id="zoomLbl"></span><button id="zoomIn" title="Phóng to (+); Z = bật/tắt 170%">+</button></div>
      <div class="boxbar" style="margin-top:6px"><span class="meta" id="sugg"></span><button id="resugg">Dò lại theo đáp án (G)</button></div>
      <div class="meta">Khung xanh là gợi ý tự động, khung đỏ là vẽ tay. Bấm vào một khung để xóa; kéo chuột để vẽ thêm. Câu nhiều ý: một khung cho mỗi ý.</div>
      <label style="margin-top:6px; color:var(--ink)"><input type="checkbox" id="alt"> Các khung là chỗ thay thế (A): mỗi khung tự nó đủ trả lời (đáp án lặp lại). Bỏ chọn nếu mỗi khung là một phần của đáp án.</label>
    </div>
    <div class="card">
      <label for="answer">Đáp án (sửa trực tiếp nếu cần)</label>
      <textarea id="answer"></textarea>
      <div class="meta" id="answerMeta"></div>
      <label style="margin-top:6px; color:var(--ink)"><input type="checkbox" id="long"> Đáp án dài (L): cần cả đoạn, nhiều bước hoặc liệt kê nhiều ý; sẽ đánh giá riêng</label>
    </div>
    <div class="card">
      <div class="btns">
        <button class="b-ok" id="bOk">1 · Đúng</button>
        <button class="b-fix" id="bFix">2 · Sửa</button>
        <button class="b-drop" id="bDrop">3 · Bỏ câu</button>
      </div>
      <div style="margin-top:8px"><label for="reason">Lý do bỏ câu (chỉ khi bỏ)</label><select id="reason"><option value="">(chọn lý do)</option></select></div>
      <div style="margin-top:8px"><label for="note">Ghi chú (tùy chọn)</label><input type="text" id="note"></div>
      <div style="margin-top:6px">Trạng thái: <span id="state" class="state"></span></div>
      <div class="msg" id="msg"></div>
      <div class="keys">Phím: 1 đúng · 2 sửa (con trỏ vào ô đáp án) · 3 bỏ câu · X xóa khung cuối · F vừa cả trang · +/−, Ctrl+lăn chuột hoặc chụm hai ngón trên touchpad để phóng/thu · Z bật/tắt 170% · G dò lại khung gợi ý · bấm vào khung để xóa · A khung là chỗ thay thế · L đáp án dài (không cần tick: tự gán khi gộp nhãn, chỉ dùng cho ngoại lệ) · ← → chuyển câu · Ctrl+Enter lưu bản sửa và sang câu tiếp · Esc thoát ô nhập</div>
    </div>
    <div class="card" id="others"></div>
    <div class="card"><details><summary>Text trang (trích tự động, có thể sai; tin ảnh hơn)</summary><pre id="pageText"></pre></details></div>
  </div>
</main>
<script>
const ITEMS = /*DATA*/;
const PRIO_NAME = /*PRIO*/;
const DROP_REASONS = /*REASONS*/;
const WORDS = /*WORDS*/;
const KEY = "review_/*ROUND*/_decisions";
const $ = (id) => document.getElementById(id);
let dec = {};
try { dec = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { dec = {}; }
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(dec)); } catch (e) { alert("Không lưu được vào trình duyệt; hãy Xuất kết quả thường xuyên."); } };
const isDone = (d) => !!d && (d.verdict === "bỏ câu" ? !!d.drop_reason : (d.evidence_bbox || []).length > 0);
let view = ITEMS.map((_, i) => i), cur = 0, boxes = [], drag = null, sugg = { boxes: [], how: "" };

for (const r of DROP_REASONS) { const o = document.createElement("option"); o.value = r; o.textContent = r; $("reason").appendChild(o); }
const prios = [...new Set(ITEMS.map(x => x.prio))].sort();
for (const p of prios) {
  const o = document.createElement("option"); o.value = p;
  o.textContent = `Ưu tiên ${p}: ${PRIO_NAME[p]} (${ITEMS.filter(x => x.prio === p).length})`; $("filter").appendChild(o);
}
function applyFilter() {
  const f = $("filter").value;
  view = ITEMS.map((x, i) => [x, i]).filter(([x]) => f === "all" || String(x.prio) === f).map(([, i]) => i);
  cur = 0; render();
}
function line(parent, cls, text) { const d = document.createElement("div"); if (cls) d.className = cls; d.textContent = text; parent.appendChild(d); return d; }
const say = (t) => { $("msg").textContent = t || ""; };

// ---- gợi ý khung: dò token đáp án trong tọa độ từ của trang; chuẩn hóa giống src/eval/score.py normalize
const EDGE = new Set([...".,;:!?\"'()[]\u201c\u201d\u2018\u2019\u2026"]);
const QMAP = { "\u2019": "'", "\u2018": "'", "\u02bc": "'", "`": "'", "\u201c": '"', "\u201d": '"' };
function stripEdge(t) { let a = 0, b = t.length; while (a < b && EDGE.has(t[a])) a++; while (b > a && EDGE.has(t[b - 1])) b--; return t.slice(a, b); }
const normToks = (s) => s.normalize("NFC").replace(/[\u2019\u2018\u02bc`\u201c\u201d]/g, c => QMAP[c]).toLowerCase()
  .replace(/_/g, " ").split(/\s+/).map(stripEdge).filter(Boolean);
const noAccent = (t) => t.normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/đ/g, "d");
const same = (key) => (P, i, T, j) => key(P[i][0]) === key(T[j]);
function findSeq(T, P, eq) {
  const hits = [];
  for (let i = 0; i + T.length <= P.length; i++) { let j = 0; while (j < T.length && eq(P, i + j, T, j)) j++; if (j === T.length) hits.push([i, i + T.length]); }
  return hits;
}
function longestRun(T, P, eq) {  // đoạn token liên tiếp dài nhất trùng giữa đáp án và trang
  let best = 0, hits = [];
  for (let i = 0; i < P.length; i++) for (let s = 0; s < T.length; s++) {
    let k = 0; while (i + k < P.length && s + k < T.length && eq(P, i + k, T, s + k)) k++;
    if (k > best) { best = k; hits = [[i, i + k]]; } else if (k && k === best) hits.push([i, i + k]);
  }
  return [best, hits];
}
function lev1(a, b) {  // khoảng cách sửa <= 1 (OCR hay sai một ký tự: "chi" -> "chl")
  if (a === b) return true;
  if (Math.abs(a.length - b.length) > 1) return false;
  let i = 0; while (i < a.length && a[i] === b[i]) i++;
  return a.slice(i + 1) === b.slice(i + 1) || a.slice(i) === b.slice(i + 1) || a.slice(i + 1) === b.slice(i);
}
function fuzzyWindow(T, P) {
  // OCR rớt từ hoặc dính từ nên không khớp liền mạch: tìm cửa sổ trên trang chứa nhiều token đáp án nhất
  // (bỏ dấu, sai 1 ký tự với từ >= 3 ký tự, hoặc từ trang dính chứa từ đáp án >= 4 ký tự)
  const keys = T.map(noAccent);
  const hit = P.map(w => { const p = noAccent(w[0]);
    return keys.findIndex(k => k === p || (k.length >= 3 && lev1(k, p)) || (k.length >= 4 && p.length > k.length && p.includes(k))); });
  const L = Math.ceil(T.length * 1.3) + 2;
  let best = 0, at = null;
  for (let i = 0; i < P.length; i++) {
    if (hit[i] < 0) continue;
    const seen = new Set(); let last = i;
    for (let j = i; j < Math.min(P.length, i + L); j++) if (hit[j] >= 0 && !seen.has(hit[j])) { seen.add(hit[j]); last = j; }
    if (seen.size > best) { best = seen.size; at = [i, last + 1]; }
  }
  if (!at) return [0, []];
  // OCR đọc ngang qua trang hai cột nên cửa sổ có thể lẫn cột kia. Cột = dải x mà mọi chữ trong cửa sổ phủ liền
  // (khe trắng rộng hơn nửa chiều cao chữ là ranh giới); mỗi cột một khung, bỏ cột chỉ có vài từ trùng ngẫu nhiên
  const win = P.slice(at[0], at[1]);
  const h = win.map(w => w[4] - w[2]).sort((a, b) => a - b)[win.length >> 1] || 10;
  const spans = [];
  for (const w of [...win].sort((a, b) => a[1] - b[1])) {
    const s = spans[spans.length - 1];
    if (s && w[1] <= s[1] + 0.5 * h) s[1] = Math.max(s[1], w[3]); else spans.push([w[1], w[3]]);
  }
  const cols = spans.map(() => []);
  for (let j = at[0]; j < at[1]; j++) {
    if (hit[j] < 0) continue;
    const cx = (P[j][1] + P[j][3]) / 2;
    cols[Math.max(0, spans.findIndex(s => cx >= s[0] && cx <= s[1]))].push(P[j]);
  }
  const n = cols.reduce((s, c) => s + c.length, 0), top = Math.max(...cols.map(c => c.length));
  return [best, cols.filter(c => c.length && (c.length === top || c.length >= Math.max(2, 0.3 * n))).map(unionBox)];
}
function unionBox(ws) {
  const PAD = 4;
  return [Math.max(0, Math.min(...ws.map(w => w[1])) - PAD), Math.max(0, Math.min(...ws.map(w => w[2])) - PAD),
          Math.max(...ws.map(w => w[3])) + PAD, Math.max(...ws.map(w => w[4])) + PAD];
}
function suggest(answer, it) {
  const page = WORDS[it.image_rel];
  if (!page) return { boxes: [], how: "không có tọa độ từ cho trang này" };
  if (!answer.trim()) return { boxes: [], how: "không có đáp án để dò" };
  const P = page.words, out = [], hows = [], MAXHIT = 6;
  let nItems = 0, repeated = false;  // một ý khớp nguyên cụm ở nhiều chỗ: gợi ý đánh dấu "chỗ thay thế"
  const exact = (T) => { let h = findSeq(T, P, same(x => x)); return h.length ? h : findSeq(T, P, same(noAccent)); };
  for (const raw of answer.split(";")) {
    const T = normToks(raw);
    if (!T.length) continue;
    nItems++;
    let hits = findSeq(T, P, same(x => x)), how = "khớp đúng";
    if (!hits.length) { hits = findSeq(T, P, same(noAccent)); how = "khớp khi bỏ dấu"; }
    if (!hits.length && raw.includes(",")) {  // Gemini hay ngăn các ý bằng dấu phẩy: dò từng ý
      const parts = raw.split(",").map(normToks).filter(t => t.length).map(exact);
      if (parts.every(h => h.length)) { hits = parts.flat(); how = "khớp từng ý tách dấu phẩy"; }
    }
    if (!hits.length) {
      const [b, h] = T.length > 1 ? longestRun(T, P, same(noAccent)) : [0, []];
      const [fb, fw] = fuzzyWindow(T, P);
      const need = T.length <= 2 ? T.length : Math.max(2, Math.ceil(T.length * 0.6));
      if (fb > b && fb >= need) {
        hows.push(`khớp gần đúng (${fb}/${T.length} từ)`);
        for (const bx of fw) if (!out.some(o => o.join() === bx.join())) out.push(bx);
        continue;
      }
      else if (b >= Math.max(2, Math.ceil(T.length / 2))) { hits = h; how = `khớp một phần (${b}/${T.length} từ)`; }
    }
    if (!hits.length) { hows.push("có ý không tìm thấy"); continue; }
    hows.push(hits.length > 1 ? `${how}, ${hits.length} chỗ` : how);
    if (hits.length > 1 && (how === "khớp đúng" || how === "khớp khi bỏ dấu")) repeated = true;
    for (const [a, b] of hits.slice(0, MAXHIT)) {
      const bx = unionBox(P.slice(a, b));
      if (!out.some(o => o.join() === bx.join())) out.push(bx);
    }
  }
  const src = page.src === "pdf" ? "text PDF" : "OCR, gần đúng";
  return { boxes: out, how: `${src} · ${[...new Set(hows)].join("; ")}`, alt: nItems === 1 && repeated && out.length > 1 };
}
const inSugg = (b) => sugg.boxes.some(s => s.join() === b.join());
function showSugg() { $("sugg").textContent = `Gợi ý: ${sugg.boxes.length} khung (${sugg.how})`; }
function resuggest() { sugg = suggest($("answer").value, ITEMS[view[cur]]); boxes = sugg.boxes.map(b => b.slice()); $("alt").checked = sugg.alt; showSugg(); drawBoxes(); }

// ---- khung bằng chứng: lưu theo pixel ảnh gốc, vẽ theo phần trăm nên không lệch khi phóng to
function drawBoxes(tmp) {
  for (const el of [...$("stage").querySelectorAll(".bx")]) el.remove();
  const img = $("img"), W = img.naturalWidth, H = img.naturalHeight;
  if (!W) return;
  const all = boxes.map((b, i) => [b, String(i + 1), false]).concat(tmp ? [[tmp, "", true]] : []);
  for (const [b, label, isTmp] of all) {
    const d = document.createElement("div"); d.className = "bx" + (isTmp ? " tmp" : inSugg(b) ? " sg" : "");
    d.style.left = (100 * b[0] / W) + "%"; d.style.top = (100 * b[1] / H) + "%";
    d.style.width = (100 * (b[2] - b[0]) / W) + "%"; d.style.height = (100 * (b[3] - b[1]) / H) + "%";
    if (label) { const s = document.createElement("span"); s.textContent = label; d.appendChild(s); }
    $("stage").appendChild(d);
  }
  $("nbox").textContent = boxes.length;
}
function toImg(e) {
  const img = $("img"), r = img.getBoundingClientRect();
  const x = Math.max(0, Math.min(r.width, e.clientX - r.left)), y = Math.max(0, Math.min(r.height, e.clientY - r.top));
  return [Math.round(x / r.width * img.naturalWidth), Math.round(y / r.height * img.naturalHeight)];
}
const rectOf = (a, b) => [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])];
$("stage").addEventListener("mousedown", (e) => { if (e.button !== 0) return; e.preventDefault(); drag = toImg(e); });
window.addEventListener("mousemove", (e) => { if (drag) drawBoxes(rectOf(drag, toImg(e))); });
window.addEventListener("mouseup", (e) => {
  if (!drag) return;
  const b = rectOf(drag, toImg(e)); drag = null;
  if (b[2] - b[0] >= 6 && b[3] - b[1] >= 6) { boxes.push(b); say(""); }
  else {  // bấm (không kéo): xóa khung nhỏ nhất chứa điểm bấm
    const [x, y] = b, area = (q) => (q[2] - q[0]) * (q[3] - q[1]);
    const inside = boxes.map((q, i) => [q, i]).filter(([q]) => x >= q[0] && x <= q[2] && y >= q[1] && y <= q[3]);
    if (inside.length) { inside.sort((u, v) => area(u[0]) - area(v[0])); boxes.splice(inside[0][1], 1); }
  }
  drawBoxes();
});
$("img").addEventListener("load", () => drawBoxes());
const undoBox = () => { boxes.pop(); drawBoxes(); };
$("undoBox").onclick = undoBox;
$("clearBox").onclick = () => { boxes = []; drawBoxes(); };
// ---- phóng/thu ảnh: zoom = "page" (vừa khít cả trang trong khung) hoặc bội số của chiều rộng khung.
// Khung bằng chứng lưu theo pixel ảnh gốc và vẽ theo phần trăm nên không đổi khi phóng/thu.
const ZSTEPS = [0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1, 1.25, 1.5, 1.7, 2, 2.5, 3];
let zoom = 1;
try { const z = localStorage.getItem("review_zoom"); if (z) zoom = z === "page" ? "page" : +z || 1; } catch (e) {}
function zoomFrac() {  // chế độ "page" quy ra bội số tương đương, để +/- đi tiếp từ đó
  const box = $("imgbox"), img = $("img"), ar = img.naturalWidth / img.naturalHeight || 0.7;
  return Math.min(1, (box.clientHeight - 4) * ar / (box.clientWidth - 4));
}
function applyZoom() {
  const box = $("imgbox"), f = zoom === "page" ? zoomFrac() : zoom;
  $("stage").style.width = Math.max(50, (box.clientWidth - 4) * f) + "px";
  $("zoomLbl").textContent = zoom === "page" ? `cả trang` : Math.round(zoom * 100) + "%";
  try { localStorage.setItem("review_zoom", String(zoom)); } catch (e) {}
}
function setZoom(z) {
  const box = $("imgbox"), cy = (box.scrollTop + box.clientHeight / 2) / Math.max(1, box.scrollHeight);
  zoom = z; applyZoom();
  box.scrollTop = cy * box.scrollHeight - box.clientHeight / 2;  // giữ vùng đang xem ở giữa khi đổi cỡ
}
const stepZoom = (dir) => {
  const f = zoom === "page" ? zoomFrac() : zoom;
  const next = dir > 0 ? ZSTEPS.find(s => s > f + 1e-6) : [...ZSTEPS].reverse().find(s => s < f - 1e-6);
  if (next) setZoom(next);
};
const toggleZoom = () => setZoom(zoom === 1.7 ? 1 : 1.7);
$("zoomPage").onclick = () => setZoom("page");
$("zoomIn").onclick = () => stepZoom(1);
$("zoomOut").onclick = () => stepZoom(-1);
$("img").addEventListener("load", applyZoom);
window.addEventListener("resize", applyZoom);
// Ctrl + lăn chuột hoặc chụm/mở hai ngón trên touchpad = phóng/thu ảnh (không phóng cả trang web).
// Trình duyệt báo cử chỉ chụm ngón là sự kiện wheel có ctrlKey, deltaY nhỏ và liên tục, nên tách hai loại:
// - nấc chuột (|deltaY| lớn, số nguyên): mỗi nấc một mức, các mức cách nhau ít nhất 180 ms;
// - touchpad: phóng/thu mượt theo deltaY (hệ số PINCH; tăng để nhạy hơn, giảm để đỡ nhạy).
const PINCH = 0.006;
let wheelAt = 0;
$("imgbox").addEventListener("wheel", (e) => {
  if (!e.ctrlKey) return;
  e.preventDefault();
  const dy = e.deltaMode === 1 ? e.deltaY * 40 : e.deltaY;  // deltaMode 1 = tính theo dòng
  if (Math.abs(dy) >= 50 && Number.isInteger(dy)) {
    const now = performance.now();
    if (now - wheelAt < 180) return;
    wheelAt = now; stepZoom(dy < 0 ? 1 : -1);
  } else {
    const f = zoom === "page" ? zoomFrac() : zoom;
    const next = Math.min(ZSTEPS[ZSTEPS.length - 1], Math.max(ZSTEPS[0], f * Math.exp(-dy * PINCH)));
    setZoom(Math.round(next * 100) / 100);
  }
}, { passive: false });
applyZoom();

const nWords = (s) => s.trim() ? s.trim().split(/\s+/).length : 0;
// giống src/eval/score.py answer_form: >= 2 ý ngăn bằng ";" là list, một ý > 25 từ (hoặc tick L) là long
const formOf = (s, tick) => s.split(";").filter(x => normToks(x).length).length > 1 ? "list"
  : (tick || normToks(s).length > 25 ? "long" : "short");
function showMeta() {
  const it = ITEMS[view[cur]];
  $("answerMeta").textContent = `Gemini: ${it.answerable ? "trả lời được" : "KHÔNG trả lời được"} · bằng chứng: ${it.evidence_type} · đáp án đang có ${nWords($("answer").value)} từ · dạng (tự gán khi gộp): ${formOf($("answer").value, $("long").checked)}`;
}
let suggTimer = null;
$("answer").addEventListener("input", () => {
  showMeta();
  if (boxes.every(inSugg)) { clearTimeout(suggTimer); suggTimer = setTimeout(resuggest, 400); }  // chưa vẽ tay: dò lại theo đáp án mới
});
$("resugg").onclick = resuggest;
function render() {
  if (!view.length) return;
  const it = ITEMS[view[cur]], d = dec[it.qid];
  $("pos").textContent = `Câu ${cur + 1}/${view.length}`;
  sugg = suggest(d && d.verdict === "sửa" ? d.answer_final : (it.answer || it.qwen || ""), it);
  boxes = d && d.evidence_bbox ? d.evidence_bbox.map(b => b.slice()) : sugg.boxes.map(b => b.slice());
  showSugg();
  if ($("img").getAttribute("src") !== it.img) { $("img").src = it.img; $("imgbox").scrollTop = 0; } else drawBoxes();
  $("where").textContent = `${it.doc} · trang ${it.page} · ${it.qid}`;
  $("question").textContent = it.question;
  const tags = $("tags"); tags.textContent = "";
  const pt = document.createElement("span"); pt.className = "tag p" + it.prio; pt.textContent = "Ưu tiên " + it.prio; tags.appendChild(pt);
  for (const r of it.reasons) { const s = document.createElement("span"); s.className = "tag"; s.textContent = r; tags.appendChild(s); }
  $("answer").value = d && d.verdict === "sửa" ? d.answer_final : it.answer;
  showMeta();
  $("long").checked = !!(d && d.long_answer);
  showMeta();
  $("alt").checked = d ? d.bbox_mode === "any" : sugg.alt;
  $("note").value = d ? (d.note || "") : "";
  $("reason").value = d && d.drop_reason ? d.drop_reason : "";
  const st = $("state");
  if (!d) { st.className = "state"; st.textContent = "chưa duyệt"; }
  else if (!isDone(d)) { st.className = "state s-thiếu"; st.textContent = d.verdict + " (còn thiếu khung bằng chứng)"; }
  else { st.className = "state s-" + d.verdict.split(" ")[0]; st.textContent = d.verdict + (d.drop_reason ? ": " + d.drop_reason : ` · ${d.evidence_bbox.length} khung`); }
  say("");
  const o = $("others"); o.textContent = "";
  line(o, "src", "Đáp án Qwen3-VL" + (it.f1q !== null ? ` (F1 với Gemini ${it.f1q})` : ""));
  line(o, "other", it.qwen || "(không có)");
  if (it.qwen) { const b = document.createElement("button"); b.textContent = "Dùng đáp án này"; b.style.marginTop = "4px";
    b.onclick = () => { $("answer").value = it.qwen; $("answer").focus(); }; o.appendChild(b); }
  const ws = Object.entries(it.workers);
  if (ws.length) { line(o, "src", "Đáp án các worker (batch hiệu chuẩn)"); for (const [a, v] of ws) line(o, "other", `${a}: ${v}`); }
  $("pageText").textContent = it.page_text || "(trống)";
  const n = ITEMS.filter(x => isDone(dec[x.qid])).length;
  $("done").textContent = `${n}/${ITEMS.length} xong`;
  $("bar").style.width = (100 * n / ITEMS.length) + "%";
}
function decide(verdict) {
  const it = ITEMS[view[cur]], txt = $("answer").value.trim(), img = $("img");
  if (verdict === "đúng" && !it.answer) { say("Gemini không trả lời câu này: nếu trang có đáp án thì gõ vào ô và bấm Ctrl+Enter; nếu không thì chọn lý do và bấm 3."); return false; }
  if (verdict === "bỏ câu" && !$("reason").value) { say("Chọn lý do bỏ câu trước, rồi bấm 3 lại."); $("reason").focus(); return false; }
  if (verdict !== "bỏ câu" && !boxes.length) { say("Cần vẽ ít nhất một khung bằng chứng trên ảnh."); return false; }
  if (verdict === "sửa" && (!txt || txt === it.answer)) { say("Sửa đáp án trong ô rồi bấm Ctrl+Enter."); $("answer").focus(); $("answer").select(); return false; }
  dec[it.qid] = { qid: it.qid, verdict,
                  answer_final: verdict === "bỏ câu" ? "" : (verdict === "đúng" ? it.answer : txt),
                  evidence_bbox: verdict === "bỏ câu" ? [] : boxes.map(b => b.slice()),
                  bbox_source: verdict === "bỏ câu" ? "" : bboxSource(),
                  image_size: [img.naturalWidth, img.naturalHeight],
                  drop_reason: verdict === "bỏ câu" ? $("reason").value : "",
                  long_answer: verdict !== "bỏ câu" && $("long").checked,
                  bbox_mode: verdict === "bỏ câu" ? "" : ($("alt").checked && boxes.length > 1 ? "any" : "all"),
                  note: $("note").value.trim(), reviewed_at: new Date().toISOString() };
  save(); return true;
}
function bboxSource() {
  const n = boxes.filter(inSugg).length;
  if (!n) return "vẽ tay";
  if (n < boxes.length) return "gợi ý + vẽ thêm";
  return n === sugg.boxes.length ? "gợi ý" : "gợi ý, bỏ bớt";
}
const go = (k) => { cur = Math.max(0, Math.min(view.length - 1, cur + k)); render(); };
$("bOk").onclick = () => { decide("đúng") && go(1); };
$("bFix").onclick = () => { decide("sửa") && go(1); };
$("bDrop").onclick = () => { decide("bỏ câu") && go(1); };
$("filter").onchange = applyFilter;
$("nextTodo").onclick = () => {
  for (let k = 1; k <= view.length; k++) { const j = (cur + k) % view.length; if (!isDone(dec[ITEMS[view[j]].qid])) { cur = j; render(); return; } }
  alert("Đã duyệt xong các câu trong bộ lọc này.");
};
$("note").addEventListener("change", () => { const it = ITEMS[view[cur]]; if (dec[it.qid]) { dec[it.qid].note = $("note").value.trim(); save(); } });
$("reason").addEventListener("change", () => $("reason").blur());
$("alt").addEventListener("change", () => {
  const it = ITEMS[view[cur]], d = dec[it.qid];
  if (d && d.verdict !== "bỏ câu") { d.bbox_mode = $("alt").checked && d.evidence_bbox.length > 1 ? "any" : "all"; save(); }
  $("alt").blur();
});
$("long").addEventListener("change", () => {
  const it = ITEMS[view[cur]], d = dec[it.qid];
  if (d && d.verdict !== "bỏ câu") { d.long_answer = $("long").checked; save(); }
  $("long").blur();
  showMeta();
});
document.addEventListener("keydown", (e) => {
  const tag = document.activeElement.tagName;
  if (tag === "TEXTAREA" || (tag === "INPUT" && document.activeElement.type === "text")) {
    if (e.key === "Enter" && e.ctrlKey) { e.preventDefault(); if (decide("sửa")) { document.activeElement.blur(); go(1); } }
    if (e.key === "Escape") document.activeElement.blur();
    return;
  }
  if (tag === "SELECT" && !["1", "3", "Escape"].includes(e.key)) return;
  const k = e.key.toLowerCase();
  if (k === "1") { decide("đúng") && go(1); }
  else if (k === "2") { e.preventDefault(); $("answer").focus(); $("answer").select(); }
  else if (k === "3") { decide("bỏ câu") && go(1); }
  else if (k === "x") undoBox();
  else if (k === "l") { $("long").checked = !$("long").checked; showMeta(); }
  else if (k === "a") $("alt").checked = !$("alt").checked;
  else if (k === "z") toggleZoom();
  else if (k === "f") setZoom(zoom === "page" ? 1 : "page");
  else if (e.key === "+" || e.key === "=") stepZoom(1);
  else if (e.key === "-") stepZoom(-1);
  else if (k === "g") resuggest();
  else if (e.key === "Escape") document.activeElement.blur();
  else if (e.key === "ArrowRight") go(1);
  else if (e.key === "ArrowLeft") go(-1);
});
$("export").onclick = () => {
  const out = { version: "review_/*ROUND*/", scope: "bằng chứng trong một trang", exported_at: new Date().toISOString(),
                total: ITEMS.length, decisions: ITEMS.filter(x => dec[x.qid]).map(x => dec[x.qid]) };
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], { type: "application/json" }));
  a.download = "review_decisions_/*ROUND*/.json"; a.click();
};
$("importBtn").onclick = () => $("importFile").click();
$("importFile").onchange = async (e) => {
  const f = e.target.files[0]; if (!f) return;
  try { const j = JSON.parse(await f.text()); for (const d of j.decisions || []) dec[d.qid] = d; save(); render(); alert(`Đã nạp ${(j.decisions || []).length} kết quả.`); }
  catch (err) { alert("File không đọc được: " + err.message); }
};
render();
</script></body></html>
"""


def write_html(rows, rnd):
    items = [{k: v for k, v in r.items() if k != "img_abs"} for r in rows]
    words = page_words([{"document_id": r["document_id"], "page_id": r["page"], "image": r["image_rel"]} for r in rows],
                       OUT_DIR / "page_words_ocr_cache.json")
    dump = lambda o: json.dumps(o, ensure_ascii=False).replace("</", "<\\/")
    html = (HTML.replace("/*DATA*/", dump(items)).replace("/*PRIO*/", dump(PRIO_NAME))
            .replace("/*REASONS*/", dump(DROP_REASONS)).replace("/*WORDS*/", dump(words)).replace("/*ROUND*/", rnd))
    out = OUT_DIR / f"review_{rnd}.html"
    out.write_text(html, encoding="utf-8")
    return out


def write_xlsx(rows, rnd):
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
    except ImportError:
        return None
    wb = Workbook()
    ws = wb.active
    ws.title = "Duyệt nhãn"
    ws.append(["STT", "Ưu tiên", "Lý do", "Tài liệu", "Trang", "Câu hỏi", "Đáp án Gemini", "Trả lời được",
               "Loại bằng chứng", "Đáp án Qwen3-VL", "F1 với Qwen", "Đáp án các worker", "Ảnh trang", "qid"])
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="D9D9D9")
    for i, r in enumerate(rows, 1):
        ws.append([i, r["prio"], "; ".join(r["reasons"]), r["doc"], r["page"], r["question"], r["answer"],
                   "có" if r["answerable"] else "không", r["evidence_type"], r["qwen"] or "",
                   "" if r["f1q"] is None else r["f1q"], " | ".join(f"{a}: {v}" for a, v in sorted(r["workers"].items())),
                   "mở ảnh", r["qid"]])
        for col in (6, 7, 10, 12):  # ô chữ: '+', '++' không bị hiểu là công thức
            ws.cell(ws.max_row, col).data_type = "s"
        link = ws.cell(ws.max_row, 13)
        link.hyperlink = r["img_abs"].as_uri()
        link.font = Font(color="0563C1", underline="single")
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    out = OUT_DIR / f"review_{rnd}.xlsx"
    wb.save(out)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="v1", choices=sorted(ROUNDS))
    rnd = ap.parse_args().round
    rows = build_rows(rnd)
    html = write_html(rows, rnd)
    xlsx = write_xlsx(rows, rnd)
    print(f"Trang duyệt: {html}")
    if xlsx:
        print(f"Excel dự phòng: {xlsx.relative_to(ROOT)}")
    counts = {}
    for r in rows:
        counts[r["prio"]] = counts.get(r["prio"], 0) + 1
    for p in sorted(counts):
        print(f"  ưu tiên {p} ({PRIO_NAME[p]}): {counts[p]} câu")


if __name__ == "__main__":
    main()
