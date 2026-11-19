"""Duyệt đề xuất loại bằng chứng (evidence_type) của Codex và gộp thành file loại bằng chứng cuối.

Bối cảnh: scope mục 9. Codex kiểm 187 câu (47 câu nháp Gemini bỏ trống, 140 câu Gemini ghi "table") trong
data/processed/visr_bench_vi/codex_etype_20261118/. Người duyệt xem:
  - mọi câu nhóm "missing" (không có nhãn Gemini để đối chiếu);
  - câu nhóm "table_check" mà Codex không đồng ý với Gemini hoặc độ tin cậy khác "high".
Câu "table_check" còn lại (Codex và Gemini cùng ghi table, độ tin cậy high) lấy theo đề xuất.

Chạy từ gốc repo:
    python -X utf8 -m src.data_prep.etype_review build    # -> WORK/review_etype.html
    python -X utf8 -m src.data_prep.etype_review merge    # đọc WORK/etype_decisions.json -> data/qa/visr_bench_vi_r2_etype.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "data" / "processed" / "visr_bench_vi" / "codex_etype_20261118"
OUT = ROOT / "data" / "qa" / "visr_bench_vi_r2_etype.jsonl"
RAW_REL = "../../../raw/visr_bench_vi/"     # từ WORK tới ảnh trang gốc
TYPES = ("text", "table", "figure")


def load() -> list:
    inp = {}
    for l in (WORK / "etype_input.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            x = json.loads(l)
            inp[x["qid"]] = x
    rows = []
    for l in (WORK / "etype_proposals.jsonl").read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        p = json.loads(l)
        x = dict(inp[p["qid"]])
        x.update({k: p[k] for k in ("proposed_evidence_type", "confidence", "agrees_with_gemini",
                                    "layout_table", "mixed", "reason", "note")})
        x["need_review"] = needs_review(x)
        rows.append(x)
    assert len(rows) == len(inp) == len({r["qid"] for r in rows}), "đề xuất và đầu vào không khớp"
    rows.sort(key=lambda r: (not r["need_review"], r["document_id"], r["page_id"], r["qid"]))
    return rows


def needs_review(x: dict) -> bool:
    return x["group"] == "missing" or x["agrees_with_gemini"] is False or x["confidence"] != "high"


def build() -> None:
    rows = load()
    data = [{
        "qid": r["qid"], "group": r["group"], "doc": r["document_id"][:8], "page": r["page_id"],
        "q": r["question"], "a": r["answer"], "form": r["answer_form"],
        "crop": f"etype_check/{r['qid']}.jpg", "img": RAW_REL + r["image"], "size": r["image_size"],
        "bbox": r["evidence_bbox"], "gem": r["gemini_evidence_type"], "prop": r["proposed_evidence_type"],
        "conf": r["confidence"], "layout": r["layout_table"], "mixed": r["mixed"],
        "reason": r["reason"], "note": r["note"], "need": r["need_review"],
    } for r in rows]
    n_need = sum(d["need"] for d in data)
    html = PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False)).replace("__NNEED__", str(n_need))
    out = WORK / "review_etype.html"
    out.write_text(html, encoding="utf-8")
    print(f"Ghi {out.relative_to(ROOT)}: {len(data)} câu, cần duyệt {n_need}")


def merge(dec_path: Path) -> None:
    rows = load()
    dec = {d["qid"]: d for d in json.loads(dec_path.read_text(encoding="utf-8"))["decisions"]}
    unknown = set(dec) - {r["qid"] for r in rows}
    assert not unknown, f"qid lạ trong file quyết định: {sorted(unknown)[:5]}"
    missing = [r["qid"] for r in rows if r["need_review"] and r["qid"] not in dec]
    if missing:
        sys.exit(f"Dừng: còn {len(missing)} câu cần duyệt chưa có quyết định, ví dụ {missing[:3]}")
    out, src, change = [], Counter(), Counter()
    for r in rows:
        if r["qid"] in dec:
            et = dec[r["qid"]]["evidence_type"]
            s = "human" if et == r["proposed_evidence_type"] else "human_override"
        else:
            et, s = r["proposed_evidence_type"], "codex_gemini_agree"
        assert et in TYPES, (r["qid"], et)
        src[s] += 1
        change[f"{r['gemini_evidence_type']} -> {et}"] += 1
        out.append({"qid": r["qid"], "evidence_type": et, "source": s,
                    "gemini_evidence_type": r["gemini_evidence_type"], "codex_evidence_type": r["proposed_evidence_type"],
                    "codex_confidence": r["confidence"]})
    OUT.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out), encoding="utf-8")
    print(f"Ghi {OUT.relative_to(ROOT)}: {len(out)} câu")
    print("Nguồn:", dict(src))
    print("Gemini -> cuối:", dict(change))


PAGE = r"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Duyệt loại bằng chứng</title>
<style>
:root{--bg:#f6f6f4;--card:#fff;--fg:#1d1d1b;--mut:#6b6b66;--line:#ddd;--acc:#1f5fbf;--ok:#1d7a3a;--warn:#b5651d}
@media (prefers-color-scheme:dark){:root{--bg:#18181a;--card:#222225;--fg:#ececea;--mut:#9a9a95;--line:#3a3a3e;--acc:#7aa8ff;--ok:#5fc27e;--warn:#e0a060}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,Segoe UI,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:8px 16px;display:flex;gap:12px;flex-wrap:wrap;align-items:center}
main{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(320px,1fr);gap:16px;padding:16px}
@media (max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px}
.crop{max-width:100%;border:1px solid var(--line);display:block}
.page{position:relative;display:inline-block;max-width:100%}.page img{max-width:100%;display:block}
.box{position:absolute;border:2px solid #e53935;background:rgba(229,57,53,.08)}
button{font:inherit;padding:6px 12px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg);cursor:pointer}
button.sel{outline:3px solid var(--acc)}button.t{min-width:110px;font-weight:600}
.mut{color:var(--mut);font-size:13px}.tag{display:inline-block;padding:1px 8px;border-radius:10px;border:1px solid var(--line);font-size:12px;margin-right:4px}
.dis{color:var(--warn);font-weight:600}.done{color:var(--ok);font-weight:600}
h2{font-size:18px;margin:4px 0 8px}
</style></head><body>
<header>
 <strong>Duyệt loại bằng chứng</strong>
 <label><input type="radio" name="f" value="need" checked> Cần duyệt (__NNEED__)</label>
 <label><input type="radio" name="f" value="all"> Tất cả</label>
 <span id="pos" class="mut"></span><span id="prog" class="mut"></span>
 <button id="prev">←</button><button id="next">→</button>
 <button id="nextUn">Câu chưa duyệt kế tiếp (N)</button>
 <button id="exp">Xuất etype_decisions.json</button>
</header>
<main>
 <section class="card">
  <div id="cropWrap"><img id="crop" class="crop" alt="ảnh crop vùng bằng chứng"></div>
  <p><button id="tog">Xem cả trang (P)</button></p>
  <div id="pageWrap" hidden><div class="page" id="page"><img id="pimg" alt="trang"></div></div>
 </section>
 <section class="card">
  <div class="mut" id="meta"></div>
  <h2 id="q"></h2>
  <div><span class="mut">Đáp án:</span> <span id="a"></span></div>
  <p id="tags"></p>
  <p><span class="mut">Lý do của Codex:</span> <span id="reason"></span></p>
  <p id="noteP" class="mut"></p>
  <p><button class="t" data-t="text">1 · text</button> <button class="t" data-t="table">2 · table</button> <button class="t" data-t="figure">3 · figure</button></p>
  <p><button id="acc">Enter · đồng ý với Codex</button> <button id="clr">Xóa quyết định</button></p>
  <p>Trạng thái: <span id="st"></span></p>
  <div class="mut">
   <p><b>text</b>: chữ chạy, tiêu đề, danh sách, dòng "nhãn: giá trị" đứng riêng. <b>table</b>: lưới hàng × cột, muốn đọc đúng phải căn hàng và cột (kể cả bảng không kẻ viền nhưng cột thẳng hàng rõ). <b>figure</b>: biểu đồ, sơ đồ, ảnh, ảnh chụp màn hình, con dấu, chữ trong hình.</p>
   <p>Phím: 1/2/3 chọn loại và sang câu kế · Enter đồng ý với Codex · ←/→ chuyển câu · N câu chưa duyệt kế tiếp · P bật/tắt cả trang.</p>
  </div>
 </section>
</main>
<script>
const DATA=__DATA__;
const KEY="review_etype_decisions";
let dec={};try{dec=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(dec))}catch(e){}};
const $=id=>document.getElementById(id);
let filt="need",i=0;
const list=()=>DATA.filter(d=>filt==="all"||d.need);
function render(){
 const L=list();if(!L.length)return;i=Math.max(0,Math.min(i,L.length-1));const d=L[i];
 $("pos").textContent=`Câu ${i+1}/${L.length}`;
 const need=DATA.filter(x=>x.need),nd=need.filter(x=>dec[x.qid]).length;
 $("prog").textContent=`· đã duyệt ${nd}/${need.length} câu cần duyệt`;
 $("meta").textContent=`${d.doc} · trang ${d.page} · ${d.qid} · dạng ${d.form}`;
 $("q").textContent=d.q;$("a").textContent=d.a;
 $("tags").innerHTML=`<span class="tag">nhóm ${d.group}</span><span class="tag">Gemini: ${d.gem??"trống"}</span>`+
  `<span class="tag ${d.gem&&d.gem!==d.prop?"dis":""}">Codex: ${d.prop} · ${d.conf}</span>`+
  (d.layout?`<span class="tag">bảng dàn trang</span>`:"")+(d.mixed?`<span class="tag">nhiều loại vùng</span>`:"");
 $("reason").textContent=d.reason;$("noteP").textContent=d.note?("Ghi chú: "+d.note):"";
 $("crop").src=d.crop;
 const P=$("page");P.querySelectorAll(".box").forEach(b=>b.remove());$("pimg").src=d.img;
 const [W,H]=d.size;for(const b of d.bbox){const e=document.createElement("div");e.className="box";
  e.style.left=100*b[0]/W+"%";e.style.top=100*b[1]/H+"%";e.style.width=100*(b[2]-b[0])/W+"%";e.style.height=100*(b[3]-b[1])/H+"%";P.appendChild(e)}
 const c=dec[d.qid];
 document.querySelectorAll("button.t").forEach(b=>b.classList.toggle("sel",(c?c.evidence_type:null)===b.dataset.t));
 $("st").innerHTML=c?`<span class="done">đã chọn ${c.evidence_type}</span>${c.evidence_type!==d.prop?' <span class="dis">(khác Codex)</span>':""}`:(d.need?"chưa duyệt":"không bắt buộc duyệt (Codex và Gemini cùng ghi table, độ tin cậy high)");
}
function decide(t){const d=list()[i];dec[d.qid]={qid:d.qid,evidence_type:t,codex_evidence_type:d.prop,reviewed_at:new Date().toISOString()};save();i++;render()}
function nextUn(){const L=list();for(let k=1;k<=L.length;k++){const j=(i+k)%L.length;if(L[j].need&&!dec[L[j].qid]){i=j;render();return}}}
document.querySelectorAll("button.t").forEach(b=>b.onclick=()=>decide(b.dataset.t));
$("acc").onclick=()=>decide(list()[i].prop);
$("clr").onclick=()=>{delete dec[list()[i].qid];save();render()};
$("prev").onclick=()=>{i--;render()};$("next").onclick=()=>{i++;render()};$("nextUn").onclick=nextUn;
$("tog").onclick=()=>{$("pageWrap").hidden=!$("pageWrap").hidden};
document.querySelectorAll("input[name=f]").forEach(r=>r.onchange=()=>{filt=r.value;i=0;render()});
$("exp").onclick=()=>{const need=DATA.filter(x=>x.need),miss=need.filter(x=>!dec[x.qid]).length;
 if(miss&&!confirm(`Còn ${miss} câu cần duyệt chưa có quyết định. Vẫn xuất?`))return;
 const blob=new Blob([JSON.stringify({version:"review_etype",exported_at:new Date().toISOString(),total:Object.keys(dec).length,decisions:Object.values(dec)},null,1)],{type:"application/json"});
 const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="etype_decisions.json";a.click()};
document.addEventListener("keydown",e=>{if(e.target.tagName==="INPUT"&&e.target.type!=="radio")return;
 if(e.key==="1")decide("text");else if(e.key==="2")decide("table");else if(e.key==="3")decide("figure");
 else if(e.key==="Enter"){e.preventDefault();decide(list()[i].prop)}
 else if(e.key==="ArrowLeft"){i--;render()}else if(e.key==="ArrowRight"){i++;render()}
 else if(e.key==="n"||e.key==="N")nextUn();else if(e.key==="p"||e.key==="P")$("tog").click()});
render();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "merge"])
    ap.add_argument("--decisions", type=Path, default=WORK / "etype_decisions.json")
    a = ap.parse_args()
    build() if a.cmd == "build" else merge(a.decisions)


if __name__ == "__main__":
    main()
