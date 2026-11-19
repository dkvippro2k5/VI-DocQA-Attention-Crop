"""LLM-judge có đáp án chuẩn (scope mục 6, mục 9; sửa đổi 18/11/2026).

Judge nhận câu hỏi, các đáp án chuẩn được chấp nhận và câu trả lời của model, trả về đúng/sai kèm một câu lý do.
Judge không xem ảnh. Một prompt cho mọi bộ và mọi nhánh (JUDGE_VERSION); temperature 0; đầu ra JSON.

Cache: mỗi bộ (model judge, JUDGE_VERSION, câu hỏi, đáp án chuẩn, câu trả lời) chỉ chấm một lần, lưu ở
results/judge/cache_<model>.jsonl. Hai nhánh trả lời giống hệt nhau thì nhận cùng một phán quyết.

Kiểm độ tin cậy: người chấm tay một mẫu (mặc định 50 câu `short` của VisR-Bench dev, seed cố định) mà không thấy
phán quyết của judge, rồi tính Cohen's κ. Scope: chỉ dùng judge khi κ ≥ 0,6; không đạt thì quay về EM.

Backend:
  llamacpp    model mở chạy cục bộ bằng llama-server (llama.cpp) trên Kaggle; đây là backend chính (scope, 18/11/2026).
              Server có API kiểu OpenAI ở LLAMACPP_BASE_URL (mặc định http://127.0.0.1:8080/v1). `--model` là nhãn ghi
              vào cache (tên file GGUF), không phải tên gửi đi.
  gemini      gọi thẳng Gemini API (GEMINI_API_KEY trong .env); gói miễn phí chỉ 20 lượt/ngày nên không dùng được
  openrouter  gọi model Gemini qua OpenRouter (OPENROUTER_API_KEY, OPENROUTER_BASE_URL trong .env)

Chạy từ gốc repo:
    python -X utf8 -m src.eval.judge models --backend gemini                     # liệt kê model khả dụng
    python -X utf8 -m src.eval.judge sample --pred results/B0_base/dev_visr_bench_vi.jsonl   # mẫu + trang chấm tay
    python -X utf8 -m src.eval.judge run --pred results/B0_base/dev_visr_bench_vi.jsonl --model <id> --backend gemini
    python -X utf8 -m src.eval.judge kappa --human <kappa_decisions.json> --judged results/B0_base/dev_visr_bench_vi_judge.jsonl
    python -X utf8 -m src.eval.judge pack --branches B0_base   # code + kết quả nhánh + mẫu κ -> data/processed/kaggle/da2-judge.zip
Notebook Kaggle chạy judge cục bộ: notebooks/judge_local.ipynb.
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "results" / "judge"
KAPPA_DIR = ROOT / "data" / "processed" / "judge_kappa"
JUDGE_VERSION = "j1"
KAPPA_MIN = 0.6

# Prompt viết bằng tiếng Anh để dùng chung cho cả bốn bộ (hai bộ tiếng Anh, hai bộ tiếng Việt).
# Dấu tiếng Việt được tính là một phần của từ (đúng với quy tắc "không bỏ dấu" của chỉ số chính).
SYSTEM_PROMPT = """You grade answers to visual question answering about images and document pages.
You see the question, the accepted gold answer(s), and a model's prediction. You do NOT see the image; trust the gold answer.

Mark the prediction CORRECT only if all of these hold:
1. It states the information in a gold answer (any one of the accepted answers is enough). For a gold answer listing several items separated by ";", every item must be present.
2. It does not also give a conflicting answer. A prediction that offers several candidate values, names or options where only one can be right is INCORRECT, even if one of them matches.
3. Extra words are allowed: restating the question, copying the surrounding sentence or clause, or a short explanation, as long as the gold information is clearly the answer being given and nothing added contradicts it.

Treat as equivalent: letter case, punctuation, spacing, word order that keeps the meaning, number formatting (05 = 5, 3,75 = 3.75, 1.000 = 1000), a unit present or omitted when unambiguous, and common abbreviations.

Vietnamese diacritics are part of the word: a word whose tone mark or vowel mark differs from the gold (e.g. "Nam" vs "Năm", "tầng" vs "tăng") is a different word. Exception: if the gold answer itself has no diacritics (e.g. names written without accents), ignore diacritics.

Mark INCORRECT: a different value, name, number or date; missing items of a list answer; a refusal or "not found"; an answer about something else.

Reply with JSON only: {"verdict": "correct" | "incorrect", "reason": "<one short sentence>"}"""

USER_TEMPLATE = """Question: {question}
Accepted gold answer(s): {gold}
Prediction: {pred}"""

# Bản Flash ổn định (không preview) mới nhất trên OpenRouter ngày 18/11/2026. Model judge phải giữ nguyên cho mọi nhánh;
# đổi model thì phải chấm lại mọi nhánh và đo lại κ.
DEFAULT_MODEL = {"gemini": "gemini-3.8-flash", "openrouter": "google/gemini-3.8-flash", "llamacpp": "local"}

# Giới hạn đầu ra của judge cục bộ đúng khuôn JSON (llama-server đổi schema thành grammar khi sinh).
VERDICT_SCHEMA = {"type": "object",
                  "properties": {"verdict": {"type": "string", "enum": ["correct", "incorrect"]},
                                 "reason": {"type": "string"}},
                  "required": ["verdict", "reason"]}


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "").strip()


def load_env() -> dict:
    env = {}
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*)\s*$", line)
            if m and not line.lstrip().startswith("#"):
                env[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    for k in ("GEMINI_API_KEY", "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "LLAMACPP_BASE_URL"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


def _post(url: str, body: dict, headers: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def call_judge(backend: str, model: str, env: dict, user: str) -> str:
    if backend == "llamacpp":
        base = env.get("LLAMACPP_BASE_URL") or "http://127.0.0.1:8080/v1"
        body = {"model": model, "temperature": 0, "seed": 0, "max_tokens": 200,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "verdict", "schema": VERDICT_SCHEMA}},
                # tắt chế độ suy nghĩ (Qwen3.x); template không có biến này thì bỏ qua
                "chat_template_kwargs": {"enable_thinking": False}}
        d = _post(base.rstrip("/") + "/chat/completions", body, {}, timeout=600)
        return d["choices"][0]["message"].get("content") or ""
    if backend == "gemini":
        key = env.get("GEMINI_API_KEY")
        if not key:
            sys.exit("Thiếu GEMINI_API_KEY trong .env")
        body = {"systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
        d = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent", body,
                  {"x-goog-api-key": key})
        return "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
    key, base = env.get("OPENROUTER_API_KEY"), (env.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1")
    if not key:
        sys.exit("Thiếu OPENROUTER_API_KEY trong .env")
    body = {"model": model, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]}
    d = _post(base.rstrip("/") + "/chat/completions", body, {"Authorization": f"Bearer {key}"})
    return d["choices"][0]["message"]["content"]


def parse_verdict(text: str) -> tuple[int | None, str]:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)   # phòng khi model vẫn sinh phần suy nghĩ
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    v = str(d.get("verdict", "")).strip().lower()
    if v in ("correct", "incorrect"):
        return int(v == "correct"), str(d.get("reason", ""))[:300]
    return None, text[:300]


def item_key(model: str, question: str, answers: list, pred: str) -> str:
    s = json.dumps([model, JUDGE_VERSION, nfc(question), sorted(nfc(a) for a in answers), nfc(pred)], ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def judge_rows(rows: list, backend: str, model: str, workers: int) -> dict:
    """Chấm các dòng (cần question, answers, pred); trả về {key: (verdict, reason)}, dùng và ghi cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"cache_{model.replace('/', '_')}.jsonl"
    cache = {}
    if cache_path.exists():
        for l in cache_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                c = json.loads(l)
                if c["verdict"] is not None:
                    cache[c["key"]] = (c["verdict"], c["reason"])
    todo = {}
    for r in rows:
        k = item_key(model, r["question"], r["answers"], r["pred"])
        if k not in cache:
            todo[k] = r
    print(f"{len(rows)} câu, {len(rows) - sum(1 for r in rows if item_key(model, r['question'], r['answers'], r['pred']) in todo)} có trong cache, cần chấm {len(todo)}")
    env, lock, stop = load_env(), threading.Lock(), threading.Event()

    def work(item):
        k, r = item
        if stop.is_set():
            return None
        user = USER_TEMPLATE.format(question=nfc(r["question"]), gold=" | ".join(nfc(a) for a in r["answers"]),
                                    pred=nfc(r["pred"]) or "(empty)")
        verdict, reason, raw = None, "", ""
        for attempt in range(5):
            try:
                raw = call_judge(backend, model, env, user)
                verdict, reason = parse_verdict(raw)
                if verdict is not None:
                    break
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")
                reason = f"HTTP {e.code}"
                if e.code == 429 and "PerDay" in body:
                    # hết hạn mức theo ngày (gói miễn phí): thử lại vô ích, dừng cả lượt
                    stop.set()
                    reason = "HTTP 429 hết hạn mức ngày"
                    break
                if e.code not in (429, 500, 502, 503, 504):
                    break
            except Exception as e:  # mạng chập chờn: thử lại
                reason = type(e).__name__
            time.sleep(2 ** attempt)
        if verdict is None:
            return None              # không ghi lỗi vào cache, lần chạy sau sẽ chấm lại
        rec = {"key": k, "model": model, "judge_version": JUDGE_VERSION, "verdict": verdict, "reason": reason,
               "at": datetime.now().isoformat(timespec="seconds")}
        with lock:
            with open(cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cache[k] = (verdict, reason)
        return verdict

    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(work, todo.items()))
    if res:
        print(f"Đã chấm {len(res)}; lỗi (không đọc được phán quyết) {sum(v is None for v in res)}")
    if stop.is_set():
        print("DỪNG: hết hạn mức gọi API theo ngày của key (gói miễn phí). Câu chưa chấm sẽ được chấm ở lần chạy sau.")
    return cache


def judge_file(path, backend: str, model: str, workers: int = 4, col: str = "pred") -> dict:
    """Chấm một file kết quả của nhánh; ghi <tên>_judge.jsonl cạnh file gốc; trả về tóm tắt."""
    p = Path(path).resolve()
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        r["answers"] = r.get("answers") or [r["gold"]]
        r["pred"] = r.get(col, "")
    t0 = time.time()
    cache = judge_rows(rows, backend, model, workers)
    out = p.with_name(p.stem + "_judge.jsonl")
    n_ok = n_missing = 0
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            v, why = cache.get(item_key(model, r["question"], r["answers"], r["pred"]), (None, ""))
            n_missing += v is None
            n_ok += v == 1
            f.write(json.dumps({"qid": r["qid"], "judge": v, "judge_reason": why, "judge_model": model,
                                "judge_version": JUDGE_VERSION, "judge_backend": backend}, ensure_ascii=False) + "\n")
    sec = round(time.time() - t0, 1)
    print(f"{out.name}: đúng {n_ok}/{len(rows)}, thiếu phán quyết {n_missing}, {sec} s")
    return {"file": out.name, "n": len(rows), "correct": n_ok, "missing": n_missing, "sec": sec}


def cmd_run(a) -> None:
    model = a.model or DEFAULT_MODEL[a.backend]
    for p in a.pred:
        judge_file(p, a.backend, model, a.workers, a.col)


def cmd_models(a) -> None:
    env = load_env()
    if a.backend == "gemini":
        key = env.get("GEMINI_API_KEY") or sys.exit("Thiếu GEMINI_API_KEY trong .env")
        req = urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                                     headers={"x-goog-api-key": key})
        for m in json.load(urllib.request.urlopen(req, timeout=30)).get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []) and "gemini" in m["name"]:
                print(m["name"].removeprefix("models/"), "|", m.get("displayName"))
    else:
        base = env.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        d = json.load(urllib.request.urlopen(base.rstrip("/") + "/models", timeout=30))
        for m in d.get("data", []):
            if m["id"].startswith("google/gemini"):
                print(m["id"], "|", m.get("name"))


def cmd_sample(a) -> None:
    """Lấy mẫu chấm tay (seed cố định) và dựng trang chấm; không gọi API, không có phán quyết judge trong trang."""
    rows = [json.loads(l) for l in Path(a.pred).read_text(encoding="utf-8").splitlines() if l.strip()]
    pool = sorted((r for r in rows if not a.form or r.get("answer_form") == a.form), key=lambda r: r["qid"])
    random.Random(a.seed).shuffle(pool)
    pick = pool[:a.n]
    KAPPA_DIR.mkdir(parents=True, exist_ok=True)
    sample = [{"qid": r["qid"], "dataset": r.get("dataset"), "question": r["question"],
               "answers": r.get("answers") or [r["gold"]], "pred": r["pred"]} for r in pick]
    meta = {"source": Path(a.pred).as_posix(), "form": a.form, "n": len(sample), "seed": a.seed,
            "created": date.today().isoformat(), "judge_version": JUDGE_VERSION}
    (KAPPA_DIR / "kappa_sample.json").write_text(json.dumps({"meta": meta, "items": sample}, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
    print(f"Mẫu {len(sample)} câu ({a.form or 'mọi dạng'}, seed {a.seed}) -> {(KAPPA_DIR / 'kappa_sample.json').relative_to(ROOT)}")
    build_page(None)


def build_page(codex_path: Path | None) -> None:
    """Dựng trang chấm tay từ kappa_sample.json; có codex_path thì kèm đề xuất của Codex (không bao giờ kèm judge)."""
    smp = json.loads((KAPPA_DIR / "kappa_sample.json").read_text(encoding="utf-8"))
    items, meta = smp["items"], dict(smp["meta"])
    if codex_path:
        cx = {}
        for l in codex_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                d = json.loads(l)
                assert d["verdict"] in ("correct", "incorrect"), d
                cx[d["qid"]] = d
        miss = [it["qid"] for it in items if it["qid"] not in cx]
        if miss:
            sys.exit(f"Đề xuất Codex thiếu {len(miss)} câu, ví dụ {miss[:3]}")
        items = [dict(it, codex={k: cx[it["qid"]].get(k) for k in ("verdict", "confidence", "reason")}) for it in items]
        meta["codex_file"] = codex_path.name
    html = KAPPA_PAGE.replace("__DATA__", json.dumps(items, ensure_ascii=False)).replace(
        "__META__", json.dumps(meta, ensure_ascii=False)).replace("__GUIDE__", GUIDE_VI)
    out = KAPPA_DIR / "review_kappa.html"
    out.write_text(html, encoding="utf-8")
    print(f"Trang chấm tay ({'có' if codex_path else 'không có'} đề xuất Codex) -> {out.relative_to(ROOT)}")


def cmd_page(a) -> None:
    build_page(Path(a.codex) if a.codex else None)


def cohen_kappa(x: list, y: list) -> float:
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    px, py = sum(x) / n, sum(y) / n
    pe = px * py + (1 - px) * (1 - py)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def cmd_kappa(a) -> None:
    judged = {}
    for l in Path(a.judged).read_text(encoding="utf-8").splitlines():
        if l.strip():
            d = json.loads(l)
            judged[d["qid"]] = d
    compute_kappa(Path(a.human), judged)


def compute_kappa(human_path: Path, judged: dict) -> dict:
    """κ giữa nhãn người (kappa_decisions.json) và phán quyết judge ({qid: {judge, judge_reason, judge_model, ...}})."""
    human = {d["qid"]: d for d in json.loads(Path(human_path).read_text(encoding="utf-8"))["decisions"]}
    sample = json.loads((KAPPA_DIR / "kappa_sample.json").read_text(encoding="utf-8"))
    qids = [it["qid"] for it in sample["items"]]
    miss_h = [q for q in qids if q not in human]
    miss_j = [q for q in qids if judged.get(q, {}).get("judge") is None]
    if miss_h or miss_j:
        sys.exit(f"Thiếu: người {len(miss_h)} câu, judge {len(miss_j)} câu")
    h = [int(human[q]["verdict"] == "correct") for q in qids]
    j = [int(judged[q]["judge"]) for q in qids]
    k = cohen_kappa(h, j)
    disagree = [{"qid": q, "human": hv, "judge": jv, "judge_reason": judged[q]["judge_reason"],
                 "human_note": human[q].get("note", "")} for q, hv, jv in zip(qids, h, j) if hv != jv]
    extra = {}
    hd = list(human.values())
    if any("codex_verdict" in d for d in hd):
        c = [int(human[q]["codex_verdict"] == "correct") for q in qids]
        extra = {"human_reviewed_from_codex": True,
                 "human_changed_codex": sum(hv != cv for hv, cv in zip(h, c)),
                 "kappa_judge_vs_codex": round(cohen_kappa(c, j), 4),
                 "kappa_human_vs_codex": round(cohen_kappa(c, h), 4)}
    res = {"date": date.today().isoformat(), "n": len(qids), "kappa": round(k, 4), **extra,
           "agreement": round(sum(a == b for a, b in zip(h, j)) / len(qids), 4),
           "human_correct": sum(h), "judge_correct": sum(j),
           "judge_model": judged[qids[0]]["judge_model"], "judge_version": judged[qids[0]]["judge_version"],
           "threshold": KAPPA_MIN, "passed": k >= KAPPA_MIN, "sample": sample["meta"], "disagreements": disagree}
    model_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", res["judge_model"])
    out = CACHE_DIR / f"kappa_{res['judge_version']}_{model_tag}.json"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"κ = {k:.3f}, đồng thuận {100 * res['agreement']:.1f}% (người đúng {sum(h)}, judge đúng {sum(j)}) "
          f"-> {'ĐẠT' if res['passed'] else 'KHÔNG ĐẠT'} ngưỡng {KAPPA_MIN}; bất đồng {len(disagree)} câu -> {out.name}")
    return res


def cmd_pack(a) -> None:
    """Đóng gói cho notebook judge_local: code, kết quả các nhánh (results/<nhánh>/dev_*.jsonl, test_*.jsonl; không gồm
    *_judge), cache judge đã có, mẫu κ và nhãn người. Không gồm ảnh: judge không xem ảnh."""
    import zipfile
    files = {f for f in (ROOT / "src").rglob("*.py") if "__pycache__" not in f.parts}
    for b in a.branches:
        for f in (ROOT / "results" / b).glob("*.jsonl"):
            if f.stem.startswith(("dev_", "test_")) and not f.stem.endswith("_judge"):
                files.add(f)
    files |= set(CACHE_DIR.glob("cache_*.jsonl"))
    for name in ("kappa_sample.json", "kappa_decisions.json"):
        if (KAPPA_DIR / name).exists():
            files.add(KAPPA_DIR / name)
    out = ROOT / "data" / "processed" / "kaggle" / "da2-judge.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            z.write(f, f.relative_to(ROOT).as_posix())
    print(f"{out.relative_to(ROOT)}: {len(files)} file, {out.stat().st_size / 1e6:.1f} MB")
    for f in sorted(files):
        if "src" not in f.relative_to(ROOT).parts[:1]:
            print("  ", f.relative_to(ROOT).as_posix())


GUIDE_VI = """<p><b>Đúng</b> khi câu trả lời nêu được thông tin của đáp án chuẩn (một trong các đáp án chấp nhận là đủ; đáp án nhiều ý thì phải đủ mọi ý), và không đưa thêm một đáp án mâu thuẫn. Được phép dài hơn: nhắc lại câu hỏi, chép cả câu/điều khoản quanh đáp án, giải thích ngắn, miễn là thông tin chuẩn rõ ràng là câu trả lời và không có gì mâu thuẫn.</p>
<p><b>Coi như nhau:</b> hoa/thường, dấu câu, khoảng trắng, thứ tự từ không đổi nghĩa, cách viết số (05 = 5, 3,75 = 3.75), có/không có đơn vị khi không mơ hồ, viết tắt thông dụng.</p>
<p><b>Dấu tiếng Việt là một phần của từ:</b> sai dấu thanh hoặc dấu mũ/móc ("Nam" và "Năm") là từ khác. Trừ khi chính đáp án chuẩn viết không dấu.</p>
<p><b>Sai:</b> giá trị/tên/số/ngày khác; liệt kê nhiều ứng viên mà chỉ một cái đúng; thiếu ý; từ chối hoặc "không tìm thấy"; trả lời chuyện khác.</p>
<p>Chấm như judge: chỉ dựa vào câu hỏi, đáp án chuẩn và câu trả lời, không cần xem ảnh.</p>"""

KAPPA_PAGE = r"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chấm tay đo kappa</title>
<style>
:root{--bg:#f6f6f4;--card:#fff;--fg:#1d1d1b;--mut:#6b6b66;--line:#ddd;--acc:#1f5fbf;--ok:#1d7a3a;--bad:#b3261e}
@media (prefers-color-scheme:dark){:root{--bg:#18181a;--card:#222225;--fg:#ececea;--mut:#9a9a95;--line:#3a3a3e;--acc:#7aa8ff;--ok:#5fc27e;--bad:#ff8a80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,Segoe UI,sans-serif}
header{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--line);padding:8px 16px;display:flex;gap:12px;flex-wrap:wrap;align-items:center}
main{max-width:900px;margin:0 auto;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px;margin-bottom:14px}
.lab{color:var(--mut);font-size:13px;margin-top:10px}.val{white-space:pre-wrap;word-break:break-word}
button{font:inherit;padding:8px 16px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg);cursor:pointer}
button.ok.sel{outline:3px solid var(--ok)}button.bad.sel{outline:3px solid var(--bad)}
input[type=text]{width:100%;font:inherit;padding:6px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)}
.mut{color:var(--mut);font-size:13px}
</style></head><body>
<header><strong>Chấm tay đo κ</strong><span id="pos" class="mut"></span><span id="prog" class="mut"></span>
<button id="prev">←</button><button id="next">→</button><button id="exp">Xuất kappa_decisions.json</button></header>
<main>
<div class="card">
 <div class="mut" id="meta"></div>
 <div class="lab">Câu hỏi</div><div class="val" id="q"></div>
 <div class="lab">Đáp án chuẩn (chấp nhận)</div><div class="val" id="g"></div>
 <div class="lab">Câu trả lời của model</div><div class="val" id="p"></div>
 <p style="margin-top:16px"><button class="ok" id="bOk">1 · Đúng</button> <button class="bad" id="bBad">2 · Sai</button></p>
 <div class="lab">Ghi chú (tùy chọn, nhất là khi phân vân)</div><input type="text" id="note">
 <div id="cx" class="card" style="margin:14px 0 0" hidden><div class="lab" style="margin-top:0">Đề xuất của Codex (chỉ dựa trên chữ, không thấy judge)</div><div id="cxv"></div><div class="mut" id="cxr"></div><p class="mut">Enter: đồng ý với Codex</p></div>
</div>
<div class="card mut">__GUIDE__<p>Phím: 1 đúng · 2 sai · ←/→ chuyển câu. Không có phán quyết của judge trong trang này.</p></div>
</main>
<script>
const DATA=__DATA__, META=__META__, KEY="judge_kappa_"+META.judge_version+"_"+META.seed;
let dec={};try{dec=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(dec))}catch(e){}};
const $=id=>document.getElementById(id);let i=0;
function render(){i=Math.max(0,Math.min(i,DATA.length-1));const d=DATA[i],c=dec[d.qid];
 $("pos").textContent=`Câu ${i+1}/${DATA.length}`;$("prog").textContent=`· đã chấm ${Object.keys(dec).length}/${DATA.length}`;
 $("meta").textContent=`${d.dataset||""} · ${d.qid}`;$("q").textContent=d.question;$("g").textContent=d.answers.join("  |  ");$("p").textContent=d.pred||"(trống)";
 if(d.codex){$("cx").hidden=false;$("cxv").innerHTML=`<b>${d.codex.verdict==="correct"?"Đúng":"Sai"}</b> · ${d.codex.confidence||""}`;$("cxr").textContent=d.codex.reason||""}else{$("cx").hidden=true}
 $("bOk").classList.toggle("sel",c&&c.verdict==="correct");$("bBad").classList.toggle("sel",c&&c.verdict==="incorrect");$("note").value=c?c.note||"":""}
function decide(v){const d=DATA[i];dec[d.qid]={qid:d.qid,verdict:v,note:$("note").value,at:new Date().toISOString()};
 if(d.codex){dec[d.qid].codex_verdict=d.codex.verdict;dec[d.qid].agrees_codex=v===d.codex.verdict}save();i++;render()}
$("bOk").onclick=()=>decide("correct");$("bBad").onclick=()=>decide("incorrect");
$("note").oninput=()=>{const d=DATA[i];if(dec[d.qid]){dec[d.qid].note=$("note").value;save()}};
$("prev").onclick=()=>{i--;render()};$("next").onclick=()=>{i++;render()};
$("exp").onclick=()=>{const miss=DATA.length-Object.keys(dec).length;if(miss&&!confirm(`Còn ${miss} câu chưa chấm. Vẫn xuất?`))return;
 const b=new Blob([JSON.stringify({meta:META,exported_at:new Date().toISOString(),decisions:Object.values(dec)},null,1)],{type:"application/json"});
 const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="kappa_decisions.json";a.click()};
document.addEventListener("keydown",e=>{if(e.target.id==="note")return;if(e.key==="1")decide("correct");else if(e.key==="2")decide("incorrect");
 else if(e.key==="Enter"&&DATA[i].codex){e.preventDefault();decide(DATA[i].codex.verdict)}
 else if(e.key==="ArrowLeft"){i--;render()}else if(e.key==="ArrowRight"){i++;render()}});
render();
</script></body></html>
"""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--pred", nargs="+", required=True)
    r.add_argument("--col", default="pred")
    r.add_argument("--backend", choices=["llamacpp", "gemini", "openrouter"], default="llamacpp")
    r.add_argument("--model", default=None)
    r.add_argument("--workers", type=int, default=4)
    m = sub.add_parser("models")
    m.add_argument("--backend", choices=["gemini", "openrouter"], default="gemini")
    s = sub.add_parser("sample")
    s.add_argument("--pred", required=True)
    s.add_argument("--n", type=int, default=50)
    s.add_argument("--seed", type=int, default=20261118)
    s.add_argument("--form", default="short", help="dạng đáp án lấy mẫu; để trống = mọi dạng")
    pk = sub.add_parser("pack")
    pk.add_argument("--branches", nargs="+", default=["B0_base"], help="thư mục nhánh trong results/ cần chấm")
    pg = sub.add_parser("page")
    pg.add_argument("--codex", default=None, help="jsonl đề xuất của Codex (qid, verdict, confidence, reason)")
    k = sub.add_parser("kappa")
    k.add_argument("--human", required=True)
    k.add_argument("--judged", required=True)
    a = ap.parse_args()
    {"run": cmd_run, "models": cmd_models, "sample": cmd_sample, "page": cmd_page, "kappa": cmd_kappa,
     "pack": cmd_pack}[a.cmd](a)


if __name__ == "__main__":
    main()
