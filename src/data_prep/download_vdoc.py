"""Tải dataset V-Doc (trannhiem/TranNhiem-Vietnamese-DocumentImage-Reasoning) từ Hugging Face.

Kích thước trên HF (revision 4394a35, 2026-07-23):
    preview/preview-000.parquet   ~100 MB   300 dòng mẫu, có ảnh thumbnail
    full/full-00000.parquet       ~480 MB   64.516 bản ghi, chỉ có đường dẫn ảnh + conversations
    data/vdoc.jsonl               ~1,1 GB   cùng nội dung với full parquet (dạng jsonl)
    shards/images-0000{0..5}.tar  ~9,5 GB   ảnh trang tài liệu, giải nén ra images/...

Mặc định chỉ tải preview + metadata (full parquet). Ảnh trang rất nặng nên phải chọn rõ
`--parts images`, có thể giới hạn số shard bằng `--shards`.

Ví dụ (chạy từ gốc repo):
    python -m src.data_prep.download_vdoc
    python -m src.data_prep.download_vdoc --parts images --shards 0
    python -m src.data_prep.download_vdoc --parts preview meta images
"""

import argparse
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "trannhiem/TranNhiem-Vietnamese-DocumentImage-Reasoning"
NUM_SHARDS = 6
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "vdoc"

# Mỗi phần ứng với một tập allow_patterns của snapshot_download
PART_PATTERNS = {
    "preview": ["preview/*.parquet"],
    "meta": ["full/*.parquet"],
    "jsonl": ["data/vdoc.jsonl"],
}


def shard_name(idx: int) -> str:
    return f"shards/images-{idx:05d}.tar"


def done_marker(out_dir: Path, idx: int) -> Path:
    return out_dir / f"{shard_name(idx)}.done"


def extract_shard(tar_path: Path, out_dir: Path, idx: int, keep_tar: bool) -> int:
    """Giải nén một shard vào out_dir (thành images/...). Trả về số file đã giải nén."""
    with tarfile.open(tar_path) as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        # filter="data" chặn đường dẫn tuyệt đối, "../", symlink ra ngoài out_dir
        tar.extractall(out_dir, members=members, filter="data")
    # Đánh dấu đã giải nén để lần chạy sau không tải lại shard đã xóa .tar
    done_marker(out_dir, idx).write_text(str(len(members)), encoding="utf-8")
    if not keep_tar:
        tar_path.unlink()
    return len(members)


def main() -> None:
    # Console Windows mặc định cp1252, không in được tiếng Việt
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Tải dataset V-Doc từ Hugging Face.")
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=[*PART_PATTERNS, "images"],
        default=["preview", "meta"],
        help="Các phần cần tải (mặc định: preview meta).",
    )
    parser.add_argument(
        "--shards",
        nargs="+",
        type=int,
        default=list(range(NUM_SHARDS)),
        help=f"Chỉ số shard ảnh cần tải, 0..{NUM_SHARDS - 1} (mặc định: tất cả).",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--revision",
        default="main",
        help="Branch/commit trên HF. Nên ghim commit sha để tái lập được.",
    )
    parser.add_argument("--keep-tar", action="store_true", help="Giữ file .tar sau khi giải nén.")
    parser.add_argument("--no-extract", action="store_true", help="Chỉ tải shard, không giải nén.")
    args = parser.parse_args()

    bad = [s for s in args.shards if not 0 <= s < NUM_SHARDS]
    if bad:
        parser.error(f"Shard không hợp lệ: {bad}")

    # Lấy commit sha cụ thể để mọi phần tải về cùng một phiên bản dataset
    sha = HfApi().dataset_info(REPO_ID, revision=args.revision).sha
    print(f"Dataset {REPO_ID} @ {sha}")

    shards = []
    if "images" in args.parts:
        for i in sorted(set(args.shards)):
            if done_marker(args.out_dir, i).exists():
                print(f"Bỏ qua {shard_name(i)}: đã giải nén ở lần chạy trước")
            else:
                shards.append(i)

    patterns = ["README.md"]
    for part in args.parts:
        patterns += PART_PATTERNS.get(part, [])
    patterns += [shard_name(i) for i in shards]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=sha,
        local_dir=args.out_dir,
        allow_patterns=patterns,
    )

    if not args.no_extract:
        for i in shards:
            tar_path = args.out_dir / shard_name(i)
            n = extract_shard(tar_path, args.out_dir, i, args.keep_tar)
            print(f"Đã giải nén {tar_path.name}: {n} file")

    # Lưu thông tin phiên bản cạnh dữ liệu để tái lập
    info = {
        "repo_id": REPO_ID,
        "revision": sha,
        "parts": args.parts,
        "shards": sorted(set(args.shards)) if "images" in args.parts else [],
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (args.out_dir / "DOWNLOAD_INFO.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Xong. Dữ liệu ở {args.out_dir}")


if __name__ == "__main__":
    main()
