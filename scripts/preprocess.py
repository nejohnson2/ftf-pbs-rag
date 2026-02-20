"""Step 1: OCR all PDFs and generate editable markdown files.

Usage:
    python scripts/preprocess.py [--force]

What it does:
  1. Reads config.yaml for paths and preprocessing settings
  2. Scans the archive folder, extracts metadata from the path structure
  3. Writes/updates data/processed/metadata.json (stub entries only — does NOT
     overwrite existing entries that you have manually edited)
  4. For each document where include=True, runs Docling OCR and writes
     a cleaned markdown file to data/processed/markdown/{doc_id}.md

After running, you should:
  - Open data/processed/metadata.json and fill in:
      - "title": human-readable report title
      - "online_url": Dropbox or other link to the document
      - "include": set to false for any documents to exclude
  - Review the markdown files in data/processed/markdown/
  - Delete or comment out sections you want removed (references, annexes, etc.)
  - Run scripts/build_index.py to ingest into the vector database

Options:
  --force   Re-run OCR even for documents that already have a markdown file
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger

from config import get_config
from ingestion.ocr import extract_to_markdown
from ingestion.preprocessor import clean_markdown
from ingestion.scanner import scan_archive


def load_existing_metadata(path: Path) -> dict[str, dict]:
    """Load existing metadata.json as a dict keyed by doc_id."""
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return {d["doc_id"]: d for d in data.get("documents", [])}
    return {}


def save_metadata(path: Path, documents: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"documents": documents}, f, indent=2, ensure_ascii=False)
    logger.info(f"Metadata written to {path} ({len(documents)} documents)")


def main(force: bool = False) -> None:
    cfg = get_config()

    archive_root = ROOT / cfg.preprocessing.docs_root
    if not archive_root.exists():
        logger.error(f"Archive not found: {archive_root}")
        sys.exit(1)

    output_dir = ROOT / cfg.preprocessing.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = ROOT / cfg.preprocessing.metadata_output

    # ── Scan archive ──────────────────────────────────────────────────────────
    logger.info("Scanning archive...")
    scanned = scan_archive(archive_root)
    logger.info(f"Found {len(scanned)} documents ({sum(1 for d in scanned if d['include'])} included)")

    # ── Merge with existing metadata (preserve manual edits) ──────────────────
    existing = load_existing_metadata(metadata_path)
    merged: list[dict] = []

    for doc in scanned:
        doc_id = doc["doc_id"]
        if doc_id in existing:
            # Preserve manually edited fields
            saved = existing[doc_id]
            doc["title"] = saved.get("title", doc["title"])
            doc["online_url"] = saved.get("online_url", doc["online_url"])
            doc["include"] = saved.get("include", doc["include"])
            doc["notes"] = saved.get("notes", doc["notes"])
        merged.append(doc)

    save_metadata(metadata_path, merged)

    # ── OCR and preprocess ────────────────────────────────────────────────────
    included = [d for d in merged if d.get("include", True)]
    logger.info(f"Running OCR on {len(included)} documents...")

    success, skipped, failed = 0, 0, 0

    for doc in included:
        md_path = ROOT / doc["markdown_path"]

        if md_path.exists() and not force:
            logger.info(f"  SKIP {doc['doc_id']} (markdown exists; use --force to re-run)")
            skipped += 1
            continue

        pdf_path = ROOT / doc["file_path"]
        if not pdf_path.exists():
            logger.warning(f"  MISSING {doc['doc_id']}: {pdf_path}")
            failed += 1
            continue

        logger.info(f"  OCR  {doc['doc_id']} ({pdf_path.name})")
        try:
            raw_markdown = extract_to_markdown(pdf_path)
            cleaned = clean_markdown(raw_markdown, cfg.preprocessing)
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(cleaned, encoding="utf-8")
            success += 1
        except Exception as e:
            logger.error(f"  FAIL {doc['doc_id']}: {e}")
            failed += 1

    logger.info(
        f"\nPreprocessing complete: {success} processed, {skipped} skipped, {failed} failed"
    )
    logger.info(
        "\nNext steps:\n"
        "  1. Edit data/processed/metadata.json — fill in 'title' and 'online_url' for each document\n"
        "  2. Review/edit markdown files in data/processed/markdown/\n"
        "  3. Run: python -m ingestion.build_index"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PDFs to editable markdown")
    parser.add_argument("--force", action="store_true", help="Re-run OCR even if markdown exists")
    args = parser.parse_args()
    main(force=args.force)
