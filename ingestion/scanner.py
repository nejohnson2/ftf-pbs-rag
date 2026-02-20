"""Scans the document archive and extracts metadata from the folder structure.

The archive encodes three metadata layers in its path:
  Country / Phase (I, II, 1, 2) / SurveyType (Baseline, Midline, Endline, Interim)

Handles all documented structural inconsistencies:
- Roman vs. Arabic phase numerals
- Countries with no Phase subfolder (Liberia, Malawi, Mozambique)
- Flat Phase-level files (Ethiopia Phase II, Kenya Phase II)
- Combined folders (Ghana "Baseline and Midline")
- DOCX files that have a PDF equivalent (DOCX is skipped)
- Non-English documents (flagged include=False)
"""

import re
from pathlib import Path
from typing import Optional

COUNTRIES = {
    "Bangladesh", "Cambodia", "Ethiopia", "Ghana", "Guatemala", "Haiti",
    "Honduras", "Kenya", "Liberia", "Malawi", "Mali", "Mozambique",
    "Nepal", "Nigeria", "Rwanda", "Senegal", "Tajikistan", "Tanzania",
    "Uganda", "Zambia",
}

PHASE_MAP: dict[str, int] = {
    "Phase I": 1,
    "Phase II": 2,
    "Phase 1": 1,
    "Phase 2": 2,
}

SURVEY_TYPE_MAP: dict[str, str] = {
    "Baseline": "baseline",
    "Interim": "interim",
    "Midline": "midline",
    "Endline": "endline",
    "Baseline and Midline": "baseline_midline",
}

# Files known to be non-English — excluded from ingestion
NON_ENGLISH_FILENAMES: set[str] = {
    "Mozambique DHS 2011 (Portuguese).pdf",
}


def _infer_doc_type(filename: str) -> str:
    lower = filename.lower()
    if any(kw in lower for kw in ["key_findings", "key findings", "kfr", "fact_sheet", "fact sheet"]):
        return "key_findings"
    if any(kw in lower for kw in ["sow", "study protocol", "sample design", "survey methods"]):
        return "planning"
    if any(kw in lower for kw in ["dhs", "cfsns", "readme"]):
        return "reference"
    return "full_report"


def _infer_year(filename: str) -> Optional[int]:
    match = re.search(r"(20\d{2}|19\d{2})", filename)
    return int(match.group(0)) if match else None


def _parse_path(file_path: Path, archive_root: Path) -> Optional[dict]:
    """Extract metadata from a file's position in the archive tree."""
    try:
        rel = file_path.relative_to(archive_root.parent)
    except ValueError:
        return None

    parts = list(rel.parts)

    # Find country in path parts
    country = next((p for p in parts if p in COUNTRIES), None)
    if country is None:
        return None

    country_idx = parts.index(country)
    # subfolders between country and filename
    subfolders = parts[country_idx + 1 : -1]

    phase: Optional[int] = None
    survey_type: Optional[str] = None

    for folder in subfolders:
        if folder in PHASE_MAP:
            phase = PHASE_MAP[folder]
        elif folder in SURVEY_TYPE_MAP:
            survey_type = SURVEY_TYPE_MAP[folder]

    filename = file_path.name
    return {
        "country": country,
        "phase": phase,
        "survey_type": survey_type,
        "doc_type": _infer_doc_type(filename),
        "year": _infer_year(filename),
        "is_english": filename not in NON_ENGLISH_FILENAMES,
        "filename": filename,
        "file_path": str(rel),
    }


def _find_docx_duplicates(all_files: list[Path]) -> set[Path]:
    """Return DOCX paths that have a sibling PDF with the same stem."""
    pdf_keys = {(f.parent, f.stem) for f in all_files if f.suffix.lower() == ".pdf"}
    return {
        f for f in all_files
        if f.suffix.lower() == ".docx" and (f.parent, f.stem) in pdf_keys
    }


def _make_doc_id(meta: dict, existing_ids: set[str]) -> str:
    country = (meta["country"] or "unknown").lower().replace(" ", "_")
    phase = f"p{meta['phase']}" if meta["phase"] else "nophase"
    survey = meta["survey_type"] or "unknown"
    doc_type = meta["doc_type"]
    base = f"{country}_{phase}_{survey}_{doc_type}"
    doc_id, counter = base, 2
    while doc_id in existing_ids:
        doc_id = f"{base}_{counter}"
        counter += 1
    return doc_id


def scan_archive(archive_root: Path) -> list[dict]:
    """Walk the archive and return a list of document metadata dicts.

    Args:
        archive_root: Path to the top-level archive folder
                      ("Archived Population-Based Survey Reports")

    Returns:
        List of dicts, one per document, ready to be written to metadata.json
    """
    all_files = [
        f for f in archive_root.rglob("*")
        if f.is_file()
        and f.suffix.lower() in (".pdf", ".docx")
        and not f.name.startswith(".")
    ]

    skip_docx = _find_docx_duplicates(all_files)

    documents: list[dict] = []
    existing_ids: set[str] = set()

    for f in sorted(all_files):
        if f in skip_docx:
            continue

        meta = _parse_path(f, archive_root)
        if meta is None:
            continue

        is_english = meta.pop("is_english")
        doc_id = _make_doc_id(meta, existing_ids)
        existing_ids.add(doc_id)

        documents.append({
            "doc_id": doc_id,
            "file_path": meta["file_path"],
            "markdown_path": f"data/processed/markdown/{doc_id}.md",
            "country": meta["country"],
            "phase": meta["phase"],
            "survey_type": meta["survey_type"],
            "doc_type": meta["doc_type"],
            "year": meta["year"],
            "title": "",          # fill in manually
            "online_url": None,   # fill in manually
            "include": is_english,
            "notes": "" if is_english else "Non-English document — excluded from ingestion",
        })

    return documents
