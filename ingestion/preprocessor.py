"""Text preprocessing for extracted markdown.

Strips noisy boilerplate sections from survey reports before chunking:
  - Table of contents
  - References / bibliography sections
  - Appendices (especially survey instruments and indicator tables)
  - Short orphan paragraphs (page headers, footers, captions)

The cleaned output is written to data/processed/markdown/ where it can be
manually reviewed and edited before ingestion into the vector store.
"""

import re
from config import PreprocessingConfig


# ── Patterns that mark the START of sections to remove ───────────────────────

# Table of contents: typically a heading followed by dot-leader lines
_TOC_START = re.compile(
    r"(?:^|\n)(#+\s*)?(table\s+of\s+contents|contents|list\s+of\s+(tables|figures|boxes|acronyms|abbreviations))\s*\n",
    re.IGNORECASE,
)

# References / bibliography
_REFS_START = re.compile(
    r"(?:^|\n)(#+\s*)?(references?|bibliography|works\s+cited)\s*\n",
    re.IGNORECASE,
)

# Appendices
_APPENDIX_START = re.compile(
    r"(?:^|\n)(#+\s*)?appendix\b",
    re.IGNORECASE,
)

# Survey instruments
_INSTRUMENT_START = re.compile(
    r"(?:^|\n)(#+\s*)?(survey\s+instrument|questionnaire|data\s+collection\s+form)\b",
    re.IGNORECASE,
)


def _remove_section_from(text: str, pattern: re.Pattern, keep_before: bool = True) -> str:
    """Find the first match of pattern and truncate or remove from that point.

    If keep_before is True, everything before the match is kept.
    If keep_before is False, the section heading is kept but its body removed.
    """
    match = pattern.search(text)
    if match:
        return text[: match.start()].rstrip()
    return text


def _remove_toc(text: str) -> str:
    """Remove a table of contents block.

    Detects the TOC by the presence of dot-leaders (.....) or page-number
    lines following a contents heading, and removes until the next real section.
    """
    # Strategy: find TOC heading, then drop lines until we hit a blank line
    # followed by a non-dotty heading
    match = _TOC_START.search(text)
    if not match:
        return text

    before = text[: match.start()]
    after = text[match.end():]

    # Drop lines that look like TOC entries (contain two or more dots in a row,
    # or end with a page number pattern like "... 12")
    toc_line = re.compile(r"^.{0,120}\.{2,}.*\d+\s*$|^\s*\d+\s*$", re.MULTILINE)
    # Find where TOC ends: first line that doesn't match a TOC entry pattern
    # and is followed by paragraph text
    lines = after.split("\n")
    non_toc_start = 0
    for i, line in enumerate(lines):
        if line.strip() == "":
            continue
        if not toc_line.match(line):
            non_toc_start = i
            break

    remainder = "\n".join(lines[non_toc_start:])
    return (before + "\n" + remainder).strip()


def _clean_short_paragraphs(text: str, min_length: int) -> str:
    """Drop standalone paragraphs shorter than min_length characters.

    These are typically page headers, footers, figure captions, or
    isolated page numbers that Docling preserves.
    """
    paragraphs = re.split(r"\n{2,}", text)
    kept = []
    for para in paragraphs:
        stripped = para.strip()
        # Always keep headings (lines starting with #)
        if stripped.startswith("#"):
            kept.append(para)
        elif len(stripped) >= min_length:
            kept.append(para)
    return "\n\n".join(kept)


def _normalize_whitespace(text: str) -> str:
    """Collapse 3+ consecutive blank lines into 2."""
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_markdown(text: str, cfg: PreprocessingConfig) -> str:
    """Apply all enabled cleaning passes to extracted markdown.

    Args:
        text: Raw markdown from OCR extraction
        cfg:  Preprocessing config section

    Returns:
        Cleaned markdown string
    """
    if cfg.remove_toc:
        text = _remove_toc(text)

    if cfg.remove_references_section:
        text = _remove_section_from(text, _REFS_START)

    if cfg.remove_appendices:
        text = _remove_section_from(text, _APPENDIX_START)

    if cfg.remove_survey_instruments:
        text = _remove_section_from(text, _INSTRUMENT_START)

    text = _clean_short_paragraphs(text, cfg.min_paragraph_length)
    text = _normalize_whitespace(text)

    return text
