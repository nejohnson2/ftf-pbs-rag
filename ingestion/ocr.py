"""OCR and text extraction using Docling.

Converts PDFs to structured markdown. Docling handles:
- Scanned PDFs (OCR via tesseract)
- Born-digital PDFs (direct text extraction)
- Tables, figures, and section headers

Run as part of scripts/preprocess.py — NOT imported at runtime on Heroku.
"""

from pathlib import Path
from loguru import logger


def extract_to_markdown(pdf_path: Path) -> str:
    """Extract text from a PDF and return as markdown string.

    Uses Docling for best-in-class extraction of structured documents.
    Falls back to a basic text extraction if Docling fails.
    """
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import PdfFormatOption

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        result = converter.convert(str(pdf_path))
        markdown = result.document.export_to_markdown()
        logger.info(f"Docling extracted {len(markdown):,} chars from {pdf_path.name}")
        return markdown

    except ImportError:
        logger.error(
            "docling is not installed. Run: pip install -r requirements-ingest.txt"
        )
        raise
    except Exception as e:
        logger.warning(f"Docling failed on {pdf_path.name}: {e}. Falling back to pdfplumber.")
        return _fallback_extract(pdf_path)


def _fallback_extract(pdf_path: Path) -> str:
    """Minimal fallback using pdfplumber if Docling fails."""
    try:
        import pdfplumber

        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(f"## Page {i}\n\n{text}")
        return "\n\n".join(pages)
    except ImportError:
        logger.error("pdfplumber not installed either. Cannot extract text.")
        return ""
    except Exception as e:
        logger.error(f"Fallback extraction failed on {pdf_path.name}: {e}")
        return ""
