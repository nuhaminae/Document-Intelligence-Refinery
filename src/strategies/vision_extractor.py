# src/strategies/vision_extractor.py
# Modified after Peer's explainer

"""
Strategy C: Vision-Augmented extraction with OCR.

Best for:
- scanned PDFs
- image-heavy documents
- low character-density documents
- cases where FastText or LayoutAware fails Gate 2 quality checks

This replacement removes the copied triage-confidence behavior and computes
post-extraction quality from actual OCR output.
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import pytesseract
from pdf2image import convert_from_path
from PIL import ImageOps

from src.models.models import (
    LDU,
    DocumentProfile,
    ExtractedDocument,
    LDUType,
    PageIndex,
    ProvenanceChain,
    StrategyType,
)
from src.utils.extraction_quality import score_extracted_document

BBox = Tuple[float, float, float, float]


class VisionExtractor:
    """
    Strategy C: Vision-Augmented OCR extraction.

    This implementation uses pdf2image + pytesseract. It does not assume Vision
    is automatically successful; the output is still checked by Gate 2 quality
    scoring before the router accepts it.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def extract(self, profile: DocumentProfile) -> ExtractedDocument:
        ldus: List[LDU] = []
        provenance: List[ProvenanceChain] = []
        page_indexes: List[PageIndex] = []

        pages = self._convert_pdf_to_images()

        for page_num, page in enumerate(pages, start=1):
            processed_img = self._preprocess_image(page)
            ocr_data = pytesseract.image_to_data(
                processed_img,
                lang=os.getenv("TESSERACT_LANG", "amh+eng"),
                output_type=pytesseract.Output.DICT,
            )

            page_ldus = self._build_page_ldus_from_ocr(
                ocr_data=ocr_data,
                page_num=page_num,
            )

            ldus.extend(page_ldus)

            for ldu in page_ldus:
                content = ldu.text or ""
                content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

                provenance.append(
                    ProvenanceChain(
                        ldu_id=ldu.ldu_id,
                        strategy_used=StrategyType.vision_augmented,
                        source_bbox=ldu.bbox,
                        source_page=page_num,
                        transformations=["pdf2image", "pytesseract_ocr_grouped"],
                        confidence_score=0.0,
                        content_hash=content_hash,
                    )
                )

            page_indexes.append(
                PageIndex(
                    page_number=page_num,
                    ldus=page_ldus,
                    char_density=profile.char_density,
                    whitespace_ratio=profile.whitespace_ratio,
                    layout_signature={
                        "x0": [ldu.bbox[0] for ldu in page_ldus],
                        "x1": [ldu.bbox[2] for ldu in page_ldus],
                        "y0": [ldu.bbox[1] for ldu in page_ldus],
                        "y1": [ldu.bbox[3] for ldu in page_ldus],
                    },
                )
            )

        extracted_doc = ExtractedDocument(
            document_id=profile.document_id,
            strategy_used=StrategyType.vision_augmented,
            content_blocks=ldus,
            provenance_chain=provenance,
            extraction_confidence=0.0,
            page_indexes=page_indexes,
        )

        quality = score_extracted_document(extracted_doc)

        extracted_doc.extraction_confidence = quality.extraction_confidence_remeasured
        extracted_doc.output_quality = quality.model_dump(mode="json")

        for prov in extracted_doc.provenance_chain:
            prov.confidence_score = quality.extraction_confidence_remeasured

        return extracted_doc

    def _convert_pdf_to_images(self):
        """
        Convert PDF pages to images.

        On Windows, set POPPLER_PATH in .env if needed:
            POPPLER_PATH=C:/poppler-25.12.0/Library/bin

        On Linux/Colab/Mac, poppler is usually discoverable from PATH.
        """

        poppler_path = os.getenv("POPPLER_PATH")

        kwargs = {"dpi": int(os.getenv("PDF_RENDER_DPI", "300"))}
        if poppler_path:
            kwargs["poppler_path"] = poppler_path

        return convert_from_path(self.pdf_path, **kwargs)

    @staticmethod
    def _preprocess_image(page):
        """
        Convert image to a high-contrast grayscale/bitonal image for OCR.
        """

        img = ImageOps.grayscale(page)
        img = img.point(lambda x: 0 if x < 140 else 255, "1")
        img = img.resize((img.width * 2, img.height * 2))
        return img

    def _build_page_ldus_from_ocr(
        self,
        ocr_data: Dict,
        page_num: int,
    ) -> List[LDU]:
        """
        Group OCR words into paragraphs.

        Uses Tesseract block/par/line IDs when available. Falls back to vertical
        bucketing if those fields are missing.
        """

        grouped_lines = self._group_words_into_lines(ocr_data)
        paragraphs = self._merge_lines_into_paragraphs(grouped_lines)

        page_ldus: List[LDU] = []

        for j, paragraph in enumerate(paragraphs):
            text = paragraph["text"].strip()
            if not text:
                continue

            ldu_id = f"ldu_{page_num}_{j}"
            bbox = paragraph["bbox"]

            page_ldus.append(
                LDU(
                    ldu_id=ldu_id,
                    type=LDUType.paragraph,
                    text=text,
                    table_data=None,
                    figure_ref=None,
                    bbox=bbox,
                    page_number=page_num,
                )
            )

        return page_ldus

    def _group_words_into_lines(self, data: Dict) -> List[Dict]:
        """
        Return line objects:
            [{"text": "...", "bbox": (x0, y0, x1, y1), "line_key": ...}, ...]
        """

        line_groups: Dict[Tuple, List[Dict]] = defaultdict(list)

        words = data.get("text", [])

        for i, word in enumerate(words):
            word = str(word).strip()
            if not word:
                continue

            try:
                conf = float(data.get("conf", ["-1"])[i])
            except (TypeError, ValueError):
                conf = -1.0

            # Ignore very low-confidence OCR noise.
            if conf != -1.0 and conf < 20:
                continue

            left = int(data.get("left", [0])[i])
            top = int(data.get("top", [0])[i])
            width = int(data.get("width", [0])[i])
            height = int(data.get("height", [0])[i])

            word_record = {
                "text": word,
                "bbox": (
                    float(left),
                    float(top),
                    float(left + width),
                    float(top + height),
                ),
            }

            block_num = data.get("block_num", [None])[i]
            par_num = data.get("par_num", [None])[i]
            line_num = data.get("line_num", [None])[i]

            if block_num is not None and par_num is not None and line_num is not None:
                line_key = (block_num, par_num, line_num)
            else:
                line_key = ("y_bucket", round(top / 10))

            line_groups[line_key].append(word_record)

        lines: List[Dict] = []

        for line_key, line_words in line_groups.items():
            sorted_words = sorted(line_words, key=lambda w: w["bbox"][0])
            text = " ".join(w["text"] for w in sorted_words)
            bbox = self._merge_bboxes([w["bbox"] for w in sorted_words])

            lines.append(
                {
                    "line_key": line_key,
                    "text": " ".join(text.split()),
                    "bbox": bbox,
                }
            )

        lines.sort(key=lambda line: (line["bbox"][1], line["bbox"][0]))
        return lines

    def _merge_lines_into_paragraphs(self, lines: Sequence[Dict]) -> List[Dict]:
        """
        Merge adjacent OCR lines into loose paragraphs.

        A new paragraph starts when the vertical gap is large.
        """

        if not lines:
            return []

        paragraphs: List[Dict] = []

        current_texts: List[str] = []
        current_bboxes: List[BBox] = []
        previous_bottom = None

        for line in lines:
            text = line["text"]
            bbox = line["bbox"]
            top = bbox[1]
            bottom = bbox[3]

            starts_new_paragraph = False

            if previous_bottom is not None:
                vertical_gap = top - previous_bottom
                if vertical_gap > 35:
                    starts_new_paragraph = True

            if starts_new_paragraph and current_texts:
                paragraphs.append(
                    {
                        "text": " ".join(current_texts),
                        "bbox": self._merge_bboxes(current_bboxes),
                    }
                )
                current_texts = []
                current_bboxes = []

            current_texts.append(text)
            current_bboxes.append(bbox)
            previous_bottom = bottom

        if current_texts:
            paragraphs.append(
                {
                    "text": " ".join(current_texts),
                    "bbox": self._merge_bboxes(current_bboxes),
                }
            )

        return paragraphs

    @staticmethod
    def _merge_bboxes(bboxes: Sequence[BBox]) -> BBox:
        x0 = min(b[0] for b in bboxes)
        y0 = min(b[1] for b in bboxes)
        x1 = max(b[2] for b in bboxes)
        y1 = max(b[3] for b in bboxes)
        return (x0, y0, x1, y1)
