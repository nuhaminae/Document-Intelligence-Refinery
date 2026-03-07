# src/strategies/vision_extractor.py
# script to define extraction strategies with OCR + LayoutLMv3

import hashlib
from typing import List

import pytesseract
from pdf2image import convert_from_path
from PIL import ImageOps
from transformers import LayoutLMv3Tokenizer

from src.models.models import (
    LDU,
    DocumentProfile,
    ExtractedDocument,
    LDUType,
    PageIndex,
    ProvenanceChain,
    StrategyType,
)


class VisionExtractor:
    """
    Strategy C: Vision-Augmented (OCR + LayoutLMv3)
    - Best for scanned PDFs, low-density text, or complex layouts.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.tokenizer = LayoutLMv3Tokenizer.from_pretrained(
            "microsoft/layoutlmv3-base"
        )

    def extract(self, profile: DocumentProfile) -> ExtractedDocument:
        ldus: List[LDU] = []
        provenance: List[ProvenanceChain] = []
        page_indexes: List[PageIndex] = []

        # Convert PDF pages to images
        pages = convert_from_path(
            self.pdf_path, dpi=300, poppler_path=r"C:/poppler-25.12.0/Library/bin"
        )

        for page_num, page in enumerate(pages, start=1):
            # Preprocess image
            img = ImageOps.grayscale(page)
            img = img.point(lambda x: 0 if x < 140 else 255, "1")
            img = img.resize((img.width * 2, img.height * 2))

            # OCR with bounding boxes
            # Language: Amharic + English
            data = pytesseract.image_to_data(
                img, lang="amh+eng", output_type=pytesseract.Output.DICT
            )

            page_ldus: List[LDU] = []

            for i, word in enumerate(data["text"]):
                if word.strip():
                    left, top, width, height = (
                        data["left"][i],
                        data["top"][i],
                        data["width"][i],
                        data["height"][i],
                    )
                    bbox = (
                        int(left * 1000 / img.width),
                        int(top * 1000 / img.height),
                        int((left + width) * 1000 / img.width),
                        int((top + height) * 1000 / img.height),
                    )

                    # Build LDU
                    ldu_id = f"ldu_{page_num}_{i}"
                    content_hash = hashlib.sha256(word.encode("utf-8")).hexdigest()

                    ldu = LDU(
                        ldu_id=ldu_id,
                        type=LDUType.paragraph,
                        text=word,
                        table_data=None,
                        figure_ref=None,
                        bbox=bbox,
                        page_number=page_num,
                    )
                    ldus.append(ldu)
                    page_ldus.append(ldu)

                    # --- Provenance ---
                    provenance.append(
                        ProvenanceChain(
                            ldu_id=ldu_id,
                            strategy_used=StrategyType.vision_augmented,
                            source_bbox=bbox,
                            source_page=page_num,
                            transformations=["pytesseract_ocr"],
                            confidence_score=profile.triage_confidence,
                            content_hash=content_hash,
                        )
                    )

            # Collect x_max values from OCR bounding boxes
            x_max_values = [ldu.bbox[2] for ldu in page_ldus]
            bbox_distribution = {"x_max": x_max_values}

            # --- Build PageIndex ---
            page_indexes.append(
                PageIndex(
                    page_number=page_num,
                    ldus=page_ldus,
                    char_density=profile.char_density,
                    whitespace_ratio=profile.whitespace_ratio,
                    layout_signature=bbox_distribution,
                )
            )

        # Return ExtractedDocument
        return ExtractedDocument(
            document_id=profile.document_id,
            strategy_used=StrategyType.vision_augmented,
            content_blocks=ldus,
            provenance_chain=provenance,
            extraction_confidence=profile.triage_confidence,
        )
