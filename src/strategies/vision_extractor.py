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
    - Groups OCR words into lines and paragraphs for coherent LDUs.
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

            # --- Group words into lines ---
            lines = {}
            for i, word in enumerate(data["text"]):
                if word.strip():
                    top = data["top"][i]
                    line_key = round(top / 10)  # bucket by vertical position
                    if line_key not in lines:
                        lines[line_key] = []
                    lines[line_key].append(word)

            # --- Merge lines into paragraphs ---
            paragraphs = []
            current_para = []
            for _, words in sorted(lines.items(), key=lambda kv: kv[0]):
                line_text = " ".join(words).strip()
                if not line_text:
                    if current_para:
                        paragraphs.append(" ".join(current_para))
                        current_para = []
                else:
                    current_para.append(line_text)
            if current_para:
                paragraphs.append(" ".join(current_para))

            page_ldus: List[LDU] = []

            # --- Emit LDUs at paragraph level ---
            for j, para in enumerate(paragraphs):
                ldu_id = f"ldu_{page_num}_{j}"
                content_hash = hashlib.sha256(para.encode("utf-8")).hexdigest()
                bbox = (0, 0, 1000, 1000)  # simplified bbox for paragraph

                ldu = LDU(
                    ldu_id=ldu_id,
                    type=LDUType.paragraph,
                    text=para,
                    table_data=None,
                    figure_ref=None,
                    bbox=bbox,
                    page_number=page_num,
                )
                ldus.append(ldu)
                page_ldus.append(ldu)

                provenance.append(
                    ProvenanceChain(
                        ldu_id=ldu_id,
                        strategy_used=StrategyType.vision_augmented,
                        source_bbox=bbox,
                        source_page=page_num,
                        transformations=["pytesseract_ocr_grouped"],
                        confidence_score=profile.triage_confidence,
                        content_hash=content_hash,
                    )
                )

            # --- Build PageIndex ---
            x_max_values = [ldu.bbox[2] for ldu in page_ldus]
            bbox_distribution = {"x_max": x_max_values}
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
            page_indexes=page_indexes,
        )
