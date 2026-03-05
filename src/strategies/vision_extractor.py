# src/strategies/vision_extractor.py

from typing import List

import requests

from src.models.models import (
    LDU,
    DocumentProfile,
    ExtractedDocument,
    LDUType,
    ProvenanceChain,
    StrategyType,
)


class VisionExtractor:
    """
    Strategy C: Vision-Augmented (Chunkr API)
    - Best for scanned PDFs, image-heavy documents, charts/figures.
    """

    def __init__(self, pdf_path: str, api_url: str = "http://localhost:8000"):
        self.pdf_path = pdf_path
        self.api_url = api_url

    def extract(self, profile: DocumentProfile) -> ExtractedDocument:
        ldus: List[LDU] = []
        provenance: List[ProvenanceChain] = []

        # Send PDF to Chunkr API
        with open(self.pdf_path, "rb") as f:
            response = requests.post(
                f"{self.api_url}/process",
                files={"file": f},
            )
        response.raise_for_status()
        result = response.json()

        # Traverse Chunkr JSON output
        for page in result.get("pages", []):
            page_num = page.get("page_number", 0)
            for block in page.get("blocks", []):
                ldu_id = f"ldu_{page_num}_{block.get('id')}"
                bbox = tuple(block.get("bbox", [0, 0, 0, 0]))
                block_type = block.get("type")

                if block_type == "text":
                    ldus.append(
                        LDU(
                            ldu_id=ldu_id,
                            type=LDUType.paragraph,
                            text=block.get("text"),
                            table_data=None,
                            figure_ref=None,
                            bbox=bbox,
                            page_number=page_num,
                        )
                    )
                elif block_type == "table":
                    ldus.append(
                        LDU(
                            ldu_id=ldu_id,
                            type=LDUType.table,
                            text=None,
                            table_data=block.get("cells"),
                            figure_ref=None,
                            bbox=bbox,
                            page_number=page_num,
                        )
                    )
                elif block_type == "figure":
                    ldus.append(
                        LDU(
                            ldu_id=ldu_id,
                            type=LDUType.figure,
                            text=None,
                            table_data=None,
                            figure_ref=block.get("image_ref"),
                            bbox=bbox,
                            page_number=page_num,
                        )
                    )

                provenance.append(
                    ProvenanceChain(
                        ldu_id=ldu_id,
                        strategy_used=StrategyType.vision_augmented,
                        source_bbox=bbox,
                        source_page=page_num,
                        transformations=["chunkr_parse"],
                        confidence_score=profile.triage_confidence,
                    )
                )

        return ExtractedDocument(
            document_id=profile.document_id,
            strategy_used=StrategyType.vision_augmented,
            content_blocks=ldus,
            provenance_chain=provenance,
            extraction_confidence=profile.triage_confidence,
        )
