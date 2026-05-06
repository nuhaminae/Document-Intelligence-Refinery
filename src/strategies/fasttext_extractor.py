# src/strategies/fasttext_extractor.py
# Modified after Peer's explainer

"""
Strategy A: FastText extraction with pdfplumber.

Best for:
- native digital PDFs
- simple or mostly single-column layouts
- documents with a usable character stream

This file now implements Gate 2 post-extraction scoring:
- it runs pdfplumber,
- builds LDUs and provenance,
- measures actual output quality,
- writes the remeasured confidence back to the ExtractedDocument.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import pdfplumber

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


class FastTextExtractor:
    """
    Strategy A: FastText using pdfplumber.

    This extractor intentionally stays cheap and local. It should be the first
    attempt for native digital documents with a usable character stream.

    Gate 2 note:
    The extraction confidence returned by this class is no longer copied from
    profile.triage_confidence. It is computed from the actual extracted output.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def extract(self, profile: DocumentProfile) -> ExtractedDocument:
        ldus: List[LDU] = []
        provenance: List[ProvenanceChain] = []
        page_indexes: List[PageIndex] = []

        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_ldus = self._extract_page_ldus(page, page_num)

                ldus.extend(page_ldus)

                for ldu in page_ldus:
                    content = self._ldu_content_for_hash(ldu)
                    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

                    provenance.append(
                        ProvenanceChain(
                            ldu_id=ldu.ldu_id,
                            strategy_used=StrategyType.fasttext,
                            source_bbox=ldu.bbox,
                            source_page=page_num,
                            transformations=["pdfplumber_fast_text_extraction"],
                            # Temporary value. Updated after Gate 2 scoring.
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
            strategy_used=StrategyType.fasttext,
            content_blocks=ldus,
            provenance_chain=provenance,
            # Gate 2 will set this below.
            extraction_confidence=0.0,
            page_indexes=page_indexes,
        )

        quality = score_extracted_document(extracted_doc)

        extracted_doc.extraction_confidence = quality.extraction_confidence_remeasured
        extracted_doc.output_quality = quality.model_dump(mode="json")

        # Update per-LDU provenance confidence to the post-extraction score.
        for prov in extracted_doc.provenance_chain:
            prov.confidence_score = quality.extraction_confidence_remeasured

        return extracted_doc

    def _extract_page_ldus(self, page, page_num: int) -> List[LDU]:
        """
        Extract LDUs from a page.

        Preferred path:
            Use pdfplumber words and approximate line bboxes.

        Fallback path:
            Use page.extract_text() lines with coarse page-level bbox.
        """

        words = page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=True,
        )

        if words:
            return self._ldus_from_words(words, page_num)

        return self._ldus_from_text_fallback(page, page_num)

    def _ldus_from_words(self, words: Sequence[Dict], page_num: int) -> List[LDU]:
        """
        Group pdfplumber words into approximate lines using their vertical position.

        This gives better provenance than assigning every line the full-page bbox.
        """

        line_buckets: Dict[int, List[Dict]] = defaultdict(list)

        for word in words:
            top = float(word.get("top", 0.0))
            line_key = round(top / 4.0)
            line_buckets[line_key].append(word)

        raw_lines = []

        for _, line_words in sorted(line_buckets.items(), key=lambda kv: kv[0]):
            sorted_words = sorted(line_words, key=lambda w: float(w.get("x0", 0.0)))
            text = " ".join(str(w.get("text", "")).strip() for w in sorted_words)
            text = " ".join(text.split())

            if not text:
                continue

            bbox = self._merge_word_bboxes(sorted_words)
            raw_lines.append((text, bbox))

        return self._group_lines_into_ldus(raw_lines, page_num)

    def _ldus_from_text_fallback(self, page, page_num: int) -> List[LDU]:
        """
        Fallback when word-level extraction is unavailable.

        This uses a coarse page-level bbox. Gate 2 will warn about coarse/duplicated
        bbox quality if too many blocks share the same bbox.
        """

        text = page.extract_text() or ""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        page_bbox = (0.0, 0.0, float(page.width), float(page.height))

        raw_lines = [(line, page_bbox) for line in lines]
        return self._group_lines_into_ldus(raw_lines, page_num)

    def _group_lines_into_ldus(
        self,
        raw_lines: Sequence[Tuple[str, BBox]],
        page_num: int,
    ) -> List[LDU]:
        """
        Convert text lines into LDUs.

        Consecutive bullet/numbered lines are grouped as a single paragraph.
        Captions are tagged as caption LDUs.
        """

        ldus: List[LDU] = []
        i = 0

        while i < len(raw_lines):
            line, bbox = raw_lines[i]
            ldu_id = f"ldu_{page_num}_{len(ldus)}"

            lower = line.lower()

            if lower.startswith(("figure", "fig.", "fig ", "table")):
                ldu = LDU(
                    ldu_id=ldu_id,
                    type=LDUType.caption,
                    text=line,
                    table_data=None,
                    figure_ref=None,
                    bbox=bbox,
                    page_number=page_num,
                )
                ldus.append(ldu)
                i += 1
                continue

            if self._is_list_item(line):
                list_items = [line]
                list_bboxes = [bbox]

                j = i + 1
                while j < len(raw_lines):
                    next_line, next_bbox = raw_lines[j]
                    if not self._is_list_item(next_line):
                        break

                    list_items.append(next_line)
                    list_bboxes.append(next_bbox)
                    j += 1

                merged_text = "\n".join(list_items)
                merged_bbox = self._merge_bboxes(list_bboxes)

                ldu = LDU(
                    ldu_id=ldu_id,
                    type=LDUType.paragraph,
                    text=merged_text,
                    table_data=None,
                    figure_ref=None,
                    bbox=merged_bbox,
                    page_number=page_num,
                )
                ldus.append(ldu)
                i = j
                continue

            ldu = LDU(
                ldu_id=ldu_id,
                type=LDUType.paragraph,
                text=line,
                table_data=None,
                figure_ref=None,
                bbox=bbox,
                page_number=page_num,
            )
            ldus.append(ldu)
            i += 1

        return ldus

    @staticmethod
    def _is_list_item(line: str) -> bool:
        stripped = line.strip()

        if not stripped:
            return False

        if stripped.startswith(("-", "•", "*")):
            return True

        first = stripped[0]
        return first.isdigit()

    @staticmethod
    def _merge_word_bboxes(words: Sequence[Dict]) -> BBox:
        x0 = min(float(w.get("x0", 0.0)) for w in words)
        y0 = min(float(w.get("top", 0.0)) for w in words)
        x1 = max(float(w.get("x1", 0.0)) for w in words)
        y1 = max(float(w.get("bottom", 0.0)) for w in words)
        return (x0, y0, x1, y1)

    @staticmethod
    def _merge_bboxes(bboxes: Sequence[BBox]) -> BBox:
        x0 = min(b[0] for b in bboxes)
        y0 = min(b[1] for b in bboxes)
        x1 = max(b[2] for b in bboxes)
        y1 = max(b[3] for b in bboxes)
        return (x0, y0, x1, y1)

    @staticmethod
    def _ldu_content_for_hash(ldu: LDU) -> str:
        if ldu.text:
            return ldu.text

        if ldu.table_data:
            return str(ldu.table_data)

        if ldu.figure_ref:
            return ldu.figure_ref

        return ""
