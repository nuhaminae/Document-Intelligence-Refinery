# src/strategies/layout_extractor.py
# Modified after Peer's explainer

"""
Strategy B: Layout-Aware extraction with Docling.

Best for:
- multi-column PDFs
- table-heavy PDFs
- mixed layout documents
- documents where FastText loses structure

This replacement adds Gate 2 post-extraction scoring so the router can decide
whether LayoutAware actually produced usable structured output.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Iterable, List, Optional, Tuple

from docling.document_converter import DocumentConverter

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


class LayoutExtractor:
    """
    Strategy B: Layout-Aware extraction using Docling.

    Gate 2 note:
    The extraction confidence returned by this class is computed from actual
    Docling output, not copied from profile.triage_confidence.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def extract(self, profile: DocumentProfile) -> ExtractedDocument:
        ldus: List[LDU] = []
        provenance: List[ProvenanceChain] = []
        page_indexes: List[PageIndex] = []

        converter = DocumentConverter()
        result = converter.convert(self.pdf_path)
        doc = result.document

        items = list(self._iter_docling_items(doc))

        if items:
            ldus, provenance, page_indexes = self._build_from_docling_items(
                items=items,
                profile=profile,
            )
        else:
            logging.warning(
                "Docling returned no iterable layout items for %s. "
                "Falling back to markdown export.",
                profile.document_id,
            )
            ldus, provenance, page_indexes = self._build_from_markdown_fallback(
                doc=doc,
                profile=profile,
            )

        extracted_doc = ExtractedDocument(
            document_id=profile.document_id,
            strategy_used=StrategyType.layout_aware,
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

    def _iter_docling_items(self, doc: Any) -> Iterable[Any]:
        """
        Robustly iterate over Docling document items.

        Different Docling versions may yield:
        - item
        - (item, level)
        - (item, level, parent)

        This normalizes to item.
        """

        if not hasattr(doc, "iterate_items"):
            return []

        normalized_items = []

        try:
            for raw in doc.iterate_items():
                if isinstance(raw, tuple) and raw:
                    normalized_items.append(raw[0])
                else:
                    normalized_items.append(raw)
        except Exception as exc:
            logging.warning("Docling iterate_items failed: %s", exc)
            return []

        return normalized_items

    def _build_from_docling_items(
        self,
        items: List[Any],
        profile: DocumentProfile,
    ) -> Tuple[List[LDU], List[ProvenanceChain], List[PageIndex]]:
        ldus: List[LDU] = []
        provenance: List[ProvenanceChain] = []
        page_indexes: List[PageIndex] = []

        page_to_ldus = {}

        for idx, item in enumerate(items):
            page_num = self._extract_page_number(item)
            block_type = self._extract_block_type(item)
            bbox = self._extract_bbox(item)
            text = self._extract_text(item)

            if not text and block_type != LDUType.table:
                continue

            ldu_id = f"ldu_{page_num}_{idx}"

            table_data = None
            figure_ref = None

            if block_type == LDUType.table:
                table_data = self._extract_table_data(item)
                text_for_hash = str(table_data) if table_data else text
            elif block_type == LDUType.figure:
                figure_ref = self._extract_figure_ref(item)
                text_for_hash = figure_ref or text
            else:
                text_for_hash = text

            content_hash = hashlib.sha256(text_for_hash.encode("utf-8")).hexdigest()

            ldu = LDU(
                ldu_id=ldu_id,
                type=block_type,
                text=(
                    text if block_type in {LDUType.paragraph, LDUType.caption} else None
                ),
                table_data=table_data,
                figure_ref=figure_ref,
                bbox=bbox,
                page_number=page_num,
            )

            ldus.append(ldu)
            page_to_ldus.setdefault(page_num, []).append(ldu)

            provenance.append(
                ProvenanceChain(
                    ldu_id=ldu_id,
                    strategy_used=StrategyType.layout_aware,
                    source_bbox=bbox,
                    source_page=page_num,
                    transformations=["docling_layout_parse"],
                    confidence_score=0.0,
                    content_hash=content_hash,
                )
            )

        for page_num, page_ldus in sorted(page_to_ldus.items()):
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

        return ldus, provenance, page_indexes

    def _build_from_markdown_fallback(
        self,
        doc: Any,
        profile: DocumentProfile,
    ) -> Tuple[List[LDU], List[ProvenanceChain], List[PageIndex]]:
        """
        Fallback when Docling layout items are unavailable.

        This preserves some content but uses coarse bboxes and page 1. Gate 2 will
        warn if provenance is too coarse or insufficient.
        """

        markdown = ""

        try:
            if hasattr(doc, "export_to_markdown"):
                markdown = doc.export_to_markdown()
        except Exception as exc:
            logging.warning("Docling markdown export failed: %s", exc)

        lines = [line.strip() for line in markdown.splitlines() if line.strip()]

        ldus: List[LDU] = []
        provenance: List[ProvenanceChain] = []

        for i, line in enumerate(lines):
            ldu_type = (
                LDUType.caption
                if line.lower().startswith(("table", "figure", "fig"))
                else LDUType.paragraph
            )
            ldu_id = f"ldu_1_{i}"
            bbox = (0.0, 0.0, 1000.0, 1000.0)
            content_hash = hashlib.sha256(line.encode("utf-8")).hexdigest()

            ldu = LDU(
                ldu_id=ldu_id,
                type=ldu_type,
                text=line,
                table_data=None,
                figure_ref=None,
                bbox=bbox,
                page_number=1,
            )
            ldus.append(ldu)

            provenance.append(
                ProvenanceChain(
                    ldu_id=ldu_id,
                    strategy_used=StrategyType.layout_aware,
                    source_bbox=bbox,
                    source_page=1,
                    transformations=["docling_markdown_fallback"],
                    confidence_score=0.0,
                    content_hash=content_hash,
                )
            )

        page_indexes = [
            PageIndex(
                page_number=1,
                ldus=ldus,
                char_density=profile.char_density,
                whitespace_ratio=profile.whitespace_ratio,
                layout_signature={
                    "x0": [ldu.bbox[0] for ldu in ldus],
                    "x1": [ldu.bbox[2] for ldu in ldus],
                    "y0": [ldu.bbox[1] for ldu in ldus],
                    "y1": [ldu.bbox[3] for ldu in ldus],
                },
            )
        ]

        return ldus, provenance, page_indexes

    @staticmethod
    def _extract_page_number(item: Any) -> int:
        for attr in ("page_no", "page_number", "page"):
            value = getattr(item, attr, None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass

        prov = getattr(item, "prov", None)
        if prov:
            try:
                first = prov[0] if isinstance(prov, list) else prov
                page_no = getattr(first, "page_no", None)
                if page_no is not None:
                    return int(page_no)
            except Exception:
                pass

        return 1

    @staticmethod
    def _extract_block_type(item: Any) -> LDUType:
        label = getattr(item, "label", None)
        item_type = getattr(item, "type", None)

        raw = label or item_type or item.__class__.__name__
        raw_text = str(raw).lower()

        if "table" in raw_text:
            return LDUType.table

        if "figure" in raw_text or "picture" in raw_text or "image" in raw_text:
            return LDUType.figure

        if "caption" in raw_text:
            return LDUType.caption

        return LDUType.paragraph

    @staticmethod
    def _extract_text(item: Any) -> str:
        for attr in ("text", "caption", "content"):
            value = getattr(item, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()

        if hasattr(item, "export_to_markdown"):
            try:
                value = item.export_to_markdown()
                if isinstance(value, str):
                    return value.strip()
            except Exception:
                pass

        return ""

    @staticmethod
    def _extract_table_data(item: Any) -> Optional[List[List[str]]]:
        cells = getattr(item, "cells", None)
        if isinstance(cells, list) and cells:
            return [
                [str(cell) for cell in row] if isinstance(row, list) else [str(row)]
                for row in cells
            ]

        if hasattr(item, "export_to_dataframe"):
            try:
                df = item.export_to_dataframe()
                rows = [list(map(str, df.columns))]
                rows.extend(df.astype(str).values.tolist())
                return rows
            except Exception:
                pass

        return None

    @staticmethod
    def _extract_figure_ref(item: Any) -> Optional[str]:
        for attr in ("image_ref", "uri", "caption", "text"):
            value = getattr(item, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()

        return None

    @staticmethod
    def _extract_bbox(item: Any) -> BBox:
        bbox = getattr(item, "bbox", None)

        if bbox is not None:
            normalized = LayoutExtractor._normalize_bbox(bbox)
            if normalized:
                return normalized

        prov = getattr(item, "prov", None)
        if prov:
            try:
                first = prov[0] if isinstance(prov, list) else prov
                prov_bbox = getattr(first, "bbox", None)
                normalized = LayoutExtractor._normalize_bbox(prov_bbox)
                if normalized:
                    return normalized
            except Exception:
                pass

        return (0.0, 0.0, 0.0, 0.0)

    @staticmethod
    def _normalize_bbox(value: Any) -> Optional[BBox]:
        if value is None:
            return None

        if isinstance(value, (tuple, list)) and len(value) == 4:
            try:
                x0, y0, x1, y1 = [float(v) for v in value]
                return (x0, y0, x1, y1)
            except (TypeError, ValueError):
                return None

        # Some libraries expose bbox as object fields.
        try:
            x0 = float(getattr(value, "l", getattr(value, "x0", 0.0)))
            y0 = float(getattr(value, "t", getattr(value, "y0", 0.0)))
            x1 = float(getattr(value, "r", getattr(value, "x1", 0.0)))
            y1 = float(getattr(value, "b", getattr(value, "y1", 0.0)))
            return (x0, y0, x1, y1)
        except Exception:
            return None
