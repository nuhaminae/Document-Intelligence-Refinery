# src/agents/triage.py
# script to define preliminary document classification

from typing import Dict

import pdfplumber

from src.models.models import DocumentProfile, LayoutComplexity, OriginType


class TriageAgent:
    """
    Triage Agent:
    - Analyses document metadata and page metrics
    - Produces a DocumentProfile
    - Guides extraction strategy routing
    """

    def __init__(self):
        pass

    def compute_metrics(self, file_path: str) -> Dict:
        """
        Compute document metrics using pdfplumber:
            - char_density: characters per unit page area
            - whitespace_ratio: estimated whitespace vs text area
            - bbox_distribution: x_max values for column detection
            - image_area_ratio: ratio of image area to page area
            - has_font_metadata: whether embedded font info exists
        """
        total_chars, total_area, whitespace_area, image_area = 0, 0, 0, 0
        bbox_distribution = {"x_max": []}
        has_font_metadata = False

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_area = page.width * page.height
                total_area += page_area

                chars = page.chars
                total_chars += len(chars)

                # Collect x_max values for column detection
                if chars:
                    x_max_values = [c["x1"] for c in chars]
                    bbox_distribution["x_max"].extend(x_max_values)

                # Estimate text area as sum of character widths
                text_area = sum((c["x1"] - c["x0"]) for c in chars)
                whitespace_area += max(page_area - text_area, 0)

                # Image area ratio
                for img in page.images:
                    img_area = (img["x1"] - img["x0"]) * (img["y1"] - img["y0"])
                    image_area += img_area

            # Check for font metadata
            if hasattr(pdf, "doc") and hasattr(pdf.doc, "info"):
                has_font_metadata = "Font" in str(pdf.doc.info)

        char_density = total_chars / total_area if total_area else 0
        whitespace_ratio = whitespace_area / total_area if total_area else 0
        image_area_ratio = image_area / total_area if total_area else 0

        return {
            "char_density": char_density,
            "whitespace_ratio": whitespace_ratio,
            "bbox_distribution": bbox_distribution,
            "image_area_ratio": image_area_ratio,
            "has_font_metadata": has_font_metadata,
            "file_path": file_path,
        }

    def analyse_document(self, document_id: str, metrics: Dict) -> DocumentProfile:
        """
        metrics: dict containing
            - char_density (float)
            - whitespace_ratio (float)
            - bbox_distribution (dict with x_min, x_max, y_min, y_max lists)
            - origin_type_hint (str: 'digital' or 'scanned')
            - domain_hint (optional str)
        """

        # --- Origin type ---
        origin_type = OriginType.digital  # default
        if metrics.get("origin_type_hint") == "scanned":
            origin_type = OriginType.scanned
        elif metrics["image_area_ratio"] > 0.3 and metrics["char_density"] > 0:
            origin_type = OriginType.mixed
        elif metrics.get("has_font_metadata") and metrics["char_density"] > 0.001:
            origin_type = OriginType.form_fillable

        # --- Layout complexity ---
        layout_complexity = LayoutComplexity.single_column  # default
        x_range = metrics["bbox_distribution"].get(
            "x_max", []
        )  # gets list that contains x_max or an empty list

        # Uses bounding box distribution (x_max) to detect multiple columns
        # if there is more than one x_max and the difference between them is greater than 100
        if len(x_range) > 1 and (max(x_range) - min(x_range)) > 100:
            layout_complexity = LayoutComplexity.multi_column

        # High whitespace ratio → likely figure‑heavy
        # if whitespace ratio is greater than 50%
        if metrics["whitespace_ratio"] > 0.5:
            layout_complexity = LayoutComplexity.figure_heavy

        # Very low character density → table‑heavy
        # if character density is less than 0.0005
        if metrics["char_density"] < 0.0005:
            layout_complexity = LayoutComplexity.table_heavy
        if metrics["whitespace_ratio"] > 0.5 and metrics["char_density"] < 0.0005:
            layout_complexity = LayoutComplexity.mixed

        # --- Domain hint ---
        domain_hint = metrics.get("domain_hint") or self._classify_domain(
            metrics["file_path"]
        )

        # --- Confidence scoring ---
        confidence = self._compute_confidence(metrics)

        return DocumentProfile(
            document_id=document_id,
            origin_type=origin_type,
            layout_complexity=layout_complexity,
            domain_hint=domain_hint,
            char_density=metrics["char_density"],
            whitespace_ratio=metrics["whitespace_ratio"],
            bbox_distribution=metrics["bbox_distribution"],
            triage_confidence=confidence,
            file_path=metrics.get("file_path"),
        )

    def _classify_domain(self, file_path: str) -> str:
        """
        Lightweight domain classifier based on filename keywords.
        """
        fname = file_path.lower()
        if any(word in fname for word in ["invoice", "budget", "tax", "finance"]):
            return "financial"
        if any(word in fname for word in ["contract", "law", "court"]):
            return "legal"
        if any(word in fname for word in ["experiment", "data", "research"]):
            return "technical"
        if any(word in fname for word in ["patient", "diagnosis", "treatment"]):
            return "medical"
        return "general"

    # -- Confidence scoring helper function --
    def _compute_confidence(self, metrics: Dict) -> float:
        """
        Multi-signal confidence score based on density, whitespace, image ratio, and font metadata.
        """
        density = metrics["char_density"]
        whitespace = metrics["whitespace_ratio"]
        image_ratio = metrics.get("image_area_ratio", 0)
        has_fonts = metrics.get("has_font_metadata", False)

        # High density + low whitespace + min image ratio → high confidence (0.9)
        if density > 0.0015 and whitespace < 0.3 and image_ratio < 0.5 and has_fonts:
            return 0.9
        # Medium density → medium confidence (0.75)
        elif 0.0005 <= density <= 0.0015:
            return 0.75
        # Otherwise → lower confidence (0.6)
        else:
            return 0.6
