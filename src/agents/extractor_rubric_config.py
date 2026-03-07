# src/agents/extractor_rubric_config.py
# script to define extraction router that runs on a rubric file
# it gives the same output as src/agents/extractor.py
# but takes configuration from rubrics/extraction_rules.yaml

import json
import logging
import os
import time

import yaml

from src.agents.triage import TriageAgent
from src.models.models import (
    DocumentProfile,
    ExtractedDocument,
    LayoutComplexity,
    OriginType,
    StrategyType,
)
from src.strategies.fasttext_extractor import FastTextExtractor
from src.strategies.layout_extractor import LayoutExtractor
from src.strategies.vision_extractor import VisionExtractor

logging.basicConfig(level=logging.INFO)


class RubricDrivenExtractionRouter:
    """
    Routes documents to the appropriate extraction strategy
    based on DocumentProfile thresholds and confidence scores.
    Gets thresholds and rules from rubric/extraction_rules.yaml
    Saves triage profiles and logs audit trail into ledger.
    """

    def __init__(
        self,
        rubric_path="rubric/extraction_rules.yaml",
        input_dir="data",
        profiles_dir=".refinery/profiles",
        ledger_path=".refinery/extraction_ledger.jsonl",
    ):
        with open(rubric_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # extract thresholds and rules
        self.thresholds = self.config["thresholds"]
        self.confidence_rules = self.config["confidence_scoring"]
        self.chunking_rules = self.config["chunking_rules"]
        self.max_cost_usd = self.config["budget_guard"]["max_cost_per_document_usd"]
        self.escalation_policy = self.config["budget_guard"]["escalation_policy"]

        self.input_dir = input_dir
        self.profiles_dir = profiles_dir
        self.ledger_path = ledger_path
        os.makedirs(self.profiles_dir, exist_ok=True)

    def build_profiles(self):
        """Build DocumentProfiles from PDFs in data/ using TriageAgent."""
        triage = TriageAgent()
        profiles = []

        for fname in os.listdir(self.input_dir):
            if fname.endswith(".pdf"):
                doc_id = os.path.splitext(fname)[0]
                file_path = os.path.join(self.input_dir, fname)
                metrics = triage.compute_metrics(file_path)
                metrics["origin_type_hint"] = (
                    "scanned" if "scan" in fname.lower() else "digital"
                )
                metrics["domain_hint"] = triage._classify_domain(file_path)
                metrics["file_path"] = file_path
                profile = triage.analyse_document(doc_id, metrics)
                profiles.append(profile)
        return profiles

    def select_strategy(self, profile: DocumentProfile) -> StrategyType:
        """
        Pragmatic router logic using density, whitespace, image ratio, and font metadata.
        Rules are defined in rubric/extraction_rules.yaml
        """

        density = profile.char_density
        whitespace = profile.whitespace_ratio
        image_ratio = getattr(profile, "image_area_ratio", 0)
        has_fonts = getattr(profile, "has_font_metadata", False)

        if (
            density < self.thresholds["char_density"]["low"]
            or whitespace > self.thresholds["whitespace_ratio"]["high"]
        ):
            return StrategyType.vision_augmented

        if profile.layout_complexity in [
            LayoutComplexity.multi_column,
            LayoutComplexity.table_heavy,
        ]:
            return StrategyType.layout_aware
        if (
            has_fonts
            and density > self.thresholds["char_density"]["high"]
            and profile.origin_type == OriginType.digital
        ):
            return StrategyType.layout_aware

        if (
            density >= self.thresholds["char_density"]["high"]
            and whitespace < self.thresholds["whitespace_ratio"]["low"]
            and image_ratio < self.thresholds["image_area_ratio"]["max_fasttext"]
        ):
            return StrategyType.fasttext

        return StrategyType.fasttext

    def escalate_strategy(self, current_strategy: StrategyType) -> StrategyType:
        """
        Escalates the extraction strategy based on the current strategy.
        - If current_strategy is StrategyType.fasttext, returns StrategyType.layout_aware.
        - If current_strategy is StrategyType.layout_aware, returns StrategyType.vision_augmented.
        - Otherwise, returns the current strategy unchanged.
        """
        if current_strategy == StrategyType.fasttext:
            return StrategyType.layout_aware
        if current_strategy == StrategyType.layout_aware:
            return StrategyType.vision_augmented
        return current_strategy

    def run(self):
        """
        Orchestrates the extraction pipeline, including:

        1. Building document profiles using TriageAgent
        2. Selecting the appropriate extraction strategy based on profile metrics
        3. Running extraction with the selected strategy
        4. Confidence-gated escalation to a more robust strategy if needed
        5. Saving extracted document profiles to JSON
        6. Appending to the extraction ledger with cost estimates and provenance chains
        """
        profiles = self.build_profiles()
        for profile in profiles:
            strategy = self.select_strategy(profile)
            logging.info(f"Document {profile.document_id} → strategy {strategy.value}")
            extractor = self._get_extractor(strategy, profile.file_path)
            start_time = time.time()
            extracted_doc: ExtractedDocument = extractor.extract(profile)
            runtime_sec = round(time.time() - start_time, 2)
            tokens = len(json.dumps(extracted_doc.model_dump())) // 4
            cost_usd = (tokens / 1000) * 0.01

            if (
                extracted_doc.extraction_confidence < 0.75
                and cost_usd <= self.max_cost_usd
            ):
                strategy = self.escalate_strategy(strategy)
                extractor = self._get_extractor(strategy, profile.file_path)
                extracted_doc = extractor.extract(profile)

            profile_path = os.path.join(
                self.profiles_dir, f"{profile.document_id}.json"
            )
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile.model_dump(), f, indent=2)

            ledger_entry = {
                "document_id": profile.document_id,
                "strategy_used": strategy.value,
                "extraction_confidence": extracted_doc.extraction_confidence,
                "provenance_chain": [
                    prov.model_dump() for prov in extracted_doc.provenance_chain
                ],
                "cost_estimate": {
                    "tokens": tokens,
                    "runtime_sec": runtime_sec,
                    "usd": round(cost_usd, 4),
                },
            }
            with open(self.ledger_path, "a", encoding="utf-8") as ledger_file:
                ledger_file.write(json.dumps(ledger_entry) + "\n")

            print(f"Processed {profile.document_id} with {strategy.value}")

    def _get_extractor(self, strategy: StrategyType, file_path: str):
        """
        Returns an extractor object based on the given strategy and file path.
        Args:
            strategy (StrategyType): Extraction strategy to use.
            file_path (str): Path to the input PDF file.
        Returns:
            An extractor object (FastTextExtractor, LayoutExtractor, VisionExtractor)
        Raises:
            ValueError: If the given strategy is not supported.
        """
        if strategy == StrategyType.fasttext:
            return FastTextExtractor(file_path)
        elif strategy == StrategyType.layout_aware:
            return LayoutExtractor(file_path)
        elif strategy == StrategyType.vision_augmented:
            return VisionExtractor(file_path)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


if __name__ == "__main__":
    router = RubricDrivenExtractionRouter(input_dir="data")
    router.run()
