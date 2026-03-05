# src/agents/extractor.py
# script to define extraction router

import json
import os
import time

from src.agents.triage import TriageAgent
from src.models.models import DocumentProfile, ExtractedDocument, StrategyType
from src.strategies.fasttext_extractor import FastTextExtractor
from src.strategies.layout_extractor import LayoutExtractor
from src.strategies.vision_extractor import VisionExtractor


class ExtractionRouter:
    """
    Routes documents to the appropriate extraction strategy
    based on DocumentProfile thresholds and confidence scores.
    Saves triage profiles and logs audit trail into ledger.
    """

    def __init__(
        self,
        input_dir="data",
        profiles_dir=".refinery/profiles",
        ledger_path=".refinery/extraction_ledger.jsonl",
        confidence_threshold=0.75,
    ):
        self.input_dir = input_dir
        self.profiles_dir = profiles_dir
        self.ledger_path = ledger_path
        self.confidence_threshold = confidence_threshold
        os.makedirs(self.profiles_dir, exist_ok=True)

    def build_profiles(self):
        """
        Build DocumentProfiles from PDFs in data/ using TriageAgent.
        """
        profiles = []
        triage = TriageAgent()

        for fname in os.listdir(self.input_dir):
            if fname.endswith(".pdf"):
                doc_id = os.path.splitext(fname)[0]
                file_path = os.path.join(self.input_dir, fname)

                # Compute metrics with pdfplumber
                metrics = triage.compute_metrics(file_path)
                metrics["origin_type_hint"] = (
                    "scanned" if "scan" in fname.lower() else "digital"
                )
                metrics["domain_hint"] = "general"
                metrics["file_path"] = file_path

                # Analyse document to produce a profile
                profile = triage.analyse_document(doc_id, metrics)
                profiles.append(profile)

        return profiles

    def select_strategy(self, profile: DocumentProfile) -> StrategyType:
        """
        Router logic: decide which strategy to use based on profile metrics.
        """
        # --- Strategy A: FastText ---
        if profile.char_density >= 0.0015 and profile.whitespace_ratio < 0.3:
            return StrategyType.fasttext

        # --- Strategy B: Layout-Aware ---
        if 0.0005 <= profile.char_density < 0.0015:
            if profile.layout_complexity in ["multi_column", "table_heavy"]:
                return StrategyType.layout_aware

        # --- Strategy C: Vision-Augmented ---
        if profile.char_density < 0.0005 or profile.whitespace_ratio > 0.6:
            return StrategyType.vision_augmented

        # --- Default fallback ---
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
            extractor = self._get_extractor(strategy, profile.file_path)

            # Run extraction
            start_time = time.time()
            extracted_doc: ExtractedDocument = extractor.extract(profile)
            runtime_sec = round(time.time() - start_time, 2)

            # Confidence-gated escalation
            if extracted_doc.extraction_confidence < self.confidence_threshold:
                print(
                    f"Low confidence ({extracted_doc.extraction_confidence}) for {profile.document_id}, escalating..."
                )
                strategy = self.escalate_strategy(strategy)
                extractor = self._get_extractor(strategy, profile.file_path)
                extracted_doc = extractor.extract(profile)

            # Save profile JSON
            profile_path = os.path.join(
                self.profiles_dir, f"{profile.document_id}.json"
            )
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile.model_dump(), f, indent=2)

            # Append to ledger
            ledger_entry = {
                "document_id": profile.document_id,
                "strategy_used": strategy.value,
                "extraction_confidence": extracted_doc.extraction_confidence,
                "provenance_chain": [
                    prov.model_dump() for prov in extracted_doc.provenance_chain
                ],
                "cost_estimate": {
                    "tokens": len(json.dumps(extracted_doc.model_dump())) // 4,
                    "runtime_sec": runtime_sec,
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
    router = ExtractionRouter(input_dir="data")
    router.run()
