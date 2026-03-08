# src/agents/indexer.py
# Script to build a PageIndex tree with LLM-generated section summaries
# Uses Gemma:2b via Ollama for summarisation
# Consumes outputs from chunker.py in .refinery/chunks/
# Saves artifacts to .refinery/pageindex/

import csv
import json
import logging
import os
import time
from typing import Any, Dict, List

import requests

# Create a persistent HTTP session for Ollama
ollama_session = requests.Session()

# Configure logging once, with timestamps and levels
logging.basicConfig(
    level=logging.DEBUG,  # change to INFO if you want less detail
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def ollama_is_running() -> bool:
    """
    Check if the Ollama server is running locally.
    """
    try:
        ollama_session.get("http://localhost:11434")
        return True
    except requests.exceptions.ConnectionError:
        return False


def generate_summary(text: str, scope: str = "section") -> str:
    """
    Generate a summary using Gemma:2b via Ollama HTTP API.
    Uses a persistent session to avoid reconnect overhead.
    """
    if not ollama_is_running():
        logging.warning("⚠️ Ollama is not running. Please start it with 'ollama serve'.")

    if not text:
        return "No content available for summary."

    start = time.time()
    try:
        prompt = f"Summarize this {scope}: {text}"
        response = ollama_session.post(
            "http://localhost:11434/api/generate",
            json={"model": "gemma:2b", "prompt": prompt},
            stream=True,
        )
        response.raise_for_status()
        output = ""
        for line in response.iter_lines():
            if line:
                data = line.decode("utf-8")
                if '"response"' in data:
                    part = data.split('"response":"')[1].split('"')[0]
                    output += part
        elapsed = time.time() - start
        logging.debug(f"Summary for {scope} took {elapsed:.2f} seconds")
        return output.strip()
    except Exception as e:
        elapsed = time.time() - start
        logging.error(f"LLM summarisation failed after {elapsed:.2f}s: {e}")
        return f"Summary unavailable (error: {e})"


class Indexer:
    """
    Loads chunk JSON files and builds a hierarchical PageIndex tree.
    Each node in the tree is enriched with LLM-generated summaries.
    Produces per-document PageIndex JSONs, a global index, and a CSV catalog.
    """

    def __init__(self, chunks_dir=".refinery/chunks", output_dir=".refinery/pageindex"):
        self.chunks_dir = chunks_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def build_pageindex(
        self, doc_id: str, chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build a hierarchical PageIndex tree from chunks.
        Summarises only at the section and document level (skips chunk summaries).
        Logs timing for each stage.
        """
        pageindex = {"document_id": doc_id, "pages": []}
        pages_map: Dict[int, Dict[str, Any]] = {}

        # Group chunks into sections by page
        for chunk in chunks:
            page_num = chunk.get("metadata", {}).get("page_number", -1)
            if page_num not in pages_map:
                pages_map[page_num] = {"page_number": page_num, "sections": []}

            section_header = chunk["metadata"].get("header") or "Untitled Section"
            section_content = chunk["content"]
            if isinstance(section_content, list):
                # Flatten list into a string
                section_content = " ".join(str(item) for item in section_content)

            section_node = {
                "header": section_header,
                "content": [section_content],  # always a string
                "section_summary": "",
                "subsections": [],
            }

            pages_map[page_num]["sections"].append(section_node)

        # Section-level summaries
        start_sections = time.time()
        for page in pages_map.values():
            for section in page["sections"]:
                aggregated_text = " ".join(section["content"])
                section["section_summary"] = generate_summary(
                    aggregated_text, scope="section"
                )
        elapsed_sections = time.time() - start_sections
        logging.info(f"Section-level summaries took {elapsed_sections:.2f} seconds")

        # Whole-document summary
        start_doc = time.time()
        all_text = " ".join(
            " ".join(section["content"])
            for page in pages_map.values()
            for section in page["sections"]
        )
        pageindex["document_summary"] = generate_summary(all_text, scope="document")
        elapsed_doc = time.time() - start_doc
        logging.info(f"Whole-document summary took {elapsed_doc:.2f} seconds")

        pageindex["pages"] = list(pages_map.values())
        return pageindex

    def build_index(self) -> Dict[str, Any]:
        """
        Build per-document PageIndex trees and a global index.
        Logs total time per document.
        """
        global_index = {}

        for fname in os.listdir(self.chunks_dir):
            if not fname.endswith("_chunks.json"):
                continue
            doc_id = fname.replace("_chunks.json", "")
            file_path = os.path.join(self.chunks_dir, fname)

            with open(file_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            logging.info(f"Building PageIndex for {doc_id} from {file_path}")

            start_doc_index = time.time()
            pageindex = self.build_pageindex(doc_id, chunks)
            elapsed_doc_index = time.time() - start_doc_index
            logging.info(f"Indexing {doc_id} took {elapsed_doc_index:.2f} seconds")

            # Save per-document PageIndex
            out_path = os.path.join(self.output_dir, f"{doc_id}_pageindex.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(pageindex, f, indent=2)
            logging.info(f"Saved PageIndex → {out_path}")

            # Collect executive summary for global catalog
            global_index[doc_id] = {
                "document_id": doc_id,
                "document_summary": pageindex.get("document_summary", ""),
                "page_count": len(pageindex["pages"]),
            }

        # Save global index
        global_path = os.path.join(self.output_dir, "global_pageindex.json")
        with open(global_path, "w", encoding="utf-8") as f:
            json.dump(global_index, f, indent=2)
        logging.info(f"Saved global PageIndex → {global_path}")

        return global_index

    def list_summaries(self, global_index: Dict[str, Any]) -> None:
        """
        Print all document executive summaries in a clean table format.
        """
        print("\n=== Global Executive Summaries ===")
        print(f"{'Document ID':<30} | {'Pages':<5} | Summary")
        print("-" * 80)
        for doc_id, entry in global_index.items():
            summary = entry.get("document_summary", "")
            page_count = entry.get("page_count", "?")
            print(f"{doc_id:<30} | {page_count:<5} | {summary[:150]}...")
        print("-" * 80)

    def export_summaries_csv(
        self, global_index: Dict[str, Any], filename: str = "global_summaries.csv"
    ) -> str:
        """
        Export all executive summaries to a CSV file for use in Excel/Power BI.
        """
        csv_path = os.path.join(self.output_dir, filename)
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Document ID", "Page Count", "Executive Summary"])
            for doc_id, entry in global_index.items():
                writer.writerow(
                    [
                        entry.get("document_id", doc_id),
                        entry.get("page_count", ""),
                        entry.get("document_summary", ""),
                    ]
                )
        logging.info(f"Exported global summaries → {csv_path}")
        return csv_path


if __name__ == "__main__":
    indexer = Indexer()
    global_index = indexer.build_index()
    indexer.list_summaries(global_index)
    csv_file = indexer.export_summaries_csv(global_index)
    print(f"CSV catalog saved at: {csv_file}")
