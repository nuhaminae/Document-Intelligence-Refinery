# src/agents/fact_extractor.py
# Extracts numerical/tabular facts from PageIndex JSONs
# Stores them in a SQLite backend for structured queries

import json
import logging
import os
import re
import sqlite3
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO)


class FactExtractor:
    """
    FactExtractor loads PageIndex artifacts and extracts numerical/tabular facts.
    Stores them in a SQLite database for structured querying.
    """

    def __init__(
        self, pageindex_dir=".refinery/pageindex", db_path=".refinery/facts.db"
    ):
        self.pageindex_dir = pageindex_dir
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite schema for facts."""
        cur = self.conn.cursor()
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS facts (
            doc_id TEXT,
            page_number INT,
            header TEXT,
            key TEXT,
            value REAL,
            unit TEXT,
            source TEXT
        )
        """
        )
        self.conn.commit()

    def _extract_numbers(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract numerical values and units from text using regex.
        Returns a list of {key, value, unit}.
        """
        results = []
        # Example regex: captures "Education budget 1.2B ETB"
        pattern = re.compile(r"([A-Za-z ]+)\s*([\d\.,]+)\s*([A-Za-z%]+)?")
        for match in pattern.finditer(text):
            key = match.group(1).strip()
            value_str = match.group(2).replace(",", "")
            try:
                value = float(value_str)
            except ValueError:
                continue
            unit = match.group(3) or ""
            results.append({"key": key, "value": value, "unit": unit})
        return results

    def extract_from_pageindex(self, doc_id: str, pageindex: Dict[str, Any]) -> None:
        """Extract facts from a single PageIndex JSON and insert into SQLite."""
        cur = self.conn.cursor()
        for page in pageindex.get("pages", []):
            page_num = page.get("page_number")
            for section in page.get("sections", []):
                header = section.get("header", "")
                content = " ".join(section.get("content", []))
                facts = self._extract_numbers(content)
                for fact in facts:
                    cur.execute(
                        """
                    INSERT INTO facts (doc_id, page_number, header, key, value, unit, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            doc_id,
                            page_num,
                            header,
                            fact["key"],
                            fact["value"],
                            fact["unit"],
                            f"{doc_id}, page {page_num}, section {header}",
                        ),
                    )
        self.conn.commit()

    def build_fact_table(self) -> None:
        """Process all PageIndex JSONs and populate the SQLite fact table."""
        for fname in os.listdir(self.pageindex_dir):
            if not fname.endswith("_pageindex.json"):
                continue
            doc_id = fname.replace("_pageindex.json", "")
            file_path = os.path.join(self.pageindex_dir, fname)
            with open(file_path, "r", encoding="utf-8") as f:
                pageindex = json.load(f)
            logging.info(f"Extracting facts from {doc_id}")
            self.extract_from_pageindex(doc_id, pageindex)
        logging.info("Fact table built successfully.")


if __name__ == "__main__":
    extractor = FactExtractor()
    extractor.build_fact_table()
    print("Fact table built at .refinery/facts.db")

