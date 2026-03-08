# src/agents/query_agent.py
# LangGraph agent with 3 tools: pageindex_navigate, semantic_search, structured_query
# Integrates AuditAgent for claim verification

import json
import logging
import os
from typing import Any, Dict, List

from src.agents.audit_agent import AuditAgent 

logging.basicConfig(level=logging.INFO)


class QueryAgent:
    """
    LangGraph-style agent with three tools:
    - pageindex_navigate: traverse PageIndex trees
    - semantic_search: embedding-based semantic retrieval
    - structured_query: metadata/content filtering
    Integrated with AuditAgent for claim verification.
    """

    def __init__(
        self,
        pageindex_dir=".refinery/pageindex",
        edge_tabs: List[Dict[str, Any]] = None,
    ):
        self.pageindex_dir = pageindex_dir
        self.global_index_path = os.path.join(pageindex_dir, "global_pageindex.json")
        self.pageindexes: Dict[str, Any] = {}
        self.global_index: Dict[str, Any] = {}
        self.active_doc_id: str = None
        self._load_artifacts()
        if edge_tabs:
            self._set_active_doc(edge_tabs)

        # Attach AuditAgent
        self.audit_agent = AuditAgent()

    def _load_artifacts(self) -> None:
        """Load all per-document PageIndex JSONs and the global index."""
        for fname in os.listdir(self.pageindex_dir):
            if fname.endswith("_pageindex.json"):
                doc_id = fname.replace("_pageindex.json", "")
                file_path = os.path.join(self.pageindex_dir, fname)
                with open(file_path, "r", encoding="utf-8") as f:
                    self.pageindexes[doc_id] = json.load(f)

        if os.path.exists(self.global_index_path):
            with open(self.global_index_path, "r", encoding="utf-8") as f:
                self.global_index = json.load(f)

        logging.info(f"Loaded {len(self.pageindexes)} PageIndex trees")
        logging.info(f"Loaded global index with {len(self.global_index)} documents")

    def _set_active_doc(self, edge_tabs: List[Dict[str, Any]]) -> None:
        """Set active document ID based on Edge tab metadata."""
        for tab in edge_tabs:
            if tab.get("isCurrent"):
                url = tab.get("pageUrl", "")
                fname = os.path.basename(url).replace(".pdf", "")
                self.active_doc_id = fname.replace(" ", "-")
                logging.info(f"Active document set to: {self.active_doc_id}")

    def format_audit_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Print audit results in a clean table format.
        """
        print("\n=== Claim Verification Results ===")
        print(f"{'Document ID':<20} | {'Page':<5} | {'Header':<25} | {'Source'}")
        print("-" * 80)
        for r in results:
            doc_id = r.get("doc_id", "")
            page = r.get("page_number", "")
            header = r.get("header", "")
            source = r.get("source", "Unverifiable")
            print(f"{doc_id:<20} | {str(page):<5} | {header:<25} | {source}")
        print("-" * 80)

    # --- Tool 1: PageIndex Navigation ---
    def pageindex_navigate(
        self, doc_id: str = None, page_number: int = None
    ) -> List[Dict[str, Any]]:
        doc_id = doc_id or self.active_doc_id
        if not doc_id or doc_id not in self.pageindexes:
            return []
        pages = self.pageindexes[doc_id].get("pages", [])
        if page_number is not None:
            pages = [p for p in pages if p.get("page_number") == page_number]
        return pages

    # --- Tool 2: Semantic Search ---
    def semantic_search(
        self, query: str, doc_id: str = None, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Semantic search across section summaries (delegated to AuditAgent for verification)."""
        results = self.audit_agent.verify_claim(query, top_k=top_k)
        return results

    # --- Tool 3: Structured Query ---
    def structured_query(
        self, type_filter: str, doc_id: str = None
    ) -> List[Dict[str, Any]]:
        doc_id = doc_id or self.active_doc_id
        if not doc_id or doc_id not in self.pageindexes:
            return []
        results = []
        for page in self.pageindexes[doc_id].get("pages", []):
            for section in page.get("sections", []):
                if section.get("header", "").lower() == type_filter.lower():
                    results.append(
                        {
                            "document_id": doc_id,
                            "page_number": page.get("page_number"),
                            "header": section.get("header", ""),
                            "section_summary": section.get("section_summary", ""),
                        }
                    )
        return results


if __name__ == "__main__":
    agent = QueryAgent()
    print("QueryAgent ready. Use pageindex_navigate, semantic_search, structured_query interactively.")

