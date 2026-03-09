# src/agents/audit_agent.py
# Audit agent for claim verification
# Uses SQLite FactTable and ChromaDB vector store
# Returns either source citation or "unverifiable"

import logging
import sqlite3
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)


class AuditAgent:
    """
    AuditAgent verifies claims against structured facts (SQLite)
    and semantic search (ChromaDB). Returns citations or 'unverifiable'.
    """

    def __init__(
        self, db_path=".refinery/facts.db", vector_dir=".refinery/vector_store"
    ):
        # SQLite FactTable
        self.conn = sqlite3.connect(db_path)

        # Vector store
        self.client = chromadb.PersistentClient(path=vector_dir, settings=Settings())
        self.collection = self.client.get_or_create_collection(name="ldu_collection")

        # Embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def verify_claim(self, claim: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Verify a claim against FactTable and Vector Store.
        Returns list of citations or 'unverifiable'.
        """
        results = []

        # 1. Check SQLite FactTable
        cur = self.conn.cursor()
        cur.execute(
            "SELECT doc_id, page_number, header, key, value, unit, source FROM facts"
        )
        for row in cur.fetchall():
            doc_id, page_number, header, key, value, unit, source = row
            if claim.lower() in key.lower():
                results.append(
                    {
                        "doc_id": doc_id,
                        "page_number": page_number,
                        "header": header,
                        "fact": f"{key} = {value} {unit}",
                        "source": source,
                    }
                )

        # 2. If no structured facts, check vector store
        if not results:
            query_emb = self.model.encode([claim]).tolist()
            vs_results = self.collection.query(
                query_embeddings=query_emb, n_results=top_k
            )
            for doc, meta in zip(
                vs_results["documents"][0], vs_results["metadatas"][0]
            ):
                results.append(
                    {
                        "doc_id": meta["doc_id"],
                        "page_number": meta["page_number"],
                        "header": meta["header"],
                        "summary": doc,
                        "source": f"{meta['doc_id']}, page {meta['page_number']}, section {meta['header']}",
                    }
                )

        # 3. If still empty, mark as unverifiable
        if not results:
            results.append({"claim": claim, "source": "Unverifiable"})

        return results


if __name__ == "__main__":
    agent = AuditAgent()
    print("AuditAgent ready. Connect and call verify_claim(claim) in interactive mode.")
