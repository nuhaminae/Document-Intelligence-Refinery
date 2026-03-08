# src/agents/vector_ingestor.py
# Ingests PageIndex LDUs into a vector store (ChromaDB)
# Enables semantic search across the entire corpus

import json
import logging
import os
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)


class VectorIngestor:
    """
    VectorIngestor loads PageIndex artifacts and ingests section summaries
    into a ChromaDB vector store for semantic search.
    """

    def __init__(
        self, pageindex_dir=".refinery/pageindex", db_dir=".refinery/vector_store"
    ):
        self.pageindex_dir = pageindex_dir
        self.db_dir = db_dir
        os.makedirs(self.db_dir, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.db_dir, settings=Settings())
        self.collection = self.client.get_or_create_collection(name="ldu_collection")

        # Embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def ingest_pageindex(self, doc_id: str, pageindex: Dict[str, Any]) -> None:
        """Embed section summaries and insert into ChromaDB."""
        texts, ids, metas = [], [], []

        for page in pageindex.get("pages", []):
            for idx, section in enumerate(page.get("sections", [])):
                summary = section.get("section_summary", "")
                if not summary:
                    continue
                texts.append(summary)
                # Ensure IDs are unique by including section index
                ids.append(
                    f"{doc_id}_p{page.get('page_number')}_{idx}_{section.get('header')}"
                )
                metas.append(
                    {
                        "doc_id": doc_id,
                        "page_number": page.get("page_number"),
                        "header": section.get("header", ""),
                    }
                )

        if texts:
            embeddings = self.model.encode(texts).tolist()
            self.collection.add(
                documents=texts, embeddings=embeddings, metadatas=metas, ids=ids
            )
            logging.info(f"Ingested {len(texts)} sections from {doc_id}")

    def build_vector_store(self) -> None:
        """Process all PageIndex JSONs and ingest into ChromaDB."""
        for fname in os.listdir(self.pageindex_dir):
            if not fname.endswith("_pageindex.json"):
                continue
            doc_id = fname.replace("_pageindex.json", "")
            file_path = os.path.join(self.pageindex_dir, fname)
            with open(file_path, "r", encoding="utf-8") as f:
                pageindex = json.load(f)
            self.ingest_pageindex(doc_id, pageindex)
        logging.info("Vector store ingestion complete.")

    def semantic_query(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Perform semantic search across the vector store."""
        query_emb = self.model.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_emb, n_results=top_k)
        output = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            output.append(
                {
                    "doc_id": meta["doc_id"],
                    "page_number": meta["page_number"],
                    "header": meta["header"],
                    "summary": doc,
                }
            )
        return output


if __name__ == "__main__":
    ingestor = VectorIngestor()
    ingestor.build_vector_store()
    print("Vector store built at .refinery/vector_store")
