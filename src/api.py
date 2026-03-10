from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import logging

from src.agents.audit_agent import AuditAgent
from src.agents.query_agent import QueryAgent

app = FastAPI(title="Document Intelligence API")

# Initialise agents
query_agent = QueryAgent()
audit_agent = AuditAgent()

# --- Root health check ---
@app.get("/")
def root():
    return {"status": "ok", "message": "Document Intelligence API is running"}

# --- Request models ---
class QueryRequest(BaseModel):
    query: str

class ClaimRequest(BaseModel):
    claim: str

# --- Query Endpoints ---
@app.post("/query")
def run_query(req: QueryRequest):
    results = query_agent.semantic_search(req.query)
    return {"results": results}

@app.get("/navigate/{doc_id}/{page_number}")
def navigate(doc_id: str, page_number: int):
    return query_agent.pageindex_navigate(doc_id=doc_id, page_number=page_number)

@app.get("/structured/{doc_id}/{type_filter}")
def structured(doc_id: str, type_filter: str):
    return query_agent.structured_query(type_filter, doc_id=doc_id)

# --- Audit Mode Endpoint ---
@app.post("/audit")
def audit_claim(req: ClaimRequest):
    results = audit_agent.verify_claim(req.claim)
    return {"claim": req.claim, "verification": results}

# --- Pipeline Endpoints ---
@app.post("/pipeline/run")
def run_pipeline():
    """
    Run the full sequential pipeline inside the container.
    Equivalent to docker-compose run pipeline.
    """
    try:
        subprocess.run(
            [
                "bash", "-c",
                "python -m src.utils.preprocessor && "
                "python -m src.agents.extractor_rubric_config && "
                "python -m src.agents.extract_docs && "
                "python -m src.agents.chunker && "
                "python -m src.agents.indexer && "
                "python -m src.agents.fact_extractor && "
                "python -m src.agents.vector_ingestor"
            ],
            check=True
        )
        return {"status": "success", "message": "Pipeline executed successfully"}
    except subprocess.CalledProcessError as e:
        logging.error(f"Pipeline failed: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/pipeline/step/{step_name}")
def run_pipeline_step(step_name: str):
    """
    Run a single pipeline step by name.
    Example: POST /pipeline/step/chunker
    """
    try:
        subprocess.run(["python", f"-m", f"src.agents.{step_name}"], check=True)
        return {"status": "success", "message": f"Step {step_name} executed successfully"}
    except subprocess.CalledProcessError as e:
        logging.error(f"Step {step_name} failed: {e}")
        return {"status": "error", "message": str(e)}
