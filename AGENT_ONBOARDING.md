# Agent Onboarding (5-Minute Start)

## 1) Project in One Sentence
This project is a safety-production multimodal agent that combines image/text understanding with case retrieval and legal-clause retrieval to generate explainable hazard analysis.

## 2) What This Repo Delivers
- Multimodal query handling (text + optional image)
- Intent routing (`greeting` / `legal_only` / `hazard_analysis`)
- Hybrid RAG (case retrieval + legal retrieval)
- Structured output for analysis and recommendation
- Web interaction via Gradio and ingestion APIs

## 3) Core Tech Stack
- Python backend: FastAPI, LangChain, ChromaDB, Transformers, PyMuPDF, python-docx
- Model serving: LLaMA-Factory OpenAI-compatible API
- Optional service: TypeScript + Express (`web-service-app`)

## 4) Read Order (Do Not Skip)
1. `README.md`
2. `DEVELOPMENT.md`
3. `scripts/start_all_services.sh`
4. `src/agent/workflow.py`
5. `src/retriever/clip_engine.py`
6. `src/data_processor/extractor.py`
7. `scripts/run_ingestion_api.py`
8. `TODO.md`

## 5) Fast Verification Path
1. Start services:
   - `bash scripts/start_all_services.sh`
2. Open WebUI and run one no-image legal query.
3. Run one image-based hazard analysis query.
4. Confirm outputs include retrieval evidence and structured fields.

## 6) Module Boundaries (Quick Map)
- `src/agent/workflow.py`: orchestration and intent routing
- `src/retriever/clip_engine.py`: image-based similar-case retrieval
- `src/data_processor/extractor.py`: PDF/Word extraction and structuring
- `scripts/run_ingestion_api.py`: knowledge ingestion and CRUD APIs
- `scripts/run_agent_webui.py`: user-facing web interaction

## 7) Common Failure Points
- Model API path/port misconfiguration
- Missing dependencies in fresh environment
- Offline mode fallback reducing retrieval quality
- Service startup race when dependencies are not ready

## 8) First Tasks for a New Agent
- Read `TODO.md` and pick the highest-priority unfinished item.
- Before edits, note impacted modules in your handoff note.
- After changes, provide reproducible outputs (`.csv` / `.md` / `.png`).
- Update status in `TODO.md` and complete `docs/engineering/HANDOFF_TEMPLATE.md`.

## 9) Source of Truth
- High-detail architecture and runtime details:
  - `reports/project_understanding_v2/PROJECT_UNDERSTANDING_REPORT_V2.md`
