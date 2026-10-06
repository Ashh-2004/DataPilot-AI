# DataPilot AI

DataPilot AI is a local-first Business Intelligence and Decision Intelligence assistant. It uses a natural-language question to retrieve data with a read-only DuckDB query, then answers conversationally with key insights, limitations, follow-up questions, and optional charts through Streamlit. SQL is an internal execution detail, not the user-facing response.

## Run with Docker Compose

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Open Streamlit at <http://localhost:8501>, the API docs at <http://localhost:8000/docs>, and Grafana at <http://localhost:3000>. Pull the configured Ollama model once before querying:

```powershell
docker compose exec ollama ollama pull llama3.2
```

Upload a CSV or JSON file in the UI. The filename becomes the DuckDB table name. The API also accepts `POST /query` with `{"question":"..."}` and exposes Prometheus metrics at `/metrics`.

Anomaly notifications are optional. Set `N8N_WEBHOOK_URL` only after creating and activating an n8n webhook workflow at that path; a failed notification is logged and does not fail the data query.

The Streamlit interface maintains recent conversation context, so follow-up questions such as "What about last month?" or "Why is that higher?" can refer to earlier turns. The assistant is grounded in the uploaded schema and query results; it will not invent columns or modify the dataset.

## Architecture

The compiled LangGraph has exactly three sequential nodes: `PlannerAgent`, `ExecutorAgent`, and `AnalyzerAgent`. Database access is isolated in `DuckDBTool` and `Neo4jTool`. Analyzer anomalies post a dummy Slack-shaped payload to the configured n8n webhook.
