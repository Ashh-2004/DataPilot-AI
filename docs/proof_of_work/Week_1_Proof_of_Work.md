# Proof of Work — Week 1: Project Planning and System Architecture

**Student Name:** M Ashish Ramana  
**USN:** 1BI25MC060  
**Institution:** Bangalore Institute of Technology (Department of Master of Computer Applications)  
**Subject:** Project Work (Subject Code: MPRJ384)  
**Project Title:** DataPilot AI : Autonomous Business Intelligence Platform  
**Reporting Period:** 07-09-2026 to 14-09-2026  

---

## 1. Executive Summary

During Week 1, the foundational groundwork for **DataPilot AI** was established. The goal of DataPilot AI is to deliver an autonomous, agent-driven Business Intelligence (BI) platform that ingests raw tabular datasets, cleans and validates them, discovers domain schemas, accepts natural language queries, and automatically generates deep analytical reports and visualizations. 

Key milestones achieved during this week include finalizing the project requirements, designing the end-to-end data analytics pipeline architecture, establishing supported multi-format data ingestion specs, defining module boundaries, and selecting the foundational technology stack.

---

## 2. Weekly Deliverables & Implementation Mapping

| # | Weekly Report Deliverable | Implementation Details & Codebase Artifacts | Status |
|---|---------------------------|---------------------------------------------|--------|
| **1** | **Requirement Analysis & Workflow Finalization** | Defined core capabilities: automated dataset ingestion, quality profiling, domain identification, multi-agent query execution, and automated report generation. | Completed |
| **2** | **End-to-End Data Analytics Pipeline Design** | Mapped pipeline stages: Data Ingestion $\rightarrow$ Dataset Validation & Cleaning $\rightarrow$ Business Domain Classification $\rightarrow$ Schema Discovery & Knowledge Graph $\rightarrow$ Natural Language Planner/Executor $\rightarrow$ Natural Language Analyzer & Report Builder. Mapped in [`app/analysis/pipeline.py`](file:///d:/Projects/DataPilot/app/analysis/pipeline.py). | Completed |
| **3** | **Supported Data Sources Review** | Evaluated parser engines for standard tabular formats: CSV, Excel (`.xlsx`), JSON, Parquet, and SQL database engines. | Completed |
| **4** | **High-Level System Architecture & Module Design** | Structured codebase into decoupled packages: <br>• [`app/agents`](file:///d:/Projects/DataPilot/app/agents) (Planner, Executor, Analyzer) <br>• [`app/analysis`](file:///d:/Projects/DataPilot/app/analysis) (Pipeline, Profiler, Classifier, Report Builder) <br>• [`app/services`](file:///d:/Projects/DataPilot/app/services) (Cleaning, Domain Classifier, Schema Profile, SQL Guardrails) <br>• [`app/frontend`](file:///d:/Projects/DataPilot/app/frontend) (Streamlit dashboard) | Completed |
| **5** | **Technology Stack & Environment Setup** | Finalized tech stack: Python 3.10+, DuckDB (in-memory analytical engine), Ollama / Llama 3.2 (local LLM runtime), Streamlit, pandas/polars, Pydantic, and pytest. Configured in [`pyproject.toml`](file:///d:/Projects/DataPilot/pyproject.toml) and [`docker-compose.yml`](file:///d:/Projects/DataPilot/docker-compose.yml). | Completed |

---

## 3. High-Level Architecture Overview

```mermaid
graph TD
    A[Raw Data File / Source] --> B[Data Ingestion Module]
    B --> C[Data Health & Cleaning Service]
    C --> D[Domain Classification & Schema Discovery]
    D --> E[DuckDB Analytical Storage]
    
    SubGraph Multi-Agent Engine
        F[Planner Agent] --> G[Executor Agent]
        G --> H[Analyzer Agent]
    End
    
    E --> F
    H --> I[Automated Visualization & PDF/HTML Report Builder]
    I --> J[Streamlit Dashboard / User Interface]
```

---

## 4. Key Codebase References

- **Pipeline Orchestrator Entrypoint:** [`app/analysis/pipeline.py`](file:///d:/Projects/DataPilot/app/analysis/pipeline.py)
- **Data Models & State Definitions:** [`app/analysis/dataclasses.py`](file:///d:/Projects/DataPilot/app/analysis/dataclasses.py)
- **Main Application Launcher:** [`main.py`](file:///d:/Projects/DataPilot/main.py)
- **Dependency & Build Configuration:** [`pyproject.toml`](file:///d:/Projects/DataPilot/pyproject.toml), [`docker-compose.yml`](file:///d:/Projects/DataPilot/docker-compose.yml)

---

## 5. Summary of Outcomes & Verification

- **Architectural Blueprint:** Clear separation of concerns between data processing, agentic reasoning, SQL generation, and frontend presentation.
- **Environment Verification:** Successfully initialized Python virtual environment with dependencies installed, and validated containerization configuration using Docker Compose.
