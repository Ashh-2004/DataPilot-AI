"""Planner agent: converts natural language into a structured execution plan."""

import json
from typing import Any

from langchain_ollama import ChatOllama

from app.mcp.duckdb_tool import DuckDBTool


class PlannerAgent:
    """Uses Ollama to produce a constrained plan for the executor."""

    def __init__(self, duckdb_tool: DuckDBTool, model: str, base_url: str) -> None:
        self.duckdb_tool = duckdb_tool
        self.llm = ChatOllama(model=model, base_url=base_url, temperature=0, num_ctx=2048, num_predict=256)

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Create a plan containing intent, entities, and SQL."""
        schema = self.duckdb_tool.schema_context()
        if schema == "No tables are loaded.":
            raise ValueError("Upload a CSV or JSON file before asking a question.")
        if self._is_overview_question(state["question"]):
            overview = self.duckdb_tool.dataset_overview()
            table = overview["tables"][0]["name"]
            return {
                "dataset_overview": overview,
                "plan": {
                    "intent": "dataset_overview",
                    "entities": [table],
                    "sql_hint": f'SELECT * FROM "{table}" LIMIT 25',
                },
            }
        history = state.get("history", [])
        prompt = (
            "You are the data retrieval specialist inside a conversational data analyst. "
            "Return JSON only with keys intent, entities, sql_hint. "
            "Use only tables and columns in this schema. SQL must be a single read-only DuckDB query. "
            "Do not use SQL placeholders, parameters, question marks, or multiple statements. "
            "Do not wrap the SQL in markdown; put literal values directly in the query. "
            "Resolve references such as 'that', 'those', 'the previous result', and omitted subjects "
            "using the conversation history. Select columns needed to support a useful answer, including "
            "grouping columns when the user asks for a comparison.\n"
            f"Schema:\n{schema}\nConversation:\n{history}\nQuestion: {state['question']}"
        )
        response = self.llm.invoke(prompt)
        content = response.content if isinstance(response.content, str) else str(response.content)
        plan = json.loads(content[content.find("{") : content.rfind("}") + 1])
        if not isinstance(plan, dict):
            raise ValueError("Planner returned an invalid plan.")
        entities = plan.get("entities", [])
        if isinstance(entities, str):
            plan["entities"] = [entities]
        elif entities is None:
            plan["entities"] = []
        elif not isinstance(entities, list):
            plan["entities"] = []
        return {"plan": plan}

    @staticmethod
    def _is_overview_question(question: str) -> bool:
        """Route broad walkthrough requests without asking the model for multiple queries."""
        normalized = question.casefold()
        markers = (
            "walk me through",
            "give me an overview",
            "dataset overview",
            "describe the dataset",
            "explain this dataset",
            "tell me about this dataset",
            "what is in this dataset",
            "summarize the dataset",
            "explore this dataset",
        )
        return any(marker in normalized for marker in markers)
