"""Neo4j MCP tools for lightweight entity and query memory."""

from typing import Any

from neo4j import GraphDatabase


class Neo4jTool:
    """Owns all Neo4j access used by the pipeline."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def record_query(self, question: str, plan: dict[str, Any]) -> None:
        """Persist a query and its intent for future context."""
        with self.driver.session() as session:
            session.run(
                "MERGE (q:Query {question: $question}) SET q.intent = $intent, q.sql_hint = $sql_hint",
                question=question,
                intent=plan.get("intent", ""),
                sql_hint=plan.get("sql_hint", ""),
            ).consume()

    def context(self, entities: list[str]) -> list[dict[str, Any]]:
        """Read known entities related to the requested question."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (q:Query) WHERE any(entity IN $entities WHERE q.question CONTAINS entity) "
                "RETURN q.question AS question, q.intent AS intent ORDER BY q.question LIMIT 5",
                entities=entities,
            )
            return [dict(record) for record in result]

    def close(self) -> None:
        """Close the Neo4j driver."""
        self.driver.close()
