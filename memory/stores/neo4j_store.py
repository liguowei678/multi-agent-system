from neo4j import GraphDatabase
from config.settings import NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD

AGENT_LABELS = {
    "user_profile": "AgentMemory_UserProfile",
    "preference": "AgentMemory_Preference",
    "event": "AgentMemory_Preference",
}


class Neo4jStore:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def add_relationship(self, subject: str, relation: str, obj: str,
                         label: str = "AgentMemory_Preference", props: dict = None) -> None:
        safe_label = label.replace(":", "")
        query = (
            f"MERGE (s:{safe_label} {{name: $subject}}) "
            f"MERGE (o:{safe_label} {{name: $obj}}) "
            f"MERGE (s)-[r:{relation}]->(o) "
            f"SET r += $props"
        )
        with self.driver.session() as session:
            session.run(query, subject=subject, obj=obj,
                        props=props or {}, relation=relation)

    def search_relations(self, node_name: str, label: str = None,
                         depth: int = 2) -> list[dict]:
        """模糊匹配: 查询词包含节点名, 或查询词包含关系类型"""
        lbl = label or "AgentMemory_Preference"
        query = (
            f"MATCH path = (n:{lbl})-[r*1..{depth}]-(connected) "
            f"WHERE $name CONTAINS n.name OR $name CONTAINS connected.name "
            f"OR ANY(rel IN r WHERE $name CONTAINS type(rel)) "
            f"RETURN [rel in relationships(path) | type(rel)] as relations, "
            f"[node in nodes(path) | node.name] as nodes "
            f"LIMIT 10"
        )
        with self.driver.session() as session:
            results = session.run(query, name=node_name)
            return [{"relations": r["relations"], "nodes": r["nodes"]} for r in results]

    def delete_node(self, node_name: str, label: str = None) -> None:
        lbl = label or "AgentMemory_Preference"
        with self.driver.session() as session:
            session.run(f"MATCH (n:{lbl} {{name: $name}}) DETACH DELETE n", name=node_name)

    def close(self):
        self.driver.close()
