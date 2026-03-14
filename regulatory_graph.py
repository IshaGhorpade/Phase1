from neo4j import GraphDatabase


driver = GraphDatabase.driver(
    "neo4j://127.0.0.1:7687",
    auth=("neo4j", "password")
)


def insert_rule(rule, source):

    query = """
    MERGE (r:Regulation {name:$source})
    CREATE (rule:Rule {text:$text})
    MERGE (r)-[:HAS_RULE]->(rule)
    """

    with driver.session() as session:

        session.run(
            query,
            source=source,
            text=rule
        )


def close_driver():
    driver.close()