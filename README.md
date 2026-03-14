See guys...
Simply run pipeline.py and edit your Neo4j URL and password then go to Neo4j and run the below to queries to check if the vectors are formed neatly 

MATCH (r:Regulation)-[:HAS_RULE]->(rule)
RETURN r.name, count(rule)


MATCH (r:Regulation)-[:HAS_RULE]->(rule)
RETURN r, rule
LIMIT 20


nd that's it... see you laterrrrr!!!😁
