CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE;
CREATE CONSTRAINT rel_predicate_exists IF NOT EXISTS FOR ()-[r:MEMORY]-() REQUIRE r.predicate IS NOT NULL;
CREATE CONSTRAINT rel_value_score_exists IF NOT EXISTS FOR ()-[r:MEMORY]-() REQUIRE r.value_score IS NOT NULL;
CREATE CONSTRAINT rel_memory_text_exists IF NOT EXISTS FOR ()-[r:MEMORY]-() REQUIRE r.memory_text IS NOT NULL;
CREATE INDEX rel_value_score IF NOT EXISTS FOR ()-[r:MEMORY]-() ON (r.value_score);
CREATE INDEX rel_created_at IF NOT EXISTS FOR ()-[r:MEMORY]-() ON (r.created_at);
CREATE INDEX rel_predicate IF NOT EXISTS FOR ()-[r:MEMORY]-() ON (r.predicate);
