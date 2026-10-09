"""
Query text from the W3C SPARQL 1.1 test suite.

Source: https://www.w3.org/2009/sparql/docs/tests/summary.html

A subset of that suite: one test per language construct introduced in SPARQL
1.1, chosen so every new pattern is represented once. Extracted mechanically
from the published summary page, keyed by its test identifier there. Do not
edit these strings by hand.
"""

TITLES: dict[str, str] = {
    "property-path-pp01": "(pp01) Simple path",
    "property-path-pp02": "(pp02) Star path",
    "property-path-pp08": "(pp08) Reverse path",
    "property-path-pp09": "(pp09) Reverse sequence path",
    "property-path-pp10": "(pp10) Path with negation",
    "property-path-pp14": "(pp14) Star path over foaf:knows",
    "property-path-pp21": "(pp21) Diamond -- :p+",
    "property-path-pp28a": "(pp28a) Diamond, with loop -- (:p/:p)?",
    "property-path-pp30": "(pp30) Operator precedence 1",
    "property-path-pp34": "(pp34) Named Graph 1",
    "bind-bind01": "bind01 - BIND",
    "bindings-inline1": "Inline VALUES graph pattern",
    "exists-exists01": "Exists with one constant",
    "negation-subset-by-exclusion-nex-1": "Subsets by exclusion (NOT EXISTS)",
    "subquery-subquery01": "sq01 - Subquery within graph pattern",
    "subquery-subquery08": "sq08 - Subquery with aggregate",
    "aggregates-agg01": "COUNT 1",
    "aggregates-agg-sum-01": "SUM",
    "aggregates-agg-min-02": "MIN with GROUP BY",
    "aggregates-agg-max-01": "MAX",
    "aggregates-agg-avg-01": "AVG",
    "aggregates-agg-sample-01": "SAMPLE",
    "aggregates-agg-groupconcat-03": "GROUP_CONCAT with SEPARATOR",
    "project-expression-projexp01": "Expression is equality",
    "service-service1": "SERVICE test 1",
    "functions-abs01": "ABS()",
    "functions-concat01": "CONCAT()",
    "functions-length01": "STRLEN()",
    "functions-substring01": "SUBSTR() (3-argument)",
    "functions-replace01": "REPLACE()",
    "functions-iri01": "IRI()/URI()",
    "functions-bnode01": "BNODE(str)",
    "functions-strdt01": "STRDT()",
    "functions-isnumeric01": "isNumeric()",
    "functions-year": "YEAR()",
    "functions-md5-01": "MD5()",
    "functions-struuid01": "STRUUID() pattern match",
    "functions-if01": "IF()",
    "functions-coalesce01": "COALESCE()",
    "functions-in01": "IN 1",
    "functions-notin01": "NOT IN 1",
    "negation-full-minuend": "Subtraction with MINUS from a fully bound minuend",
    "bindings-values1": "Post-query VALUES with subj-var, 1 row",
    "bindings-values4": "Post-query VALUES with 2 obj-vars, 1 row with UNDEF",
    "aggregates-agg-avg-02": "AVG with GROUP BY",
    "service-service6": "SERVICE test 6",
    "construct-constructwhere01": "constructwhere01 - CONSTRUCT WHERE",
    "delete-data-dawg-delete-data-01": "Simple DELETE DATA 1",
    "delete-where-dawg-delete-where-01": "Simple DELETE WHERE 1",
    "basic-update-insert-data-spo-named1": "Simple insert data named 1",
    "update-silent-load-silent": "LOAD SILENT",
    "clear-dawg-clear-all-01": "CLEAR ALL",
    "update-silent-create-silent": "CREATE SILENT iri",
    "drop-dawg-drop-all-01": "DROP ALL",
    "copy-copy01": "COPY 1",
    "move-move01": "MOVE 1",
    "add-add01": "ADD 1",
    "delete-dawg-delete-with-01": "Simple DELETE 1 (WITH)",
    "delete-dawg-delete-using-01": "Simple DELETE 1 (USING)",
}

QUERIES: dict[str, str] = {
    "property-path-pp01": """\
prefix ex:	<http://www.example.org/schema#>
prefix in:	<http://www.example.org/instance#>

select * where {
in:a ex:p1/ex:p2/ex:p3 ?x
}
""",
    "property-path-pp02": """\
prefix ex:	<http://www.example.org/schema#>
prefix in:	<http://www.example.org/instance#>

select * where {
in:a (ex:p1/ex:p2/ex:p3)* ?x
}
""",
    "property-path-pp08": """\
prefix ex:	<http://www.example.org/schema#>
prefix in:	<http://www.example.org/instance#>

ask {
in:b ^ex:p in:a
}
""",
    "property-path-pp09": """\
prefix ex:	<http://www.example.org/schema#>
prefix in:	<http://www.example.org/instance#>

select  * where {
in:c ^(ex:p1/ex:p2) ?x
}
""",
    "property-path-pp10": """\
prefix ex:	<http://www.example.org/schema#>
prefix in:	<http://www.example.org/instance#>

select * where {
in:a !(ex:p1|ex:p2) ?x
}
""",
    "property-path-pp14": """\
PREFIX : <http://example.org/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

SELECT *
WHERE { ?X foaf:knows* ?Y } 
ORDER BY ?X ?Y
""",
    "property-path-pp21": """\
prefix : <http://example/> 



select * where {

    :a :p+ ?z

}
""",
    "property-path-pp28a": """\
prefix : <http://example/> 



select * where {

    :a (:p/:p)? ?t

}
""",
    "property-path-pp30": """\
prefix :  <http://www.example.org/>
select ?t
where {
  :a :p1|:p2/:p3|:p4 ?t
}
""",
    "property-path-pp34": """\
prefix :  <http://www.example.org/>
select ?t
where {
  GRAPH <ng-01.ttl> {
    ?s :p1* ?t }
}
""",
    "bind-bind01": """\
PREFIX : <http://example.org/> 

SELECT ?z
{
  ?s ?p ?o .
  BIND(?o+10 AS ?z)
}
""",
    "bindings-inline1": """\
PREFIX dc:   <http://purl.org/dc/elements/1.1/> 
PREFIX :     <http://example.org/book/> 
PREFIX ns:   <http://example.org/ns#> 

SELECT ?book ?title ?price
{
   VALUES ?book { :book1 }
   ?book dc:title ?title ;
         ns:price ?price .
}
""",
    "exists-exists01": """\
prefix ex: <http://www.example.org/>

select * where {
?s ?p ?o
filter exists {?s ?p ex:o}
}
""",
    "negation-subset-by-exclusion-nex-1": """\
PREFIX ex: <http://www.w3.org/2009/sparql/docs/tests/data-sparql11/negation#>
SELECT ?animal { 
  ?animal a ex:Animal 
  FILTER NOT EXISTS { ?animal a ex:Insect } 
}
""",
    "subquery-subquery01": """\
prefix ex:	<http://www.example.org/schema#>
prefix in:	<http://www.example.org/instance#>

select  ?x ?p where {
graph ?g {
{select * where {?x ?p ?y}}
}
}
""",
    "subquery-subquery08": """\
prefix ex:	<http://www.example.org/schema#>
prefix in:	<http://www.example.org/instance#>

select ?x ?max where {
{select (max(?y) as ?max) where {?x ex:p ?y} } 
?x ex:p ?max
}
""",
    "aggregates-agg01": """\
PREFIX : <http://www.example.org>

SELECT (COUNT(?O) AS ?C)
WHERE { ?S ?P ?O }
""",
    "aggregates-agg-sum-01": """\
PREFIX : <http://www.example.org/>
SELECT (SUM(?o) AS ?sum)
WHERE {
	?s :dec ?o
}
""",
    "aggregates-agg-min-02": """\
PREFIX : <http://www.example.org/>
SELECT ?s (MIN(?o) AS ?min)
WHERE {
	?s ?p ?o
}
GROUP BY ?s
""",
    "aggregates-agg-max-01": """\
PREFIX : <http://www.example.org/>
SELECT (MAX(?o) AS ?max)
WHERE {
	?s ?p ?o
}
""",
    "aggregates-agg-avg-01": """\
PREFIX : <http://www.example.org/>
SELECT (AVG(?o) AS ?avg)
WHERE {
	?s :dec ?o
}
""",
    "aggregates-agg-sample-01": """\
PREFIX : <http://www.example.org/>
ASK {
	{
		SELECT (SAMPLE(?o) AS ?sample)
		WHERE {
			?s :dec ?o
		}
	}
	FILTER(?sample = 1.0 || ?sample = 2.2 || ?sample = 3.5)
}
""",
    "aggregates-agg-groupconcat-03": """\
PREFIX : <http://www.example.org/>
ASK {
	{SELECT (GROUP_CONCAT(?o;SEPARATOR=":") AS ?g) WHERE {
		[] :p1 ?o
	}}
	FILTER(?g = "1:22" || ?g = "22:1")
}
""",
    "project-expression-projexp01": """\
prefix ex:	<http://www.example.org/schema#>
prefix in:	<http://www.example.org/instance#>

select ?x ?y ?z ((?y = ?z) as ?eq) where {
  ?x ex:p ?y .
  ?x ex:q ?z
}
""",
    "service-service1": """\
# SERVICE join with pattern in the default graph

PREFIX : <http://example.org/> 

SELECT ?s ?o1 ?o2 
{
  ?s ?p1 ?o1 .
  SERVICE <http://example.org/sparql> {
    ?s ?p2 ?o2
  }
}
""",
    "functions-abs01": """\
PREFIX : <http://example.org/>
SELECT * WHERE {
	?s :num ?num
	FILTER(ABS(?num) >= 2)
}
""",
    "functions-concat01": """\
PREFIX : <http://example.org/>
SELECT (CONCAT(?str1,?str2) AS ?str) WHERE {
	:s6 :str ?str1 .
	:s7 :str ?str2 .
}
""",
    "functions-length01": """\
PREFIX : <http://example.org/>
SELECT ?str (STRLEN(?str) AS ?len) WHERE {
	?s :str ?str
}
""",
    "functions-substring01": """\
PREFIX : <http://example.org/>
SELECT ?s ?str (SUBSTR(?str,1,1) AS ?substr) WHERE {
	?s :str ?str
}
""",
    "functions-replace01": """\
PREFIX : <http://example.org/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
SELECT ?s (REPLACE(?str,"[^a-z0-9]", "-") AS ?new) WHERE {
	?s :str ?str
}
""",
    "functions-iri01": """\
BASE <http://example.org/>
SELECT (URI("uri") AS ?uri) (IRI("iri") AS ?iri)
WHERE {}
""",
    "functions-bnode01": """\
PREFIX : <http://example.org/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
SELECT ?s1 ?s2
(BNODE(?s1) AS ?b1) (BNODE(?s2) AS ?b2)
WHERE {
	?a :str ?s1 .
	?b :str ?s2 .
	FILTER (?a = :s1 || ?a = :s3)
	FILTER (?b = :s1 || ?b = :s3)
}
""",
    "functions-strdt01": """\
PREFIX : <http://example.org/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
SELECT ?s (STRDT(?str,xsd:string) AS ?str1) WHERE {
	?s :str ?str
	FILTER(LANGMATCHES(LANG(?str), "en"))
}
""",
    "functions-isnumeric01": """\
PREFIX : <http://example.org/>
SELECT ?s ?num WHERE {
	?s ?p ?num
	FILTER isNumeric(?num)
}
""",
    "functions-year": """\
PREFIX : <http://example.org/>
SELECT ?s (YEAR(?date) AS ?x) WHERE {
	?s :date ?date
}
""",
    "functions-md5-01": """\
PREFIX : <http://example.org/>
SELECT (MD5(?l) AS ?hash) WHERE {
	:s1 :str ?l
}
""",
    "functions-struuid01": """\
PREFIX : <http://example.org/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
SELECT (STRLEN(?uuid) AS ?length)
WHERE {
	BIND(STRUUID() AS ?uuid)
	FILTER(ISLITERAL(?uuid) && REGEX(?uuid, "^[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}$", "i"))
}
""",
    "functions-if01": """\
BASE <http://example.org/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
SELECT ?o (IF(lang(?o) = "ja", true, false) AS ?integer)
WHERE {
	?s ?p ?o
}
""",
    "functions-coalesce01": """\
PREFIX : <http://example.org/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
SELECT
	(COALESCE(?x, -1) AS ?cx)     # error when ?x is unbound -> -1
	(COALESCE(?o/?x, -2) AS ?div) # error when ?x is unbound or zero -> -2
	(COALESCE(?z, -3) AS ?def)    # always unbound -> -3
	(COALESCE(?z) AS ?err)        # always an error -> unbound
WHERE {
	?s :p ?o .
	OPTIONAL {
		?s :q ?x
	}
}
""",
    "functions-in01": """\
ASK {
	FILTER(2 IN (1, 2, 3))
}
""",
    "functions-notin01": """\
ASK {
	FILTER(2 NOT IN ())
}
""",
    "negation-full-minuend": """\
prefix : <http://example/>

select ?a ?b ?c {
  ?a :p1 ?b; :p2 ?c
  MINUS {
    ?d a :Sub
    OPTIONAL { ?d :q1 ?b }
    OPTIONAL { ?d :q2 ?c }
  }
}
order by ?a
""",
    "bindings-values1": """\
PREFIX dc:   <http://purl.org/dc/elements/1.1/> 
PREFIX :     <http://example.org/book/> 
PREFIX ns:   <http://example.org/ns#> 

SELECT ?book ?title ?price
{
   ?book dc:title ?title ;
         ns:price ?price .
}
VALUES ?book {
 :book1
}
""",
    "bindings-values4": """\
# bindings with one element UNDEF

PREFIX : <http://example.org/> 

SELECT ?s ?o1 ?o2
{
  ?s ?p1 ?o1 .
  ?s ?p2 ?o2 .
} VALUES (?o1 ?o2) {
 ("Alan" UNDEF)
}
""",
    "aggregates-agg-avg-02": """\
PREFIX : <http://www.example.org/>
SELECT ?s (AVG(?o) AS ?avg)
WHERE {
	?s ?p ?o
}
GROUP BY ?s
HAVING (AVG(?o) <= 2.0)
""",
    "service-service6": """\
# SERVICE with one optional and a nested SERVICE. This query depends in the capabilities of the example1.org endpoint

PREFIX : <http://example.org/> 

SELECT ?s ?o1 ?o2
{
  SERVICE <http://example1.org/sparql> {
  ?s ?p ?o1 .
  OPTIONAL {
	SERVICE SILENT <http://invalid.endpoint.org/sparql> {
    ?s ?p2 ?o2 }
  }
}
}
""",
    "construct-constructwhere01": """\
PREFIX : <http://example.org/>

CONSTRUCT WHERE { ?s ?p ?o}
""",
    "delete-data-dawg-delete-data-01": """\
PREFIX     : <http://example.org/> 
PREFIX foaf: <http://xmlns.com/foaf/0.1/> 

DELETE DATA 
{
  :a foaf:knows :b .
}
""",
    "delete-where-dawg-delete-where-01": """\
PREFIX     : <http://example.org/> 
PREFIX foaf: <http://xmlns.com/foaf/0.1/> 

DELETE WHERE
{
  :a foaf:knows ?b .
}
""",
    "basic-update-insert-data-spo-named1": """\
PREFIX : <http://example.org/ns#>

INSERT DATA { GRAPH <http://example.org/g1> { :s :p :o } }
""",
    "update-silent-load-silent": """\
LOAD SILENT <somescheme://www.example.com/THIS-GRAPH-DOES-NOT-EXIST/>
""",
    "clear-dawg-clear-all-01": """\
PREFIX     : <http://example.org/> 

CLEAR ALL
""",
    "update-silent-create-silent": """\
CREATE SILENT GRAPH <http://example.org/g1>
""",
    "drop-dawg-drop-all-01": """\
PREFIX     : <http://example.org/> 

DROP ALL
""",
    "copy-copy01": """\
PREFIX : <http://example.org/>
COPY DEFAULT TO :g1
""",
    "move-move01": """\
PREFIX : <http://example.org/>
MOVE DEFAULT TO :g1
""",
    "add-add01": """\
PREFIX : <http://example.org/>
ADD DEFAULT TO :g1
""",
    "delete-dawg-delete-with-01": """\
PREFIX     : <http://example.org/> 
PREFIX foaf: <http://xmlns.com/foaf/0.1/> 

WITH <http://example.org/g1>
DELETE 
{
  ?s ?p ?o .
}
WHERE
{
  :a foaf:knows ?s .
  ?s ?p ?o
}
""",
    "delete-dawg-delete-using-01": """\
PREFIX     : <http://example.org/> 
PREFIX foaf: <http://xmlns.com/foaf/0.1/> 

DELETE 
{
  ?s ?p ?o .
}
USING <http://example.org/g2>
WHERE
{
  :a foaf:knows ?s .
  ?s ?p ?o 
}
""",
}
