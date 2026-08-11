# TODO: SPARQL the query builders cannot express

Every construct listed here is one the `qleverlux.SPARQLQueryBuilder` /
`SPARQLSyntaxTerms` classes have no way to produce. Each entry names the
conformance test it comes from, shows the query that should be buildable once
the gap is closed, and sketches the API that would close it.

The list is derived from two sources, and is exhaustive with respect to both:

| Source | Examples | Built | Gaps |
| --- | --- | --- | --- |
| [SPARQL 1.0 examples](https://www.w3.org/2001/sw/DataAccess/rq23/examples.html) | 68 | 51 | 17 |
| [SPARQL 1.1 test suite](https://www.w3.org/2009/sparql/docs/tests/summary.html) | 59 selected | 42 | 17 |

The 34 gap entries reduce to the 22 distinct features below.

## How to close one

1. Add the API to the builder classes.
2. Write the builder function in `tests/test_w3c_examples.py` or
   `tests/test_sparql11_patterns.py`, registered with `@example("<test id>")`.
3. Delete the entry from that module's `UNSUPPORTED` list.

`test_every_w3c_example_is_accounted_for` and
`test_every_selected_test_is_accounted_for` fail if a test appears in neither
list, so a gap cannot be dropped silently. The `UNSUPPORTED` probes also fail
the moment the API appears, which is the reminder to do step 2.

## Priority for this repository

The middletier only ever issues read-only SELECT queries, so most of this list
is latent capability rather than something LUX needs.

- **Plausibly useful**: `MINUS`, `VALUES` over a variable list, `ASK`.
  `ASK` in particular is the natural form for the HAL existence probes that
  `middletier_config.py:565` currently writes as `SELECT ?uri … LIMIT 1` — though
  note those are hand-written strings today, not built with these classes.
- **Not used here**: `CONSTRUCT`, `DESCRIBE`, `BASE`, dataset clauses, and all
  twelve update operations. Worth implementing only for completeness, or if the
  builders get reused outside this middletier.

---

# 1. Query forms

Only SELECT and a DELETE/INSERT update are implemented (`SPARQLSelectQuery`,
`SPARQLUpdateQuery`). The other three query forms have no class at all.

## 1.1 ASK

Tests: `Q94`, `Q97`

```sparql
PREFIX foaf:    <http://xmlns.com/foaf/0.1/>
ASK  { ?x foaf:name  "Alice" }
```

```python
class SPARQLAskQuery(SPARQLQuery):
    """prefixes + WHERE, no projection or solution modifiers."""
```

Everything needed is already on `SPARQLQuery`: `add_prefix`, `set_where_pattern`.
`get_text` is `prefixes + "\nASK " + where`.

## 1.2 CONSTRUCT

Tests: `Q77`, `Q80`, `Q82`, `Q83`, `Q85`

```sparql
PREFIX foaf:    <http://xmlns.com/foaf/0.1/>
PREFIX vcard:   <http://www.w3.org/2001/vcard-rdf/3.0#>
CONSTRUCT   { <http://example.org/person#Alice> vcard:FN ?name }
WHERE       { ?x foaf:name ?name }
```

```python
class SPARQLConstructQuery(SPARQLQuery):
    def set_construct_template(self, graph_pattern: SPARQLGraphPattern) -> Self: ...
```

The template is a plain group of triples, so `SPARQLGraphPattern` can carry it -
but note it must reject the OPTIONAL/UNION/GRAPH/SERVICE flags, which are not
legal in a CONSTRUCT template. `Q85` also needs solution modifiers
(`ORDER BY desc(?hits) LIMIT 2`), so those belong on this class too, not only on
SELECT.

## 1.3 CONSTRUCT WHERE

Test: `construct-constructwhere01`

```sparql
PREFIX : <http://example.org/>

CONSTRUCT WHERE { ?s ?p ?o}
```

The short form where the template is the pattern. A flag on the CONSTRUCT class
(`SPARQLConstructQuery(where_form=True)`) rather than a separate type.

## 1.4 DESCRIBE

Tests: `Q87`, `Q88`, `Q89`, `Q90`, `Q91`

```sparql
DESCRIBE <http://example.org/>
```

```sparql
PREFIX foaf:   <http://xmlns.com/foaf/0.1/>
DESCRIBE ?x ?y <http://example.org/>
WHERE    {?x foaf:knows ?y}
```

```python
class SPARQLDescribeQuery(SPARQLQuery):
    def add_resources(self, resources: str | Iterable[str]) -> Self: ...
```

The WHERE clause is optional here, which no existing class allows - `Q87` is a
complete query with no pattern at all.

---

# 2. Base and dataset

## 2.1 BASE

Tests: `Q5`, `Q6`

```sparql
BASE    <http://example.org/book/>
PREFIX  dc: <http://purl.org/dc/elements/1.1/>
SELECT  $title
WHERE   { <book1>  dc:title  ?title }
```

```python
class SPARQLQuery:
    def set_base(self, iri: str) -> Self: ...
```

Rendered before the prefixes. Belongs on `SPARQLQuery` so every query form gets
it. Cheap, and a prerequisite for any query using relative IRIs.

## 2.2 FROM and FROM NAMED

Tests: `Q56`, `Q59`, `Q63`

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT  ?name
FROM    <http://example.org/foaf/aliceFoaf>
WHERE   { ?x foaf:name ?name }
```

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX dc: <http://purl.org/dc/elements/1.1/>

SELECT ?who ?g ?mbox
FROM <http://example.org/dft.ttl>
FROM NAMED <http://example.org/alice>
FROM NAMED <http://example.org/bob>
WHERE
{
   ?g dc:publisher ?who .
   GRAPH ?g { ?x foaf:mbox ?mbox }
}
```

```python
class SPARQLQuery:
    def add_from(self, iri: str) -> Self: ...
    def add_from_named(self, iri: str) -> Self: ...
```

Both lists render after the projection and before WHERE, `FROM` first. Applies
to SELECT, CONSTRUCT, DESCRIBE and ASK alike.

---

# 3. Graph patterns

## 3.1 MINUS

Test: `negation-full-minuend`

```sparql
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
```

```python
GraphPattern(minus=True)
```

This is the one real pattern gap. It slots straight into the existing flag
mechanism next to `optional`, `union` and `not_exists`: another branch in
`emit_into` writing `MINUS {`, and the same mutual-exclusion check the
`not_exists` flag already performs. `MINUS` is not interchangeable with
`FILTER NOT EXISTS` - they differ when the inner pattern shares no variables
with the outer one - so the existing `not_exists` flag does not cover it.

## 3.2 SERVICE SILENT

Test: `service-service6`

```sparql
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
```

```python
GraphPattern(service="<http://…/sparql>", silent=True)
```

One extra keyword in the SERVICE branch of `emit_into`. `SILENT` makes a failing
remote endpoint yield no results instead of failing the whole query.

Note: rdflib cannot parse this test's nested SERVICE (it hits the recursion
limit), so the test is xfailed for reference validity. A builder test for
`SERVICE SILENT` should use a single, non-nested SERVICE block.

---

# 4. Inline data

## 4.1 VALUES after the WHERE clause

Test: `bindings-values1`

```sparql
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
```

`Values` can only be added to a `GraphPattern` (`add_value`), which puts it
inside the braces. The post-query position is a different thing: it constrains
the whole solution sequence.

```python
class SPARQLSelectQuery(SPARQLQuery):
    def add_values(self, values: Values) -> Self: ...
```

Rendered after the solution modifiers, i.e. last of all.

## 4.2 VALUES over a variable list, with UNDEF

Test: `bindings-values4`

```sparql
PREFIX : <http://example.org/>

SELECT ?s ?o1 ?o2
{
  ?s ?p1 ?o1 .
  ?s ?p2 ?o2 .
} VALUES (?o1 ?o2) {
 ("Alan" UNDEF)
}
```

`Values(values, name)` binds one variable to a flat list of terms. The general
form binds a *tuple* of variables to a list of rows, where any cell may be
`UNDEF`.

```python
Values(["?o1", "?o2"], rows=[('"Alan"', UNDEF)])
```

This needs a decision rather than just code: either widen `Values` to accept a
sequence of names plus a sequence of rows (keeping the current single-variable
call working), or add a second class and leave `Values` alone. The single
variable form is the one the middletier uses (`sparql.py` binds `?var` to one
URI), so whatever shape is chosen must keep that call site untouched.

---

# 5. Update operations

`SPARQLUpdateQuery` covers exactly `DELETE { } INSERT { } WHERE { }`. The
remaining update forms in the suite have no support. None of them is used by the
middletier, which never writes.

## 5.1 INSERT DATA / DELETE DATA

Tests: `basic-update-insert-data-spo-named1`, `delete-data-dawg-delete-data-01`

```sparql
PREFIX : <http://example.org/ns#>

INSERT DATA { GRAPH <http://example.org/g1> { :s :p :o } }
```

```sparql
PREFIX     : <http://example.org/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

DELETE DATA
{
  :a foaf:knows :b .
}
```

Ground triples, no WHERE clause and no variables permitted.

```python
class SPARQLUpdateQuery(SPARQLQuery):
    def set_insert_data(self, graph_pattern: SPARQLGraphPattern) -> Self: ...
    def set_delete_data(self, graph_pattern: SPARQLGraphPattern) -> Self: ...
```

## 5.2 DELETE WHERE

Test: `delete-where-dawg-delete-where-01`

```sparql
PREFIX     : <http://example.org/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

DELETE WHERE
{
  :a foaf:knows ?b .
}
```

The short form where the template is the pattern - the update analogue of
CONSTRUCT WHERE.

## 5.3 WITH

Test: `delete-dawg-delete-with-01`

```sparql
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
```

Names the graph the operation applies to. Renders before `DELETE`.

## 5.4 USING

Test: `delete-dawg-delete-using-01`

```sparql
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
```

The update-side equivalent of `FROM`: the dataset the WHERE clause matches
against. `USING NAMED` exists too and should come with it.

## 5.5 Graph management

Tests: `update-silent-load-silent`, `clear-dawg-clear-all-01`,
`update-silent-create-silent`, `drop-dawg-drop-all-01`, `copy-copy01`,
`move-move01`, `add-add01`

```sparql
LOAD SILENT <somescheme://www.example.com/THIS-GRAPH-DOES-NOT-EXIST/>
```

```sparql
CLEAR ALL
```

```sparql
CREATE SILENT GRAPH <http://example.org/g1>
```

```sparql
DROP ALL
```

```sparql
PREFIX : <http://example.org/>
COPY DEFAULT TO :g1
```

```sparql
PREFIX : <http://example.org/>
MOVE DEFAULT TO :g1
```

```sparql
PREFIX : <http://example.org/>
ADD DEFAULT TO :g1
```

Seven one-line operations sharing a shape: an optional `SILENT`, and a graph
reference that is either `DEFAULT`, `NAMED`, `ALL` or `GRAPH <iri>`. Best done as
one small class parameterized by keyword rather than seven classes:

```python
class SPARQLGraphManagementQuery(SPARQLQuery):
    def __init__(self, operation: str, target: str, destination: str = "",
                 silent: bool = False) -> None: ...
```

Note these parse with `parseUpdate`, not `parseQuery`; the reference-validity
test already routes them accordingly.

---

# 6. Sharp edges that are not missing features

These already work, but behave in a way worth fixing or documenting.

## 6.1 `OrderBy` with several variables emits invalid SPARQL

```python
OrderBy(["?a", "?b"]).get_text()   # 'ASC(?a ?b)' - does not parse
```

`GroupBy(["?a", "?b"])` correctly means *two grouping variables*, so the same
call shape on `OrderBy` reads as *two ordering conditions* - but `ASC` takes a
single expression, so the result is rejected. Two conditions must be added
separately:

```python
query.add_order_by(OrderBy(["?a"]))
query.add_order_by(OrderBy(["?b"]))   # ORDER BY ASC(?a) ASC(?b)
```

Either reject a multi-element list in `OrderBy`, or have `add_order_by` expand
one into several conditions. Covered by
`test_builder_reproduces_sparql11_test[property-path-pp14-...]`, which documents
the workaround.

## 6.2 Positive `EXISTS` has no flag

`GraphPattern(not_exists=True)` renders `FILTER NOT EXISTS { … }`, but there is
no counterpart for a positive `EXISTS`. It is reachable by composition:

```python
inner = GraphPattern()
inner.add_triples(Triple("?s", "?p", "ex:o"))
where.add_filter(Filter(f"EXISTS {inner.get_text(1)}"))
```

which is what `test_sparql11_patterns.build_positive_exists` does. An `exists=True`
flag alongside `not_exists=True` would make it symmetric.

## 6.3 `Values` renders a bare token list

`Values` passes each value through `in_brackets`, which wraps anything starting
with `http` and leaves everything else alone. Literals therefore have to be
pre-quoted by the caller, and a value like `"http://…"` intended as a string
literal would be emitted as an IRI. Fine for the middletier's use (URIs only),
worth knowing before reusing it.
