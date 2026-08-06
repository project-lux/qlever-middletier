# LUX Middle Tier code for Qlever

## Installation

### Install Qlever

See the Qlever installation instructions.

For MacOS:
* Install homebrew: see https://brew.sh/
* brew tap qlever-dev/qlever
* brew install qlever

For Linux, from source:
* See [installing qlever](installing_qlever.md)

Install the Qlever UI:  (optional but very useful)

* With docker: just wait and do `qlever ui`
* Locally:
  * git clone https://github.com/qlever-dev/qlever-ui.git
  * cd qlever-ui
  * npm install
  * npm run build
  * python3 -m venv QLEVERUI
  * source QLEVERUI/bin/activate
  * Check pyproject.toml for any changes to dependencies but...
  * pip install django==5.2.12 django-environ==0.13.0 djangorestframework django-import-export==4.4.0 gunicorn==25.1.0 markdown requests==2.32.5 whitenoise[brotli]==6.12.0 pyyaml
  * python manage.py makemigrations --merge && python manage.py migrate
  * ./manage.py createsuperuser
  * ./manage.py runserver localhost:8146

* Login to the admin console and configure the lux endpoint to use localhost:7010

### Index and Serve the LUX data

Create a directory for the Qlever data:
* mkdir qlever

Copy the configuration files from the repo:
* cp files/Qleverfile files/Qleverfile-ui.yml lux.settings.json qlever/

Move the data somewhere accessible (and edit Qleverfile to point to it)
* mkdir data
* cp /path/to/triples/data/*gz data/

Index the data, and create the materialized views:
* cd qlever
* qlever index
* ... wait ...

Serve the data via SPARQL at :7010:
* qlever start

Start the UI locally, rather than via docker:
* ???



## Running qleverlux

Install luxql:  https://github.com/project-lux/luxql/
And pip install -e .

python ./qleverlux/middletier.py --help


## Notes ... ignore from here

```sparql
PREFIX lux: <https://lux.collections.yale.edu/ns/>
PREFIX ogc: <http://www.opengis.net/rdf#>
PREFIX osmrel: <https://www.openstreetmap.org/relation/>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX osmkey: <https://www.openstreetmap.org/wiki/Key:>
PREFIX geof: <http://www.opengis.net/def/function/geosparql/>
PREFIX qlss: <https://qlever.cs.uni-freiburg.de/spatialSearch/>

SELECT ?where ?coords WHERE {
  BIND( "POINT(174.763336 -36.848461)"^^geo:wktLiteral AS ?akl )

  SERVICE qlss: {
    _:config  qlss:algorithm qlss:s2 ;
              qlss:left ?akl ;
              qlss:right ?coords ;
              qlss:numNearestNeighbors 20 ;
              qlss:maxDistance 5000 ;
              qlss:bindDistance ?dist_left_right ;
              qlss:payload ?where  .
    {
      ?where lux:placeDefinedBy ?coords .
    }
  }
}
```


Query to generate a materialized view for item words:

```sparql
PREFIX lux: <https://lux.collections.yale.edu/ns/> 
SELECT ?word ?uri ?score ?tf WHERE { 
  { ?uri lux:itemPrimaryName ?text BIND (14 AS ?weight) } 
  UNION 
  { ?uri lux:recordText ?text BIND (5 AS ?weight) } 
  UNION 
  { ?uri lux:itemAny/lux:primaryName ?text BIND (1 AS ?weight) } 
  ?uri a lux:Item . 
  GRAPH ?tf { ?text ql:has-word ?word } 
  BIND (?tf * ?weight AS ?score) }
```

And then the query:

```sparql
PREFIX view: <https://qlever.cs.uni-freiburg.de/materializedView/>
PREFIX lux: <https://lux.collections.yale.edu/ns/>

SELECT ?subject (SUM(?s1 + ?s2 + ?s3) AS ?score) WHERE {
  SERVICE view:itemWords { [ view:column-word "dort" ; view:column-uri ?subject; view:column-score ?s1 ] }
  SERVICE view:itemWords { [ view:column-word "turner" ; view:column-uri ?subject; view:column-score ?s2 ] }
  SERVICE view:itemWords { [ view:column-word "painting" ; view:column-uri ?subject; view:column-score ?s3 ] }

} GROUP BY ?subject ORDER BY DESC(?score)
```
