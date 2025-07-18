# LUX Middle Tier code for Qlever

## Installation

### Install Qlever

See installing_qlever.md

### Set up Data

...

## Running qleverlux

Install luxql:  https://github.com/project-lux/luxql/
And pip install -e .

python ./qleverlux/middletier.py --help




## query notes

geo_search = """
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
"""

geo_search2 = """
PREFIX lux: <https://lux.collections.yale.edu/ns/>
PREFIX ogc: <http://www.opengis.net/rdf#>
PREFIX osmrel: <https://www.openstreetmap.org/relation/>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX osmkey: <https://www.openstreetmap.org/wiki/Key:>
PREFIX geof: <http://www.opengis.net/def/function/geosparql/>
PREFIX qlss: <https://qlever.cs.uni-freiburg.de/spatialSearch/>
SELECT ?where ?centroid WHERE {
  BIND( "POINT (-1.5 53.608273)"^^geo:wktLiteral AS ?akl )
  SERVICE qlss: {
    _:config qlss:algorithm qlss:s2 ;
              qlss:left ?akl ;
              qlss:right ?centroid ;
              qlss:numNearestNeighbors 10000 ;
              qlss:maxDistance 70000 ;
              qlss:bindDistance ?dist_left_right ;
              qlss:payload ?where, ?coords .
    {
      ?where lux:placeDefinedBy ?coords .
      BIND(geof:centroid(?coords) AS ?centroid)
    }
  }
}
"""
