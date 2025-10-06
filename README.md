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




## Query Optimization notes


### UNION vs Property Path OR

UNION and property path | expressions are equivalent in timing

```
SELECT (COUNT(*) AS ?count)
WHERE {
   {
      SELECT ?uri
      WHERE       {
         ?uri a lux:Item .
         {
            ?uri lux:agentOfItemBeginning <https://lux.collections.yale.edu/data/person/03ed33fa-1b1c-4fe0-9cba-614698dc94cd> .
         }
         UNION
         {
            ?uri lux:agentOfItemEncounter <https://lux.collections.yale.edu/data/person/03ed33fa-1b1c-4fe0-9cba-614698dc94cd> .
         }
         UNION
         {
            ?uri lux:agentInfluenceOfItemBeginning <https://lux.collections.yale.edu/data/person/03ed33fa-1b1c-4fe0-9cba-614698dc94cd> .
         }
      }
      GROUP BY ?uri   }
}
```

is the same timing as the slightly tidier but harder to generate from the LUX JSON syntax:

```
SELECT (COUNT(*) AS ?count)
WHERE {
   {
      SELECT ?uri
      WHERE       {
         ?uri a lux:Item ;
			lux:agentOfItemBeginning | lux:agentOfItemEncounter | lux:agentInfluenceOfItemBeginning
			<https://lux.collections.yale.edu/data/person/03ed33fa-1b1c-4fe0-9cba-614698dc94cd> .
      }
      GROUP BY ?uri   }
}
```

(98ms cold to count 78,322 objects, 66ms cold to generate first 100 entries)



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
