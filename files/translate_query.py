import os
import re
import json

templates = {}
queries = {}
query_by_scope = {}

with open("builder.js") as fh:
    chunk = []
    acc = False
    qbs = False
    for line in fh:
        line = line.strip()

        if line.startswith("const keyFuncNameMap"):
            qbs = True
        elif qbs and line.endswith("{"):
            scope = line.split(":")[0].strip()
            query_by_scope[scope] = {}
            qscope = query_by_scope[scope]
        elif qbs and line.startswith('"lux:'):
            try:
                name, q = line.split(": ")
            except Exception:
                print(line)
                continue
            name = name[1:-1]
            q = q.split(".")[1]
            qscope[name] = q

        if line.startswith("const queryBuilders"):
            # start reading
            acc = True
            qbs = False
        elif acc and len(chunk) < 4:
            chunk.append(line)
        elif acc:
            # process chunk and start new one
            if chunk[0] == "}":
                break
            print(chunk)
            key = f"lux:{chunk[0].split(':')[1].strip()[:-1]}"
            if "queries." in chunk[1]:
                qname = chunk[1].split("queries.")[1].split("(")[0].strip()
            else:
                qname = "-"
            href = chunk[2].replace("return `$", "'")
            href = href.replace("config.", "").replace(";", "")
            href = href.replace("`", "'").replace("${", "{").replace("{idEnc}", "{id}")[1:-1]
            chunk = [line]
            templates[key] = href
            queries[key] = qname

with open("hal_link_templates.json", "w") as outh:
    outh.write(json.dumps(templates, indent=2))
with open("hal_link_queries.json", "w") as outh:
    outh.write(json.dumps(queries, indent=2))
with open("query_by_scope.json", "w") as outh:
    outh.write(json.dumps(query_by_scope, indent=2))

cre = re.compile(" = \((.+?)\) =>")
keyre = re.compile("([a-zA-Z0-9_]+):")
files = os.listdir("queries")
files.sort()
for f in files:
    if f.endswith(".js"):
        print(f)
        with open(f"queries/{f}") as fh:
            variable = None
            qlines = ["{"]
            for line in fh:
                line = line.strip()
                if line.startswith("const "):
                    # beginning of query
                    m = cre.search(line)
                    if m:
                        variable = m.groups()[0]
                    else:
                        print(f"Failed to match regex on: {line}")
                        break
                elif line in ["})", "});"]:
                    qlines.append("}")
                    query = "".join(qlines)
                    try:
                        js = eval(f"{query}")
                    except Exception as e:
                        print(f"Failed to evaluate query {f}: {query}")
                        print(f"Error: {e}")
                        break

                    with open(f"queries/{f}on", "w") as outh:
                        outh.write(json.dumps(js, indent=2))

                elif variable is not None:
                    # middle of query
                    m = keyre.search(line)
                    if m:
                        key = m.groups()[0]
                        line = line.replace(key, f'"{key}"')
                    if variable in line:
                        line = line.replace(variable, '"URI-HERE"')

                    qlines.append(line)
