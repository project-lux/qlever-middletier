import os
import sys
import gzip

import ujson as json
import psycopg2
from psycopg2.extras import RealDictCursor, Json

conn = psycopg2.connect(user="ubuntu", dbname="ubuntu")
table = "lux_data_cache"


def make_cursor():
    return conn.cursor(cursor_factory=RealDictCursor)


def make_table():
    qry = f"""CREATE TABLE public.lux_data_cache (
        identifier uuid PRIMARY KEY,
        data jsonb NOT NULL);"""

    with make_cursor() as cursor:
        try:
            cursor.execute(qry)
            conn.commit()
        except Exception as e:
            print(f"Make table failed: {e}")
            conn.rollback()


def write_record(data, identifier, cursor):
    jdata = Json(data)
    qnames = ["data", "identifier"]
    qvals = (jdata, identifier)
    qd = dict(zip(qnames, qvals))
    qps = [qn for qn in qnames if qd[qn] is not None]
    qvs = tuple([qv for qv in qvals if qv is not None])
    pholders = ",".join(["%s"] * len(qps))
    qpstr = ",".join(qps)

    try:
        qry = f"""INSERT INTO {table} ({qpstr}) VALUES ({pholders})"""
        cursor.execute(qry, qvs)
    except Exception as e:
        # Could be a psycopg2.errors.UniqueViolation if we're trying to insert without delete
        # logger.critical(f"DATA: {data}")
        print(f"Failed to upsert!: {e}?\n{qpstr} = {qvs}")


if "--init" in sys.argv:
    make_table()
    print("Made table")
    sys.exit(0)


my_slice = int(sys.argv[1])
max_slice = 12
files = [x for x in os.listdir(".") if x.endswith(".jsonl.gz")]
use_gzip = True
if not files:
    files = [x for x in os.listdir(".") if x.endswith(".jsonl")]
    use_gzip = False
files.sort()

for f in files[my_slice::max_slice]:
    print(f)
    if use_gzip:
        opener = gzip.open
    else:
        opener = open

    with opener(f) as fh:
        x = 0
        for line in fh:
            cursor = make_cursor()
            js = json.loads(line)
            data = js["json"]
            ident = data["id"].rsplit("/", 1)[-1]
            write_record(data, ident, cursor)
            x += 1
            if not x % 5000:
                conn.commit()
                print(x)
conn.commit()
conn.close()
