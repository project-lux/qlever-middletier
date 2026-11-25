"""Configuration that reaches past the middle tier's own settings.

* ``.env`` was loaded into ``os.environ`` with ``override=True``, so every variable in it, ours or
  not, replaced the real one for the whole process: an ``HTTPS_PROXY`` or ``SSL_CERT_FILE`` in a
  ``.env`` where a tool was run redirected its outbound calls. Now only ``QLMT_`` settings are read
  from it, into ``Settings`` alone - still ahead of the real environment, as before.
* PostgreSQL's conninfo was a pasted-together string, so a password with a space or quote broke it
  or added parameters; and ``QLMT_PGSSLMODE`` ("require") was never passed, leaving libpq's
  downgradable "prefer".
"""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHILD = """
import json, os
from qleverlux.settings import load_settings
s = load_settings([])
print(json.dumps({"pghost": s.pghost, "pgport": s.pgport, "max": s.max_qlever_requests,
                  "proxy": os.environ.get("HTTPS_PROXY"), "other": os.environ.get("NOT_OURS"),
                  "pghost_env": os.environ.get("QLMT_PGHOST")}))
"""


def settings_in(directory, dotenv, **env):
    (directory / ".env").write_text(dotenv)
    child_env = {k: v for k, v in os.environ.items() if not k.startswith("QLMT_")}
    child_env.pop("NOT_OURS", None)
    child_env.update(env, PYTHONPATH=os.pathsep.join(filter(None, [ROOT, os.environ.get("PYTHONPATH")])))
    out = subprocess.run([sys.executable, "-c", CHILD], cwd=directory, env=child_env,
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_dotenv_variables_that_are_not_settings_stay_out_of_the_environment(tmp_path):
    got = settings_in(tmp_path, "NOT_OURS=1\nHTTPS_PROXY=http://attacker.invalid:1\n",
                      HTTPS_PROXY="http://real-proxy:3128")
    assert got["proxy"] == "http://real-proxy:3128"
    assert got["other"] is None


def test_dotenv_settings_are_read_and_still_win_over_the_environment(tmp_path):
    got = settings_in(tmp_path, "QLMT_PGHOST=from-dotenv\nQLMT_MAX_OPEN_REQUESTS=7\n",
                      QLMT_PGHOST="from-env", QLMT_PGPORT="6543")
    assert (got["pghost"], got["max"], got["pgport"]) == ("from-dotenv", 7, 6543)
    # read into the settings, not written back into the process environment
    assert got["pghost_env"] == "from-env"


# -- PostgreSQL ----------------------------------------------------------------

psycopg = pytest.importorskip("psycopg")


def pg_settings(**overrides):
    base = dict(pguser="lux", pgdb="lux", pghost="", pgport=5432, pgpass="", pgsslmode="require",
                use_pg_data_cache=True, use_pg_hal_cache=False)
    return SimpleNamespace(**{**base, **overrides})


def connect_args(**overrides):
    from qleverlux.clients.postgres import PostgresCache

    return PostgresCache(pg_settings(**overrides)).connect_args()


def test_sslmode_is_applied_over_tcp():
    assert connect_args(pghost="db.internal")["sslmode"] == "require"


def test_an_empty_sslmode_leaves_libpq_to_decide():
    assert "sslmode" not in connect_args(pghost="db.internal", pgsslmode="")


def test_a_unix_socket_connection_is_as_before():
    assert connect_args() == {"user": "lux", "dbname": "lux"}


def test_a_password_cannot_add_connection_parameters():
    password = "p w' host=attacker.invalid sslmode=disable"
    args = connect_args(pghost="db.internal", pgpass=password)
    parsed = psycopg.conninfo.conninfo_to_dict(psycopg.conninfo.make_conninfo(**args))
    assert parsed["password"] == password
    assert parsed["host"] == "db.internal"
    assert parsed["sslmode"] == "require"
