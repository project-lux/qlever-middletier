"""Middle tier settings: environment variables, then CLI overrides.

Every setting is declared exactly once, as a model field. Its environment
variable is ``QLMT_`` + the field name unless the field carries an explicit
alias, and its command-line flag is the field name with underscores turned into
hyphens unless it is listed in ``NO_CLI``. Precedence is env (including
``.env``, which wins over the real environment) first, then any flag actually
passed on the command line.

The old version declared each setting twice - once reading ``os.getenv`` and
once as an ``argparse`` argument whose default was the env value - which is
where the ``mt_queue_size`` / ``queue_size`` shadowing came from. There is now
one name per setting: the short one the rest of the code already used.

Unknown flags are tolerated (uvicorn's, hypercorn's) and kept in
``remaining_args``.
"""

from __future__ import annotations

import argparse
import os
import sys
from getpass import getuser
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BeforeValidator, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if "--pgpass" in sys.argv:
    print("Don't put passwords in the command line, as they're visible via ps")
    print("Instead use the .env file with QLMT_PGPASS")
    sys.exit(1)

loaded = load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"), override=True)
if not loaded:
    print(f"Warning: No .env file found in {os.getcwd()}")


def _strict_bool(v):
    """Only the exact string "true" is true, matching the original getenv checks."""
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() == "true"


Bool = Annotated[bool, BeforeValidator(_strict_bool)]

#: Settings with no command-line flag: secrets, and knobs only ever set by env.
NO_CLI = frozenset(
    {
        "pgpass",
        "pgsslmode",
        "pgtable_hal",
        "lmdb_binary_uuid_keys",
        "lmdb_key_format",
        "lmdb_json_path",
        "facet_page_length",
    }
)

#: Flags whose spelling does not follow from the field name.
CLI_ALIASES = {"use_pg_hal_cache": "--use-postgres-hal-cache"}


class Settings(BaseSettings):
    """Every knob the middle tier has. Pure data - no I/O beyond reading env."""

    model_config = SettingsConfigDict(env_prefix="QLMT_", extra="ignore")

    # -- where config and queries live
    config_path: str = "config"
    queries_path: str = "queries"
    hal_cache_path: str = "hal_cache"
    cms_path: str = "cms"
    search_config: str = ""

    # -- PostgreSQL record / HAL cache
    pguser: str = Field(default_factory=getuser)
    pgpass: str = ""
    pghost: str = ""
    pgport: int = 5432
    pgdb: str = Field(default_factory=getuser)
    pgtable: str = "lux_data_cache"
    pgsslmode: str = "require"
    pgtable_hal: str = "hal_data_cache"
    use_pg_data_cache: Bool = False

    # -- LMDB record cache
    lmdb_path: str = ""
    use_lmdb_data_cache: Bool = False
    lmdb_binary_uuid_keys: Bool = True
    #: "uuid" (16 raw bytes), "text" (the identifier as UTF-8), or "qid" (a
    #: Wikidata Q number plus the store's character for the record's type).
    #: Empty falls back to lmdb_binary_uuid_keys: true is uuid, false text.
    lmdb_key_format: str = ""
    lmdb_json_path: str = ""

    # -- QLever SPARQL endpoint
    qlproto: str = "http"
    qlhost: str = "localhost"
    qlport: int = 7010
    qlpath: str = "sparql"
    qlever_timeout: int = 30
    max_qlever_connections: int = 20
    max_qlever_requests: int = Field(default=64, alias="QLMT_MAX_OPEN_REQUESTS")
    use_httpx: Bool = Field(default=True, alias="QLMT_USEHTTPX")

    # -- where the middle tier listens
    mthost: str = "0.0.0.0"
    mtport: int = 5000
    mtproto: str = "https"
    mtpath: str = ""
    cert_name: str = Field(default="qleverlux", alias="QLMT_CERTNAME")
    log_level: str = Field(default="info", alias="QLMT_LOGLEVEL")

    # -- URI rewriting: data URIs in, this deployment's URIs out
    data_uri: str = Field(
        default="https://lux.collections.yale.edu/", alias="QLMT_DATAURI"
    )
    replace_proto: str = "https"
    replace_host: str = Field(
        default="qleverlux.collections.yale.edu", alias="QLMT_EXTERNAL_HOST"
    )
    replace_port: int = Field(default=-1, alias="QLMT_EXTERNAL_PORT")
    replace_path: str = Field(default="", alias="QLMT_EXTERNAL_PATH")
    #: A record's URI in the data, after data_uri. ``{class}`` is the record
    #: class from the URL path and ``{id}`` its identifier. Data whose URIs
    #: carry no class (Wikidata's, "{id}") get it from the record's type.
    record_path: str = "data/{class}/{id}"

    # -- search behaviour
    page_length: int = Field(default=20, alias="QLMT_PAGELENGTH")
    facet_page_length: int = Field(default=20, alias="QLMT_FACET_PAGELENGTH")
    facet_delay: int = 0
    portal: str = ""
    use_stopwords: Bool = Field(default=True, alias="QLMT_USESTOPWORDS")

    # -- HAL link cache
    use_pg_hal_cache: Bool = Field(
        default=False, alias="QLMT_USE_POSTGRES_HAL_CACHE"
    )
    use_disk_hal_cache: Bool = True

    # -- AI query translation
    ai_translate_enabled: Bool = Field(default=True, alias="QLMT_AI_TRANSLATE")
    #: "gemini", "lmstudio" or "openai". Empty infers from the other settings:
    #: a gemini model name, else an endpoint means openai, else disabled.
    ai_translate_backend: str = ""
    ai_translate_model: str = ""
    ai_translate_project: str = ""
    ai_translate_region: str = "global"
    ai_translate_api_key: str = ""
    #: Base URL for the openai backend, or host:port for lmstudio.
    ai_translate_api_endpoint: str = ""
    ai_thinking_budget: int = 2000
    #: Sampling settings, shared by every backend.
    ai_translate_temperature: float = 0.8
    ai_translate_top_p: float = 0.95
    ai_translate_max_tokens: int = 36000
    #: Ask the backend to constrain output to JSON. Not every OpenAI-compatible
    #: server supports it, so the backend retries without it if it is rejected.
    ai_translate_json_mode: Bool = True
    ai_translate_timeout: int = 120

    # -- server tuning
    backlog: int = 512
    queue_size: int = 128
    max_app_queue_size: int = Field(default=128, alias="QLMT_APP_QUEUE_SIZE")
    workers: int = 4
    read_timeout: int = 30

    #: Flags this process did not recognise, left for uvicorn/hypercorn.
    remaining_args: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _derive(self):
        if not self.lmdb_path:
            self.use_lmdb_data_cache = False
        if not self.lmdb_key_format:
            self.lmdb_key_format = "uuid" if self.lmdb_binary_uuid_keys else "text"
        if self.lmdb_key_format not in ("uuid", "text", "qid"):
            raise ValueError(
                f"QLMT_LMDB_KEY_FORMAT must be uuid, text or qid, not {self.lmdb_key_format!r}"
            )
        if "{id}" not in self.record_path:
            raise ValueError(f"QLMT_RECORD_PATH has no {{id}}: {self.record_path!r}")
        return self

    # -- computed ---------------------------------------------------------

    @property
    def _port_suffix(self) -> str:
        """An upstream load balancer may publish a different port."""
        return f":{self.replace_port}" if self.replace_port > 0 else ""

    @property
    def mt_uri(self) -> str:
        """This deployment's own base URI, what responses must use."""
        return (
            f"{self.replace_proto}://{self.replace_host}"
            f"{self._port_suffix}{self.replace_path}"
        )

    @property
    def sparql_endpoint(self) -> str:
        port = f":{self.qlport}" if self.qlport > 0 else ""
        return f"{self.qlproto}://{self.qlhost}{port}/{self.qlpath}"

    # -- reporting --------------------------------------------------------

    def print_config(self):
        print("---- Middletier Configuration ----")
        print()
        print(f"Postgres:       {self.pghost or 'localhost'}:{self.pgport}/{self.pgdb}")
        print(f"QLever:         {self.sparql_endpoint}")
        print(f"LMDB:           {self.lmdb_path} ({self.lmdb_key_format} keys)")
        print(f"Use HTTPX:      {self.use_httpx}")
        print()
        print(f"Postgres HAL:   {self.use_pg_hal_cache}")
        print(f"Disk HAL:       {self.use_disk_hal_cache}")
        print(f"Internal URI:   {self.data_uri}")
        print(f"External URI:   {self.mt_uri}")
        print(f"Record path:    {self.record_path}")
        print()
        print(f"Queue Size:     {self.queue_size}")
        print(f"App Queue Size: {self.max_app_queue_size}")
        print(f"Backlog size:   {self.backlog}")
        print(f"Read timeout:   {self.read_timeout}")
        print(f"SPARQL timeout: {self.qlever_timeout}")
        print(f"Workers:        {self.workers}")
        print(f"Max QLever conns:{self.max_qlever_connections}")
        print(f"Max QLever reqs: {self.max_qlever_requests}")
        if self.portal:
            print(f"Portal:         {self.portal}")
        print()


def build_parser() -> argparse.ArgumentParser:
    """An argparse parser derived from the model fields.

    Defaults are suppressed, so a flag that is not passed leaves the value from
    the environment alone. Booleans get ``--flag`` / ``--no-flag``, which the
    hand-written ``store_true`` arguments could not express.
    """
    parser = argparse.ArgumentParser(prog="qleverlux")
    for name, field in Settings.model_fields.items():
        if name in NO_CLI or name == "remaining_args":
            continue
        flag = CLI_ALIASES.get(name, "--" + name.replace("_", "-"))
        help_text = f"{name} (env: QLMT_{name.upper()})"
        if field.annotation is bool or name.startswith("use_"):
            parser.add_argument(
                flag,
                dest=name,
                action=argparse.BooleanOptionalAction,
                default=argparse.SUPPRESS,
                help=help_text,
            )
        else:
            parser.add_argument(
                flag,
                dest=name,
                type=(
                    field.annotation
                    if field.annotation in (int, float, str)
                    else str
                ),
                default=argparse.SUPPRESS,
                help=help_text,
            )
    return parser


def load_settings(argv=None) -> Settings:
    """Environment (and ``.env``) first, then whatever flags were passed."""
    settings = Settings()
    args, rest = build_parser().parse_known_args(argv)
    for key, value in vars(args).items():
        setattr(settings, key, value)
    settings.remaining_args = rest
    if not settings.lmdb_path:
        settings.use_lmdb_data_cache = False
    return settings
