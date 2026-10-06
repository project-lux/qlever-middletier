"""The FastAPI application and the object graph behind it.

``MiddleTier`` owns one instance of everything: the settings, the compiled
query catalogue, the outbound clients and the services built on top of them.
It is constructed by the lifespan handler, so connection pools are opened
inside the running event loop, and it is reachable from a route through
``Depends(get_middletier)`` rather than a module-level global.

``app`` is importable without building any of it - constructing the catalogue
means reading ~90 query files and compiling them to SPARQL, which happens when
the server starts, not when the module is imported.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from qleverlux.api import router as api_router
from qleverlux.clients.ai import build_ai_backend
from qleverlux.clients.postgres import PostgresCache
from qleverlux.clients.qlever import QLeverClient
from qleverlux.clients.record_cache import HalCache, RecordCache
from qleverlux.errors import install_error_handlers
from qleverlux.presentation.uris import UriRewriter
from qleverlux.query.catalogue import QueryCatalogue
from qleverlux.services.ai_translate import AiTranslateService
from qleverlux.services.cms import CmsService
from qleverlux.services.facets import FacetService
from qleverlux.services.hal import HalService
from qleverlux.services.records import RecordService
from qleverlux.services.related import RelatedListService
from qleverlux.services.search import SearchService
from qleverlux.services.stats import StatsService
from qleverlux.settings import load_settings


class MiddleTier:
    """Everything the API needs, wired together once."""

    def __init__(self, settings=None, catalogue=None):
        self.settings = settings if settings is not None else load_settings()
        self.settings.print_config()
        self.catalogue = (
            catalogue if catalogue is not None else QueryCatalogue(self.settings)
        )

        self.uris = UriRewriter(self.settings.data_uri, self.settings.mt_uri)
        self.qlever = QLeverClient(self.settings)
        self.postgres = PostgresCache(self.settings)
        self.record_cache = RecordCache(self.settings, self.postgres)
        self.hal_cache = HalCache(self.settings, self.postgres)
        self.ai_backend = build_ai_backend(self.settings, self.catalogue)

        common = (self.settings, self.catalogue, self.qlever, self.uris)
        self.search = SearchService(*common)
        self.facets = FacetService(*common)
        self.related = RelatedListService(*common)
        self.stats = StatsService(self.settings, self.catalogue, self.qlever)
        self.hal = HalService(
            self.settings, self.catalogue, self.qlever, self.hal_cache
        )
        self.records = RecordService(
            self.settings, self.record_cache, self.hal, self.uris
        )
        self.ai = AiTranslateService(self.settings, self.catalogue, self.ai_backend)
        self.cms = CmsService(self.settings)

    async def start(self):
        """Open the connection pools. Must run inside the event loop."""
        self.qlever.connect()
        await self.record_cache.connect()

    async def aclose(self):
        await self.qlever.aclose()
        await self.record_cache.aclose()
        self.ai_backend.close()


def create_app(middletier=None) -> FastAPI:
    """Build the application. ``middletier`` is for tests that supply their own."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        mt = middletier if middletier is not None else MiddleTier()
        app.state.mt = mt
        await mt.start()
        try:
            yield
        finally:
            await mt.aclose()

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router)
    install_error_handlers(app)
    return app


#: The ASGI application. ``hypercorn qleverlux.app:app``
app = create_app()
