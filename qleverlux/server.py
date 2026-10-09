"""Running the middle tier.

Three ways to start it, sharing one set of server settings:

* ``serve_dev()``    - uvicorn, plain HTTP, single process. The dev loop.
* ``serve_https()``  - hypercorn with HTTP/2 and TLS, single process.
* ``run_workers()``  - hypercorn, multiple worker processes.

Connection pools are opened by the application's lifespan handler, so none of
these has to remember to do it.
"""

from __future__ import annotations

import asyncio

import uvloop

from qleverlux.settings import load_settings

#: Port the development server listens on.
DEV_PORT = 5001


def _hypercorn_config(settings, *, tls=True, workers=None):
    from hypercorn.config import Config as HyperConfig

    hconfig = HyperConfig()
    hconfig.bind = [f"{settings.mthost}:{settings.mtport}"]
    hconfig.loglevel = settings.log_level
    hconfig.accesslog = "-"
    hconfig.errorlog = "-"
    if tls:
        hconfig.certfile = f"files/{settings.cert_name}.pem"
        hconfig.keyfile = f"files/{settings.cert_name}-key.pem"
    hconfig.queue_size = settings.queue_size
    hconfig.backlog = settings.backlog
    hconfig.read_timeout = settings.read_timeout
    hconfig.max_app_queue_size = settings.max_app_queue_size
    if workers is not None:
        hconfig.workers = workers
        hconfig.worker_class = "uvloop"
        hconfig.reload = False
        hconfig.application_path = "qleverlux.app:app"
    return hconfig


def serve_dev(settings=None, port=DEV_PORT):
    """uvicorn over plain HTTP, for development."""
    import uvicorn

    from qleverlux.app import app

    settings = settings if settings is not None else load_settings()

    async def main():
        uvloop.install()
        config = uvicorn.Config(app)
        config.host = settings.mthost
        config.port = port
        server = uvicorn.Server(config)
        await server.serve()

    print(f"Starting uvicorn http server on {settings.mthost}:{port} ...")
    asyncio.run(main())


def serve_https(settings=None):
    """hypercorn with HTTP/2 and TLS, single process."""
    from hypercorn.asyncio import serve as hypercorn_serve

    from qleverlux.app import app

    settings = settings if settings is not None else load_settings()
    hconfig = _hypercorn_config(settings)

    async def main():
        uvloop.install()
        await hypercorn_serve(app, hconfig)

    print("Starting hypercorn https/2 server...")
    asyncio.run(main())


def run_workers(settings=None):
    """hypercorn with several worker processes."""
    from hypercorn.run import run as hypercorn_run

    settings = settings if settings is not None else load_settings()
    uvloop.install()
    print(f"Starting hypercorn with {settings.workers} workers...")
    hypercorn_run(_hypercorn_config(settings, workers=settings.workers))


def main(argv=None):
    """``python -m qleverlux.server [--dev | --workers]``"""
    import sys

    argv = sys.argv[1:] if argv is None else argv
    settings = load_settings()
    if "--dev" in argv:
        serve_dev(settings)
    elif "--workers" in argv:
        run_workers(settings)
    else:
        serve_https(settings)


if __name__ == "__main__":
    main()
