import asyncio
import uvloop
from hypercorn.config import Config as HyperConfig
from hypercorn.asyncio import serve as hypercorn_serve
import qleverlux.middletier
from qleverlux.middletier import QLeverLuxMiddleTier, app


async def main(mt):
    mt_config = mt.config
    uvloop.install()
    hconfig = HyperConfig()
    hconfig.bind = [f"0.0.0.0:{mt_config.mtport}"]
    hconfig.loglevel = mt_config.log_level
    hconfig.accesslog = "-"
    hconfig.errorlog = "-"
    hconfig.certfile = f"files/{mt_config.cert_name}.pem"
    hconfig.keyfile = f"files/{mt_config.cert_name}-key.pem"
    hconfig.queue_size = 256
    hconfig.backlog = 2048
    hconfig.read_timeout = 120
    hconfig.max_app_queue_size = 256
    await mt.connect_to_postgres()
    mt.connect_to_qlever()
    await hypercorn_serve(app, hconfig)


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting hypercorn https/2 server...")
    mt = QLeverLuxMiddleTier()
    qleverlux.middletier.mt = mt
    asyncio.run(main(mt))
