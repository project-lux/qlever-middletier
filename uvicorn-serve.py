import asyncio
import uvloop
import uvicorn

import qleverlux.middletier
from qleverlux.middletier import QLeverLuxMiddleTier, app


async def main(mt):
    uvloop.install()
    config = uvicorn.Config(app)
    config.host = "0.0.0.0"
    config.port = 5001
    mt.start()
    server = uvicorn.Server(config)
    await server.serve()


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting hypercorn https/2 server...")
    mt = QLeverLuxMiddleTier()
    qleverlux.middletier.mt = mt
    asyncio.run(main(mt))
