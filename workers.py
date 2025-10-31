import sys
import uvloop
from hypercorn.config import Config as HyperConfig
from hypercorn.run import run as hypercorn_run
import qleverlux.middletier
from qleverlux.middletier import QLeverLuxMiddleTier

# This gets called once per worker process
mt = QLeverLuxMiddleTier()
qleverlux.middletier.mt = mt

if __name__ == "__main__":
    # Whereas this gets called exactly once
    mt_config = mt.config
    uvloop.install()
    hconfig = HyperConfig()
    hconfig.bind = [f"0.0.0.0:{mt_config.mtport}"]
    hconfig.loglevel = mt_config.log_level
    hconfig.accesslog = "-"
    hconfig.errorlog = "-"
    hconfig.certfile = f"files/{mt_config.cert_name}.pem"
    hconfig.keyfile = f"files/{mt_config.cert_name}-key.pem"
    hconfig.queue_size = mt_config.queue_size
    hconfig.backlog = mt_config.backlog
    hconfig.read_timeout = mt_config.read_timeout
    hconfig.max_app_queue_size = mt_config.max_app_queue_size
    hconfig.workers = mt_config.workers
    hconfig.worker_class = "uvloop"
    hconfig.reload = False
    me = sys.argv[0].replace("./", "").replace(".py", "")
    hconfig.application_path = "qleverlux.middletier:app"
    hypercorn_run(hconfig)
