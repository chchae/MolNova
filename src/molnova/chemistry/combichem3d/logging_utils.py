"""Flushed console and append-only run logging, inherited by Linux workers."""
import logging
from pathlib import Path
import sys
import time

logger = logging.getLogger('molnova.combichem3d')
TRACE = 5
logging.addLevelName(TRACE,'TRACE')


def trace(message,*args):
    logger.log(TRACE,message,*args)


def configure(output, level='INFO'):
    # Repeated CLI calls in the same interpreter must not duplicate handlers.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(TRACE if level=='TRACE' else level)
    logger.propagate = False
    root=Path(output)
    root.mkdir(parents=True,exist_ok=True)
    formatter=logging.Formatter('%(asctime)s [combichem-3d] %(levelname)s pid=%(process)d %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
    for handler in (logging.StreamHandler(sys.stdout),logging.FileHandler(root/'run.log',mode='a',encoding='utf-8')):
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def progress(info, iteration, workers):
    remaining=max(0.,(info.get('deadline') or 0)-time.time())
    logger.info('progress iteration=%s attempted=%s unique=%s remaining=%.1fs',
                iteration,info['attempted'],info['unique_count'],remaining)
