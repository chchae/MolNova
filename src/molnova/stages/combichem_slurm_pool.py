"""One Slurm node step supplies pure calculators; no SQLite or task claims."""
import hmac
import json
import multiprocessing
import os
from pathlib import Path
import signal
import socket
import sys
import time


def calculate(listener, root, token):
    from molnova.stages.combichem_compute import serve
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    while not (root/'stop').exists():
        try:
            connection, _ = listener.accept()
        except socket.timeout:
            continue
        with connection:
            connection.settimeout(30)
            with connection.makefile('rw', encoding='utf-8') as stream:
                identity = stream.readline(4096).strip()
                if not hmac.compare_digest(identity, token):
                    continue
                connection.settimeout(None)
                try:
                    serve(stream, stream)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # Disconnected claim owners retain their own journal/cursor.


def main(root, slots):
    from molnova.stages.combichem_slurm import atomic_save
    from molnova.chemistry.combichem3d.output import versions
    host = os.environ['SLURMD_NODENAME']
    root = Path(root)/host
    token = (root/'token').read_text().strip()
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('', 0)); listener.listen(slots); listener.settimeout(1.)
        workers = [multiprocessing.get_context('fork').Process(target=calculate, args=(listener,root,token))
                   for _ in range(slots)]
        for worker in workers:
            worker.start()
        atomic_save(root/'endpoint.json', dict(host=host, port=listener.getsockname()[1],
                    job_id=os.environ['SLURM_JOB_ID'], slots=slots, versions=versions()))
        failed = False
        try:
            while not stopping and not (root/'stop').exists():
                if any(worker.exitcode is not None for worker in workers):
                    failed = True
                    break
                time.sleep(.2)
        finally:
            (root/'stop').touch()
            deadline = time.monotonic()+10
            for worker in workers:
                worker.join(max(0, deadline-time.monotonic()))
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                worker.join()
            if failed or any(worker.exitcode!=0 for worker in workers):
                raise RuntimeError('Slurm pure-calculator pool stopped unexpectedly')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
