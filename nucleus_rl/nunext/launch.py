"""Start one process per GPU on this machine without torchrun's TCP rendezvous.

    python -m nucleus_rl.nunext.launch --nproc 8 -- -m nucleus_rl.nunext.sft --output ... [args]

torchrun --standalone failed on Databricks Serverless GPU ("client socket has timed out ... (localhost, 0)"
while creating its rendezvous store). Here the processes meet through a file on local disk instead
(torch FileStore, NUNEXT_DIST_INIT=file://...), which needs no network. Each child gets RANK,
LOCAL_RANK, WORLD_SIZE and LOCAL_WORLD_SIZE. Rank 0's output is passed through; other ranks write
to log files whose tail is printed if they fail. If any rank fails, the others are stopped.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def tail(path: Path, lines: int = 60) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except FileNotFoundError:
        return ""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nproc", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- then the python arguments, e.g. -m module ...")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.nproc < 1 or not command:
        parser.error("need --nproc >= 1 and a command after --")

    workdir = Path(tempfile.mkdtemp(prefix="nunext-launch-"))
    store = workdir / "store"          # FileStore creates it; it must not exist yet
    print(f"[launch] {args.nproc} processes; rendezvous file {store}; rank logs in {workdir}", flush=True)
    processes, logs = [], []
    for rank in range(args.nproc):
        env = {**os.environ, "RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(args.nproc),
               "LOCAL_WORLD_SIZE": str(args.nproc), "NUNEXT_DIST_INIT": f"file://{store}"}
        env.setdefault("NCCL_SOCKET_IFNAME", "lo")    # single node: NCCL bootstrap over loopback only
        env.setdefault("GLOO_SOCKET_IFNAME", "lo")
        log = workdir / f"rank{rank}.log"
        logs.append(log)
        output = None if rank == 0 else log.open("w")
        processes.append(subprocess.Popen([sys.executable, *command], env=env, stdout=output,
                                          stderr=subprocess.STDOUT if output else None))
    code = 0
    try:
        while True:
            codes = [p.poll() for p in processes]
            failed = [(rank, c) for rank, c in enumerate(codes) if c not in (None, 0)]
            if failed:
                rank, code = failed[0]
                print(f"[launch] rank {rank} exited with code {code}; stopping the others", flush=True)
                if rank:
                    print(f"[launch] last lines of rank {rank}:\n{tail(logs[rank])}", flush=True)
                break
            if all(c == 0 for c in codes):
                break
            time.sleep(2)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        deadline = time.time() + 60
        for process in processes:
            try:
                process.wait(max(1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                process.kill()
    sys.exit(code)


if __name__ == "__main__":
    main()
