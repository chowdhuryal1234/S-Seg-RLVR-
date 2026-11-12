"""Run a command on one chosen GPU: python -m nucleus_rl.nunext.on_gpu --gpu 3 -- -m nucleus_rl.nunext.grpo ...

Sets CUDA_VISIBLE_DEVICES before the child imports torch, then replaces this process with it, so several
single-GPU runs (for example GRPO arms and seeds) can share one 8-GPU machine.
"""
from __future__ import annotations

import argparse
import os
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, help="GPU index (or comma-separated indices) to expose")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- then the python arguments")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("need a command after --")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.stdout.flush()
    os.execv(sys.executable, [sys.executable, *command])


if __name__ == "__main__":
    main()
