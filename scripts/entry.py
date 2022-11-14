"""Entrypoint in docker container."""
# pylint: disable=import-error,import-outside-toplevel,wrong-import-position
import os
import re
import sys

user = os.path.expanduser("~/")
sys.path.append(user + ".local/lib/python3.10/site-packages")

from tng_sv.cli import cli  # noqa: E402

if __name__ == "__main__":
    sys.argv[0] = re.sub(r"(-script\.pyw|\.exe)?$", "", sys.argv[0])
    sys.exit(cli())
