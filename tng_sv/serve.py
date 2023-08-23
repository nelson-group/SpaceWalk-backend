"""Webservice which provides preprocessing endpoint."""

import logging
import os
import tarfile
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import StreamingResponse

from tng_sv.api.download import download_webapp_groups, download_webapp_snapshot
from tng_sv.data.dir import get_webapp_base_path
from tng_sv.preprocessing import webapp

logger = logging.getLogger(__name__)

app = FastAPI()


def remove_file(path: str) -> None:
    """Remove a file."""
    logger.info("Removing file %(path)s", {"path": path})
    os.unlink(path)


@app.get("/v1/preprocess/{simulation_name}/{snap_id}")
async def preprocess_handler(
    background_tasks: BackgroundTasks, simulation_name: str, snap_id: int
) -> StreamingResponse:
    """Handle preprocess request."""
    name = Path(f"/tmp/{simulation_name}-{snap_id}.tar.gz")
    if not name.exists():
        download_webapp_snapshot(simulation_name, snap_id)
        download_webapp_groups(simulation_name, snap_id)
        download_webapp_snapshot(simulation_name, snap_id + 1)
        download_webapp_groups(simulation_name, snap_id + 1)
        webapp.preprocess_snap(simulation_name, snap_id)

        base_path = str(get_webapp_base_path(simulation_name))

        snapshot_n = 0
        snapdir_path = base_path + "/snapdir_" + str(snap_id + snapshot_n).zfill(3) + "/"
        logger.info("Generating tar %(name)s", {"name": name})
        with tarfile.open(str(name), mode="w:gz") as tar:
            tar.add(snapdir_path, recursive=True)

    def iter_file():
        with open(name, mode="rb") as _file:
            yield from _file

    logger.info("Streaming file %(name)s", {"name": name})
    background_tasks.add_task(remove_file, str(name))
    return StreamingResponse(iter_file(), media_type="application/tar+gzip")
