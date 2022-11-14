"""CLI entrypoint."""


import typer

from tng_sv.api.download import download_snapshot
from tng_sv.data.utils import combine_snapshot
from tng_sv.preprocessing import run_delaunay

app = typer.Typer()


@app.command()
def download(simulation_name: str = "TNG50-4-Subbox2", snapshot_idx: int = 0) -> None:
    """Download a snapshot."""
    download_snapshot(simulation_name, snapshot_idx)


@app.command()
def combine(simulation_name: str = "TNG50-4-Subbox2", snapshot_idx: int = 0) -> None:
    """Combine a snapshot."""
    combine_snapshot(simulation_name, snapshot_idx)


@app.command()
def delaunay(simulation_name: str = "TNG50-4-Subbox2", snapshot_idx: int = 0) -> None:
    """Download a snapshot."""
    run_delaunay(simulation_name, snapshot_idx)


def cli() -> int:
    """Run the main function with typer."""
    app()
    return 0


if __name__ == "__main__":
    cli()
