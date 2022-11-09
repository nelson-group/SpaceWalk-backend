"""CLI entrypoint."""


import typer

from tng_sv.api.download import download_snapshot


def main(simulation_name: str = "TNG50-4-Subbox2", snapshot_idx: int = 0) -> None:
    """Do something."""
    download_snapshot(simulation_name, snapshot_idx)


def cli() -> None:
    """Run the main function with typer."""
    typer.run(main)


if __name__ == "__main__":
    cli()
