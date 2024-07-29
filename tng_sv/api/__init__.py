"""API basic information."""
import os

BASEURL = "https://www.tng-project.org/api/"
HEADERS = {"api-key": os.getenv("TNG_TOKEN", "")}
