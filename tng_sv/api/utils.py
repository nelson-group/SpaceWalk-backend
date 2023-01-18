"""API utils."""


from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import requests
from requests import Response

from tng_sv.api import HEADERS


def get(path, params=None) -> Response:
    """Generic get wrapper to set correct headers."""
    # make HTTP GET request to path
    response = requests.get(path, params=params, headers=HEADERS, timeout=300)

    # raise exception if response code is not HTTP SUCCESS (200)
    response.raise_for_status()

    return response


def head(path, params=None) -> Response:
    """Generic get wrapper to set correct headers."""
    # make HTTP GET request to path
    response = requests.head(path, params=params, headers=HEADERS, timeout=300, allow_redirects=True)

    # raise exception if response code is not HTTP SUCCESS (200)
    response.raise_for_status()

    return response


def get_json(path, params=None) -> Dict[str, Any]:
    """Get json data.

    Raise TypeError if no json data is found.
    """
    response = get(path, params)
    if response.headers["content-type"] == "application/json":
        return response.json()  # parse json responses automatically
    raise TypeError("No json data found.")


def get_json_list(path, params=None) -> List[Dict[str, Any]]:
    """Yield before pyright."""
    return cast(List[Dict[str, Any]], get_json(path, params=params))


def get_file(  # pylint: disable=too-many-arguments
    path: str,
    pre_dir: Path = Path(""),
    pre_fix: str = "",
    post_fix: str = "",
    params=None,
    override_filename: Optional[str] = None,
) -> str:
    """Get file data.

    Saves a file to the local filesystem and returns the filename.
    Raise TypeError if no file is in response.
    """
    response = head(path, params)
    print(f"Downloading: {response.headers['content-length']} bytes")
    response = get(path, params)

    if "content-disposition" in response.headers:
        filename: str = override_filename or response.headers["content-disposition"].split("filename=")[1]
        filename = str(pre_dir.joinpath(pre_fix + filename + post_fix))
        with open(filename, "wb") as _file:
            _file.write(response.content)
        return filename  # return the filename string
    raise TypeError("No json data found.")


def get_index(data: List[Dict[str, Any]], key: str, value: str) -> Tuple[int, Dict[str, Any]]:
    """Find object with value in given key."""
    values = [element[key] for element in data]
    idx = values.index(value)
    return idx, data[idx]
