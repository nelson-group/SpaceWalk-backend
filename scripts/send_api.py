import requests
from pprint import pprint as pp

client_state = {
    "node_indices": [],
    "level_of_detail": {},
    "batch_size_lod": 500,
    "camera_information": {
        "x": 25000,
        "y": 25000,
        "z": 25000,
        "size": 500,
    }
}

simulation = "test"
snap_id = "075"
response = requests.post(f"http://127.0.0.1:8000/v1/get/splines/{simulation}/{snap_id}", json=client_state)

pp(response.status_code)
pp(response.json())
response.raise_for_status()
