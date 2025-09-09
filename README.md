# SpaceWalk - Backend

This CLI can be used to preprocess the data and get the proper data and format for the backend. This process downloads the data, preprocesses the data, saves the needed particle properties and generates the corresponding octree.

## Preliminary

Download this repo
Python >= 3.10

## Setup

```
make venv
source venv/bin/activate # to activate the virtual environment
```

## CLI Usage Download and Preprocess

```
export TNG_TOKEN="..."

tng-sv-cli web download --simulation-name TNG50-4 --snapshot-idx NR
tng-sv-cli web preprocess --simulation-name TNG50-4 --snapshot-idx NR

tng-sv-cli web batch-download --simulation-name TNG50-4 --snapshot-idx NR
tng-sv-cli web batch-preprocess --simulation-name TNG50-4 --snapshot-idx NR
```

## Dev Python Backend

```
PYTHONPATH=. fastapi dev webScripts/api/backend.py --host 0.0.0.0 --port 9999
```

## Frontend

Go into frontend repository. Maybe adjust src/index.ts:139 const url to backend ip and port

```
npm install
npm run start
```

# Credits

Originally writted by Nicolas Bender, Marc Burg, and Jonannes Maul as part of a research project at Heidelberg University.

