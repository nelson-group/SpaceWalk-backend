# API

## Important structures
CameraInformation:
- contains coordinates and size of the client's camera

ClientState:
- saves the state for already loaded node indices, level of detail, batch size and percentage of data per leaf node in order to load data dynamically batch-wise 

DataCache:
- class responsible for loading data efficiently:
    - checks if a snap is already loaded, if not it fetches it from the server's filesystem
    - keeps loaded data on server cache

## `GET /v1/get/init/{simulation}/{snap_id}`
This endpoint initializes the visualization by providing metadata (number of quantiles and their data, all available snapshots and `BoxSize`) and fetching the initial simulation data.

**How does it work?**
- scans directories to identify available snapshots (`all_possible_snaps`) by matching folder names with the pattern snapdir_
- extracts box size (`BoxSize`) metadata from files matching the pattern `groups_` using the illustris library
- uses the `DataCache` class to check if the requested simulation and snapshot data (`simulation`, `snap_id`) is already cached
- if cached, it retrieves the data directly. Otherwise, it:
    - loads several data files (`splines`, `velocities`, `densities`, etc) and structures from the filesystem based on the simulation and snapshot
    - prepares a `ListOfLeafs` object from `leafs` and `leafs_scan` arrays
    - calculates density quantiles using the densities data
    - caches the loaded data

**What is the response?**
The response is a JSON that includes:
- `density_quantiles`: A list of quantile values derived from the density data.
- `n_quantiles`: The number of quantiles available.
- `available_snaps`: A list of all possible snapshot numbers for the simulation.
- `BoxSize`: The size of the simulation box.


## `POST /v1/get/splines/{simulation}/{snap_id}`
This endpoint processes and retrieves spline data along with related information for a specific simulation and snapshot, filtered based on the client's camera view.

**How does it work?**
- retrieves cached data for the specified simulation and snapshot (`simulation`, `snap_i`d) using the `DataCache` class. This data includes:
    - `octree`: For spatial hierarchy and node traversal
    - `splines`: Cubic spline parameters
    - `velocities`, densities, coordinates, voronoi_diameter_extended
    - `particle_list_of_leafs`: Data structure that maps particles to octree leaf nodes
- traverses octree:
    - uses the client's camera position (from `CameraInformation`) to create a `ViewBox`, representing the region of interest in 3D space
    - traverses the octree to find intersecting nodes containing relevant particles (`node_indices`).
- filters and loads particles for each intersecting node:
    - retrieves particle IDs from `particle_list_of_leafs` based on the percentage of data (`client_state.percentage`) and level of detail (LOD)
    - adjusts the range of particles per node based on `batch_size_lod`
- increases level of detail:
    - updates the LOD for each node in the client state, ensuring the detail increases with each call
- extract relevant data:
    - Splines: Extracts spline parameters (`splines_a`, `splines_b`, `splines_c`, `splines_d`)
    - Physical properties: Coordinates, velocities, densities, Voronoi diameters
    - Calculates the minimum and maximum densities for the selected particles

**What is the response?**
The response is a JSON that includes:
- Data:
    - Relevant particle IDs (`relevant_ids`).
    - Coordinates, velocities, densities, splines, and Voronoi diameters for the selected particles
- Metadata:
    - Updated `level_of_detail` for the nodes
    - Density range (`min_density`, `max_density`)
    - Total number of particles (`nParticles`)
    - Density quantiles, snapshot ID (`snapnum`)

# Octree
Structure:
- the octree starts with a root node that represents the entire bounding box (the space of interest).
- each node is recursively subdivided into eight smaller cubical regions (children), dividing the space into octants.
- subdivision continues up to a maximum depth or until each node contains fewer than a specified number of points (or other criteria are met).

Storage of Data:
- particles are stored in the leaf nodes. If a node contains more particles than the allowed threshold (`size_per_leaf`), it is further subdivided.
- leaf node stores data like particle indices and values of relevant fields

Traversal:
- queries or operations (e.g. finding neighbors or retrieving data) involve traversing the octree from the root, descending into relevant nodes based on the spatial location of interest.

# Dataflow

## Trigger download from server to client
Download is triggered as soon as one of the following premises is met:
- download of the current ViewBox is finished for current percentage and LOD
- a leaf inside the ViewBox has a higher LOD than the current still with particles in it

## Fetch data from server to client
- the particles of the current LOD from every leaf are downloaded and then the LOD is increased
- the latest LOD which was loaded is saved so that changing the ViewBox will continue downloading at the last LOD which was not loaded yet per leaf