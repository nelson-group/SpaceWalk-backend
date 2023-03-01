import os

import h5py
import numpy as np

subDirs = os.walk("D:\VMShare\Documents\data\groupcats")
path, dirs, lists = next(subDirs)

for x in dirs:
    print(x)
    with h5py.File(path +"/"+x + "/combine.hdf5", mode="w") as combined:
        _group = combined.create_group("Group")
        _subhalo = combined.create_group("Subhalo")

        group_pos = np.zeros((0, 3))
        group_vel = np.zeros((0, 3))
        group_mass = np.zeros(0)
        subhalo_pos = np.zeros((0, 3))
        subhalo_vel = np.zeros((0, 3))
        subhalo_mass = np.zeros(0)

        for i in range(0, 11):
            with h5py.File(path +"/"+x+ f"/fof_subhalo_tab_0{x}.{i}.hdf5", mode="r")  as f:
                group_pos = np.concatenate((group_pos, f["Group"]["GroupPos"]), axis=0)
                group_vel = np.concatenate((group_vel, f["Group"]["GroupVel"]), axis=0)
                group_mass = np.concatenate((group_mass, f["Group"]["GroupMass"]), axis=0)
                group_ids = np.concatenate((group_mass, f["Group"]["GroupMass"]), axis=0)
                subhalo_pos = np.concatenate((subhalo_pos, f["Subhalo"]["SubhaloPos"]), axis=0)
                subhalo_vel = np.concatenate((subhalo_vel, f["Subhalo"]["SubhaloVel"]), axis=0)
                subhalo_mass = np.concatenate((subhalo_mass, f["Subhalo"]["SubhaloMass"]), axis=0)

        _group.create_dataset("GroupPos", group_pos.shape, float, group_pos)
        _group.create_dataset("GroupVel", group_vel.shape, float, group_vel)
        _group.create_dataset("GroupMass", group_mass.shape, float, group_mass)
        _subhalo.create_dataset("SubhaloPos", subhalo_pos.shape, float, subhalo_pos)
        _subhalo.create_dataset("SubhaloVel", subhalo_vel.shape, float, subhalo_vel)
        _subhalo.create_dataset("SubhaloMass", subhalo_mass.shape, float, subhalo_mass)

    combined.close()