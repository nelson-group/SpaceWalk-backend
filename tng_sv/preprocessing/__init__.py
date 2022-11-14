"""Information about preprocessing."""


from tng_sv.data.dir import get_snapshot_combination_index_path


def assert_pvpython(func):
    "Assert that we are within paraview env." ""

    def _wrapper(*arg):
        import paraview  # pylint: disable=import-error,import-outside-toplevel,unused-import

        return func(*arg)

    return _wrapper


@assert_pvpython
def run_delaunay(simulation_name: str, snapshot_idx: int) -> None:
    """Run delaunay."""
    # pylint: disable=import-error,import-outside-toplevel
    path = get_snapshot_combination_index_path(simulation_name, snapshot_idx)

    # Lazy import to prevent failing in non pvpython environment
    import vtk
    from paraview.modules.vtkPVVTKExtensionsFiltersPython import vtkPythonProgrammableFilter
    from vtkmodules.vtkFiltersCore import vtkDelaunay3D

    # create a new vtkPythonProgrammableFilter
    programmable_source = vtkPythonProgrammableFilter()
    programmable_source.SetInformationScript("")
    programmable_source.SetOutputDataSetType(4)
    programmable_source.SetPythonPath("")
    programmable_source.SetScript(
        rf"""
        from pathlib import Path

        import numpy as np
        import h5py
        from vtk.numpy_interface import algorithms as algs


        f = h5py.File("{path}", 'r')

        X = f['PartType0']['Coordinates'][:, 0]
        Y = f['PartType0']['Coordinates'][:, 1]
        Z = f['PartType0']['Coordinates'][:, 2]
        velocity = f['PartType0']['Velocities'][:]

        coordinates = algs.make_vector(X.ravel(), Y.ravel(), Z.ravel())
        output.Points = coordinates
        output.PointData.append(velocity, 'velocity')
        f.close()
        """
    )

    # create a new vtkDelaunay3D
    delaunay3d = vtkDelaunay3D()
    delaunay3d.SetAlpha(0.0)
    delaunay3d.SetAlphaLines(False)
    delaunay3d.SetAlphaTets(True)
    delaunay3d.SetAlphaTris(True)
    delaunay3d.SetAlphaVerts(False)
    delaunay3d.SetBoundingTriangulation(False)
    delaunay3d.SetInputConnection(0, programmable_source.GetOutputPort(0))
    delaunay3d.SetOffset(2.5)
    delaunay3d.SetTolerance(0.001)

    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetInputConnection(delaunay3d.GetOutputPort(0))
    writer.SetFileName(str(path).replace(".hdf5", "_delaunay.pvd"))
    writer.Write()
