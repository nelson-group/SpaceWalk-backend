"""Information about preprocessing."""


from tng_sv.data.dir import get_bound_info_file, get_delaunay_path, get_snapshot_combination_index_path
from tng_sv.data.field_type import FieldType
from tng_sv.data.part_type import PartType


def assert_pvpython(func):
    "Assert that we are within paraview env." ""

    def _wrapper(*arg):
        import paraview  # pylint: disable=import-error,import-outside-toplevel,unused-import

        return func(*arg)

    return _wrapper


@assert_pvpython
def run_delaunay(simulation_name: str, snapshot_idx: int, part_type: PartType, field_type: FieldType) -> None:
    """Run delaunay."""
    # pylint: disable=import-error,import-outside-toplevel,no-member,no-name-in-module
    path = get_snapshot_combination_index_path(simulation_name, snapshot_idx, part_type, field_type)

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

        X = f["{part_type.value}"]['Coordinates'][:, 0]
        Y = f["{part_type.value}"]['Coordinates'][:, 1]
        Z = f["{part_type.value}"]['Coordinates'][:, 2]
        values = f["{part_type.value}"]["{field_type.value}"][:]

        coordinates = algs.make_vector(X.ravel(), Y.ravel(), Z.ravel())
        output.Points = coordinates
        output.PointData.append(values, "{field_type.value}")
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

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetInputConnection(delaunay3d.GetOutputPort(0))
    writer.SetFileName(str(path).replace(f"_{snapshot_idx:03d}.hdf5", f"_{snapshot_idx:03d}_delaunay.pvd"))
    writer.Write()


@assert_pvpython
def run_resample_delaunay(simulation_name: str, snapshot_idx: int, part_type: PartType, field_type: FieldType) -> None:
    """Run resample on delaunay input data."""
    # pylint: disable=import-error,import-outside-toplevel,no-member,no-name-in-module
    path = get_delaunay_path(simulation_name, snapshot_idx, part_type, field_type)

    import vtk
    from vtkmodules.vtkFiltersCore import vtkResampleToImage

    # create a new vtkPVDReader
    # delaunay_pvd = vtkPVDReader()
    delaunay_pvd = vtk.vtkXMLUnstructuredGridReader()
    delaunay_pvd.SetFileName(path)
    delaunay_pvd.SetPointArrayStatus(field_type.value, 1)

    if part_type != PartType.GAS:
        from paraview.modules.vtkPVVTKExtensionsFiltersPython import vtkPythonProgrammableFilter

        # create a new vtkPythonProgrammableFilter
        _filter = vtkPythonProgrammableFilter()
        _filter.SetInformationScript("")
        _filter.SetOutputDataSetType(8)
        _filter.AddInputConnection(0, delaunay_pvd.GetOutputPort(0))
        _filter.SetPythonPath("")
        _filter.SetScript(
            f"""
            import numpy as np

            bh = inputs[0]

            points = bh.GetPoints()
            lpoints = points.tolist()
            bound_points = np.load("{get_bound_info_file(simulation_name)}")
            lpoints.extend(bound_points.tolist())
            points = np.array(lpoints)

            data = bh.GetPointData().GetArray("{field_type.value}")
            ldata = data.tolist()
            ldata.extend([[0, 0, 0]]*2)
            data = np.array(ldata)

            output.Points = points
            output.PointData.append(data, "{field_type.value}")
            """
        )
        port = _filter
    else:
        port = delaunay_pvd

    # create a new vtkResampleToImage
    resample_to_image = vtkResampleToImage()
    resample_to_image.SetInputConnection(0, port.GetOutputPort(0))
    resample_to_image.SetSamplingDimensions(100, 100, 100)
    resample_to_image.SetUseInputBounds(True)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetInputConnection(resample_to_image.GetOutputPort(0))
    writer.SetFileName(
        str(path).replace(f"_{snapshot_idx:03d}_delaunay.pvd", f"_{snapshot_idx:03d}_resampled_delaunay.pvd")
    )
    writer.Write()
