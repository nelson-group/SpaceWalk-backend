"""Module for operations between two vector fields"""

# pylint: disable=import-error, no-name-in-module, no-member

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from tng_sv.data.dir import get_resampled_delaunay_path, get_snapshot_index_path
from tng_sv.data.field_type import FieldType


def _load_image_data(simulation_name: str, snapshot_idx: int, field_type: FieldType) -> vtk.vtkImageData:
    """Loads the resampled image for a given simulation snapshot"""
    path = get_resampled_delaunay_path(simulation_name, snapshot_idx, field_type)

    field = vtk.vtkXMLImageDataReader()
    field.SetFileName(str(path))
    field.SetPointArrayStatus(field_type.value, 1)
    field.Update()

    return field.GetOutput()


def _image_data_to_nd_array(field: vtk.vtkImageData, field_type: FieldType) -> np.ndarray:
    """Converts a vtk image data object to a numpy array"""
    dims = field.GetDimensions()
    data_array = field.GetPointData().GetArray(field_type.value)
    data_array = vtk_to_numpy(data_array)
    return data_array.reshape(*dims, -1)


def _save_ndarray_to_image_data(
    simulation_name: str, snapshot_idx: int, data: np.ndarray, input_field: vtk.vtkImageData, experiment_name: str
) -> None:
    """Saves a given ndarray using vtkXMLImageDataWriter"""
    output_vtk_image = vtk.vtkImageData()
    output_vtk_image.SetDimensions(*data.shape[:-1])
    output_vtk_image.SetOrigin(input_field.GetOrigin())
    output_vtk_image.SetSpacing(input_field.GetSpacing())
    depth_array = numpy_to_vtk(data.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
    output_vtk_image.GetPointData().SetScalars(depth_array)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetInputData(output_vtk_image)
    writer.SetFileName(
        get_snapshot_index_path(simulation_name, snapshot_idx).joinpath(f"image_data_{experiment_name}.pvd")
    )
    writer.Write()


def scalar_product(simulation_name: str, snapshot_idx: int, field_type_1: FieldType, field_type_2: FieldType) -> None:
    """Compute the scalar product between two vector fields"""
    field_1_image = _load_image_data(simulation_name, snapshot_idx, field_type_1)
    field_2_image = _load_image_data(simulation_name, snapshot_idx, field_type_2)

    field_1 = _image_data_to_nd_array(field_1_image, field_type_1)
    field_2 = _image_data_to_nd_array(field_2_image, field_type_2)

    result = (
        field_1[:, :, :, 0] * field_2[:, :, :, 0]
        + field_1[:, :, :, 1] * field_2[:, :, :, 1]
        + field_1[:, :, :, 2] * field_2[:, :, :, 2]
    ).reshape((*field_1.shape[:-1], 1))

    _save_ndarray_to_image_data(simulation_name, snapshot_idx, result, field_1_image, "scalar_product")


def vector_angle(simulation_name: str, snapshot_idx: int, field_type_1: FieldType, field_type_2: FieldType) -> None:
    """Compute the angle between the vectors in two vector fields"""
    # field_1_image = _load_image_data(simulation_name, snapshot_idx, field_type_1)
    # field_2_image = _load_image_data(simulation_name, snapshot_idx, field_type_2)
    #
    # field_1 = _image_data_to_nd_array(field_1_image, field_type_1)
    # field_2 = _image_data_to_nd_array(field_2_image, field_type_2)

    raise NotImplementedError("I know, I'm too slow :(")
