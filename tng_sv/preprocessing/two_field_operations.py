"""Module for operations between two vector fields"""

# pylint: disable=import-error, no-name-in-module, no-member

from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from tng_sv.data.dir import get_resampled_delaunay_path, get_scalar_field_experiment_path
from tng_sv.data.field_type import FieldType
from tng_sv.data.part_type import PartType


def _load_image_data(simulation_name: str, snapshot_idx: int, field_type: FieldType) -> vtk.vtkImageData:
    """Loads the resampled image for a given simulation snapshot"""
    path = get_resampled_delaunay_path(simulation_name, snapshot_idx, PartType.GAS, field_type)

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


def _save_ndarray_to_image_data(data: np.ndarray, input_field: vtk.vtkImageData, file_name: Path) -> None:
    """Saves a given ndarray using vtkXMLImageDataWriter"""
    output_vtk_image = vtk.vtkImageData()
    output_vtk_image.SetDimensions(*data.shape[:-1])
    output_vtk_image.SetOrigin(input_field.GetOrigin())
    output_vtk_image.SetSpacing(input_field.GetSpacing())
    depth_array = numpy_to_vtk(data.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
    output_vtk_image.GetPointData().SetScalars(depth_array)

    if file_name.exists():
        file_name.unlink()

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetInputData(output_vtk_image)
    writer.SetFileName(str(file_name))
    writer.Write()


def _normalize_vector_field(field: np.ndarray) -> np.ndarray:
    """Normalizes the vectors in the given vector field"""
    result = np.zeros_like(field)
    vector_length = _compute_vector_length(field)

    existing_values = vector_length[:, :, :, 0] != 0
    result[existing_values, :] = field[existing_values, :] / vector_length[existing_values]
    return result


def _compute_scalar_product(field_1: np.ndarray, field_2: np.ndarray):
    """Computes the scalar product for two given vector fields"""
    return (
        field_1[:, :, :, 0] * field_2[:, :, :, 0]
        + field_1[:, :, :, 1] * field_2[:, :, :, 1]
        + field_1[:, :, :, 2] * field_2[:, :, :, 2]
    ).reshape((*field_1.shape[:-1], 1))


def _compute_vector_length(field_data: np.ndarray):
    """Computes the vector length for a given vector field"""
    return np.sqrt(
        np.square(field_data[:, :, :, 0]) + np.square(field_data[:, :, :, 1]) + np.square(field_data[:, :, :, 2])
    ).reshape((*field_data.shape[:-1], 1))


def scalar_product(
    simulation_name: str,
    snapshot_idx: int,
    field_type_1: FieldType,
    field_type_2: FieldType,
    normalize_vectors: bool = True,
) -> None:
    """Compute the scalar product between two vector fields"""
    field_1_image = _load_image_data(simulation_name, snapshot_idx, field_type_1)
    field_2_image = _load_image_data(simulation_name, snapshot_idx, field_type_2)

    field_1 = _image_data_to_nd_array(field_1_image, field_type_1)
    field_2 = _image_data_to_nd_array(field_2_image, field_type_2)

    if normalize_vectors:
        field_1 = _normalize_vector_field(field_1)
        field_2 = _normalize_vector_field(field_2)

    _save_ndarray_to_image_data(
        _compute_scalar_product(field_1, field_2),
        field_1_image,
        get_scalar_field_experiment_path(simulation_name, snapshot_idx, "scalar_product", field_type_1, field_type_2),
    )


def vector_angle(
    simulation_name: str,
    snapshot_idx: int,
    field_type_1: FieldType,
    field_type_2: FieldType,
    normalize_vectors: bool = True,
) -> None:
    """Compute the angle between the vectors in two vector fields"""
    field_1_image = _load_image_data(simulation_name, snapshot_idx, field_type_1)
    field_2_image = _load_image_data(simulation_name, snapshot_idx, field_type_2)

    field_1 = _image_data_to_nd_array(field_1_image, field_type_1)
    field_2 = _image_data_to_nd_array(field_2_image, field_type_2)

    if not normalize_vectors:
        scalar_product_result = _compute_scalar_product(field_1, field_2)

        field_1_vector_length = _compute_vector_length(field_1)
        field_2_vector_length = _compute_vector_length(field_2)

        denominator = field_1_vector_length * field_2_vector_length

        angle = np.zeros_like(scalar_product_result)
        angle[denominator != 0] = np.arccos(scalar_product_result[denominator != 0] / denominator[denominator != 0])
    else:
        field_1 = _normalize_vector_field(field_1)
        field_2 = _normalize_vector_field(field_2)
        angle = np.arccos(_compute_scalar_product(field_1, field_2))

    _save_ndarray_to_image_data(
        angle,
        field_1_image,
        get_scalar_field_experiment_path(simulation_name, snapshot_idx, "vector_angle", field_type_1, field_type_2),
    )
