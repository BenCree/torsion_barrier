"""
Module of package-agnostic maths utilities.

This module contains utility functions for working
with PyTorch tensors and numpy arrays.
"""
import functools
import numpy as np
import torch

TensorType = np.ndarray | torch.Tensor

__all__ = [
    "calculate_distances",
    "calculate_angles",
    "calculate_dihedrals",
]


def _switch_backend_function_wrapper(
    torch_function: callable,
    numpy_function: callable,
):
    """Wrap a function to switch between PyTorch and numpy backends.

    Parameters
    ----------
    torch_function : callable
        The function to call if the input is a PyTorch tensor.
    numpy_function : callable
        The function to call if the input is a numpy array.

    Returns
    -------
    callable
        The wrapped function.
    """

    @functools.wraps(torch_function)
    def wrapped_function(*args, **kwargs):
        if isinstance(args[0], np.ndarray):
            return numpy_function(*args, **kwargs)
        elif args[0].__module__.startswith("torch"):
            return torch_function(*args, **kwargs)
        else:
            raise NotImplementedError(
                f"Function not implemented for type {type(args[0])}"
            )
    return wrapped_function

def _calculate_distances_torch(
    source: torch.Tensor,
    destination: torch.Tensor,
):
    """
    Calculate the Euclidean distance between two sets of points.

    Parameters
    ----------
    source : torch.Tensor
        The source points.
    destination : torch.Tensor
        The destination points.

    Returns
    -------
    torch.Tensor
        The Euclidean distances between the source and destination points.
    """
    return torch.norm(source - destination, dim=1)

def _calculate_distances_numpy(
    source: np.ndarray,
    destination: np.ndarray,
):
    """
    Calculate the Euclidean distance between two sets of points.

    Parameters
    ----------
    source : np.ndarray
        The source points.
    destination : np.ndarray
        The destination points.

    Returns
    -------
    np.ndarray
        The Euclidean distances between the source and destination points.
    """
    return np.linalg.norm(source - destination, axis=1)


calculate_distances = _switch_backend_function_wrapper(
    _calculate_distances_torch,
    _calculate_distances_numpy,
)

def calculate_distances(
    source: TensorType,
    destination: TensorType,
):
    """Calculate the Euclidean distance between two sets of points.

    Parameters
    ----------
    source : torch.Tensor
        The source points.
    destination : torch.Tensor
        The destination points.

    Returns
    -------
    torch.Tensor
        The Euclidean distances between the source and destination points.
    """
    if isinstance(source, np.ndarray):
        return np.linalg.norm(source - destination, axis=1)
    return torch.norm(source - destination, dim=1)


def _calculate_angles_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
):
    """
    Calculate the angle between three sets of points.

    Parameters
    ----------
    a : torch.Tensor
        The first set of points.
    b : torch.Tensor
        The second set of points.
    c : torch.Tensor
        The third set of points.

    Returns
    -------
    torch.Tensor
        The angles between the three sets of points.
    """
    ba = a - b
    bc = c - b
    cosine_angle = torch.sum(ba * bc, dim=1) / (
        torch.norm(ba, dim=1) * torch.norm(bc, dim=1)
    )
    return torch.acos(cosine_angle)

def _calculate_angles_numpy(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
):
    """
    Calculate the angle between three sets of points.

    Parameters
    ----------
    a : np.ndarray
        The first set of points.
    b : np.ndarray
        The second set of points.
    c : np.ndarray
        The third set of points.

    Returns
    -------
    np.ndarray
        The angles between the three sets of points.
    """
    ba = a - b
    bc = c - b
    cosine_angle = np.sum(ba * bc, axis=1) / (
        np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    )
    return np.arccos(cosine_angle)

calculate_angles = _switch_backend_function_wrapper(
    _calculate_angles_torch,
    _calculate_angles_numpy,
)

def _calculate_dihedrals_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
):
    """
    Calculate the dihedral angle between four sets of points.

    Parameters
    ----------
    a : torch.Tensor
        The first set of points.
    b : torch.Tensor
        The second set of points.
    c : torch.Tensor
        The third set of points.
    d : torch.Tensor
        The fourth set of points.

    Returns
    -------
    torch.Tensor
        The dihedral angles between the four sets of points.
    """
    # Standard IUPAC construction. Only ONE plane normal is crossed with the
    # central bond; crossing BOTH (the previous behaviour) makes x and y
    # proportional to each other, so atan2 collapsed to two values whose
    # magnitude tracked |b2| rather than the torsion. See
    # tests/utils/test_tensors_geometry.py::test_dihedral_is_invariant_to_central_bond_length
    #
    # The handedness is NOT a free choice. cross(b2_hat, n1) rather than
    # cross(n1, b2_hat) is what makes this agree with RDKit's GetDihedralDeg and
    # with the grid keys torsiondrive writes through geomeTRIC; the other order
    # returns the negative of both. Measured against QCArchive TorsionNet500 on
    # 3 molecules by 24 grid points, agreement to 0.010 degrees. Getting this
    # backwards leaves every energy unchanged, because a model absorbs the sign
    # into its own coefficients, and mirrors every PHASE written out as a force
    # field parameter. See test_dihedral_matches_an_external_torsiondrive_geometry.
    b1 = b - a
    b2 = c - b
    b3 = d - c

    n1 = torch.linalg.cross(b1, b2, dim=1)
    n2 = torch.linalg.cross(b2, b3, dim=1)
    b2_hat = b2 / torch.norm(b2, dim=1, keepdim=True)
    m1 = torch.linalg.cross(b2_hat, n1, dim=1)

    x = torch.sum(n1 * n2, dim=1)
    y = torch.sum(m1 * n2, dim=1)

    return torch.atan2(y, x)


def _calculate_dihedrals_numpy(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
):
    """
    Calculate the dihedral angle between four sets of points.

    Parameters
    ----------
    a : np.ndarray
        The first set of points.
    b : np.ndarray
        The second set of points.
    c : np.ndarray
        The third set of points.
    d : np.ndarray
        The fourth set of points.

    Returns
    -------
    np.ndarray
        The dihedral angles between the four sets of points.
    """
    # See the torch implementation above for why only one normal is crossed
    # with the central bond.
    b1 = b - a
    b2 = c - b
    b3 = d - c

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    b2_hat = b2 / np.linalg.norm(b2, axis=1, keepdims=True)
    m1 = np.cross(b2_hat, n1)

    x = np.sum(n1 * n2, axis=1)
    y = np.sum(m1 * n2, axis=1)

    return np.arctan2(y, x)

calculate_dihedrals = _switch_backend_function_wrapper(
    _calculate_dihedrals_torch,
    _calculate_dihedrals_numpy,
)

