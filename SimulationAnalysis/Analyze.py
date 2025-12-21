from typing import Any
from SimulationAnalysis import LAMMPS_Data, load_data
import numpy as np
from tkinter import messagebox, filedialog
import os
import matplotlib.pyplot as plt
import numpy.typing as npt
from collections.abc import Iterable
from scipy.stats import linregress

type CoordinateType = tuple[npt.NDArray[np.floating[Any]],
                            npt.NDArray[np.floating[Any]]]


def calc_basic_coords(data: Iterable[LAMMPS_Data], ncounts: int) -> npt.NDArray[np.floating[Any]]:
    """Calculates the each coordinates.

    Parameters
    ----------
    data : LAMMPS_Data
    ncounts : int

    Returns
    -------
    result : NDArray[floating[Any]]
        The coefficients `np.array([a, b])`.
    """
    xlist: list[int] = list()
    ylist: list[float] = list()
    for datumn in data:
        displacements = datumn.get_displacements()[-ncounts:]
        norms = np.linalg.norm(displacements, axis=1)
        normmean2 = np.mean(norms ** 2)
        xlist.append(datumn.number_atoms - 1)
        ylist.append(normmean2)
    xarray: npt.NDArray[np.floating[Any]] = np.array(xlist)
    xarray = np.log10(xarray)
    yarray: npt.NDArray[np.floating[Any]] = np.array(ylist)
    yarray = np.log10(yarray)
    print(len(xarray))
    res = linregress(xarray, yarray)
    a = res.slope, res.intercept
    return np.array(a)


def calc_data(data: Iterable[LAMMPS_Data]) -> CoordinateType:
    """Calculates the coordinates.

    Parameters
    ----------
    data : LAMMPS_Data
        The data.

    Returns
    -------
    result : CoordinateType
        The coordinates.
    """
    coeffs: list[float] = list()
    cnts: list[int] = list()
    for i in  range(len(tuple(data)[0])):
        coeffs.append(calc_basic_coords(data, i + 1)[0])
        cnts.append(i + 1)
    return np.array(cnts), np.array(coeffs)


fnames = filedialog.askopenfilenames(
    title='Choose a file.',
    defaultextension='.lammpstrj',
    filetypes=(
        (
            'LAMMPS Trajetory file',
            '*.lammpstrj'
        ),
    ),
    initialdir=os.curdir
)

results: list[LAMMPS_Data] = list()
for fname in fnames:
    try:
        with open(fname) as file:
            results.append(load_data(file))
    except Exception as err:
        messagebox.showerror(
            'Error',
            'An error occured.',
            detail=str(err)
        )
    except BaseException as e:
        messagebox.showinfo(
            'Stopped',
           'The process was stopped.',
           detail=str(e)
        )
plt.plot(*calc_data(results))
plt.show()
