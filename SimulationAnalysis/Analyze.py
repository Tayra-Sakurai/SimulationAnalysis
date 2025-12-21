from typing import Any
from SimulationAnalysis import load_vectors
import numpy as np
from tkinter import messagebox, filedialog
import os
import matplotlib.pyplot as plt
import numpy.typing as npt

fname = filedialog.askopenfilename(
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

try:
    with open(fname) as file:
        vecs = load_vectors(file)
    norms: np.ndarray[tuple[int], np.dtype[np.floating[Any]]] = np.linalg.norm(vecs, axis=1)
    norms **= 2
    mlist: list[np.floating[Any]] = list()
    for num in range(1, len(norms)+1):
        nslice = norms[-num:]
        m = np.mean(nslice)
        mlist.append(m)
    mr = np.array(mlist)
    plt.plot(mr)
    plt.show()
except Exception as e:
    messagebox.showerror(
        'Error',
        'An error occured.',
        detail=str(e)
    )
