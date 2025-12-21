"""This calculates the end distance values from .lammpstrj file.

The classes can be used for other purposes.
"""

import numpy as np
from typing import Any, TextIO
import numpy.typing as npt
from collections.abc import Iterable
from enum import Enum, auto
from string import Template

type SpaceVector = np.ndarray[tuple[int], np.dtype[np.floating[Any]]]
type SpaceVectors = np.ndarray[tuple[int, int], np.dtype[np.floating[Any]]]


class LAMMPS_Atom:
    """Represents the LAMMPS Atom.

    Parameters
    ----------
    id : int
        The atome's identifier.
    xu : float
        The atome's x coordinate.
    yu : float
        The y-coordinate of the atom.
    zu : float
        The z coordinate of the atom.

    Attributes
    ----------
    coord : NDArray[floating[Any]]
        The location vector of the atom.
    """
    def __init__(
            self,
            atom_id: int,
            xu: float,
            yu: float,
            zu: float
    ) -> None:
        self.id: int = atom_id
        self.xu: float = xu
        self.yu: float = yu
        self.zu: float = zu
        self.coord: SpaceVector = np.array((xu, yu, zu))

    def __str__(self) -> str:
        return '<LAMMPS_Atom id=' + str(self.id) + '>'


class LAMMPS_Frame:
    """This is the class of a LAMMPS frame.

    Parameters
    ----------
    number_atoms : int
        The number of atoms.
    timestamp : int
        The timestamp.
    atoms : Iterable[LAMMPS_Atom]
        The chain.

    Attributes
    ----------
    end_distance : float
        The distance between the ends of the chain.

    Methods
    -------
    get_displacement(i1=0, i2=-1)
        Gets the displacement.
    """
    def __init__(
            self,
            number_atoms: int,
            timestamp: int,
            atoms: Iterable[LAMMPS_Atom]
    ) -> None:
        self.number_atoms: int = number_atoms
        self.timestamp: int = timestamp
        self.atoms: tuple[LAMMPS_Atom, ...] = tuple(atoms)
        firstatom: LAMMPS_Atom
        lastatom: LAMMPS_Atom
        for atom in atoms:
            if atom.id == 1:
                firstatom = atom
            elif atom.id == number_atoms:
                lastatom = atom
            else:
                continue
        displacement = lastatom.coord - firstatom.coord
        self.end_didtance = np.linalg.norm(displacement)

    def get_displacement(self, i1: int = 0, i2: int = -1) -> SpaceVector:
        """Gets the displacement vector from first index to second index.

        Parameters
        ----------
        i1 : int, optional
            The first index.
        i2 : int, optional
            The second index.

        Returns
        -------
        vector : ndarray[tuple[int], dtype[floating[Any]]]
            The displacement vector.
        """
        v1: SpaceVector = self.atoms[i1].coord
        v2: SpaceVector = self.atoms[i2].coord
        return v2 - v1


class LAMMPS_Data:
    """This represents the LAMMPS data frames.

    Parameters
    ----------
    frames : Iterable[LAMMPS_Frame]
        The frames.

    Attributes
    ----------
    number_atoms : int
        The number of atoms.

    Methods
    -------
    ReArray(num=0)
        Returns the distances.
    get_displacements(i1=0, i2=-1)
        Gets the displacements.
    """
    def __init__(
            self,
            frames: Iterable[LAMMPS_Frame]
    ) -> None:
        self.frames: tuple[LAMMPS_Frame, ...] = tuple(frames)
        self.number_atoms: int = self.frames[0].number_atoms

    def get_re(self, num: int = 0) -> npt.NDArray[np.floating[Any]]:
        """Returns the `Re` values of frames.

        Parameters
        ----------
        num : int, default=0
            The number of frames which are included in the array.
            If it is 0, the whole frames are used to calculate.

        Returns
        -------
        ReArray : NDArray[floating[Any]]
            The distances.
        """
        # This is the slice of frames to be used for calculattion.
        tslice: tuple[LAMMPS_Frame, ...]
        if num == 0:
            tslice = self.frames
        else:
            tslice = self.frames[-num:]
        Relist: list[np.floating[Any]] = [
            i.end_didtance for i in tslice
        ]
        ReArray: npt.NDArray[np.floating[Any]] = np.array(Relist)
        return ReArray

    def get_displacements(self, i1: int = 0, i2: int = -1) -> SpaceVectors:
        """Gets the displacements of each frame.

        Parameters
        ----------
        i1 : int, optional
            The first index.
        i2 : int, optional
            The second index.

        Returns
        -------
        spVectors : NDArray[floating[Any]]
            The vectors.

        See Also
        --------
        LAMMPS_Frame.get_displacement(i1=0,i2=-1) : For detail.
        """
        dlist: list[SpaceVector] = [
            i.get_displacement(i1, i2) for i in self.frames
        ]
        spVectors: SpaceVectors = np.array(dlist)
        return spVectors

    def __str__(self) -> str:
        return '<LAMMPS_Data: Frames=' + str(len(self.frames)) + '>'

    def __len__(self) -> int:
        return len(self.frames)


class Action (Enum):
    """The enumerate of actions.

    Attributes
    ----------
    ACTION_NONE : int
        This expresses that no action is assigned.
    BOX_BOUNDS : int
        The box bounding.
    TIMESTEP : int
        The time step.
    NUMBER_OF_ATOMS : int
        The number of atoms.
    ATOMS : int
        The atoms.
    """
    ACTION_NONE = auto()
    BOX_BOUNDS = auto()
    TIMESTEP = auto()
    NUMBER_OF_ATOMS = auto()
    ATOMS = auto()


def load_data(file: TextIO) -> LAMMPS_Data:
    """Loads the lammpstrj file to get the vectors.

    Parameters
    ----------
    file : TextIO
        The lammpstrj file.

    Returns
    -------
    values : LAMMPS_Data
        The data.
    """
    file.seek(0)
    lines: list[str] = file.readlines()
    flist: list[LAMMPS_Frame] = list()
    alist: list[LAMMPS_Atom] = list()
    action: Action = Action.ACTION_NONE
    tstep: int | None = None
    natoms: int = 0
    line_num: int = 0
    msg = Template("Loading...    $linenum / $wholenum")
    bksps = '\b' * len(msg.substitute(linenum=line_num, wholenum=len(lines)))
    print(msg.substitute(linenum=line_num, wholenum=len(lines)), end='')
    for line in lines:
        line_num += 1
        print(bksps, end='')
        bksps = '\b' * len(
            msg.substitute(linenum=line_num, wholenum=len(lines)))
        print(msg.substitute(linenum=line_num, wholenum=len(lines)), end='')
        if line.startswith('ITEM: '):
            actionName = line.removeprefix('ITEM: ').strip('\n').strip('\r')
            if actionName == 'TIMESTEP':
                action = Action.TIMESTEP
            elif actionName == 'NUMBER OF ATOMS':
                action = Action.NUMBER_OF_ATOMS
            elif actionName == 'BOX BOUNDS pp pp pp':
                action = Action.BOX_BOUNDS
            elif actionName == 'ATOMS id xu yu zu':
                action = Action.ATOMS
        else:
            if action == Action.ATOMS:
                # Atom adder.
                data: list[str] = line.split()
                aid = int(data[0])
                xu = float(data[1])
                yu = float(data[2])
                zu = float(data[3])
                alist.append(LAMMPS_Atom(aid, xu, yu, zu))
            else:
                if (
                    (natoms > 0) and
                    (tstep is not None) and
                    (len(alist) == natoms)
                ):
                    flist.append(LAMMPS_Frame(natoms, tstep, alist))
                    alist = list()
                if action == Action.TIMESTEP:
                    tstep = int(line)
                elif action == Action.NUMBER_OF_ATOMS:
                    natoms = int(line)
                else:
                    continue
    d_atoms = LAMMPS_Data(flist)
    print()
    print()
    return d_atoms
