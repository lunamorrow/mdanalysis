# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# MDAnalysis --- https://www.mdanalysis.org
# Copyright (c) 2006-2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
#
# Please cite your use of MDAnalysis in published work:
#
# R. J. Gowers, M. Linke, J. Barnoud, T. J. E. Reddy, M. N. Melo, S. L. Seyler,
# D. L. Dotson, J. Domanski, S. Buchoux, I. M. Kenney, and O. Beckstein.
# MDAnalysis: A Python package for the rapid analysis of molecular dynamics
# simulations. In S. Benthall and S. Rostrup editors, Proceedings of the 15th
# Python in Science Conference, pages 102-109, Austin, TX, 2016. SciPy.
# doi: 10.25080/majora-629e541a-00e
#
# N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein.
# MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations.
# J. Comput. Chem. 32 (2011), 2319--2327, doi:10.1002/jcc.21787
#

import numpy as np

from ..base import AnalysisBase
from .utils import get_centers_by_residue, get_orientations

class LipidOrientation(AnalysisBase):

    def __init__(self, universe, select="name ROH"):
        super().__init__(universe.universe.trajectory)
        self.selection = universe.select_atoms(select)
        self.headgroups = self.selection.split("residue")
        self.residues = self.selection.residues
        self.n_residues = len(self.residues)

    def _prepare(self):
        self.orientations = np.zeros((self.n_frames, self.n_residues))
    
    def _single_frame(self):
        box = self._trajectory.box
        coordinates = get_centers_by_residue(self.selection,
                                             box=box)
        orientations = get_orientations(self.residues,
                                        self.selection,
                                        box=box,
                                        headgroup_centers=coordinates,
                                        normalize=True)
        
        self.orientations[self._frame_index] = np.arccos(orientations[:, 2])