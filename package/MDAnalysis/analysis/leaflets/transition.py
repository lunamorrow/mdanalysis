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

from .base import BaseLeafletAnalysis
from ..distances import capped_distance

class MembraneTransition(BaseLeafletAnalysis):

    def __init__(self, universe, *args, **kwargs):
        super().__init__(universe, *args, **kwargs)

    def _prepare(self):
        ...

        self.locations = np.zeros((self.n_frames, self.n_residues))
        self.transitions = np.zeros(self.n_residues)

    def _single_frame(self):
        
        coords = self.leafletfinder.coordinates.copy()
        coords[:, :2] = 0
        upper_comp, lower_comp = self.leafletfinder.components[:2]
        upper_z = coords[upper_comp]
        lower_z = coords[lower_comp]
        thickness = upper_z.mean() - lower_z.mean()

        box = self._trajectory.box
        if box is not None:
            zone_length = box[2] / 3
        else:
            zone_length = thickness * 2
        
        row = self.locations[self._frame_index]
        for i, r in enumerate(self.headgroups):
            res = r.positions.copy()
            res[:, :2] = 0
            pairs, dists = capped_distance(coords, res,
                                           max_cutoff=zone_length,
                                           box=box,
                                           return_distances=True)
            if not len(pairs):
                # not near any leaflet
                row[i] = -10
                continue
            
            rix = pairs[:, 0]
            upper = dists[np.in1d(rix, upper_comp)]
            lower = dists[np.in1d(rix, lower_comp)]

            if not len(lower) and len(upper):
                row[i] = 0
                continue

            if not len(upper) and len(lower):
                row[i] = 2
                continue

            upper_mean = upper.mean()
            lower_mean = lower.mean()

            if upper_mean <= thickness and lower_mean <= thickness:
                row[i] = 1
            elif upper_mean < lower_mean:
                row[i] = 0
            else:
                row[i] = 2

    def _conclude(self):
        ...
        # TODO: finish
        # differences = self.locations[:-1] - self.locations[1:]
        # abs_diff = np.abs(differences)
        # np.where(abs_dif == 2)[0]
        # loc_differences = differences[:-1] - differences[1:]


    



            






