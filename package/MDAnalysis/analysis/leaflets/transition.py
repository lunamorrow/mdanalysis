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
from ..distances import capped_distance, calc_bonds, self_distance_array
from ...lib._cutil import remove_repeated_int

class MembraneTransition(BaseLeafletAnalysis):

    def __init__(self, universe, *args, **kwargs):
        super().__init__(universe, *args, **kwargs)

    def _prepare(self):
        self.locations = np.ones((self.n_frames, self.n_residues))
        self.locations *= -10


    def _single_frame(self):
        upper_i = 0
        membrane_i = 1
        lower_i = 2
        inter_i = -10
        
        box = self.universe.dimensions

        coords = self.leafletfinder.coordinates.copy()
        coords[:, :2] = 0
        upper_comp, lower_comp = self.leafletfinder.components[:2]
        upper_comp = set(list(upper_comp))
        lower_comp = set(list(lower_comp))
        
        if box is not None:
            zone_length = box[2] / 3
        else:
            upper_z = coords[upper_comp]
            lower_z = coords[lower_comp]
            thickness = calc_bonds(upper_z.mean(axis=0),
                                   lower_z.mean(axis=0),
                                   box=box)
            zone_length = thickness * 2

        row = self.locations[self._frame_index]
        res_positions = get_centers_by_residue(self.selection)

        pairs = capped_distance(res_positions,
                                self.leafletfinder.coordinates,
                                box=box, max_cutoff=zone_length,
                                return_distances=False)

        all_pairs = np.sort(pairs, axis=0)
        splix = np.where(np.ediff1d(pairs[:, 0]))[0] + 1
        plist = np.split(pairs, splix)

        res_positions[:, :2] = 0

        for pairs_ in plist:
            rix = set(pairs_[:, 1])
            i = pairs[0, 0]

            upper_idx = sorted(rix & upper_comp)
            lower_idx = sorted(rix & lower_comp)
            upper = coords[upper_idx]
            lower = coords[lower_idx]
            res = res_positions[i]

            if not len(lower) and len(upper):
                row[i] = upper_i
                continue

            if not len(upper) and len(lower):
                row[i] = lower_i
                continue

            upper = mean_unwrap_around(upper, upper[0], box)
            lower = mean_unwrap_around(lower, lower[0], box)
            ref = np.array([upper, lower, res])

            thickness, udist, ldist = self_distance_array(ref, box=box)

            if (udist <= thickness) and (ldist <= thickness):
                row[i] = membrane_i
            elif udist <= ldist:
                row[i] = upper_i
            else:
                row[i] = lower_i
        

    def _conclude(self):
        # look for 0, 1, 2 or 2, 1, 0
        # NOT -1

        self.flips = np.zeros((self.n_residues))
        self.flops = np.zeros((self.n_residues))

        if not self.n_frames:
            return

        for i, col in enumerate(self.locations.T):
            unique = remove_repeated_int(col[col != 1])
            diff = unique[:-1] - unique[1:]
            self.flips[i] = (diff == 2).sum()
            self.flops[i] = (diff == -2).sum()

        self.transitions = self.flips + self.flops
    



            






