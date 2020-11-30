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

from .base import LeafletAnalysis
from .. import distances
from ...lib.c_distances import unwrap_around
from ...lib.mdamath import norm

class LipidFlipFlop(LeafletAnalysis):
    """Quantify lipid flip-flops between leaflets.

    Parameters
    ----------

    universe : Universe
        :class:`~MDAnalysis.core.universe.Universe` object.
    select : AtomGroup or str
        A AtomGroup instance or a
        :meth:`Universe.select_atoms` selection string
        for atoms that define the lipid head groups, e.g.
        universe.atoms.PO4 or "name PO4" or "name P*"



    """

    def __init__(self, universe, *args, select="resname CHOL",
                 cutoff=50, buffer_zone=8, **kwargs):
        super().__init__(universe, *args, select=select, **kwargs)
        self.cutoff = cutoff
        self.buffer_zone = buffer_zone
        other_i = [i for i, x in enumerate(self.leafletfinder.residues)
                   if x not in self.residues]
        self.other_i = set(other_i)
    
    def _prepare(self):
        self.residue_leaflet_raw = np.zeros((self.n_frames, self.n_residues), dtype=int)
        self.bilayer_section = np.zeros((self.n_frames, self.n_residues), dtype=int)

    def _single_frame(self):

        lfer = self.leafletfinder
        box = self.selection.dimensions

        components = [set(x) for x in lfer.components]

        for i in range(self.n_residues):
            j = self._a2lf[i]
            lf_i = lfer.i2comp[j]
            self.residue_leaflet_raw[self._frame_index, i] = lf_i

            # determine which is upper or lower for mid-point calculation
            if lf_i % 2:
                top_i, bot_i = lf_i - 1, lf_i
            else:
                top_i, bot_i = lf_i, lf_i + 1

            # i_xyz = lfer.coordinates[j]
            i_xyz = self.headgroups[i].positions
            i_xyz = unwrap_around(i_xyz, i_xyz[0], box=box)

            top = np.array(list(components[top_i] & self.other_i))
            bot = np.array(list(components[bot_i] & self.other_i))
            
            top_xyz = lfer.coordinates[top]
            bot_xyz = lfer.coordinates[bot]

            i_xyz_ = i_xyz.copy()
            i_xyz_[0][2] = 0
            top_xyz_ = top_xyz.copy()
            bot_xyz_ = bot_xyz.copy()

            top_xyz_[:, 2] = 0
            bot_xyz_[:, 2] = 0

            # there's probably a better way to do this
            top_pairs, top_dists = distances.capped_distance(i_xyz_, top_xyz_, self.cutoff,
                                                            return_distances=True)
            bot_pairs, bot_dists = distances.capped_distance(i_xyz_, bot_xyz_, self.cutoff,
                                                             return_distances=True)

            top_pairs = top_pairs[np.argsort(top_dists)][:5]
            bot_pairs = bot_pairs[np.argsort(bot_dists)][:5]

            top_uw = unwrap_around(top_xyz[top_pairs[:, 1]], i_xyz, box=box)
            top_point = top_uw.mean(axis=0)
            bot_uw = unwrap_around(bot_xyz[bot_pairs[:, 1]], i_xyz, box=box)
            bot_point = bot_uw.mean(axis=0)

            top_point[[0, 1]] = 0
            bot_point[[0, 1]] = 0
            i_xyz[0][[0, 1]] = 0

            vec = top_point - bot_point
            dist = norm(vec) / 2
            mid = (dist * vec) + bot_point
            buffer = dist - self.buffer_zone
            i_xyz = np.array([i_xyz[0], i_xyz[0], i_xyz[0]])
            points = np.array([mid, top_point, bot_point])


            dist_to_z, *tbs = distances.calc_bonds(i_xyz, points, box=box)
            if dist_to_z < buffer:
                lf_i = -1
            else:
                lf_i = [top_i, bot_i][np.argmin(tbs)]
            self.bilayer_section[self._frame_index, i] = lf_i
            

    def _conclude(self):
        self.residue_leaflet = np.zeros_like(self.residue_leaflet_raw)
        self.flips = np.zeros(self.n_residues)
        self.flops = np.zeros(self.n_residues)
        self.flip_sections = np.zeros(self.n_residues)
        self.flop_sections = np.zeros(self.n_residues)

        if not self.n_frames:
            return

        for i in range(self.n_residues):
            trans = self.residue_leaflet_raw[:, i]
            self.residue_leaflet[:, i] = trans
            diff = trans[1:] - trans[:-1]
            self.flips[i] = np.sum(diff > 0)  # 0: upper, 1: lower
            self.flops[i] = np.sum(diff < 0)

            trans2 = self.bilayer_section[:, i]
            trans2 = trans2[trans2 != -1]
            if len(trans2) < 2: 
                continue
            diff2 = trans2[1:] - trans2[:-1]
            self.flip_sections[i] = np.sum(diff2 > 0)
            self.flop_sections[i] = np.sum(diff2 < 0)
        
        self.translocations = self.flips + self.flops
        self.trans_sections = self.flip_sections + self.flop_sections

        self.flips_by_attr = {}
        self.flops_by_attr = {}
        self.translocations_by_attr = {}

        self.flip_sections_by_attr = {}
        self.flop_sections_by_attr = {}
        self.trans_sections_by_attr = {}
        for each in np.unique(self.ids):
            mask = self.ids == each
            self.flips_by_attr[each] = int(sum(self.flips[mask]))
            self.flops_by_attr[each] = int(sum(self.flops[mask]))
            self.translocations_by_attr[each] = int(sum(self.translocations[mask]))
            self.flip_sections_by_attr[each] = int(sum(self.flip_sections[mask]))
            self.flop_sections_by_attr[each] = int(sum(self.flop_sections[mask]))
            self.trans_sections_by_attr[each] = int(sum(self.trans_sections[mask]))