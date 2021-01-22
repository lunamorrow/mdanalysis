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
from ...lib.util import fixedwidth_bins


class Thickness(LeafletAnalysis):

    def __init__(self, universe, *args, xmin=None, ymin=None,
                 xmax=None, ymax=None, delta=20.0, padding=5.0,
                 **kwargs):
        super().__init__(universe, *args, **kwargs)
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.delta = delta
        self.padding = padding

    def _prepare(self):
        coord = self.selection.positions
        if self.xmin is None:
            try:
                self.xmin = coord[:, 0].min() - self.padding
            except ValueError:
                msg = ("No selected atoms present at current frame. "
                       "Give xdim and ydim to set up thickness bins.")
                raise ValueError(msg) from None
        
        if self.xmax is None:
            self.xmax = coord[:, 0].max() + self.padding
        
        if self.ymin is None:
            self.ymin = coord[:, 1].min() - self.padding
        
        if self.ymax is None:
            self.ymax = coord[:, 1].max() + self.padding
        
        BINS = fixedwidth_bins(self.delta, [self.xmin, self.ymin],
                               [self.xmax, self.ymax])
        arange = np.transpose(np.vstack((BINS['min'], BINS['max'])))
        self.nx, self.ny = BINS['Nbins']

        self.xbins = np.linspace(self.xmin, self.xmax, num=self.nx+1)
        self.ybins = np.linspace(self.ymin, self.ymax, num=self.ny+1)

        self.n_pairs = self.leafletfinder.n_leaflets / 2
        self.thickness = np.zeros((self.n_frames, self.ny, self.nx))
    

    def _single_frame(self):
        upper_z = np.zeros((self.ny, self.nx))
        lower_z = np.zeros((self.ny, self.nx))

        n_upper = np.zeros((self.ny, self.nx))
        n_lower = np.zeros((self.ny, self.nx))

        upper_rix = self._relevant_lf_rix[0]
        upper = self.leafletfinder.coordinates[upper_rix]
        lower_rix = self._relevant_lf_rix[1]
        lower = self.leafletfinder.coordinates[lower_rix]

        xu = np.searchsorted(self.xbins, upper[:, 0]) - 1
        yu = np.searchsorted(self.ybins, upper[:, 1]) - 1
        xl = np.searchsorted(self.xbins, lower[:, 0]) - 1
        yl = np.searchsorted(self.ybins, lower[:, 1]) - 1

        for x, y, z in zip(xu, yu, upper[:, 2]):
            try:
                upper_z[y, x] += z
                n_upper[y, x] += 1
            except IndexError:
                # outside the bounds :(
                pass
        
        for x, y, z in zip(xl, yl, lower[:, 2]):
            try:
                lower_z[y, x] += z
                n_lower[y, x] += 1
            except IndexError:
                # outside the bounds :(
                pass

        mask = (n_upper > 0) & (n_lower > 0)
        upper_z[mask] /= n_upper[mask]
        lower_z[mask] /= n_lower[mask]

        self.thickness[self._frame_index][mask] = upper_z[mask]
        self.thickness[self._frame_index][mask] -= lower_z[mask]


        