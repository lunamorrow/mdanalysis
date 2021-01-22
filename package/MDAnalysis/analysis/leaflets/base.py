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
import logging

import numpy as np

from ..base import AnalysisBase, ProgressBar
from MDAnalysis.analysis.leaflets.leafletfinder import LeafletFinder


logger = logging.getLogger(__name__)

class BaseLeafletAnalysis(AnalysisBase):
    def __init__(self, universe, select="all",
                 leafletfinder=None,
                 leaflet_kwargs={}, 
                 group_by_attr="resnames",
                 pbc=True, update_leaflet_step=1, **kwargs):
        super().__init__(universe.universe.trajectory, **kwargs)
        self.universe = universe.universe
        self.selection = universe.select_atoms(select)
        self.headgroups = self.selection.split("residue")
        self.residues = self.selection.residues
        self.resindices = self.residues.resindices
        self.n_residues = len(self.residues)
        self.group_by_attr = group_by_attr
        self.ids = getattr(self.residues, self.group_by_attr)
        self._rix2ix = {r.resindex:i for i, r in enumerate(self.residues)}
        self._rix2id = {r.resindex:x for r, x in zip(self.residues, self.ids)}
        self.n_leaflets = self.leafletfinder.n_leaflets
        self.update_leaflet_step = update_leaflet_step

        if leafletfinder is None:
            if "select" not in leaflet_kwargs:
                leaflet_kwargs = dict(select=select, **leaflet_kwargs)
            leafletfinder = LeafletFinder(universe, **leaflet_kwargs)
        self.leafletfinder = leafletfinder


    def _update_leaflets(self):
        self.leafletfinder.run()

    def run(self, start=None, stop=None, step=None, verbose=None):
        """Perform the calculation

        Parameters
        ----------
        start : int, optional
            start frame of analysis
        stop : int, optional
            stop frame of analysis
        step : int, optional
            number of frames to skip between each analysed frame
        verbose : bool, optional
            Turn on verbosity
        """
        logger.info("Choosing frames to analyze")
        # if verbose unchanged, use class default
        if verbose is None:
            verbose = getattr(self, '_verbose', False)

        self._setup_frames(self._trajectory, start, stop, step)
        logger.info("Starting preparation")
        self._prepare()
        for i, ts in enumerate(ProgressBar(
                self._trajectory[self.start:self.stop:self.step],
                verbose=verbose)):
            self._frame_index = i
            self._ts = ts
            self.frames[i] = ts.frame
            self.times[i] = ts.time
            # logger.info("--> Doing frame {} of {}".format(i+1, self.n_frames))
            if not i % self.update_leaflet_step:
                self._update_leaflets()
            self._single_frame()
        logger.info("Finishing up")
        self._conclude()
        return self


class LeafletAnalysis(BaseLeafletAnalysis):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            assert self.residues.issubset(self.leafletfinder.residues)
        except AssertionError:
            raise ValueError("Residues selected for LeafletFinder "
                             "must include all residues selected for "
                             "leaflet-based analysis")
        
        
        lf_res = list(self.leafletfinder.residues)
        self._a2lf = np.array([lf_res.index(x) for x in self.residues])
        self._lf2a = np.ones(self.leafletfinder.n_residues, dtype=int) * -1
        self._lf2a[self._a2lf] = np.arange(self.n_residues)
        self._i2resix = {i:r.resindex
                         for i, r in enumerate(self.leafletfinder.residues)
                         if r in self.residues}
        self._lf_res_i = set(self._i2resix.keys())
        self._rix2ix = {r.resindex:i for i, r in enumerate(self.residues)}
        self._rix2id = {r.resindex:x for r, x in zip(self.residues, self.ids)}


    def _update_leaflets(self):
        self.leafletfinder.run()
        # leaflets of residues actually involved in analysis
        self._relevant_lf_rix = [sorted(self._lf_res_i & set(c))
                                 for c in self.leafletfinder.components]
        self._relevant_resix = [[self._i2resix[r] for r in c]
                                 for c in self._relevant_lf_rix]
        self._relevant_rix = [[self._rix2ix[r] for r in c]
                               for c in self._relevant_resix]
