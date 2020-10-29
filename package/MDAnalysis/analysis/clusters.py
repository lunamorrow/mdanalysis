# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
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

class Clusters:
    def __init__(self, predictor=None, **kwargs):
        if isinstance(predictor, type):
            predictor = predictor(**kwargs)
        self.predictor = predictor
        self.cluster_indices = []
        self.clusters_by_size = []
        self.outlier_indices = []
        self.data_labels = []
    
    def run(self, data):
        self.data_labels = self.predictor.fit_predict(data)
        ix = np.argsort(self.data_labels)
        indices = np.arange(len(data))
        splix = np.where(np.ediff1d(self.data_labels[ix]))[0] + 1
        self.cluster_indices = np.split(indices[ix], splix)
        self.cluster_indices = [np.sort(x) for x in self.cluster_indices]
        if self.data_labels[ix[0]] == -1:
            self.outlier_indices = self.cluster_indices.pop(0)
        self.clusters_by_size = sorted(self.cluster_indices,
                                       key=lambda x: len(x), reverse=True)

    def set_clusters(self, cluster_indices):
        self.cluster_indices = cluster_indices
        self.clusters_by_size = sorted(self.cluster_indices,
                                       key=lambda x: len(x), reverse=True)
        labels = np.zeros(sum(map(len, cluster_indices)))
        for i, cl in enumerate(cluster_indices):
            labels[cl] = i
        self.data_labels = labels