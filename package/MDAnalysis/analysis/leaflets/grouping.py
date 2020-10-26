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

import warnings

import numpy as np

from ..distances import contact_matrix
from .utils import (get_centers_by_residue, get_distances_with_projection,
                    get_orientations)

def group_by_graph(residues, headgroups, cutoff=15.0, sparse=None, box=None,
                   return_predictor=False, **kwargs):
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx is required to use this method "
                          "but is not installed. Install it with "
                          "`conda install networkx` or "
                          "`pip install networkx`.") from None
    returntype = "numpy" if not sparse else "sparse"

    coordinates = get_centers_by_residue(headgroups, box=box)

    try:
        adj = contact_matrix(coordinates, cutoff=cutoff, box=box,
                             returntype=returntype)
    except ValueError as exc:
        if sparse is None:
            warnings.warn("NxN matrix is too big. Switching to sparse matrix "
                          "method")
            adj = contact_matrix(coordinates, cutoff=cutoff, box=box,
                                 returntype="sparse")
        elif sparse is False:
            raise ValueError("NxN matrix is too big. "
                             "Use `sparse=True`") from None
        else:
            raise exc

    graph = nx.Graph(adj)
    groups = [np.sort(list(c)) for c in nx.connected_components(graph)]
    if return_predictor:
        return (groups, graph)
    else:
        return groups


def group_by_dbscan(residues, headgroups, angle_threshold=0.8,
                    return_predictor=False, cutoff=30, box=None,
                    eps=None, min_samples=10, **kwargs):
    try:
        import sklearn.cluster as skc
    except ImportError:
        raise ImportError('scikit-learn is required to use this method '
                          'but is not installed. Install it with `conda '
                          'install scikit-learn` or `pip install '
                          'scikit-learn`.') from None
    
    coordinates = get_centers_by_residue(headgroups, box=box)
    orientations = get_orientations(residues, headgroups, box=box,
                                    headgroup_centers=coordinates,
                                    normalize=True)
    dist_mat = get_distances_with_projection(coordinates, orientations,
                                             cutoff, box=box)
    if eps is None:
        eps = cutoff
    
    angles = np.dot(orientations, orientations.T)
    angles += 1
    # angles /= 2

    db = skc.DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    clusters = db.fit_predict(dist_mat / angles)
    ix = np.argsort(clusters)
    indices = np.arange(len(coordinates))
    groups = np.split(indices[ix], np.where(np.ediff1d(clusters[ix]))[0]+1)
    groups = [np.sort(x) for x in groups]
    outliers = []
    if clusters[ix[0]] == -1:
        outliers = [groups.pop(0)]
    groups.sort(key=lambda x: len(x), reverse=True)
    groups += outliers
    if return_predictor:
        return (groups, db)
    return groups
    

def group_by_spectralclustering(residues, headgroups, n_leaflets=2, delta=20,
                                cutoff=30, box=None, angle_threshold=0.8,
                                return_predictor=False):
    try:
        import sklearn.cluster as skc
    except ImportError:
        raise ImportError('scikit-learn is required to use this method '
                          'but is not installed. Install it with `conda '
                          'install scikit-learn` or `pip install '
                          'scikit-learn`.') from None
    
    coordinates = get_centers_by_residue(headgroups, box=box)
    orientations = get_orientations(residues, headgroups, box=box,
                                    headgroup_centers=coordinates,
                                    normalize=True)
    dist_mat = get_distances_with_projection(coordinates, orientations,
                                             cutoff, box=box)
    
    if delta is None:
        delta = np.max(dist_mat[dist_mat < cutoff*2]) / 3
    gau = np.exp(- dist_mat ** 2 / (2. * delta ** 2))
    # reasonably acute/obtuse angles are acute/obtuse anough
    angles = np.dot(orientations, orientations.T)
    cos = np.clip(angles, -angle_threshold, angle_threshold)
    cos += angle_threshold
    cos /= (2*angle_threshold)
    ker = gau * cos

    sc = skc.SpectralClustering(n_clusters=n_leaflets, affinity="precomputed")
    clusters = sc.fit_predict(ker)

    ix = np.argsort(clusters)
    indices = np.arange(len(coordinates))
    groups = np.split(indices[ix], np.where(np.ediff1d(clusters[ix]))[0]+1)
    groups = [np.sort(x) for x in groups]
    groups.sort(key=lambda x: len(x), reverse=True)
    if return_predictor:
        return (groups, sc)
    return groups