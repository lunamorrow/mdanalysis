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
from ..clusters import Clusters
from .utils import (get_centers_by_residue, get_distances_with_projection,
                    get_orientations)


def group_by_graph(residues, headgroups, cutoff=15.0, sparse=None, box=None,
                   coordinates=None, **kwargs):
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx is required to use this method "
                          "but is not installed. Install it with "
                          "`conda install networkx` or "
                          "`pip install networkx`.") from None
    returntype = "numpy" if not sparse else "sparse"

    if coordinates is None:
        coordinates = get_centers_by_residue(headgroups, box=box)
    else:
        assert len(coordinates) == len(residues)

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
    clusters = Clusters(graph)
    clusters.set_clusters(groups)
    return clusters


def group_by_dbscan(residues, headgroups, angle_threshold=0.8,
                    cutoff=20, box=None, coordinates=None,
                    eps=30, min_samples=20, angle_factor=1,
                    **kwargs):
    try:
        import sklearn.cluster as skc
    except ImportError:
        raise ImportError('scikit-learn is required to use this method '
                          'but is not installed. Install it with `conda '
                          'install scikit-learn` or `pip install '
                          'scikit-learn`.') from None
    
    if coordinates is None:
        coordinates = get_centers_by_residue(headgroups, box=box)
    else:
        assert len(coordinates) == len(residues)
    orientations = get_orientations(residues, headgroups, box=box,
                                    headgroup_centers=coordinates,
                                    normalize=True)
    dist_mat = get_distances_with_projection(coordinates, orientations,
                                             cutoff, box=box,
                                             angle_factor=angle_factor,
                                             average_neighbors=3)
    if eps is None:
        eps = cutoff
    
    angles = np.dot(orientations, orientations.T)
    angles *= -1
    angles = np.clip(angles, -angle_threshold, angle_threshold)
    angles += 1
    angles /= 2
    db = skc.DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    mask = dist_mat != dist_mat.max()
    data = dist_mat
    data[mask] += data[mask]* angles[mask]
    data[~mask] += cutoff

    clusters = Clusters(skc.DBSCAN, eps=eps, min_samples=min_samples,
                        metric="precomputed")
    clusters.run(data)
    return clusters    

def group_by_spectralclustering(residues, headgroups, n_leaflets=2, delta=20,
                                cutoff=30, box=None, angle_threshold=0.8,
                                angle_factor=1, coordinates=None,
                                **kwargs):
    try:
        import sklearn.cluster as skc
    except ImportError:
        raise ImportError('scikit-learn is required to use this method '
                          'but is not installed. Install it with `conda '
                          'install scikit-learn` or `pip install '
                          'scikit-learn`.') from None
    
    if coordinates is None:
        coordinates = get_centers_by_residue(headgroups, box=box)
    else:
        assert len(coordinates) == len(residues)
    orientations = get_orientations(residues, headgroups, box=box,
                                    headgroup_centers=coordinates,
                                    normalize=True)
    dist_mat = get_distances_with_projection(coordinates, orientations,
                                             cutoff, box=box,
                                             angle_factor=angle_factor)
    
    if delta is None:
        delta = np.max(dist_mat[dist_mat < cutoff*2]) / 3
    gau = np.exp(- dist_mat ** 2 / (2. * delta ** 2))
    # reasonably acute/obtuse angles are acute/obtuse anough
    angles = np.dot(orientations, orientations.T)
    cos = np.clip(angles, -angle_threshold, angle_threshold)
    cos += angle_threshold
    cos /= (2*angle_threshold)
    ker = gau * cos

    clusters = Clusters(skc.SpectralClustering, n_clusters=n_leaflets,
                        affinity="precomputed")
    clusters.run(ker)
    return clusters


def group_by_orientation(residues, headgroups, n_leaflets=2,
                         cutoff=50, box=None, min_cosine=0.5,
                         max_neighbors=30, max_dist=20,
                         min_lipids=10, angle_factor=0.5,
                         coordinates=None,
                         relax_dist=10, **kwargs):
    if coordinates is None:
        coordinates = get_centers_by_residue(headgroups, box=box)
    else:
        assert len(coordinates) == len(residues)
    orientations = get_orientations(residues, headgroups, box=box,
                                    headgroup_centers=coordinates,
                                    normalize=True)
    angles = np.dot(orientations, orientations.T)

    dist_mat = get_distances_with_projection(coordinates, orientations,
                                             cutoff, box=box,
                                             angle_factor=angle_factor,
                                             average_neighbors=min_lipids-1,
                                             max_dist=max_dist,
                                             average_orientations=True)
    
    groups = np.arange(len(coordinates))
    row_ix = np.arange(len(coordinates))
    old_min_cosine = min_cosine

    def sort_clusters():
        for i, row in enumerate(dist_mat):
            nearest = np.argsort(row)
            direction = angles[i][nearest] > min_cosine
            within_min_dist = row[nearest] <= max_dist
            neighbors = nearest[within_min_dist & direction][:max_neighbors]
            common = np.bincount(groups[neighbors]).argmax()
            groups[neighbors] = common
    
    min_cosine += 0.3
    sort_clusters()
    old_max_dist = max_dist
    
    ids, counts = np.unique(groups, return_counts=True)
    outliers = counts < min_lipids
    n_squash = len(ids[~outliers]) - n_leaflets
    if n_squash > 0:
        old_n = n_squash + 1
        n_dist = 10
        
        old_max_neighbors = max_neighbors
        n_cosine = 8
        while n_squash > 0 and (n_dist or n_cosine or old_n > n_squash):
            max_neighbors += 1
            old_n = n_squash
            sort_clusters()
            ids, counts = np.unique(groups, return_counts=True)
            outliers = counts < min_lipids
            n_squash = len(ids[~outliers]) - n_leaflets
            if n_dist:
                n_dist -= 1
                max_dist += (relax_dist * 0.1)
            if n_cosine:
                n_cosine -= 1
                min_cosine -= 0.05

    min_cosine = old_min_cosine
    max_dist = old_max_dist
    indices = [np.where(groups == i)[0] for i in ids[np.argsort(counts)]]

    if n_squash > 0:
        # assume we're keeping the largest ones
        # keep = indices[-n_leaflets:]
        # ditch = list(indices[:-n_leaflets])
        others = []
        
        # same leaflet as most nearest neighbors
        copy = dist_mat.copy()
        copy[np.diag_indices(len(copy))] = np.inf
        copy[angles < min_cosine] = np.inf
        while len(indices) > (n_leaflets+1):
            ix = indices.pop(0)
            min_dist = copy[ix].min(axis=0)
            if not sum(min_dist <= max_dist):
                others.append(ix)
                continue
            nearest = np.argsort(min_dist)
            neighbors = np.where(min_dist[nearest] <= max_dist)[0]
            neighbors = nearest[neighbors]#[:max_neighbors]
            if len(neighbors):
                counts = [sum(np.isin(neighbors, x)) for x in indices]
                cluster_id = np.argmax(counts)
                indices[cluster_id] = np.r_[indices[cluster_id], ix]
            else:
                indices.append(ix)

            indices = sorted(indices, key=lambda x: len(x))
        # special-case the last one
        ix = indices.pop(0)
        keeps = [len(x) for x in indices]
        cluster_id = np.argmin(keeps)
        indices[cluster_id] = np.r_[indices[cluster_id], ix]
        indices.extend(others)

    if n_squash < 0:
        raise NotImplementedError("Ehh haven't done this yet")

    groups = sorted(indices, key=lambda x: len(x), reverse=True)
    if len(groups) > n_leaflets:
        # combine outliers
        outliers = np.concatenate(groups[n_leaflets:])
    else:
        outliers = []

    clusters = Clusters((dist_mat, orientations))
    clusters.set_clusters(groups, outlier_indices=outliers)
    return clusters