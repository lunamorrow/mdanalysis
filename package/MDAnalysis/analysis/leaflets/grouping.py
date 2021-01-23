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
    ker = gau
    mask = ~np.isnan(cos)
    gau[mask] *= cos[mask]

    clusters = Clusters(skc.SpectralClustering, n_clusters=n_leaflets,
                        affinity="precomputed")
    clusters.run(ker)
    return clusters

def sort_clusters(sorted_dist, min_cosine, max_dist, max_neighbors, sorted_angles, groups, sorted_ix):
    # nearests = np.argsort(dist_mat, axis=1)
    # print("max_dist:", max_dist, "min_cosine::", min_cosine)
    valid = (sorted_dist <= max_dist) & (sorted_angles > min_cosine)

    # dists = np.argsort(dist_mat[valid], axis=1)
    # print(dists)
    # nearests =  indices[dists]#[:, :max_neighbors]
    # print(nearests)

    for i, valid_ in enumerate(valid):
        # order = nb_order[i]
        nearest = sorted_ix[i][valid_]
        # nearest = nb_ix[i][order][valid_[order]]
        # direction = angles[i][nearest] > min_cosine
        # within_min_dist = row[nearest] <= max_dist
        # neighbors = nearest[within_min_dist & direction][:max_neighbors]
        neighbors = nearest[:max_neighbors]

        common = np.bincount(groups[neighbors])
        common = common.argmax()
        groups[neighbors] = common
        # print(len(neighbors), len(np.unique(groups)))

def sort_clusters_(dists, angs, min_cosine, max_dist, max_neighbors, groups, new_ix):
    for i, row in enumerate(dists):
        direction = angs[i] > min_cosine
        within_min_dist = row <= max_dist
        mask = direction & within_min_dist
        nearest = new_ix[i][mask]
        # nearest = new_ix[mask]
        # nearest = nearest[np.argsort(row[mask])]
        neighbors = nearest[:max_neighbors]
        # neighbors = nearest[within_min_dist & direction][:max_neighbors]
        common = np.bincount(groups[neighbors])
        common = common.argmax()

        # ids, counts = np.unique(groups[neighbors], )
        groups[neighbors] = common

def group_by_orientation(residues, headgroups, n_leaflets=2,
                         cutoff=50, box=None, min_cosine=0.3,
                         max_neighbors=20, max_dist=40,
                         min_lipids=5, angle_factor=5,
                         coordinates=None, **kwargs):
    n_coordinates = len(coordinates)
    if coordinates is None:
        coordinates = get_centers_by_residue(headgroups, box=box)
    else:
        assert n_coordinates == len(residues)
    orientations = get_orientations(residues, headgroups, box=box,
                                    headgroup_centers=coordinates,
                                    normalize=True)
    angles = np.dot(orientations, orientations.T)

    dist_mat = get_distances_with_projection(coordinates, orientations,
                                             cutoff, box=box,
                                             angles=angles,
                                             angle_factor=angle_factor,
                                             average_neighbors=min_lipids-1,
                                             max_dist=max_dist,
                                             average_orientations=True)
    
    groups = np.arange(n_coordinates)
    old_min_cosine = min_cosine
    row_ix = np.arange(n_coordinates)

    mask = (dist_mat <= (max_dist * 2)) & (angles > 0)
    new_ix = [row_ix[x] for x in mask]
    dists = [d[x] for d, x in zip(dist_mat, new_ix)]
    angs = [a[x] for a, x in zip(angles, new_ix)]

    # dists = dist_mat
    # angs = angles
    # new_ix = row_ix

    dist_order = [np.argsort(x) for x in dists]
    dists = [d[x] for d, x in zip(dists, dist_order)]
    angs = [a[x] for a, x in zip(angs, dist_order)]
    new_ix = [i[x] for i, x in zip(new_ix, dist_order)]

    # def sort_clusters_():
    #     for i, row in enumerate(dists):
    #         direction = angs[i] > min_cosine
    #         within_min_dist = row <= max_dist
    #         mask = direction & within_min_dist
    #         nearest = new_ix[i][mask][np.argsort(row[mask])]
    #         neighbors = nearest[:max_neighbors]
    #         # neighbors = nearest[within_min_dist & direction][:max_neighbors]
    #         common = np.bincount(groups[neighbors]).argmax()
    #         groups[neighbors] = common

        # for i, row in enumerate(dist_mat):
        #     # nearest = np.argsort(row)
        #     direction = angles[i] > min_cosine
        #     within_min_dist = row <= max_dist
        #     mask = direction & within_min_dist
        #     nearest = row_ix[mask][np.argsort(row[mask])]
        #     neighbors = nearest[:max_neighbors]
        #     # neighbors = nearest[within_min_dist & direction][:max_neighbors]
        #     common = np.bincount(groups[neighbors]).argmax()
        #     groups[neighbors] = common


    # nb_ix = np.tile(groups, (n_coordinates, 1))
    # nb_order = np.argsort(dist_mat, axis=1)

    # sorted_dist = np.array([x[y] for x, y in zip(dist_mat, nb_order)])
    # sorted_angles = np.array([x[y] for x, y in zip(angles, nb_order)])
    # sorted_ix = np.array([x[y] for x, y in zip(nb_ix, nb_order)])
    
    # min_cosine += 0.3
    sort_clusters_(dists, angs, min_cosine, max_dist, max_neighbors, groups, new_ix)
    # sort_clusters(sorted_dist, min_cosine, max_dist, max_neighbors, sorted_angles, groups, sorted_ix)
    old_max_dist = max_dist
    old_min_cosine = min_cosine
    
    ids, counts = np.unique(groups, return_counts=True)
    outliers = counts < min_lipids
    n_squash = len(ids[~outliers]) - n_leaflets
    n_sorts = 3

    while n_squash > 0 and n_sorts:
        sort_clusters_(dists, angs, min_cosine, max_dist, max_neighbors, groups, new_ix)
        # sort_clusters(sorted_dist, min_cosine, max_dist, max_neighbors, sorted_angles, groups, sorted_ix)
        ids, counts = np.unique(groups, return_counts=True)
        outliers = counts < min_lipids
        n_squash = len(ids[~outliers]) - n_leaflets
        max_dist *= 1.2
        min_cosine -= 0.1
        n_sorts -= 1

    min_cosine = old_min_cosine
    max_dist = old_max_dist
    ids, counts = np.unique(groups, return_counts=True)
    outliers = counts < min_lipids
    n_squash = len(ids[~outliers]) - n_leaflets
    indices = [np.where(groups == i)[0] for i in ids[np.argsort(counts)]]

    if n_squash > 0:
        others = []
        
        # same leaflet as most nearest neighbors
        copy = dist_mat.copy()
        copy[np.diag_indices(len(copy))] = np.inf
        copy[angles < min_cosine] = np.inf

        while len(indices) > (n_leaflets+0):
            ix = indices.pop(0)
            min_dist = copy[ix].min(axis=0)
            nearest = np.argsort(min_dist)
            # neighbors = np.unique(np.where(copy[ix] <= max_dist)[1])
            # neighbors = [x for x in neighbors if x not in ix]

            neighbors = np.where(min_dist[nearest] <= max_dist)[0]
            neighbors = nearest[neighbors]
            # print(neighbors, min_dist[neighbors])
            # print(min_dist[nearest[neighbors]])
            # neighbors = nearest[neighbors]
            # neighbors = np.unique(nearest[neighbors])#[:max_neighbors]
            neighbors = [x for x in neighbors if x not in ix]
            # print(np.min(min_dist), min_dist[neighbors])
            # neighbors = nearest[neighbors][:max_neighbors]
            if len(neighbors):
                counts = [sum(np.isin(neighbors, x)) for x in indices]
                cluster_id = np.argmax(counts)
                indices[cluster_id] = np.r_[indices[cluster_id], ix]
            else:
                others.append(ix)

            indices = sorted(indices, key=len)
        indices.extend(others)

    if n_squash < 0:
        raise NotImplementedError("Ehh haven't done this yet")

    groups = sorted(indices, key=len, reverse=True)
    if len(groups) > n_leaflets:
        # combine outliers
        outliers = np.concatenate(groups[n_leaflets:])
    else:
        outliers = []

    clusters = Clusters((dist_mat, orientations))
    clusters.set_clusters(groups, outlier_indices=outliers)
    return clusters