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

from ..distances import capped_distance
from ...lib.mdamath import norm
from ...lib.c_distances import unwrap_around, mean_unwrap_around, calc_cosine_similarity

def get_centers_by_residue(selection, centers=None, box=None):
    if box is None:
        return selection.center(None, compound='residues', pbc=False)
    # res = selection.split('residue')
    # sel = [x.positions for x in res]
    # print(np.ediff1d(selection.resindices))
    splix = np.where(np.ediff1d(selection.resindices))[0]+1
    sel = np.split(selection.positions, splix)
    if centers is None:
        centers = [x[0] for x in sel]

    
    # uw = [unwrap_around(x, c, box) for x, c in zip(sel, centers)]
    # unwrapped = np.array([x.mean(axis=0) for x in uw])
    # print(unwrapped)
    unwrapped = np.array([mean_unwrap_around(x, c, box) for x, c in zip(sel, centers)])
    return unwrapped


def get_orientations(residues, headgroups, box=None, headgroup_centers=None,
                     normalize=False):
    if headgroup_centers is None:
        headgroup_centers = get_centers_by_residue(headgroups, box=box)
    other = residues.atoms - headgroups
    other_centers = get_centers_by_residue(other, centers=headgroup_centers,
                                           box=box)
    orientations = other_centers - headgroup_centers
    if normalize:
        norms = np.linalg.norm(orientations, axis=1)
        orientations /= norms.reshape(-1, 1)
    return orientations


def average_near_orientations(orientations, pairs, dists, average_neighbors=0,
                              max_dist=30, angles=None):
    if angles is None:
        angles = np.dot(orientations, orientations.T)
    
    mask = (dists <= max_dist) & (angles[tuple(pairs.T)] > 0)
    dists = dists[mask]
    pairs = pairs[mask]

    splix = np.where(np.ediff1d(pairs[:, 0]))[0] + 1
    plist = np.split(pairs, splix)
    dlist = np.split(dists, splix)

    for p, d in zip(plist, dlist):
        i = p[0, 0]
        js = p[1:, 1]  # first is self-to-self
        d = d[1:]

        dist_order = np.argsort(d)
        nearest_j = js[dist_order][:average_neighbors]
        nearest = orientations[nearest_j]
        vec = orientations[i]
        vec += nearest.sum(axis=0)
        vec /= norm(vec)
        orientations[i] = vec
    




def get_distances_with_projection(coordinates, orientations, cutoff, box=None,
                                  angle_factor=1, average_neighbors=0,
                                  max_dist=30, angles=None,
                                  average_orientations=False):
    n_coordinates = len(coordinates)
    # set up distance matrix
    filler = (angle_factor + 1) * cutoff
    dist_mat = np.ones((n_coordinates, n_coordinates)) * filler
    dist_mat[np.diag_indices(n_coordinates)] = 0
    pairs, dists = capped_distance(coordinates, coordinates, cutoff, box=box,
                                  return_distances=True)
    pi, pj = tuple(pairs.T)
    # dist_mat[pi, pj] = dists


    # split pairs + distances by residue
    splix = np.where(np.ediff1d(pairs[:, 0]))[0] + 1
    plist = np.split(pairs, splix)
    dlist = np.split(dists, splix)

    if average_orientations and average_neighbors:
        average_near_orientations(orientations, pairs, dists,
                                  average_neighbors=average_neighbors,
                                  max_dist=max_dist, angles=angles)
        # if angles is None:
        #     angles = np.dot(orientations, orientations.T)

        # row = np.arange(n_coordinates)

    # project distances onto orientation vector
    for p, d in zip(plist, dlist):
        i = p[0, 0]
        js = p[1:, 1]  # first is self-to-self
        d = d[1:]
        i_coord = coordinates[i]
        neigh_ = coordinates[js].copy()

        if box is not None:
            unwrap_around(neigh_, i_coord, box[:3])
        neigh_ -= i_coord
        
        vec = orientations[[i]]
        # if average_orientations and average_neighbors:
        #     # within_threshold = d <= max_dist
        #     # acute = angles[i][js] > 0
        #     # mask = within_threshold & acute
        #     # dist_order = np.argsort(d[mask])
        #     # nearest_j = js[mask][dist_order][:average_neighbors]




        #     dist_order = np.argsort(d)
        #     within_threshold = d[dist_order] <= max_dist
        #     acute = angles[i][js[dist_order]] > 0
        #     mask = within_threshold & acute
        #     nearest_j = js[dist_order[mask][:average_neighbors]]
        #     nearest = orientations[nearest_j]
        #     # print(angles[i][js[dist_order]])
        #     # print(nearest)

        #     neigh_orients = np.array([vec] + list(nearest))
        #     vec = neigh_orients.mean(axis=0)
        #     # vec += nearest.sum(axis=0)
        #     vec /= norm(vec[0])
        #     orientations[i] = vec

        ang_ = calc_cosine_similarity(vec, neigh_)
        # ang_ = np.nan_to_num(ang_, nan=1)
        
        proj = np.abs(d * ang_)
        half = (proj * angle_factor)[0] #/ 2
        dist_mat[i, js] = half + d
        # dist_mat[js, i] += half

    dist_mat += dist_mat.T
    dist_mat /= 2
    # dist_mat /= angle_factor + 1

    # dist_mat[dist_mat == 0] = filler
    # dist_mat[np.diag_indices(n_coordinates)] = 0
        
    return dist_mat