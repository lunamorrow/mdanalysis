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
from collections import defaultdict
import itertools

from .base import LeafletAnalysis
from ..distances import capped_distance
from ...lib.c_distances import unwrap_around

def pad_box(coordinates, box, ignore_axis=2, cutoff=30):
    axes = [x for x in [0, 1, 2] if x != ignore_axis]

    extra = []
    for ax in axes:
        box_len = box[ax]
        sub = coordinates[coordinates[:, ax] > box_len - cutoff]
        add = coordinates[coordinates[:, ax] < cutoff]
        diff = np.zeros(3)
        diff[ax] = box_len
        extra.append(sub - diff)
        extra.append(add + diff)

    x, y = axes[:2]
    xlen, ylen = box[[x, y]]
    x0 = coordinates[:, x] < cutoff
    y0 = coordinates[:, y] < cutoff
    x1 = coordinates[:, x] > xlen - cutoff
    y1 = coordinates[:, y] > ylen - cutoff

    x0y0 = coordinates[(x0 & y0)] + [x, y, 0]
    x1y1 = coordinates[(x1 & y1)] - [x, y, 0]
    x0y1 = coordinates[(x0 & y1)] + [x, -y, 0]
    x1y0 = coordinates[(x1 & y0)] + [-x, y, 0]

    extra.extend([x0y0, x1y1, x0y1, x1y0])

    coordinates = np.concatenate([coordinates] + extra)
    return coordinates



def leaflet_apl(coordinates, box=None, cutoff=30):
    from scipy.spatial import Voronoi, voronoi_plot_2d
    import matplotlib.pyplot as plt


    n_coordinates = len(coordinates)
    areas = np.zeros(n_coordinates)
    
    if box is not None:
        coordinates = pad_box(coordinates, box, cutoff=cutoff)


    # first grab neighbors in 3D
    vor = Voronoi(coordinates)
    # assert -1 not in vor.point_region[:n_coordinates]
    mask = (vor.ridge_points <= n_coordinates).any(axis=1)
    left, right = vor.ridge_points[mask].T
    pt_range = defaultdict(set)
    for i in range(n_coordinates):
        r = right[left == i]
        l = left[right == i]
        neighbors = set(itertools.chain.from_iterable([r, l]))
        pt_range[i] = neighbors
    
    for i in range(n_coordinates):
        # get neighbors, and neighbors thereof
        # we need the second degree to project onto the right plane
        nbs = [pt_range[j] for j in pt_range[i]]
        indices = set(itertools.chain.from_iterable(nbs))
        indices |= pt_range[i]

        # Voronoi surface of the flat plane. Necessary??
        try:
            indices.remove(i)
        except KeyError:
            pass
        indices = [0, *indices]
        points = coordinates[indices]
        center = points.mean(axis=0)
        points = points - center
        Mt_M = np.matmul(points.T, points)
        u, s, vh = np.linalg.linalg.svd(Mt_M)
        xy = np.matmul(points, vh[:2].T)
        vor2 = Voronoi(xy)
        headgroup_cell_int = vor2.point_region[0]
        headgroup_cell = vor2.regions[headgroup_cell_int]
        x, y = np.array([vor2.vertices[x] for x in headgroup_cell]).T
        area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
        area += (x[-1] * y[0] - y[-1] * x[0])
        lipid_area = 0.5 * np.abs(area)
        areas[i] = lipid_area
    
    return areas


        



def lipid_area(headgroup_coordinate,
               neighbor_coordinates,
               other_coordinates=None,
               box=None, plot=False):
    """
    Calculate the area of a lipid by projecting it onto a plane with
    neighboring coordinates and creating a Voronoi diagram.

    Parameters
    ----------
    headgroup_coordinate: numpy.ndarray
        Coordinate array of shape (3,) or (n, 3) of the central lipid
    neighbor_coordinates: numpy.ndarray
        Coordinate array of shape (n, 3) of neighboring lipids to the central lipid.
        These coordinates are used to construct the plane of best fit.
    other_coordinates: numpy.ndarray (optional)
        Coordinate array of shape (n, 3) of neighboring atoms to the central lipid.
        These coordinates are *not* used to construct the plane of best fit, but
        are projected onto it.
    box: numpy.ndarray (optional)
        Box of minimum cell, used for unwrapping coordinates.
    plot: bool (optional)
        Whether to plot the resulting Voronoi diagram.

    Returns
    -------
    area: float
        Area of the central lipid
    
    Raises
    ------
    ValueError
        If a Voronoi cell cannot be constructed for the central lipid, usually
        because too few neighboring lipids have been given.
    """
    from scipy.spatial import Voronoi
    
    # preprocess coordinates
    headgroup_coordinate = np.asarray(headgroup_coordinate)
    if len(headgroup_coordinate.shape) > 1:
        if box is not None:
            headgroup_coordinates = unwrap_around(headgroup_coordinate.copy(),
                                                headgroup_coordinate[0],
                                                box)
        headgroup_coordinate = headgroup_coordinates.mean(axis=0)
    if box is not None:
        neighbor_coordinates = unwrap_around(neighbor_coordinates.copy(),
                                             headgroup_coordinate,
                                             box)
        if other_coordinates is not None:
            other_coordinates = np.asarray(other_coordinates).copy()
            other_coordinates = unwrap_around(other_coordinates,
                                              headgroup_coordinate,
                                              box)
    points = np.concatenate([[headgroup_coordinate], neighbor_coordinates])
    points -= headgroup_coordinate
    center = points.mean(axis=0)
    points -= center

    Mt_M = np.matmul(points.T, points)
    u, s, vh = np.linalg.linalg.svd(Mt_M)
    # project onto plane
    if other_coordinates is not None:
        points = np.r_[points, other_coordinates-center]
    xy = np.matmul(points, vh[:2].T)
    xy -= xy[0]
    # voronoi
    vor = Voronoi(xy)
    headgroup_cell_int = vor.point_region[0]
    headgroup_cell = vor.regions[headgroup_cell_int]
    if plot:
        from scipy.spatial import voronoi_plot_2d
        import matplotlib.pyplot as plt
        fig = voronoi_plot_2d(vor, show_vertices=False, line_alpha=0.6)
        plt.plot([vor.points[0][0]], [vor.points[0][1]], 'r+', markersize=12)
        plt.show()

    if not all(vertex != -1 for vertex in headgroup_cell):
        raise ValueError("headgroup not bounded by Voronoi cell points: "
                            f"{headgroup_cell}. "
                            "Try including more neighbor points")
    # x and y should be ordered clockwise
    x, y = np.array([vor.vertices[x] for x in headgroup_cell]).T
    area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
    area += (x[-1] * y[0] - y[-1] * x[0])
    lipid_area = 0.5 * np.abs(area)

    
    # if lipid_area < 5 or lipid_area > 100:
    #     print(lipid_area)
        # fig = voronoi_plot_2d(vor, show_vertices=False, line_alpha=0.6)
        # plt.plot([vor.points[0][0]], [vor.points[0][1]], 'r+', markersize=12)
        # plt.show()
    return lipid_area


class AreaPerLipid(LeafletAnalysis):
    """
    Calculate the area of each lipid by projecting it onto a plane with
    neighboring coordinates and creating a Voronoi diagram.

    Parameters
    ----------
    """

    def __init__(self, universe, *args, cutoff=50, cutoff_other=None, select_other=None,
                 max_neighbors=100, **kwargs):
        super().__init__(universe, *args, **kwargs)
        self.max_neighbors = max_neighbors
        if select_other is None:
            self.other = (self.universe.residues - self.residues).atoms
        else:
            self.other = universe.select_atoms(select_other) - self.residues.atoms
        self.cutoff = cutoff
        if cutoff_other is None:
            cutoff_other = cutoff
        self.cutoff_other = cutoff_other
        self.unique_ids = np.unique(self.ids)
        self.resindices = self.residues.resindices
        self.rix2hg = {ag.residues[0].resindex: ag for ag in self.headgroups}
        self.n_per_res = np.array([len(x) for x in self.headgroups])

    

    def _prepare(self):
        super()._prepare()
        self.areas = np.zeros((self.n_frames, self.n_residues))
        self.areas_by_attr = []
        for i in range(self.n_leaflets):
            dct = {}
            for each in self.unique_ids:
                dct[each] = []
            self.areas_by_attr.append(dct)
    
    def _single_frame(self):
        other = self.other.positions
        box = self.universe.dimensions
        rix2lfi = {}
        components = []
        leaflets = []

        for lf_i, comp in enumerate(self.leafletfinder.components):
            ag_i = sorted(self._lf_res_i & set(comp))
            self_i = self._lf2a[ag_i]
            assert -1 not in self_i
            coords = self.leafletfinder.coordinates[ag_i]

            # areas = leaflet_apl(coords, box=box, cutoff=self.cutoff)
            # self.areas[self._frame_index][self_i] = areas

            # for rid, area in zip(self.ids[self_i], areas):
            #     self.areas_by_attr[lf_i][rid].append(area)

    #         raise ValueError()




            pairs, dists = capped_distance(coords, coords,
                                           self.cutoff,
                                           box=box,
                                           return_distances=True)
            
            if not len(pairs):
                continue
            # pairs = pairs[dist > 0]
            # dists = dists[dist > 0]
            splix = np.where(np.ediff1d(pairs[:, 0]))[0] + 1
            plist = np.split(pairs, splix)
            dlist = np.split(dists, splix)

            d_order = [np.argsort(x) for x in dlist]
            plist = [p[x] for p, x in zip(plist, d_order)]

            # splix = np.where(np.ediff1d(pi))[0]+1
            # pi = np.split(pi, splix)
            # pj = np.split(pj, splix)
            # dist = np.split(dist, splix)
            # dist_order = [np.argsort(x) for x in dist]
            # i2j = {i[0]:j[d] for i, j, d in zip(pi, pj, dist_order)}
            i2j = {p[0, 0]: p[1:, 1] for p in plist}

            for i, resi in enumerate(ag_i):
                hg_xyz = coords[i]
                # js = pj[pi == i]
                try:
                    js = i2j[i]
                except KeyError:
                    continue
                neighbor_xyz = coords[js[:self.max_neighbors]]

                rix = self.i2resix[resi]
                rid = self.rix2id[rix]
                ri = self.rix2ix[rix]
                pairs2 = capped_distance(hg_xyz, other, self.cutoff_other,
                                         box=box,
                                         return_distances=False)

                if len(pairs2):
                    other_xyz = other[np.unique(pairs2[:, 1])]
                else:
                    other_xyz = None

                area = lipid_area(hg_xyz, neighbor_xyz,
                                  other_coordinates=other_xyz,
                                  box=box)
                if area > 0:
                    self.areas[self._frame_index][ri] = area
                    self.areas_by_attr[lf_i][rid].append(area)


        # for lf_i, x in enumerate(self.leafletfinder.leaflets):
        #     ix = []
        #     atoms = []
        #     for y in x.residues.resindices:
        #         rix2lfi[y] = i
        #         if y in self.resindices:
        #             ix.append(self.rix2ix[y])
        #             atoms.extend(self.headgroups[self.rix2ix[y]])
        #     components.append(np.array(ix))
        #     leaflets.append(sum(atoms))

        # hg_coords = [unwrap_around(x.positions.copy(), x.positions.copy()[0], box)
        #              for x in self.headgroups]
        # hg_mean = np.array([x.mean(axis=0) for x in hg_coords])

        # all_wrapped = [hg_mean[x] for x in components]

        
        # for i, rix in enumerate(self.resindices):
        #     hg_xyz = hg_mean[i]
        #     try:
        #         lf_i = rix2lfi[rix]
        #     except KeyError:
        #         self.areas[self._frame_index][i] = np.nan
        #         continue
        #     potential_xyz = all_wrapped[lf_i]
        #     # hg_xyz = self.headgroups[i].positions
        #     # potential_xyz = leaflets[lf_i].positions

        #     pairs, dist = distances.capped_distance(hg_xyz,
        #                                             potential_xyz,
        #                                             self.cutoff,
        #                                             box=self.selection.dimensions,
        #                                             return_distances=True)

        #     if not len(pairs):
        #         continue            
        #     pairs = pairs[dist>0]
        #     js = np.unique(pairs[:, 1])
        #     neighbor_xyz = potential_xyz[js]

        #     # get protein / etc ones
        #     pairs2 = distances.capped_distance(hg_xyz, other, self.cutoff_other,
        #                                        box=self.selection.dimensions,
        #                                        return_distances=False)
        #     if len(pairs2):
        #         other_xyz = other[np.unique(pairs2[:, 1])]
        #     else:
        #         other_xyz = None
        #     res = self.residues[i]
        #     try:
        #         area = lipid_area(hg_xyz, neighbor_xyz,
        #                         other_coordinates=other_xyz,
        #                         box=self.selection.dimensions)
        #     except:
        #         print(i)
        #         raise ValueError()
        #     self.areas[self._frame_index][i] = area
        #     self.areas_by_attr[lf_i][self.ids[i]].append(area)

    # def _conclude(self):
    #     super()._conclude()
    #     self.mean_area_by_attr = []
    #     self.std_area_by_attr = []
    #     for n in range(self.n_leaflets):
    #         mean = {}
    #         std = {}
    #         for id_ in self.ids:
    #             values = np.array(self.areas_by_attr[n][id_])
    #             mean[id_] = values.mean()
    #             std[id_] = values.std()
    #         self.mean_area_by_attr.append(mean)
    #         self.std_area_by_attr.append(std)