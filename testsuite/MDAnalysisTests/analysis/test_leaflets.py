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
import MDAnalysis as mda
import pytest
import numpy as np

from numpy.testing import assert_equal, assert_almost_equal

from MDAnalysis.analysis.leaflets import LeafletFinder
from MDAnalysisTests.datafiles import (Martini_membrane_gro,
                                       Martini_double_membrane,
                                       DPPC_vesicle_only,
                                       DPPC_vesicle_plus,
                                       GRO_MEMPROT,
                                       XTC_MEMPROT,
                                       )


class BaseTestLeafletFinder(object):
    select = "name PO4"

    @pytest.fixture()
    def universe(self):
        return mda.Universe(self.file)

    def test_leafletfinder(self, universe, method, kwargs):
        lf = LeafletFinder(universe, select=self.select, pbc=True,
                           method=method, **kwargs)

        for found, given in zip(lf.leaflets, self.leaflet_resix):
            assert_equal(found.resindices, given,
                         err_msg="Found wrong leaflet lipids")
        

@pytest.mark.parametrize("method,kwargs", [
    ("graph", {"cutoff": 20}),
    ("spectralclustering", {"cutoff": 40}),
    ("dbscan", {"cutoff": 40})
])
class TestSinglePlanar(BaseTestLeafletFinder):
    file = Martini_membrane_gro
    leaflet_resix = [np.arange(180), np.arange(225, 405)]


@pytest.mark.parametrize("method,kwargs", [
    ("spectralclustering", {"cutoff": 40, "n_leaflets": 4}),
    ("dbscan", {"cutoff": 20, "n_leaflets": 4, "eps": 40, "angle_factor": 10})
])
class TestDoublePlanar(BaseTestLeafletFinder):
    file = Martini_double_membrane
    leaflet_resix = [np.arange(450, 630),
                     np.arange(675, 855),
                     np.arange(180),
                     np.arange(225, 405)]


class BaseTestVesicle:
    file = DPPC_vesicle_only
    select = "name PO4"
    n_leaflets = 2

    full_20 = ([0,   43,   76,  112,  141,  172,  204,
                234,  270,  301,  342,  377,  409,  441,
                474,  513,  544,  579,  621,  647,  677,
                715,  747,  771,  811,  847,  882,  914,
                951,  982, 1016, 1046, 1084, 1116, 1150,
                1181, 1210, 1246, 1278, 1312, 1351, 1375,
                1401, 1440, 1476, 1505, 1549, 1582, 1618,
                1648, 1680, 1713, 1740, 1780, 1810, 1841,
                1864, 1899, 1936, 1974, 1999, 2033, 2066,
                2095, 2127, 2181, 2207, 2243, 2278, 2311,
                2336, 2368, 2400, 2427, 2456, 2482, 2515,
                2547, 2575, 2608, 2636, 2665, 2693, 2720,
                2748, 2792, 2822, 2860, 2891, 2936, 2960,
                2992, 3017],
               [ 3,   36,   89,  139,  198,  249,  298,
                340,  388,  435,  491,  528,  583,  620,
                681,  730,  794,  831,  877,  932,  979,
                1032, 1073, 1132, 1180, 1238, 1286, 1328,
                1396, 1441, 1490, 1528, 1577, 1625, 1688,
                1742, 1782, 1839, 1910, 1945, 2005, 2057,
                2111, 2153, 2180, 2236, 2286, 2342, 2401,
                2470, 2528, 2584, 2649, 2722, 2773, 2818,
                2861, 2905, 2961])

    half_20 = ([0,   74,  134,  188,  250,  306,  362,
                452,  524,  588,  660,  736,  796,  872,
                928,  996, 1066, 1120, 1190, 1252, 1304,
                1374, 1434, 1512, 1576, 1638, 1686, 1750,
                1818, 1872, 1954, 2008, 2078, 2146, 2222,
                2296, 2346, 2398, 2460, 2524, 2590, 2646,
                2702, 2756, 2836, 2900, 2958, 3012],
               [4,   98,  228,  350,  434,  518,  614,
                696,  806,  912, 1006, 1124, 1220, 1328,
                1452, 1528, 1666, 1776, 1892, 1972, 2088,
                2174, 2264, 2410, 2520, 2626, 2766, 2854,
                2972])

    fifth_20 = ([0,  175,  355,  540,  735,  890, 1105,
                1270, 1430, 1580, 1735, 1885, 2095, 2300,
                2445, 2585, 2720, 2885, 3020],
                [5,  265,  465,  650,  915, 1095, 1325,
                    1675, 1920, 2115, 2305, 2640, 2945])

    
    @pytest.fixture()
    def universe(self):
        return mda.Universe(self.file)


@pytest.mark.parametrize("method,kwargs", [
    ("graph", {"cutoff": 25}),
    ("spectralclustering", {"cutoff": 100, "delta": 10}),
    ("dbscan", {"cutoff": 30, "angle_factor": 0.5}),
    ("dbscan", {"cutoff": 30, "eps": 10})
])
class TestVesicleFull(BaseTestVesicle):
    def test_full(self, universe, method, kwargs):
        lf = LeafletFinder(universe.atoms, select=self.select,
                           n_leaflets=self.n_leaflets, pbc=True,
                           method=method, **kwargs)
        
        for found, given in zip(lf.leaflets, self.full_20):
            assert_equal(found.residues.resindices[::20], given,
                         err_msg="Found wrong leaflet lipids")

@pytest.mark.parametrize("method,kwargs", [
    ("spectralclustering", {"cutoff": 100, "delta": 10}),
    ("dbscan", {"cutoff": 30, "eps": 10})
])
class TestVesicleHalf(BaseTestVesicle):
    def test_half(self, universe, method, kwargs):
        ag = universe.residues[::2].atoms
        lf = LeafletFinder(ag, select=self.select,
                           n_leaflets=self.n_leaflets,
                           method=method, **kwargs)
        
        for found, given in zip(lf.leaflets, self.half_20):
            assert_equal(found.resindices[::20], given,
                         err_msg="Found wrong leaflet lipids")
    

@pytest.mark.parametrize("method,kwargs", [
    ("spectralclustering", {"cutoff": 100, "delta": 10}),
])
class TestVesicleFifth(BaseTestVesicle):
    def test_fifth(self, universe, method, kwargs):
        ag = universe.residues[::5].atoms
        lf = LeafletFinder(ag, select=self.select,
                           n_leaflets=self.n_leaflets, pbc=True,
                           method=method, **kwargs)
        
        for found, given in zip(lf.leaflets, self.fifth_20):
            assert_equal(found.resindices[::20], given,
                         err_msg="Found wrong leaflet lipids")


class BaseTestMessyVesicle:
    file = DPPC_vesicle_plus
    select = "name PO4"
    n_leaflets = 2

    full_20 = ([0,   44,   77,  113,  142,  174,  206,
                238,  274,  305,  350,  385,  419,  452,
                485,  525,  556,  591,  633,  659,  689,
                728,  760,  785,  825,  862,  897,  929,
                966,  998, 1033, 1063, 1101, 1133, 1167,
                1199, 1228, 1265, 1297, 1333, 1373, 1398,
                1424, 1464, 1500, 1529, 1574, 1608, 1644,
                1674, 1706, 1739, 1767, 1809, 1839, 1870,
                1893, 1929, 1966, 2005, 2030, 2064, 2097,
                2126, 2159, 2214, 2240, 2276, 2311, 2345,
                2370, 2402, 2435, 2462, 2491, 2517, 2550,
                2583, 2612, 2646, 2674, 2703, 2731, 2758,
                2786, 2831, 2862, 2900, 2931, 2977, 3002,
                3034, 3059],
            [3,   37,   90,  140,  200,  253,  302,
                348,  396,  445,  502,  540,  595,  632,
                693,  743,  808,  846,  892,  947,  995,
                1049, 1090, 1149, 1198, 1257, 1306, 1350,
                1419, 1465, 1514, 1553, 1603, 1651, 1714,
                1769, 1811, 1868, 1940, 1975, 2036, 2088,
                2143, 2185, 2213, 2269, 2319, 2376, 2436,
                2505, 2563, 2621, 2687, 2760, 2812, 2858,
                2901, 2945, 3003])

    half_20 = ([4,   90,  186,  294,  404,  518,  614,
                712,  818,  894,  980, 1072, 1156, 1274,
                1354, 1472, 1546, 1652, 1780, 1892, 1972,
                2080, 2192, 2282, 2426, 2580, 2696, 2818,
                2932, 3064],
            [0,   74,  148,  212,  280,  338,  414,
                470,  544,  616,  684,  752,  822,  902,
                982, 1050, 1132, 1186, 1250, 1336, 1398,
                1466, 1556, 1612, 1674, 1728, 1798, 1864,
                1930, 2014, 2070, 2128, 2190, 2268, 2330,
                2380, 2442, 2490, 2546, 2608, 2674, 2728,
                2784, 2852, 2912, 2982, 3034])
    fifth_20 = ([0,  175,  360,  495,  725,  855, 1045,
                1180, 1345, 1515, 1700, 1840, 2030, 2220,
                2360, 2485, 2635, 2780, 2980],
                [5,  245,  515,  705,  980, 1295, 1535,
                1805, 2025, 2250, 2610, 2875])

    
    @pytest.fixture()
    def universe(self):
        return mda.Universe(self.file)

@pytest.mark.parametrize("method,kwargs", [
    ("dbscan", {"cutoff": 30, "eps": 10})
])
class TestMessyVesicleFull(BaseTestMessyVesicle):
    def test_full(self, universe, method, kwargs):
        lf = LeafletFinder(universe.atoms, select=self.select,
                           n_leaflets=self.n_leaflets, pbc=True,
                           method=method, **kwargs)
        
        for found, given in zip(lf.leaflets, self.full_20):
            assert_equal(found.residues.resindices[::20], given,
                         err_msg="Found wrong leaflet lipids")

@pytest.mark.parametrize("method,kwargs", [
    ("dbscan", {"cutoff": 30, "eps": 10})
])
class TestMessyVesicleHalf(BaseTestMessyVesicle):
    def test_half(self, universe, method, kwargs):
        ag = universe.residues[::2].atoms
        lf = LeafletFinder(ag, select=self.select,
                           n_leaflets=self.n_leaflets,
                           method=method, **kwargs)
        
        for found, given in zip(lf.leaflets, self.half_20):
            assert_equal(found.resindices[::20], given,
                         err_msg="Found wrong leaflet lipids")
    
