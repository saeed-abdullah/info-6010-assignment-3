# -*- coding: utf-8 -*-
"""
    test_network.py
    ~~~~~~~~~~~~~~

    Test cases for network.py
"""
from __future__ import division

import unittest2

import network

class TestGraph(unittest2.TestCase):

    def test_add_edge(self):
        g = network.Graph(3)

        # Self loop
        g.add_edge(1, 1)

        # Add same edge
        g.add_edge(1, 2)
        g.add_edge(1, 2)

        self.assertFalse(1 in g._edges[1])
        self.assertTrue(2 in g._edges[1])
        self.assertTrue(1 in g._edges[2])

        self.assertEquals(len(g._edges[1]), 1)
        self.assertEquals(len(g._edges[2]), 1)

    def test_get_clustering_coefficient(self):
        g = network.Graph(4)
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(0, 3)

        self.assertEquals(g.get_clustering_coefficient(), 0)

        g.add_edge(1, 2)
        self.assertEquals(g.get_clustering_coefficient(), 3/5)



