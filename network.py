# -*- coding: utf-8 -*-
"""
    network.py
    ~~~~~~~~~~

    Info 6010 Assignment 3:
    https://courses.cit.cornell.edu/info6010/assignment3.html

"""

from __future__ import division
import random
import itertools

class Graph:
    """
    Represents a graph.

    Each graph is represented by adjacency list where there
    might be at most one edge between two nodes and no self-loop.
    """

    def __init__(self, n):
        self._node = n

        # Set to ensure there is at most one edge.
        self._edges = [set() for x in range(n)]

    def add_edge(self, n1, n2):
        """
        Adds a new edge if n1 and n2 are not same.
        """
        # Avoids self-loop.
        if n1 != n2:
            self._edges[n1].add(n2)
            self._edges[n2].add(n1)

    def get_degree_distribution(self):
        """
        Gets degree distribution
        """

        # Get degree count
        degree = [len(x) for x in self._edges]

        # Compact and inefficient?
        return {x: degree.count(x) for x in set(degree)}

    def get_clustering_coefficient(self):
        """
        Returns clustering coefficient of the graph
        """
        triples = 0
        triangles = 0
        for x in range(self._node):
            # get triples
            # Does the exact same thing as in Paul's example but
            # uses itertools
            t = itertools.ifilter(lambda s: s[0] < s[1],
                    itertools.product(self._edges[x], repeat=2))
            for s in t:
                triples += 1
                if s[0] in self._edges[s[1]]:
                    triangles += 1

        return triangles/triples

def generate_random_network(n, d_avg=7):
    """
    Generates Erdos-Renyi network

    :param n: Number of nodes.

    :param d_avg: Average number of edges per node.

    :returns:: A Graph instance

    """
    l = d_avg * n // 2
    g = Graph(n)
    for i in range(l):
        while True:
            n1 = random.randint(0, n-1)
            n2 = random.randint(0, n-1)
            if n1 != n2:
                break
        g.add_edge(n1, n2)
    return g

def draw_degree_distribution(g, mu):
    """
    Draws the degree distribution of a graph and Poisson fit
    """

    import numpy as np
    from scipy.stats import poisson
    import matplotlib.pyplot as plt

    d = g.get_degree_distribution()

    v1 = [x/sum(d.values()) for x in d.values()]

    # sorted as we need to draw the line
    sorted_d = sorted(d.keys())
    v2 = poisson.pmf(sorted_d, mu)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    width = 0.35
    ax.bar(np.array(d.keys()) - width/2, v1, width, color='m', label='data')
    ax.plot(sorted_d, v2, 'c--', label='Poisson')

    ax.set_xlabel('degree')
    ax.set_ylabel('probability')
    ax.legend()

    plt.show()
