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
import numpy as np
import matplotlib.pyplot as plt

class Graph:
    """
    Represents a graph.

    Each graph is represented by adjacency list with
    at most one edge between two nodes and no self-loop.
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
            # uses itertools to use generators
            t = itertools.ifilter(lambda s: s[0] < s[1],
                    itertools.product(self._edges[x], repeat=2))
            for s in t:
                triples += 1
                if s[0] in self._edges[s[1]]:
                    triangles += 1

        return triangles/triples

    def get_avg_f_o_f(self):
        """
        Returns computed and expected avg number of friends of friends.
        """
        d = [len(x) for x in self._edges]
        f_o_f = [sum([len(self._edges[f_o_v])
                    for f_o_v in self._edges[v]])
                        for v in range(self._node)]

        avg_fof = np.mean(f_o_f) / np.mean(d)
        exp_fof = np.mean(d) + np.var(d) / np.mean(d)

        return avg_fof, exp_fof

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

    from scipy.stats import poisson

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

def generate_preferential_attachment_network(n):
    """
    Generates graph following preferential attachment model

    The following implementation is inspired from the networkx implementaion
    of Barabási-Albert model:
    https://github.com/networkx/networkx/blob/master/networkx/generators/random_graphs.py#L567
    Here, the number of edge from new node to existing nodes is set to one.

    """
    g = Graph(n)

    # current node
    curr = 1

    # edge end-point
    target = 0

    # weights of node in terms of edge count
    weight = []

    # For any new node n, the connection is made to one of existing
    # [0, n-1] nodes.
    while curr < n:
        # avoiding self-loop
        if curr != target:
            g.add_edge(curr, target)

            # weight in terms of edge count is updated
            weight.append(target)
            weight.append(curr)
            curr += 1

        target = random.choice(weight)

    return g
