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

def draw_log_log_degree_distribution(g):
    """
    Draws degree distribution in log-log scale
    """
    d = g.get_degree_distribution()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.plot(d.keys(), d.values(), 'm.')
    ax.set_xlabel('degree')
    ax.set_ylabel('frequency')
    ax.set_title("{0} Nodes".format(g._node))

    plt.show()

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
    ax.set_title("{0} Nodes".format(g._node))
    ax.legend()

    plt.show()

def _random_choice(seq, m, n):
    """
    Picks m unique samples from seq where none is equal to n
    """

    t = set()
    while len(t) < m:
        x = random.choice(seq)
        if x != n:
            t.add(x)
    return t

def generate_preferential_attachment_network(n, d_avg):
    """
    Generates graph following preferential attachment model

    :param n: Node count

    :param d_avg: Average edges created for each new node.

    The following implementation is inspired from the networkx implementaion
    of Barabási-Albert model:
    https://github.com/networkx/networkx/blob/master/networkx/generators/random_graphs.py#L567
    """

    g = Graph(n)

    # For any new node n, the connection is made to one of existing
    # [0, n-1] nodes.

    # current node
    curr = d_avg

    # edge end-point
    target = list(range(d_avg))

    # weights of node in terms of edge count
    weight = []

    while curr < n:
        for x in target:
            g.add_edge(curr, x)

        # weight in terms of edge count is updated
        weight.extend(target)
        weight.extend([curr] * d_avg)
        curr += 1

        # If you are using Numpy 1.7.1 version:
        # target = np.random.choice(weight, size=d_avg)
        target = _random_choice(weight, d_avg, curr)

    return g

def main():


    d_avg = 7
    node_counts = [10, 1000, 100000]
    r_nets = []
    b_nets = []

    s_format = "Node: {0} Actual: {1} Expected: {2}\n"

    # Generates networks
    for x in node_counts:
        r_nets.append(generate_random_network(x, d_avg))
        b_nets.append(generate_preferential_attachment_network(x, d_avg))

    print("Erdös-Renyi networks\n")

    for r in r_nets:
        draw_degree_distribution(r, d_avg)

    print("Friend of friend:\n")
    for r in r_nets:
        t = r.get_avg_f_o_f()
        print(s_format.format(r._node, t[0], t[1]))

    print("Clustering coefficient\n")
    for r in r_nets:
        t = r.get_clustering_coefficient()
        print(s_format.format(r._node, t, d_avg/(r._node - 1)))

    print("Barabási–Albert networks\n")

    for b in b_nets:
        draw_log_log_degree_distribution(b)

    print("Friend of friend:\n")
    for b in b_nets:
        t = b.get_avg_f_o_f()
        print(s_format.format(b._node, t[0], t[1]))

    print("Clustering coefficient\n")
    for b in b_nets:
        t = b.get_clustering_coefficient()
        print(s_format.format(b._node, t, d_avg/(b._node - 1)))

if __name__ == "__main__":
    main()
