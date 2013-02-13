# -*- coding: utf-8 -*-
"""
    network.py
    ~~~~~~~~~~

    Info 6010 Assignment 3:
    https://courses.cit.cornell.edu/info6010/assignment3.html

"""

class Graph:
    """
    Represents a graph
    """

    def __init__(self, n):
        self._node = n
        self._edges = [set() for x in range(n)]

    def add_edge(self, n1, n2):
        """
        Adds a new edge
        """
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



