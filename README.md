Third Assignment for Info 6010: https://courses.cit.cornell.edu/info6010/assignment3.html

Requirments
===

* NumPy version >= 1.6.1
* SciPy version >= 0.10.1
* matplotlib version >= 1.2.x

Run
===

From terminal, just do:

> python network.py

This command will create three Erdös-Renyi and Barabási-Albert networks with
10, 1000, 100000 nodes.

Erdös-Renyi networks
===

For Erdös-Renyi networks, the degree distribution looks like:

![10 nodes](https://raw.github.com/saeed-abdullah/info-6010-assignment-3/master/images/erdos_renyi_10_nodes.png "Erdös-Renyi network with 10 nodes")

![1000 nodes](https://raw.github.com/saeed-abdullah/info-6010-assignment-3/master/images/erdos_renyi_1000_nodes.png "Erdös-Renyi network with 1000 nodes")

![100000 nodes](https://raw.github.com/saeed-abdullah/info-6010-assignment-3/master/images/erdos_renyi_100000_nodes.png "Erdös-Renyi network with 100000 nodes")


Average friend of friends
----

| Node  | Actual | Expected |
| ----- | ------ | -------- |
| 10    |  5.28  |   5.28   |
| 1000  | 7.9073 | 7.9073 |
|100000 | 8.0036 | 8.0036 |

Clustering coefficient
---

| Node  | Actual | Expected |
| ----- | ------ | -------- |
|  10   | 0.5046 | 0.7777   |
| 1000  | 0.0083 | 0.0070  |
| 100000 |7.09e-05 | 7.00e-05 |

Barabási-Albert networks
===

For Barabási-Albert networks, the degree distribution in log-log scale looks like:

![10 nodes](https://raw.github.com/saeed-abdullah/info-6010-assignment-3/master/images/barbasi_albert_10_nodes.png "Barabási-Albert network with 10 nodes")

![1000 nodes](https://raw.github.com/saeed-abdullah/info-6010-assignment-3/master/images/barbasi_albert_1000_nodes.png "Barabási-Albert network with 1000 nodes")

![100000 nodes](https://raw.github.com/saeed-abdullah/info-6010-assignment-3/master/images/barbasi_albert_100000_nodes.png "Barabási-Albert network with 100000 nodes")


Average friend of friends
----

| Node  | Actual | Expected |
| ----- | ------ | -------- |
| 10    |  5.809 |   5.809  |
| 1000  | 26.713 | 26.713  |
|100000 | 44.03 | 44.03 |

Clustering coefficient
---

| Node  | Actual | Expected |
| ----- | ------ | -------- |
|  10   | 0.505  | 0.7777   |
| 1000  | 0.0405 | 0.0070  |
| 100000 | 0.001 | 7.00e-05 |


