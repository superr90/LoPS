import itertools
import numpy as np
import pandas as pd

from pgmpy.models import MarkovModel, MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
from copy import deepcopy
from PCalgorithm import PCskletetonData
from pgmpy.sampling import GibbsSampling
from pgmpy.inference import VariableElimination
from pgmpy.estimators import HillClimbSearch
# from pgmpy.estimators import PC

import warnings
from collections import Counter

warnings.filterwarnings("ignore")


def PC(data, sampleNumber=4000):
    data_num = data.shape[1]
    if sampleNumber > data_num:
        sampleNumber = data_num
    index = np.random.choice(a=list(range(data_num)), p=[1 / data_num] * data_num, replace=False, size=sampleNumber)
    data = data[:, index]
    G, S = PCskletetonData(data)
    # GpDAG = PCorient(G, S)
    return G


def generate_all_Markov_network(n):
    # Generate all pairs of nodes
    pairs = list(itertools.combinations(range(n), 2))
    # Number of possible edges
    m = len(pairs)
    # Store all graphs
    graphs = []
    # Generate all combinations of edges
    for edges in itertools.product([0, 1], repeat=m):
        # Create adjacency matrix
        matrix = np.zeros((n, n))
        for idx, edge in enumerate(edges):
            i, j = pairs[idx]
            matrix[i, j] = edge
            matrix[j, i] = edge  # since it's an undirected graph
        # power_matrix = np.linalg.matrix_power(matrix, len(matrix))
        # if not (power_matrix == 0).any():
        #     graphs.append(matrix)
        graphs.append(matrix)
    table = np.random.rand(*([2] * 3))
    table / table.sum()

    p1 = np.random.uniform(0, 1, size=3)
    p2 = np.random.uniform(0, 1, size=3)
    potentials = [
        {(0): np.random.uniform(0, 1, size=2), (1): np.random.uniform(0, 1, size=2),
         (2): np.random.uniform(0, 1, size=2)},
        {(1, 2): np.random.uniform(0, 1, size=4).reshape(2, 2), (0): np.random.uniform(0, 1, size=2)},
        {(0, 2): np.random.uniform(0, 1, size=4).reshape(2, 2), (1): np.random.uniform(0, 1, size=2)},
        {(0, 2): np.random.uniform(0, 1, size=4).reshape(2, 2), (1, 2): np.random.uniform(0, 1, size=4).reshape(2, 2)},
        {(0, 1): np.random.uniform(0, 1, size=4).reshape(2, 2), (2): np.random.uniform(0, 1, size=2)},
        {(0, 1): np.random.uniform(0, 1, size=4).reshape(2, 2), (1, 2): np.random.uniform(0, 1, size=4).reshape(2, 2)},
        {(0, 1): np.random.uniform(0, 1, size=4).reshape(2, 2), (0, 2): np.random.uniform(0, 1, size=4).reshape(2, 2)},
        # {(0, 1): np.random.uniform(0, 1, size=8).reshape(2, 2, 2)},
        {(0, 1, 2): np.random.uniform(0, 1, size=8).reshape(2, 2, 2)}

    ]

    return graphs, potentials


def generate_all_belief_network(upstream_n, down_stream_n):
    upstream_nodes = list(range(upstream_n))
    combinations = []

    for r in range(upstream_n + 1):  # 因为你想从0到6，所以范围是7
        combinations.extend(list(itertools.combinations(upstream_nodes, r)))
    combinations = [list(c) for c in combinations]

    graphs = []

    def all_graphs(number, connects):
        if number > down_stream_n:
            graph = np.zeros((upstream_n + down_stream_n, upstream_n + down_stream_n))
            for i in range(len(connects)):
                for c in connects[i]:
                    graph[c, i + upstream_n] = 1
            graphs.append(deepcopy(graph))
            x = 0
        else:
            for connect in combinations:
                connects.append(connect)
                all_graphs(number + 1, deepcopy(connects))
                connects = connects[:-1]

    all_graphs(1, [])
    return graphs


# Markov network sampling
def effective_sample_size(chain):
    """
    Calculate the effective sample size of a chain.

    Parameters:
    - chain : ndarray
        A 2D array of shape [num_variables, num_samples].

    Returns:
    - ess : ndarray
        An array of shape [num_variables] containing ESS for each variable.
    """

    num_vars, num_samples = chain.shape
    ess = np.zeros(num_vars)

    for v in range(num_vars):
        variogram = [np.var(chain[v, i:] - chain[v, :-i]) for i in range(1, num_samples // 2)]
        negative_autocorr = np.where(np.array(variogram) > 0)[0][0]
        ess[v] = num_samples / (1 + 2 * np.sum(np.array(variogram)[:negative_autocorr]))

    return ess


def potential(x, y, prob):
    if x == y:
        return prob
    else:
        return 1.0 - prob


def get_components(adj_matrix):
    G = np.array(adj_matrix) + np.eye(len(adj_matrix))
    power_matrix = np.linalg.matrix_power(G, len(G))
    components = []
    while np.sum(power_matrix) != 0:
        index = np.where(power_matrix != 0)[0][0]
        index = np.where(power_matrix[index, :] != 0)[0]
        components.append(deepcopy(index))
        power_matrix[index, :] = 0
        power_matrix[:, index] = 0
    return components


def construct_MarkovNetwork(adj_matrix):
    components = get_components(adj_matrix)
    models = []
    bps = []
    edge_potentials = {}
    for k, component in enumerate(components):
        models.append(MarkovNetwork())
        # add node
        for i in component:
            models[k].add_node(i)
        # add edge and potential
        for i in component:
            for j in component:
                if j > i and adj_matrix[i][j] == 1:
                    p = np.random.uniform(0, 1)
                    if p <= 0.5:
                        p = np.random.uniform(0, 0.3)
                    else:
                        p = np.random.uniform(0.7, 1)
                    potential = [[p, 1 - p], [1 - p, p]]
                    factor = DiscreteFactor(variables=[i, j],
                                            cardinality=[2, 2],
                                            values=potential)
                    models[k].add_edge(u=i, v=j)
                    models[k].add_factors(factor)
                    edge_potentials.update({(i, j): potential})
        # If it is an isolated node, add potential separately.
        if len(component) == 1:
            p = np.random.uniform(0, 1)
            if p <= 0.5:
                p = np.random.uniform(0, 0.3)
            else:
                p = np.random.uniform(0.7, 1)
            potential = [1 - p, p]
            factor = DiscreteFactor(variables=[component[0]],
                                    cardinality=[2],
                                    values=potential)
            models[k].add_factors(factor)
        bp = BeliefPropagation(models[k])
        bps.append(bp)
    # The component where the node is located
    node_components = {}
    for k, component in enumerate(components):
        for c in component:
            node_components.update({c: k})
    return components, node_components, models, bps, edge_potentials


def gibbs_sampling(num_samples, adj_matrix):
    components, node_components, models, bps, edge_potentials = construct_MarkovNetwork(adj_matrix)
    # Initialize node values
    burn_in = 2000
    sampleData = np.zeros((len(adj_matrix), num_samples + burn_in))
    for i, model in enumerate(models):
        gibbs = GibbsSampling(model)
        samples = gibbs.sample(size=num_samples + burn_in).values.T
        # samples = gibbs.sample(size=num_samples + burn_in)

        for j in range(len(components[i])):
            sampleData[components[i][j], :] = samples[j, :]

    sampleData = np.array(sampleData + 1, dtype=np.int64)
    sampleData = sampleData[:, burn_in:]
    learned_graph = PC(sampleData, sampleData.shape[1])

    return np.array(sampleData), edge_potentials, learned_graph

# generate_all_belief_network(6, 3)
