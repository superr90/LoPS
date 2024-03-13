import networkx as nx
import numpy as np
from itertools import combinations
from copy import deepcopy
import pickle
import itertools


def kl_divergence(P, Q):
    """
    Compute the KL divergence D(P || Q) for discrete distributions
    :param P: np.array, first probability distribution
    :param Q: np.array, second probability distribution
    :return: float, KL divergence of P and Q
    """
    # Ensure the probability distributions are normalized
    P = P / P.sum()
    Q = Q / Q.sum()
    # Compute the KL divergence, taking care not to divide or take the logarithm of zero
    return np.sum(P * (np.log2(P) - np.log2(Q)))


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
        graphs.append(matrix)
    table = np.random.rand(*([2] * 3))
    table / table.sum()
    return graphs


def get_connect_graph(G):
    connect_matrix = deepcopy(G)
    for n in range(1, len(connect_matrix) + 1):
        temp = np.linalg.matrix_power(G, n)
        connect_matrix += temp
    return connect_matrix


def find_separator(adj_matrix, u, v):
    leave_nodes = list(range(len(adj_matrix)))
    leave_nodes.remove(u)
    leave_nodes.remove(v)

    all_combinations = []
    for r in range(0, len(leave_nodes) + 1):
        # Generate all combinations of length r
        combinations_r = combinations(leave_nodes, r)
        all_combinations.extend(combinations_r)

    for com in all_combinations:
        temp_adj_matrix = deepcopy(adj_matrix)
        for idx in com:
            temp_adj_matrix[idx, :] = 0
            temp_adj_matrix[:, idx] = 0
        connect_matrix = get_connect_graph(temp_adj_matrix)
        if connect_matrix[u, v] == 0:
            return list(com)


def get_joint_probs(potentials):
    joint_probs = np.zeros((2, 2, 2))
    combinations = []
    for i in range(1, 4):
        com = list(itertools.combinations([0, 1, 2], i))
        combinations += com
    for i in range(2):
        for j in range(2):
            for k in range(2):
                state = [i, j, k]
                states = [[state[c] for c in co] for co in combinations]
                P = 1
                for c, co in enumerate(combinations):
                    if len(co) == 1:
                        co = co[0]
                    if co not in potentials.keys():
                        continue
                    p = np.array(potentials[co])[tuple(states[c])]
                    P *= p
                joint_probs[i, j, k] = P
    joint_probs = joint_probs / joint_probs.sum()
    return joint_probs


def generate_potential(G):
    edges = np.array(np.where(G != 0))
    potentials = {}
    for i in range(edges.shape[1]):
        s = edges[0, i]
        t = edges[1, i]
        if s >= t:
            continue
        p = np.random.uniform(0, 1)
        if p <= 0.5:
            p = np.random.uniform(0, 0.3)
        else:
            p = np.random.uniform(0.7, 1)
        p = np.array([[p, 1 - p], [1 - p, p]])
        potentials.update({(s, t): p})

    for i in range(len(G)):
        p = np.random.uniform(0.1, 0.9)
        p = np.array([p, 1 - p])
        potentials.update({(i): p})
    joint_probs = get_joint_probs(potentials)
    return joint_probs, potentials


def get_px_c(pxc, x, c):
    # get p(x|c)
    px_c = np.zeros((2, 2))
    for i in range(2):
        if x < c:
            z = np.sum(pxc[:, i])
        else:
            z = np.sum(pxc[i, :])
        for j in range(2):
            if x < c:
                px_c[j, i] = pxc[j, i] / z
            else:
                px_c[i, j] = pxc[i, j] / z
    return px_c


def get_px_c_py_c(px_c, py_c, x, y, c):
    seq = [x, y, c]

    # k is the control condition
    p_indep = np.zeros((2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                state = [-1, -1, -1]
                state[x] = i
                state[y] = j
                state[c] = k
                if x < c:
                    p1 = px_c[i, k]
                else:
                    p1 = px_c[k, j]

                if y < c:
                    p2 = py_c[j, k]
                else:
                    p2 = py_c[k, j]
                p_indep[tuple(state)] = p1 * p2
    return p_indep


def get_pxy_c(pxyc, x, y, c):
    pxy_c = np.zeros((2, 2, 2))
    for k in range(2):
        if c == 0:
            z = np.sum(pxyc[k, :, :])
        elif c == 1:
            z = np.sum(pxyc[:, k, :])
        else:
            z = np.sum(pxyc[:, :, k])
        for i in range(2):
            for j in range(2):
                state = [-1, -1, -1]
                state[x] = i
                state[y] = j
                state[c] = k
                pxy_c[tuple(state)] = pxyc[tuple(state)] / z
    return pxy_c


def global_markov_property(joint_probs, G):
    edges = list(itertools.combinations(list(range(len(G))), 2))
    thred = 1e-100
    for s, t in edges:
        if s >= t:
            continue
        saparator = find_separator(G, s, t)
        if saparator is None:
            continue
        if saparator == []:
            # If s and t are not connected, it needs to satisfy p(s)p(t)==p(s,t)
            axis_s = list(range(len(G)))
            axis_s.remove(s)
            ps = np.sum(joint_probs, axis=tuple(axis_s))

            axis_t = list(range(len(G)))
            axis_t.remove(t)
            pt = np.sum(joint_probs, axis=tuple(axis_t))

            axis_st = list(range(len(G)))
            axis_st.remove(t)
            axis_st.remove(s)
            pst = np.sum(joint_probs, axis=tuple(axis_st))

            # p(s)*p(t)
            p_indep = ps.reshape(-1, 1) * pt.reshape(1, -1)
            kl = kl_divergence(p_indep.reshape(-1), pst.reshape(-1))
            if np.abs(kl) != 0:
                return False
        else:
            c = saparator[0]
            # p(s|c)
            axis_sc = list(range(len(G)))
            axis_sc.remove(s)
            axis_sc.remove(c)
            psc = np.sum(joint_probs, axis=tuple(axis_sc))
            ps_c = get_px_c(psc, s, c)

            # p(t|c)
            axis_tc = list(range(len(G)))
            axis_tc.remove(t)
            axis_tc.remove(c)
            ptc = np.sum(joint_probs, axis=tuple(axis_tc))
            pt_c = get_px_c(ptc, t, c)

            # p(s,t|c)
            pst_c = get_pxy_c(joint_probs, s, t, c)
            # p(s|c)*p(t|c)
            p_indep = px_n_muti_py_n(ps_c, pt_c, s, t, c)

            kl = kl_divergence(p_indep.reshape(-1), pst_c.reshape(-1))
            if np.abs(kl) != 0:
                return False
    return True


def get_px_c_1(pxc, x, c):
    c = c[0]
    px_cn = np.zeros((2, 2))
    for k in range(2):
        if x < c:
            z = np.sum(pxc[:, k])
        else:
            z = np.sum(pxc[k, :])
        for i in range(2):
            if x < c:
                px_cn[i, k] = pxc[i, k] / z
            else:
                px_cn[k, i] = pxc[k, i] / z
    return px_cn


def get_px_c_2(pxc, x, c):
    px_c = np.zeros((2, 2, 2))
    for i in range(2):
        for j in range(2):
            z = 0
            for k in range(2):
                state = [-1, -1, -1]
                state[x] = k
                state[c[0]] = i
                state[c[1]] = j
                z += pxc[tuple[state]]
            for k in range(2):
                state = [-1, -1, -1]
                state[x] = k
                state[c[0]] = i
                state[c[1]] = j
                px_c[tuple[state]] = pxc[tuple[state]] / z
    return px_c


def get_px(pxc, x, c):
    return pxc


def px_n_muti_py_n(px_n, py_n, x, y, c):
    p_indep = np.zeros((2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                if x < c:
                    p1 = px_n[i, k]
                else:
                    p1 = px_n[k, i]
                if y < c:
                    p2 = py_n[j, k]
                else:
                    p2 = py_n[k, j]
                state = [-1, -1, -1]
                state[x] = i
                state[y] = j
                state[c] = k
                p_indep[tuple(state)] = p1 * p2
    return p_indep


def local_markov_property(joint_probs, G):
    thred = 1e-100
    f = [get_px, get_px_c_1, get_px_c_2]
    for s in range(len(G)):
        neighbor = np.where(G[s, :] != 0)[0]
        other_nodes = list(range(len(G)))
        other_nodes.remove(s)
        for n in neighbor:
            other_nodes.remove(n)
        if len(other_nodes) == 0:
            continue

        psn = np.sum(joint_probs, axis=tuple(other_nodes))

        ps_n = f[len(neighbor)](psn, s, neighbor)
        for t in other_nodes:
            axis_t = list(range(len(G)))
            axis_t.remove(t)
            for n in neighbor:
                axis_t.remove(n)
            ptn = np.sum(joint_probs, axis=tuple(axis_t))
            pt_n = f[len(neighbor)](ptn, t, neighbor)
            # p(s|n)*p(t|n)
            if len(neighbor) == 0:
                if s < t:
                    p_indep = ps_n.reshape(-1, 1) * pt_n.reshape(1, -1)
                else:
                    p_indep = pt_n.reshape(-1, 1) * ps_n.reshape(1, -1)
            else:
                p_indep = px_n_muti_py_n(ps_n, pt_n, s, t, neighbor[0])

            # p(s,t|n)
            if len(neighbor) == 0:
                axis_st = list(range(len(G)))
                axis_st.remove(s)
                axis_st.remove(t)
                pst_n = np.sum(joint_probs, axis=tuple(axis_st))
            else:
                pst_n = get_pxy_c(joint_probs, s, t, neighbor[0])
            kl = kl_divergence(p_indep.reshape(-1), pst_n.reshape(-1))
            if np.abs(kl) != 0:
                return False
    return True


def is_legal(joint_probs, G):
    global_property = global_markov_property(joint_probs, G)
    local_property = local_markov_property(joint_probs, G)
    connect = is_connect(joint_probs, G)
    if global_property == True and local_property == True and connect == True:
        return True
    else:
        return False


def is_connect(joint_probs, G):
    edges = np.array(np.where(G == 1)).T
    for i in range(len(edges)):
        s = edges[i][0]
        t = edges[i][1]
        if s > t:
            continue
        o = list(range(len(G)))
        o.remove(s)
        o.remove(t)
        o = o[0]
        # p(s|o)
        pso = np.sum(joint_probs, axis=t)
        ps_o = get_px_c(pso, s, o)
        # p(t|o)
        pto = np.sum(joint_probs, axis=s)
        pt_o = get_px_c(pto, t, o)
        # p(s,t|o)
        pst_o = get_pxy_c(joint_probs, s, t, o)

        # p_indep
        p_indep = px_n_muti_py_n(ps_o, pt_o, s, t, o)

        kl = kl_divergence(p_indep.reshape(-1), pst_o.reshape(-1))
        if kl < 0.2:
            return False
    return True


def generate_markov_graph():
    all_graph = generate_all_Markov_network(3)
    jonit_probs = []
    graphs = []
    for graph in all_graph:
        print("===============")
        probs = []
        while len(probs) < 10:
            joint_prob, potentials = generate_potential(graph)
            legal = is_legal(joint_prob, graph)
            if legal == True:
                kl = np.array([kl_divergence(joint_prob.reshape(-1), p.reshape(-1)) for p in probs])
                kl = np.where(kl < 0.2)[0]
                if len(kl) == 0:
                    probs.append(joint_prob)
            else:
                pass
        jonit_probs.append(probs)
        graphs.append(graph)
    result = {
        "joint_probs": jonit_probs,
        "G": graphs,
    }

    with open("data/MarkovGraph.pkl", "wb") as file:
        pickle.dump(result, file)


if __name__ == '__main__':
    generate_markov_graph()
