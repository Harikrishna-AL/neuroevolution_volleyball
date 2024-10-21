"""Implementation of NEAT.

    python neat.py --task {xor, lunar, cartpole}

See the post at https://wellecks.wordpress.com/ for details.
Parts of this implementation are based on Neat-Python.
"""
from itertools import count
import numpy as np
import math
import random
from copy import deepcopy
from collections import defaultdict

GLOBAL_COUNT = count(1)
PARTITION_COUNT = count(1)
NODE_COUNT = count(10)  # make sure this is larger than the output dimension

ELITISM = 2
CUTOFF_PCT = 0.2
COMPATIBILITY_THRESHOLD = 3.0
NODE_DIST_COEFF = 0.5
NODE_DISJOINT_COEFF = 1.0
EDGE_DIST_COEFF = 0.5
EDGE_DISJOINT_COEFF = 1.0

NODE_ADD_PROB = 0.3
NODE_DEL_PROB = 0.2
EDGE_ADD_PROB = 0.3
EDGE_DEL_PROB = 0.2

WEIGHT_MUTATE_RATE = 0.8
WEIGHT_REINIT_RATE = 0.1
ACTIVE_MUTATE_RATE = 0.01
BIAS_MUTATE_RATE = 0.7
BIAS_REINIT_RATE = 0.1
RESPONSE_MUTATE_RATE = 0.0
ACTIVATION_MUTATE_RATE = 0.01

WEIGHT_MUTATE_SCALE = 0.5
BIAS_MUTATE_SCALE = 0.5
WEIGHT_INIT_SCALE = 1.0
BIAS_INIT_SCALE = 1.0


def sigmoid(z):
    z = max(-60.0, min(60.0, 2.5 * z))
    return 1.0 / (1.0 + math.exp(-z))

def tanh(z):
    z = max(-60.0, min(60.0, 2.5 * z))
    return np.tanh(z)

def relu(z):
    return np.maximum(z, 0)

def abs(z):
    return np.abs(z)

ACTIVATIONS = [sigmoid, tanh, relu, abs]


class Node(object):
    def __init__(self, key):
        self.bias = np.random.normal(0, BIAS_INIT_SCALE)
        self.response = 1.0
        self.activation = sigmoid
        self.aggregation = np.sum
        self.key = key

    def dist(self, other):
        d = abs(self.bias - other.bias)
        if self.activation != other.activation:
            d += 1.0
        if self.aggregation != other.aggregation:
            d += 1.0
        return NODE_DIST_COEFF*d

    def mutate_(self):
        r = random.random()
        if r < BIAS_MUTATE_RATE:
            self.bias = np.clip(self.bias + np.random.normal(0, BIAS_MUTATE_SCALE), -30, 30)

        elif r < BIAS_MUTATE_RATE + BIAS_REINIT_RATE:  # re-initialize
            self.bias = np.random.normal(0, BIAS_INIT_SCALE)

        if random.random() < ACTIVATION_MUTATE_RATE:
            self.activation = random.choice(ACTIVATIONS)


class Edge(object):
    def __init__(self, u, v, weight=None):
        self.weight = np.random.normal(0, WEIGHT_INIT_SCALE) if weight is None else weight
        self.uv = (u, v)
        self.active = True

    def dist(self, other):
        d = (abs(self.weight - other.weight)) + float(self.active != other.active)
        d = d * EDGE_DIST_COEFF
        return d

    def mutate_(self):
        r = random.random()
        if r < WEIGHT_MUTATE_RATE:  # perturb
            self.weight = np.clip(self.weight + np.random.normal(0, WEIGHT_MUTATE_SCALE), -30, 30)

        elif r < WEIGHT_MUTATE_RATE + WEIGHT_REINIT_RATE:  # re-initialize
            self.weight = np.random.normal(0, WEIGHT_INIT_SCALE)

        if random.random() < ACTIVE_MUTATE_RATE:
            self.active = random.random() < 0.5


class Genome(object):
    def __init__(self, key, input_size, output_size):
        self.nodes = {}
        self.edges = {}
        self.key = key
        self.input_keys = [-1 * i for i in range(1, input_size + 1)]
        self.output_keys = [i for i in range(output_size)]
        for k in self.output_keys:  # Initialize output nodes.
            self.nodes[k] = Node(k)
        for u in self.input_keys:   # Initialize input to output connections.
            for v in self.output_keys:
                self.edges[(u, v)] = Edge(u, v)

    def dist(self, other):
        d = self._nodes_dist(other) + self._edges_dist(other)
        return d

    def _nodes_dist(self, other):
        disjoint_nodes = 0
        d = 0.0
        if self.nodes or other.nodes:
            for k2 in other.nodes:
                if k2 not in self.nodes:
                    disjoint_nodes += 1
            for k1, n1 in self.nodes.items():
                n2 = other.nodes.get(k1)
                if n2 is None:
                    disjoint_nodes += 1
                else:
                    d += n1.dist(n2)
            max_nodes = max(len(self.nodes), len(other.nodes))
            d = (d + (NODE_DISJOINT_COEFF * disjoint_nodes)) / max_nodes
        return d

    def _edges_dist(self, other):
        d = 0.0
        disjoint_edges = 0
        if self.edges or other.edges:
            for k2 in other.edges:
                if k2 not in self.edges:
                    disjoint_edges += 1

            for k1, e1 in self.edges.items():
                e2 = other.edges.get(k1)
                if e2 is None:
                    disjoint_edges += 1
                else:
                    d += e1.dist(e2)
            max_edges = max(len(self.edges), len(other.edges))
            d = (d + (EDGE_DISJOINT_COEFF * disjoint_edges)) / max_edges
        return d

    def crossover_edges(self, other, child):
        for key, edge_p1 in self.edges.items():
            edge_p2 = other.edges.get(key)
            if edge_p2 is None:
                child.edges[key] = deepcopy(edge_p1)
            else:
                child.edges[key] = crossover(edge_p1, edge_p2, Edge(key[0], key[1]),
                                             attrs=['weight', 'active'])
        return child

    def crossover_nodes(self, other, child):
        for key, node_p1 in self.nodes.items():
            node_p2 = other.nodes.get(key)
            if node_p2 is None:
                child.nodes[key] = deepcopy(node_p1)
            else:
                child.nodes[key] = crossover(node_p1, node_p2, Node(next(NODE_COUNT)),
                                             attrs=['bias', 'response', 'activation', 'aggregation'])
        return child

    def mutate_(self):
        self._mutate_add_node_()
        self._mutate_del_node_()
        self._mutate_add_edge_()
        self._mutate_del_edge_()
        self._mutate_node_properties()
        self._mutate_edge_properties()

    def _mutate_add_node_(self):
        if random.random() < NODE_ADD_PROB:
            if len(self.edges) == 0:
                return
            edge_to_split = random.choice(list(self.edges.values()))
            edge_to_split.active = False

            new_node = Node(next(NODE_COUNT))
            self.nodes[new_node.key] = new_node

            edge_u_to_new = Edge(edge_to_split.uv[0], new_node.key, weight=1.0)
            self.edges[edge_u_to_new.uv] = edge_u_to_new

            edge_new_to_v = Edge(new_node.key, edge_to_split.uv[1], weight=edge_to_split.weight)
            self.edges[edge_new_to_v.uv] = edge_new_to_v

    def _mutate_del_node_(self):
        if random.random() < NODE_DEL_PROB:
            available_nodes = [k for k in self.nodes.keys() if k not in self.output_keys]
            if available_nodes:
                del_key = random.choice(available_nodes)
                edges_to_delete = set()
                for k, v in self.edges.items():
                    if del_key in k:
                        edges_to_delete.add(k)
                for key in edges_to_delete:
                    del self.edges[key]
                del self.nodes[del_key]

    def _mutate_add_edge_(self):
        if random.random() < EDGE_ADD_PROB:
            possible_outputs = list(self.nodes.keys())
            out_node = random.choice(possible_outputs)
            in_node = random.choice(possible_outputs + self.input_keys)
            key = (in_node, out_node)
            stop = False
            if key in self.edges:
                stop = True
            if in_node in self.output_keys and out_node in self.output_keys:
                stop = True
            if creates_cycle(self.edges.keys(), in_node, out_node):
                stop = True
            if not stop:
                self.edges[key] = Edge(in_node, out_node)

    def _mutate_del_edge_(self):
        if random.random() < EDGE_DEL_PROB:
            if len(self.edges) > 0:
                key = random.choice(list(self.edges.keys()))
                del self.edges[key]

    def _mutate_edge_properties(self):
        for edge in self.edges.values():
            edge.mutate_()

    def _mutate_node_properties(self):
        for node in self.nodes.values():
            node.mutate_()

class Partition(object):
    def __init__(self, key, members=[], representative=None):
        self.key = key
        self.members = members
        self.representative = representative

    def find_representative(self, gids, population):
        # New representative is the closest candidate from `gids` to the current representative.
        candidates = []
        for gid in gids:
            d = self.representative.dist(population.gid_to_genome[gid])
            candidates.append((d, population.gid_to_genome[gid]))
        _, new_rep = min(candidates, key=lambda x: x[0])
        return new_rep


class Partitions(object):
    def __init__(self):
        self.pid_to_partition = {}
        self.gid_to_pid = {}

    def new_partition(self, pid, members, representative):
        if pid is None:
            pid = next(PARTITION_COUNT)
        p = Partition(pid, members, representative)
        self.pid_to_partition[pid] = p

    def closest_representative(self, genome):
        candidates = []
        for pid, p in self.pid_to_partition.items():
            d = p.representative.dist(genome)
            if d < COMPATIBILITY_THRESHOLD:
                candidates.append((d, pid))
        if candidates:
            _, pid = min(candidates, key=lambda x: x[0])
        else:
            pid = None
        return pid

    def adjust_fitnesses(self, fitnesses):
        partition_adjusted_fitnesses = {}
        min_fitness = min(fitnesses.values())
        max_fitness = max(fitnesses.values())
        fitness_range = max(1.0, max_fitness - min_fitness)  # NOTE: 1.0 arbitrary
        for pid, partition in self.pid_to_partition.items():
            msf = np.mean([fitnesses[m] for m in partition.members])
            af = (msf - min_fitness) / fitness_range
            partition_adjusted_fitnesses[pid] = af
        return partition_adjusted_fitnesses

    def next_partition_sizes(self, partition_adjusted_fitnesses, pop_size):
        """Decide partition sizes for the next generation by fitness. Based on Neat-Python."""
        previous_sizes = {pid: len(p.members) for pid, p in self.pid_to_partition.items()}
        af_sum = sum(partition_adjusted_fitnesses.values())
        sizes = {}
        min_species_size = 2
        for pid in partition_adjusted_fitnesses:
            if af_sum > 0:
                s = max(min_species_size, partition_adjusted_fitnesses[pid]/af_sum*pop_size)
            else:
                s = min_species_size

            d = (s - previous_sizes[pid]) * 0.5
            c = int(round(d))
            size = previous_sizes[pid]
            if abs(c) > 0:
                size += c
            elif d > 0:
                size += 1
            elif d < 0:
                size -= 1
            sizes[pid] = size

        normalizer = pop_size / sum(sizes.values())
        sizes = {pid: max(min_species_size, int(round(size*normalizer)))
                 for pid, size in sizes.items()}
        return sizes


class Population(object):
    def __init__(self):
        self.gid_to_genome = {}
        self.gid_to_ancestors = {}

    @classmethod
    def initial_population(cls, args):
        pop = cls()
        for _ in range(args.population_size):
            gid = next(GLOBAL_COUNT)
            pop.gid_to_genome[gid] = Genome(gid, args.input_size, args.output_size)
            pop.gid_to_ancestors[gid] = tuple()
        return pop

    def partition(self, initial_partitions):
        """Partition population by similarity."""
        unpartitioned = set(self.gid_to_genome.keys())
        new_partitions = Partitions()

        # Find new representatives (retain the old partitions ids).
        for pid, p in initial_partitions.pid_to_partition.items():
            new_rep = p.find_representative(unpartitioned, self)
            new_partitions.new_partition(pid, members=[new_rep.key], representative=new_rep)
            unpartitioned.remove(new_rep.key)

        # Add remaining members to partitions by finding the partition with the
        # most similar representative; if none exist then create a new partition.
        while unpartitioned:
            gid = unpartitioned.pop()
            g = self.gid_to_genome[gid]
            pid = new_partitions.closest_representative(g)
            if pid is None:
                new_partitions.new_partition(pid, members=[gid], representative=g)
            else:
                new_partitions.pid_to_partition[pid].members.append(gid)
        return new_partitions


class EvalNode(object):
    def __init__(self, node_id, activation, aggregation, bias, incoming_connections):
        self.node_id = node_id
        self.activation = activation
        self.aggregation = aggregation
        self.bias = bias
        self.incoming_connections = incoming_connections


class Network(object):
    def __init__(self, input_keys, output_keys, eval_nodes):
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.eval_nodes = eval_nodes

    @classmethod
    def make_network(cls, genome):
        edges = [edge.uv for edge in genome.edges.values() if edge.active]
        required_nodes = cls._required_nodes(edges, genome.input_keys, genome.output_keys)
        layers = cls._make_layers(required_nodes, edges, genome.input_keys)
        eval_nodes = cls._make_eval_nodes(layers, edges, genome)
        net = cls(genome.input_keys, genome.output_keys, eval_nodes)
        return net

    @staticmethod
    def _required_nodes(edges, input_keys, output_keys):
        # Identify nodes that are required for output by working backwards from output nodes.
        required = set(output_keys)
        seen = set(output_keys)
        while True:
            layer = set(u for (u, v) in edges if v in seen and u not in seen)
            if not layer:
                break
            layer_nodes = set(u for u in layer if u not in input_keys)
            if not layer_nodes:
                break
            required = required.union(layer_nodes)
            seen = seen.union(layer)
        return required

    @staticmethod
    def _make_layers(required_nodes, edges, input_keys):
        layers = []
        seen = set(input_keys)
        while True:
            # Find candidate nodes for the next layer that connect a seen node to an unseen node.
            candidates = set(v for (u, v) in edges if u in seen and v not in seen)
            # Keep only required nodes whose entire input set is contained in seen.
            layer = set()
            for w in candidates:
                if w in required_nodes and all(u in seen for (u, v) in edges if v == w):
                    layer.add(w)
            if not layer:
                break

            layers.append(layer)
            seen = seen.union(layer)
        return layers

    @staticmethod
    def _make_eval_nodes(layers, edges, genome):
        eval_nodes = []
        for layer in layers:
            for node in layer:
                incoming_conns = [(u, genome.edges[u, v].weight) for u, v in edges if v == node]
                eval_node = EvalNode(node_id=node,
                                     activation=genome.nodes[node].activation,
                                     aggregation=genome.nodes[node].aggregation,
                                     bias=genome.nodes[node].bias,
                                     incoming_connections=incoming_conns)
                eval_nodes.append(eval_node)
        return eval_nodes

    def forward(self, x):
        values = {k: 0.0 for k in self.input_keys + self.output_keys}
        for k, v in zip(self.input_keys, x):
            values[k] = v

        for eval_node in self.eval_nodes:
            node_inputs = []
            for i, w in eval_node.incoming_connections:
                node_inputs.append((values[i] * w))
            agg = eval_node.aggregation(node_inputs)
            values[eval_node.node_id] = eval_node.activation(eval_node.bias + agg)

        outputs = [values[n] for n in self.output_keys]
        return outputs


def next_generation(fitnesses, population, partitions):
    partition_adjusted_fitnesses = partitions.adjust_fitnesses(fitnesses)
    sizes = partitions.next_partition_sizes(partition_adjusted_fitnesses, len(population.gid_to_genome))

    new_population = Population()
    for pid, p in partitions.pid_to_partition.items():
        size = sizes[pid]
        # Sort in order of descending fitness and remove low-fitness members.
        old_members = sorted(list(p.members), key=lambda x: fitnesses[x], reverse=True)
        for gid in old_members[:ELITISM]:
            new_population.gid_to_genome[gid] = population.gid_to_genome[gid]
            size -= 1
        cutoff = max(int(math.ceil(CUTOFF_PCT*len(old_members))), 2)
        old_members = old_members[:cutoff]

        # Generate new members.
        while size > 0:
            size -= 1
            gid1, gid2 = random.choice(old_members), random.choice(old_members)
            child = new_child(population.gid_to_genome[gid1],
                              population.gid_to_genome[gid2],
                              fitnesses[gid1],
                              fitnesses[gid2])
            new_population.gid_to_genome[child.key] = child
            new_population.gid_to_ancestors[child.key] = (gid1, gid2)

    return new_population


def crossover(obj1, obj2, obj_new, attrs):
    for attr in attrs:
        if random.random() > 0.5:
            setattr(obj_new, attr, getattr(obj1, attr))
        else:
            setattr(obj_new, attr, getattr(obj2, attr))
    return obj_new


def new_child(p1, p2, f1, f2):
    child = Genome(next(GLOBAL_COUNT), len(p1.input_keys), len(p1.output_keys))
    if f1 < f2:
        p1, p2 = p2, p1
    child = p1.crossover_edges(p2, child)
    child = p1.crossover_nodes(p2, child)
    child.mutate_()
    return child


def creates_cycle(edges, u, v):
    # check if there is a v->u path
    if u == v:
        return True

    graph = defaultdict(list)
    for i, j in edges:
        graph[i].append(j)

    if v not in graph:
        return False

    seen = set()
    queue = [v]
    while len(queue) > 0:
        curr = queue[0]
        queue = queue[1:]
        seen.add(curr)
        if curr == u:
            return True

        for child in graph[curr]:
            if child not in seen:
                queue.append(child)
    return False


def run(eval_population_fn, args):
    population = Population.initial_population(args)
    partitions = population.partition(initial_partitions=Partitions())
    stats = defaultdict(list)
    gen = 0
    while True:
        gen += 1

        # Evaluate fitness
        fitnesses = eval_population_fn(population)
        stats['max'].append(np.max(list(fitnesses.values())))
        stats['mean'].append(np.mean(list(fitnesses.values())))

        # Check stop condition
        if args.stop_criterion(list(fitnesses.values())) >= args.stop_threshold:
            gid, fitness = max(list(fitnesses.items()), key=lambda x: x[1])
            visualize(population.gid_to_genome[gid], stats)
            if args.task != 'xor':
                eval_pop = Population()
                eval_pop.gid_to_genome[gid] = population.gid_to_genome[gid]
                gym_eval_population(env, eval_pop, render=True)
            break
        print("fitness\tmean %.3f\tmax %.3f\npopulation %d in %d partitions\n" %
              (np.mean(list(fitnesses.values())),
               np.max(list(fitnesses.values())),
               len(population.gid_to_genome),
               len(partitions.pid_to_partition)))

        # Create next generation
        population = next_generation(fitnesses, population, partitions)
        partitions = population.partition(initial_partitions=partitions)


def xor_eval_population(population):
    xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]

    fitnesses = {}
    for gid, g in population.gid_to_genome.items():
        net = Network.make_network(g)
        fitness = 4.0
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.forward(xi)
            fitness -= (output[0] - xo[0])**2
        fitnesses[gid] = fitness
    return fitnesses


def gym_eval_population(env, population, render=False):
    fitnesses = {}
    for gid, g in population.gid_to_genome.items():
        net = Network.make_network(g)

        episode_reward = 0
        num_episodes_per_eval = 3
        for i in range(num_episodes_per_eval):
            done = False
            t = 0
            state = env.reset()
            while not done:
                output = net.forward(state)
                if 'LunarLander' in env.__repr__():
                    action = np.argmax(output)
                elif 'CartPole' in env.__repr__():
                    action = int(output[0] > 0.5)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                t += 1
                if render:
                    env.render()

        render = False
        fitnesses[gid] = episode_reward/num_episodes_per_eval
    return fitnesses


def visualize(candidate, stats):
    import matplotlib.pyplot as plt
    import networkx as nx
    plt.plot(stats['max'], label='max')
    plt.plot(stats['mean'], label='mean')
    plt.legend()
    plt.show()

    V = candidate.nodes
    E = candidate.edges
    def a_names(a):
        if 'sigmoid' in a.__name__:
            return 'σ()'
        return a.__name__ + '()'

    def color(v):
        if v in candidate.input_keys:
            return "green"
        if v in candidate.output_keys:
            return "red"
        return "blue"
    v_label = {k: '%s' % (a_names(v.activation)) for k, v in V.items()}
    e_label = {k: np.round(v.weight, 3) for k, v in E.items()}

    G = nx.DiGraph()
    G.add_nodes_from(V)
    G.add_edges_from(E)

    paths = nx.all_pairs_shortest_path_length(G)
    for v, ps in list(paths):
        ps = deepcopy(ps)
        if all([u not in ps for u in candidate.output_keys]):  # no path to the output layer
            G.remove_node(v)
            v_label_ = deepcopy(v_label)
            if v in v_label_:
                del v_label[v]
            e_label_ = deepcopy(e_label)
            for i, j in e_label_:
                if i == v or j == v:
                    del e_label[i, j]
    pos = nx.drawing.nx_agraph.pygraphviz_layout(G, prog='dot')
    pos = {k: (v[0], -v[1]) for k, v in pos.items()}
    nx.draw(G, pos=pos, node_color=[color(k) for k in G.nodes])
    nx.draw_networkx_labels(G, labels=v_label, pos=pos)
    nx.draw_networkx_edge_labels(G, edge_labels=e_label, pos=pos)
    plt.show()


if __name__ == '__main__':
    import argparse
    import gym
    from functools import partial
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['xor', 'cartpole', 'lunar'], required=True)

    args = parser.parse_args()

    args.stop_criterion = np.max
    if args.task == 'xor':
        eval_func = xor_eval_population
        args.stop_threshold = 4.0-1e-3
        args.input_size = 2
        args.output_size = 1

    if args.task == 'cartpole':
        env = gym.make('CartPole-v1')
        eval_func = partial(gym_eval_population, env)
        args.stop_threshold = 200
        env._max_episode_steps = 500
        args.input_size = 4
        args.output_size = 1
        args.stop_criterion = np.mean

    if args.task == 'lunar':
        env = gym.make('LunarLander-v2')
        eval_func = partial(gym_eval_population, env)
        args.stop_threshold = 250
        args.input_size = 8
        args.output_size = 4

    args.population_size = 250

    run(eval_func, args)