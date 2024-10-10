import jax
import jax.numpy as jnp
import random
from flax.struct import dataclass

@dataclass
class GenomeData:
    nodes : jnp.ndarray
    connections : jnp.ndarray
    innovation_count : int
    node_count : int
    key : jnp.ndarray
    matrix : jnp.ndarray



class Genome:
    def __init__(self, nodes=None, connections=None, innovation_count=0, node_count=0, config=None):
        self.nodes = jnp.empty((0, 2), dtype=jnp.int32) if nodes is None else jnp.array(nodes)
        self.connections = jnp.empty((0, 5), dtype=jnp.float32) if connections is None else jnp.array(connections)
        self.innovation_count = innovation_count
        self.node_count = node_count
        self.node_values = {int(node[0].item()): 0.0 for node in self.nodes}
        self.key = jax.random.PRNGKey(0)
        self.config = config

        def _init_pop(key):
            input_num = 12
            output_num = 3

            nodes = jnp.zeros((input_num + output_num, 2))
            nodes = nodes.at[:input_num, 1].set(0)
            nodes = nodes.at[input_num:, 1].set(1)
            nodes = nodes.at[:, 0].set(jnp.arange(input_num + output_num))

            conn = jnp.zeros((input_num * output_num, 5))
            conn = conn.at[:, 0].set(jnp.tile(jnp.arange(input_num, dtype=jnp.int32), output_num))
            conn = conn.at[:, 1].set(jnp.repeat(jnp.arange(input_num, input_num + output_num, dtype=jnp.int32), input_num))

            conn = conn.at[:, 2].set(jnp.nan)
            conn = conn.at[:, 3].set(jnp.arange(input_num * output_num, dtype=jnp.int32))
            conn = conn.at[:, 4].set(jax.random.bernoulli(key, p=0.5, shape=(input_num * output_num,)))

            conn = conn.at[:, 2].set(jax.random.uniform(key, shape=(input_num * output_num,), minval=-1.0, maxval=1.0))
            
            source_nodes = conn[:, 0].astype(int)
            target_nodes = conn[:, 1].astype(int)
            weights = conn[:, 2]

            matrix = jnp.zeros((input_num + output_num, input_num + output_num))

            enabled_connections = conn[:, 4] == 1.0
            matrix = matrix.at[source_nodes, target_nodes].add(enabled_connections * weights)


            return GenomeData(nodes, conn, input_num * output_num, input_num + output_num, key, matrix)
        
        def _forward(genome: GenomeData, obs: jnp.ndarray):
            n_nodes = genome.nodes.shape[0]
            n_inputs = 12
            n_outputs = 3
            node_activations = jnp.zeros(n_nodes)
            node_activations = node_activations.at[: n_inputs].set(obs)

        
            for i in range(n_inputs, n_nodes):
                wieghted_sum = jnp.dot(node_activations, genome.matrix[:, i])
                node_activations = node_activations.at[i].set(jax.nn.relu(wieghted_sum))
            
            output = node_activations[-n_outputs:]
            return output
        
        def __add_node(genome: GenomeData):
            node_type_map = {"input": 0, "output": 1, "hidden": 2}
            node_type_numeric = node_type_map["hidden"]

            new_id = genome.node_count
            new_node = jnp.array([[new_id, node_type_numeric]])  
            new_nodes = jnp.concatenate((genome.nodes, new_node), axis=0)

            new_matrix = jnp.zeros((genome.node_count + 1, genome.node_count + 1))
            new_matrix = new_matrix.at[:genome.node_count, :genome.node_count].set(genome.matrix)
            # genome.node_count += 1
            return new_id, GenomeData(new_nodes, genome.connections, genome.innovation_count, genome.node_count + 1, genome.key, new_matrix)
        
        def __add_connection(genome: GenomeData, in_node, out_node, weight):
            new_connection = jnp.array([[in_node, out_node, weight, genome.innovation_count, 1.0]]) 
            new_connections = jnp.concatenate((genome.connections, new_connection), axis=0) 
            # genome.innovation_count += 1
            matrix = genome.matrix.at[int(in_node), int(out_node)].set(weight)

            return genome.innovation_count + 1, GenomeData(genome.nodes, new_connections, genome.innovation_count + 1, genome.node_count, genome.key, matrix)
        
        self.add_node = __add_node
        self.add_connection = __add_connection

        def __mutate_add_node(genome: GenomeData):
            connection_to_split = genome.connections[jax.random.randint(genome.key, shape=(), minval=0, maxval=len(genome.connections))]
            connection_to_split = connection_to_split.at[4].set(0.0)

            new_node_id, genome = self.add_node(genome)

            inv_num , genome = self.add_connection(genome, connection_to_split[0], new_node_id, 1.0)
            inv_num, genome = self.add_connection(genome, new_node_id, connection_to_split[1], connection_to_split[2])

            return genome, inv_num

        def __mutate_add_connection(genome: GenomeData, weight_range=(-1.0, 1.0)):
            # Sample two nodes
            node_input = genome.nodes[jax.random.randint(genome.key, shape=(), minval=0, maxval=len(genome.nodes))]
            node_output = genome.nodes[jax.random.randint(genome.key, shape=(), minval=0, maxval=len(genome.nodes))]

            weight = jax.random.uniform(genome.key, shape=(), minval=weight_range[0], maxval=weight_range[1])

            nodes_are_different = node_input[0] != node_output[0]

            new_connection = jnp.array([[node_input[0], node_output[0], weight, genome.innovation_count, nodes_are_different * 1.0]])
            new_connections = jnp.concatenate((genome.connections, new_connection), axis=0)

            matrix = genome.matrix.at[int(node_input[0]), int(node_output[0])].set(weight)
            return GenomeData(genome.nodes, new_connections, genome.innovation_count + 1, genome.node_count, genome.key, matrix), genome.innovation_count + 1
        
        def _express(genome: GenomeData):
            """Converts genome to weight matrix and activation vector"""
            nodes = genome.nodes
            connections = genome.connections

            n_nodes = nodes.shape[0]
            weight_mat = jnp.zeros((n_nodes, n_nodes))

            # enabled_connections = jnp.where(connections[:, 4] == 1.0)[0]

            for conn in connections:

                # if conn[4]:
                source, target, weight = jnp.int32(conn[0]), jnp.int32(conn[1]), conn[2]
                weight_mat = weight_mat.at[source, target].set(weight)
            
            return GenomeData(nodes, connections, genome.innovation_count, genome.node_count, genome.key, weight_mat)

        self._init_pop = jax.jit(jax.vmap(_init_pop))
        self._forward = jax.jit(jax.vmap(_forward))
        self.mutate_add_node = __mutate_add_node
        self.mutate_add_connection = __mutate_add_connection
        self.express = jax.jit(jax.vmap(_express))

    def init_pops(self, keys):
        return self._init_pop(keys)
    
    def forward_pops(self, pops, obs):
        return self._forward(pops, obs)

    def mutate(self, genome: GenomeData):
        nodes = genome.nodes
        connections = genome.connections
        n_conn = connections.shape[0]

        disabled = jnp.where(connections[:, 4] == 0.0)[0]
        reenable = jax.random.bernoulli(jax.random.PRNGKey(0), p=self.config['prob_enable'], shape=(disabled.shape[0],))
        connections = connections.at[disabled, 4].set(reenable)

        mutated_weights = jax.random.bernoulli(jax.random.PRNGKey(0), p=self.config['prob_weight_mut'], shape=(n_conn,))
        weights_change = jax.random.uniform(jax.random.PRNGKey(0), shape=(n_conn,), minval=-1.0, maxval=1.0) * mutated_weights
        connections = connections.at[:, 2].add(weights_change)

        connections = connections.at[:, 2].set(jnp.clip(connections[:, 2], -self.config['clamp_weights'], self.config["clamp_weights"]))

        genome = GenomeData(nodes, connections, genome.innovation_count, genome.node_count, genome.key, genome.matrix)

        if jax.random.uniform(jax.random.PRNGKey(0), shape=(), minval=0.0, maxval=1.0) < self.config["prob_add_node"] and jnp.any(connections[:, 4] == 1.0):
            genome, innov = self.mutate_add_node(GenomeData(genome.nodes, genome.connections, genome.innovation_count, genome.node_count, genome.key, genome.matrix))

        if jax.random.uniform(jax.random.PRNGKey(0), shape=(), minval=0.0, maxval=1.0) < self.config["prob_add_connection"]:
            genome, innov = self.mutate_add_connection(GenomeData(genome.nodes, genome.connections, genome.innovation_count, genome.node_count, genome.key, genome.matrix))

        return genome, innov
    
    def crossover(self, parent1: GenomeData, parent2: GenomeData):
        aConn = parent1.connections.copy()[:, 3]
        bConn = parent2.connections.copy()[:, 3]
        # child = GenomeData(parent1.nodes.copy(), parent1.connections.copy(), parent1.innovation_count, parent1.node_count, parent1.key, parent1.matrix.copy())
        matching, g1, g2 = jnp.intersect1d(aConn, bConn, return_indices=True)

        bProb = 0.5
        bGenes = jax.random.bernoulli(jax.random.PRNGKey(0), p=bProb, shape=(len(matching),))

        # child.connections.at[g1[bGenes[0]], 2] = parent2.connections[g2[bGenes[0]], 2]
        cross_connections = parent1.connections.at[g1[bGenes], 2].set(parent2.connections[g2[bGenes], 2])
        # update the matrix
        # batch_nodes = parent1.nodes.unsquzze(0)
        
        # child_matrix = self.express(parent1.nodes.unsqueeze(0), cross_connections.unsqueeze(0))

        return GenomeData(parent1.nodes.copy(), cross_connections, parent1.innovation_count, parent1.node_count, parent1.key, parent1.matrix.copy())

    # def express(self, nodes, connections):
    #     """Converts genome to weight matrix and activation vector"""
    #     n_nodes = nodes.shape[0]
    #     weight_mat = jnp.zeros((n_nodes, n_nodes))

    #     for conn in connections:
    #         if conn[4]:
    #             source, target, weight = int(conn[0]), int(conn[1]), conn[2]
    #             weight_mat = weight_mat.at[source, target].set(weight)
        
    #     return weight_mat
    
    # def add_node(self, node_type):
    #     node_type_map = {"input": 0, "output": 1, "hidden": 2}
    #     node_type_numeric = node_type_map[node_type]

    #     new_id = self.node_count
    #     new_node = jnp.array([[new_id, node_type_numeric]])  
    #     self.nodes = jnp.concatenate((self.nodes, new_node), axis=0) 
    #     self.node_count += 1
    #     return new_id
    
    # def add_connection(self, in_node, out_node, weight):
    #     new_connection = jnp.array([[in_node, out_node, weight, self.innovation_count, 1.0]]) 
    #     self.connections = jnp.concatenate((self.connections, new_connection), axis=0) 
    #     self.innovation_count += 1
    #     return self.innovation_count 
    
    # def mutate_add_node(self):
    #     connection_to_split = self.connections[jax.random.randint(jax.random.PRNGKey(0), shape=(), minval=0, maxval=len(self.connections))]
    #     connection_to_split = connection_to_split.at[-1].set(0.0)
    #     new_node_id = self.add_node("hidden")
    #     self.add_connection(connection_to_split[0], new_node_id, 1.0)
    #     self.add_connection(new_node_id, connection_to_split[1], connection_to_split[2])

    # def mutate_add_connection(self, weight_range=(-1.0, 1.0)):
    #     node_input = self.nodes[jax.random.randint(jax.random.PRNGKey(0), shape=(), minval=0, maxval=len(self.nodes))]
    #     node_output = self.nodes[jax.random.randint(jax.random.PRNGKey(0), shape=(), minval=0, maxval=len(self.nodes))]

    #     if node_input[0] != node_output[0]:
    #         weight = jax.random.uniform(jax.random.PRNGKey(0), shape=(), minval=weight_range[0], maxval=weight_range[1])
    #         self.add_connection(node_input[0], node_output[0], weight)

    def get_weight(self):
        num_nodes = self.nodes.shape[0]

        weight_mat = jnp.zeros((num_nodes, num_nodes))

        for conn in self.connections:
            if conn[3]:
                source, target, weight = int(conn[0]), int(conn[1]), conn[2]
                weight_mat = weight_mat.at[source, target].set(weight)
        
        return weight_mat
    
    def topological_sort(self):
        input_nodes = self.nodes[self.nodes[:, 1] == 0][:, 0]
        output_nodes = self.nodes[self.nodes[:, 1] == 1][:, 0]
        hidden_nodes = self.nodes[self.nodes[:, 1] == 2][:, 0]

        sorted_nodes = jnp.concatenate((input_nodes, hidden_nodes, output_nodes), axis=0)
        return sorted_nodes
    
    def reorder_weight_matrix(self, weight_mat):
        sorted_order = self.topological_sort()
        sorted_matrix = weight_mat[sorted_order][:, sorted_order]
        self.matrix = sorted_matrix
        return sorted_matrix, sorted_order
        
    def net_to_mat(self):
        weight_mat = self.get_weight()
        sorted_weight_mat, sorted_order = self.reorder_weight_matrix(weight_mat)
        return sorted_weight_mat, sorted_order
   
    # def forward(self, obs):
    #     for node_id  in range(self.node_count):
    #         self.node_values[node_id] = jnp.float32(0.0)

    #     input_nodes = [node[0] for node in self.nodes if node[1] == 0]

    #     for i, node_id in enumerate(input_nodes):
    #         self.node_values[int(node_id)] = obs[i]

    #     for connection in self.connections:
    #         in_node, out_node, weight, enabled = int(connection[0]), int(connection[1]), connection[2], connection[4]

    #         if enabled:
    #             # print(in_node, out_node, weight)
    #             self.node_values[out_node] += self.node_values[in_node] * weight

    #     for node in self.nodes:
    #         node_id, node_type = node[0], node[1]
    #         if node_type == 1 or node_type == 2:
    #             self.node_values[int(node_id)] = jax.nn.relu(self.node_values[int(node_id)])
        
    #     output_nodes = [node[0] for node in self.nodes if node[1] == 1]
    #     output_values = jnp.array([self.node_values[int(node_id)] for node_id in output_nodes])

    #     return output_values
        
    def forward(self, obs, n_inputs, n_outputs):
        n_nodes = self.nodes.shape[0]
    
        node_activations = jnp.zeros(n_nodes)
        node_activations = node_activations.at[: n_inputs].set(obs)

        for i in range(n_inputs, n_nodes):
            wieghted_sum = jnp.dot(node_activations, self.matrix[:, i])
            node_activations = node_activations.at[i].set(jax.nn.relu(wieghted_sum))
        
        output = node_activations[-n_outputs:]
        return output
    
    # def backward(self, loss):
    #     pass

    

    
    
    def init_population(self):
        # Initialize a simple population
        input_num = 3
        output_num = 2

        for _ in range(input_num):
            self.add_node("input")

        for _ in range(output_num):
            self.add_node("output")

        for i in range(input_num):
            for j in range(output_num):
                self.add_connection(i, j + input_num, random.uniform(-1.0, 1.0))

    def visualize(self, nodes: jnp.ndarray, connections: jnp.ndarray):
        import networkx as nx
        import matplotlib.pyplot as plt

        # nodes = genome.nodes
        # connections = genome.connections

        G = nx.DiGraph()
        for node in nodes:
            G.add_node(int(node[0]), type=int(node[1]))  

        for connection in connections:
            if connection[4] == 1.0:
                G.add_edge(int(connection[0]), int(connection[1]), weight=float(connection[2]))

        pos = nx.spring_layout(G)
        node_colors = ['green' if node[1] == 0 else 'blue' if node[1] == 1 else 'orange' for node in nodes] 
        edge_colors = ['black' for connection in connections]  

        nx.draw(
            G, 
            pos, 
            with_labels=True, 
            node_color=node_colors, 
            edge_color=edge_colors, 
            edge_cmap=plt.cm.Blues, 
            edge_vmin=-1.0, 
            edge_vmax=1.0, 
            width=2.0 
        )
        connections = connections[connections[:, 4] == 1.0]
        edge_labels = {(int(connection[0]), int(connection[1])): round(float(connection[2]), 2) for connection in connections}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')


        plt.show()


#Test
envs = 100
obs_size = 12
config = {
    "prob_enable": 0.5,
    "prob_weight_mut": 0.8,
    "clamp_weights": 5.0,
    "prob_add_node": 0.5,
    "prob_add_connection": 0.5
}
# example_genome = Genome(config=config)
# pops = example_genome.init_pops(jax.random.split(jax.random.PRNGKey(0), envs))
# # create random observation of 3 values and for 10 environments
# obs = jax.random.uniform(jax.random.PRNGKey(0), shape=(envs, obs_size), minval=-1.0, maxval=1.0)

# pops = example_genome.express(pops)
# # pops.matrix = matrices
# # assign the matrix to the population


# output = example_genome.forward_pops(pops, obs)
# print(output)
# gen = GenomeData(pops.nodes[0], pops.connections[0], pops.innovation_count[0], pops.node_count[0], pops.key[0], pops.matrix[0])
# # gen, _ = example_genome.mutate(gen)
# parent1 = GenomeData(pops.nodes[1], pops.connections[1], pops.innovation_count[1], pops.node_count[1], pops.key[1], pops.matrix[1])
# child = example_genome.crossover(parent1, gen)
# example_genome.visualize(child.nodes, child.connections)
# print(gen.nodes.shape)
# print(gen.connections.shape)

