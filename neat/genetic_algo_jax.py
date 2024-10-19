from genome import Genome, GenomeData
from utils import manage_specie_shape

import jax
import jax.numpy as jnp
from jax import lax
import random


class GeneticEvolution:
    def __init__(self, population_size, genome, obs_size, mutation_rate=0.5, crossover_rate=0.5):
        self.population_size = population_size
        self.genome = genome
        self.mutation_rate = mutation_rate
        self.cross_rate = crossover_rate
        self.obs_size = obs_size
        self.keys = jax.random.split(jax.random.PRNGKey(0), self.population_size)
        # self.population = genome.init_pops(self.keys)
        self.population = []
        self.rewards = []

        self.distance_vmap = jax.vmap(self.compatibility_distance, in_axes=(None,0))

    import numpy as np

    # Assuming 'population' is a GenomeData object
    def convert_population_to_numpy(self,population):
        arr_population = []
        for i in range(population.nodes.shape[0]):
            genome = GenomeData(population.nodes[i], population.connections[i], population.innovation_count[i], population.node_count[i], population.key[i], population.matrix[i])
            arr_population.append(genome)
        
        return arr_population

    def convert_population_to_genome(self,population):
        nodes = []
        connections = []
        innovation_count = []
        node_count = []
        key = []
        matrix = []
        for i in range(len(population)):
            nodes.append(population[i].nodes)
            connections.append(population[i].connections)
            innovation_count.append(population[i].innovation_count)
            node_count.append(population[i].node_count)
            key.append(population[i].key)
            matrix.append(population[i].matrix)
        
        return GenomeData(jnp.array(nodes), jnp.array(connections), jnp.array(innovation_count), jnp.array(node_count), jnp.array(key), jnp.array(matrix))

    def ask(self):
        length = len(self.population)
        if length == 0:
            self.population = self.genome.init_pops(self.keys)
            self.population = self.convert_population_to_numpy(self.population)
            # print(self.population)
            
        else:
            self.population = self.rank_population()
            # TO DO 
            self.species = self.speciate()
            self.population = self.evolve()
        
        return self.population

    
    def tell(self, rewards):
        self.rewards = rewards
        self.fitnesses = rewards
        for i in range(len(self.population)):
            # print(self.population[i])
            genome = GenomeData(self.population[i].nodes, self.population[i].connections, self.population[i].innovation_count, self.population[i].node_count, self.population[i].key, self.population[i].matrix, fitness=rewards[i])
            # self.population[i].fitness = rewards[i]
            self.population[i] = genome

    def rank(self, population):
        fitnesses = jnp.array([genome.fitness for genome in population])
        sorted_population = [population[idx] for idx in jnp.argsort(fitnesses)[::-1]]
        return sorted_population


    def rank_population(self):
        sorted_population = self.rank(self.population)
        return sorted_population
    
    def compatibility_distance(self, connections1, connections2, c1=1, c2=1, c3=0.4):
        if len(connections1.shape) == 0 or len(connections2.shape) == 0:
            return jnp.inf
        
        max_innov = jnp.max(jnp.concatenate([connections1[:,3], connections2[:,3]]), axis=0, keepdims=True)

        N = jnp.max(jnp.array([connections1.shape[0], connections2.shape[0]]))

        innovations1 = connections1[:, 3]
        innovations2 = connections2[:, 3]

        # print("Innovations1: ",innovations1)
        # print("Innovations2: ",innovations2)
        
        excess_genes = 0
        disjoint_genes = 0
        weight_diff_sum = 0.0
        matching_genes = 0

        max_innov = jnp.array(max_innov, int)
        max_innov = max_innov[0]
        max_innov = jax.lax.stop_gradient(max_innov)

        def loop_body(i, carry):
            excess_genes, disjoint_genes, weight_diff_sum, matching_genes = carry

            # Check if gene i exists in both connections1 and connections2
            in1 = jnp.any(innovations1 == i)
            in2 = jnp.any(innovations2 == i)

            # Find matching weights if both have gene i, otherwise return 0
            def get_weight_diff(_):
                idx1 = jnp.argmax(innovations1 == i)
                idx2 = jnp.argmax(innovations2 == i)
                return jnp.abs(connections1[idx1, 2] - connections2[idx2, 2])

            weight_diff_sum = lax.cond(
                in1 & in2,
                lambda _: weight_diff_sum + get_weight_diff(_),
                lambda _: weight_diff_sum,
                operand=None
            )

            # Update matching genes count
            matching_genes = lax.cond(
                in1 & in2,
                lambda _: matching_genes + 1,
                lambda _: matching_genes,
                operand=None
            )

            # Update excess genes
            excess_genes = lax.cond(
                (in1 | in2) & ((i > jnp.max(innovations1)) | (i > jnp.max(innovations2))),
                lambda _: excess_genes + 1,
                lambda _: excess_genes,
                operand=None
            )

            # Update disjoint genes
            disjoint_genes = lax.cond(
                # (in1 | in2) & (i <= jnp.max(innovations1)) & (i <= jnp.max(innovations2)),
                (in1 | in2) & (in1 != in2) & ((i <= jnp.max(innovations1)) | (i <= jnp.max(innovations2))),
                lambda _: disjoint_genes + 1,
                lambda _: disjoint_genes,
                operand=None
            )

            return excess_genes, disjoint_genes, weight_diff_sum, matching_genes

        # Use lax.fori_loop for the loop
        initial_carry = (0, 0, 0.0, 0)
        excess_genes, disjoint_genes, weight_diff_sum, matching_genes = lax.fori_loop(
            0, max_innov + 1, loop_body, initial_carry)
 
        avg_weight_diff = lax.cond(
        matching_genes > 0,
        lambda _: weight_diff_sum / matching_genes,
        lambda _: 0.0,
        operand=None
        )
        # Calculate the compatibility distance
        # print("Excess genes: ",excess_genes)
        # print("Disjoint genes: ",disjoint_genes)
        # print("Average weight difference: ",avg_weight_diff)
        # print("Excess genes: ",excess_genes)
        # print("Disjoint genes: ",disjoint_genes)
        # print("Average weight difference: ",avg_weight_diff)
        # print("Matching genes: ",matching_genes)
        # print("N: ",N)
        distance = c1 * (excess_genes / N) + c2 * (disjoint_genes / N) + c3 * (avg_weight_diff/ matching_genes)
        return distance

    def get_enabled_connections(self, connections):
        return connections[connections[:, 4] == 1.0]
    
    def speciate(self):
        species = []

        for genome in self.population:
            genome_connections = self.get_enabled_connections(genome.connections)
            distances = self.distance_vmap(genome_connections, jnp.array([manage_specie_shape(self.get_enabled_connections(specie[0].connections), genome_connections.shape[0]) for specie in species]))
            # print("Distances: ",distances)
            # print("Distances shape: ",distances.shape)
            found = jnp.any(distances < 1)
            if found:
                specie_index = jnp.argmax(distances < 1)
                species[specie_index].append(genome)
            else:
                species.append([genome])

        # adjusted fitness for each species
        for s in range(len(species)):
            specie = species[s]
            specie_size  = len(specie)
            for g in range(len(specie)):
                genome = specie[g]
                genome = GenomeData(genome.nodes, genome.connections, genome.innovation_count, genome.node_count, genome.key, genome.matrix, fitness=genome.fitness / specie_size)
                specie[g] = genome

            # rank the species
            specie = self.rank(specie)
            species[s] = specie

        return species

    def eval_fitness(self,genome, env):
        reward = 0
        obs = env.reset()
        done = False
        total_reward = 0
        idx = 0
        while not done:
            action = self.run(genome, obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            idx += 1
            # print(reward)
            if idx > 1000:
                break

        return total_reward

    def evolve(self):
        new_population = []
        for specie in self.species:
            # specie = self.rank_population(specie)
            top_n = max(1, len(specie) // 5)
            new_specie = specie[:top_n]
            print("Specie length: ",len(specie))    
            while len(new_specie) < len(specie):
                parent1_idx = random.randint(0, top_n-1)
                parent2_idx = random.randint(0, top_n-1)

                parent1 = specie[parent1_idx]
                parent2 = specie[parent2_idx]
                cross_chances = random.random()
                # print("Cross chances: ",cross_chances)
                if cross_chances < self.cross_rate:
                    child = self.genome.crossover(parent1, parent2)
                else:
                    child, _ = self.genome.mutate(parent1)
                
                new_specie.append(child)
            
            new_population.extend(new_specie)

        return new_population


    def train(self, env, num_generations):
        for i in range(num_generations):
            self.evolve(env)
            print(f"Generation {i} completed")
            print(f"Best fitness: {self.eval_fitness(self.population[0], env=env)}")
            # print(f"Average fitness: {sum([self.eval_fitness(genome, env=env) for genome in self.population]) / self.population_size}")
            self.population[0].visualize()
            print("\n")

        return self.population[5]

class Policy:
    def __init__(self, genome, pops):
        self.genome = genome  
        self.pops = pops
        self.forward = self._forward
        n_nodes = jnp.zeros((len(self.pops),1))

        for i in range(len(self.pops)):
            n_nodes = n_nodes.at[i,0].set(self.pops[i].nodes.shape[0])

        self.n_nodes = n_nodes
        self.matrices = jnp.array(self.genome.express_all(self.pops, n_nodes))

        def _slice_activation(activations, n_nodes):
            """Helper function to slice activations based on n_nodes."""
            start_idx = n_nodes - 3
            return jax.lax.dynamic_slice(activations, (jnp.int32(start_idx),), (3,))

        self.slice_activation = _slice_activation

    def _forward(self, obs):
        activations = self.genome.forward_pops(self.matrices, obs)
        return jax.vmap(self.slice_activation)(activations, self.n_nodes[:, 0])


# num_envs = 10
# obs_size = 12
# config = {
#     "prob_enable": 0.5,
#     "prob_weight_mut": 0.8,
#     "clamp_weights": 5.0,
#     "prob_add_node": 0.5,
#     "prob_add_connection": 0.5,
#     "input_num": obs_size,
#     "output_num": 3
# }
# test_genome = Genome(config=config)
# evolver = GeneticEvolution(num_envs, test_genome, 12)
# pops = evolver.ask()
# genome1 = pops[0]
# genome2 = pops[1]

# dist = evolver.compatibility_distance(genome1.connections, genome2.connections)
# print("Distance: ",dist)
    
