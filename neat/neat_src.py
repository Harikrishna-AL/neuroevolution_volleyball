from genome import GenomeData, Genome


import jax
import jax.numpy as jnp
from jax import lax, vmap
import random
import numpy as np

def to_jax(genome):
    """Converts the GenomeData attributes to a tuple of JAX arrays."""
    return np.array([genome.nodes, genome.connections, genome.innovation_count, genome.node_count, genome.key, genome.matrix, genome.fitness])

@staticmethod
def from_jax(data):
    """Converts a tuple of JAX arrays back into a GenomeData object."""
    return GenomeData(jnp.asarray(data.nodes), jnp.asarray(data.connections), int(data.innovation_count), int(data.node_count), jnp.asarray(data.key), jnp.asarray(data.matrix), float(data.fitness))

def convert_population_to_numpy(population):
        arr_population = []
        for i in range(population.nodes.shape[0]):
            genome = GenomeData(
                population.nodes[i],
                population.connections[i],
                population.innovation_count[i],
                population.node_count[i],
                population.key[i],
                population.matrix[i],
            )
            arr_population.append(genome)

        return arr_population


def _speciate_fn(population, compatibility_threshold, species_target):
    def add_to_species(genome, species):
        # Compatibility check as a JAX function
        def compatibility_check(s):
            return _compatibility_distance_fn(genome, s[0]) < compatibility_threshold

        # Use JAX's dynamic conditions
        found_species = jax.lax.cond(
            jnp.any(jnp.array([compatibility_check(s) for s in species])),  # This needs to be a JAX operation
            lambda _: jax.lax.dynamic_update_index_in_dim(
                species, genome, jnp.size(species) - 1, axis=0  # Use axis=0 instead of 0 for clarity
            ),
            lambda _: jnp.append(species, jnp.expand_dims(genome, axis=0)),  # Append new species, expand dims
            operand=None
        )
        return found_species

    species = []  # Initialize species as an empty JAX array with dtype=object
    for genome in population:
        # genome = to_jax(genome)
        # print(genome)
        species = add_to_species(genome, species)
    
    # Adjust compatibility threshold based on species count
    compatibility_threshold = jax.lax.cond(
        jnp.size(species) < species_target,
        lambda _: compatibility_threshold * 0.9,
        lambda _: compatibility_threshold * 1.1,
        operand=None
    )

    return species

def _compatibility_distance_fn(genome1, genome2, c1=1.0, c2=1.0, c3=0.4):
    g1 = genome1.connections
    g2 = genome2.connections
    N = max(len(g1), len(g2))
    
    matching = jnp.sum(jnp.equal(g1[:, 3], g2[:, 3]))
    disjoint = len(g1) + len(g2) - 2 * matching
    excess = jnp.abs(len(g1) - len(g2))
    
    weight_diff = jnp.sum(jnp.abs(g1[:, 2] - g2[:, 2]) * jnp.equal(g1[:, 3], g2[:, 3]))
    
    # Use jax.lax.cond for the conditional statement
    avg_weight_diff = lax.cond(matching > 0,
                               lambda _: weight_diff / matching,
                               lambda _: jnp.array(0.0),
                               operand=None)
    
    return (c1 * excess / N) + (c2 * disjoint / N) + (c3 * avg_weight_diff)


class GeneticEvolution:
    def __init__(self, population_size, genome, obs_size, mutation_rate=0.5, crossover_rate=0.5):
        self.population_size = population_size
        self.genome = genome
        self.mutation_rate = mutation_rate
        self.cross_rate = crossover_rate
        self.obs_size = obs_size
        self.keys = jax.random.split(jax.random.PRNGKey(0), self.population_size)
        self.population = self.convert_population_to_numpy(self.init_population())
        self.rewards = []
        self.species = []
        self.compatibility_threshold = 0.2
        self.species_target = 5

    def init_population(self):
        return self.genome.init_pops(self.keys)
    
    def convert_population_to_numpy(self, population):
        arr_population = []
        for i in range(population.nodes.shape[0]):
            genome = GenomeData(
                population.nodes[i],
                population.connections[i],
                population.innovation_count[i],
                population.node_count[i],
                population.key[i],
                population.matrix[i],
            )
            arr_population.append(genome)

        return arr_population

    def ask(self):
        if not self.species:
            self.species = self.speciate()
        else:
            self.population = self.evolve()
            self.species = self.speciate()
        return self.population

    def tell(self, rewards):
        self.rewards = rewards
        for i in range(len(self.population)):
            # print(self.population[i])
            genome = GenomeData(
                self.population[i].nodes,
                self.population[i].connections,
                self.population[i].innovation_count,
                self.population[i].node_count,
                self.population[i].key,
                self.population[i].matrix,
                fitness=rewards[i],
            )
            # self.population[i].fitness = rewards[i]
            self.population[i] = genome

    def speciate(self):
        species = []
        for genome in self.population:
            found_species = False
            # print(genome)
            for s in species:
                if self.compatibility_distance(genome, s[0]) < self.compatibility_threshold:
                    s.append(genome)
                    found_species = True
                    break
            if not found_species:
                species.append([genome])
        
        # Adjust species target
        if len(species) < self.species_target:
            self.compatibility_threshold *= 0.5
        elif len(species) > self.species_target:
            self.compatibility_threshold *= 2
        
        return species

    def compatibility_distance(self, genome1, genome2, c1=1.0, c2=1.0, c3=0.4):
        # Implement NEAT-specific distance metric
        g1 = genome1.connections
        g2 = genome2.connections
        
        N = max(len(g1), len(g2))
        
        matching = sum(1 for conn1 in g1 for conn2 in g2 if conn1[3] == conn2[3])
        disjoint = len(g1) + len(g2) - 2 * matching
        excess = abs(len(g1) - len(g2))
        
        weight_diff = sum(abs(conn1[2] - conn2[2]) for conn1 in g1 for conn2 in g2 if conn1[3] == conn2[3])
        avg_weight_diff = weight_diff / matching if matching > 0 else 0
        
        # print("Excess: ", excess, "Disjoint: ", disjoint, "Avg weight diff: ", avg_weight_diff)
        # print("Compatibility distance: ", (c1 * excess / N) + (c2 * disjoint / N) + (c3 * avg_weight_diff))
        return (c1 * excess / N) + (c2 * disjoint / N) + (c3 * avg_weight_diff)

    def evolve(self):
        new_population = []
        print("Species: ", len(self.species))
        for species in self.species:
            # Sort species by fitness
            new_species = []
            species.sort(key=lambda x: x.fitness, reverse=True)
            
            # Elitism: keep the top 20% performing individual
            new_species.extend(species[:len(species) // 5])
        
            # Fitness sharing
            adjusted_fitnesses = self.fitness_sharing(species)
            
            # Selection and reproduction
            while len(new_species) < len(species):
                parent1 = self.tournament_selection(species, adjusted_fitnesses)
                parent2 = self.tournament_selection(species, adjusted_fitnesses)
                
                if random.random() < self.cross_rate:
                    child = self.genome.crossover(parent1, parent2)
                else:
                    child, _ = self.genome.mutate(parent1)
                
                new_species.append(child)

            new_population.extend(new_species)
            # print("new pop: ", new_population)
        
        # Ensure population size remains constant
        while len(new_population) < self.population_size:
            new_population.append(random.choice(new_population))
        
        return new_population[:self.population_size]

    def fitness_sharing(self, species):
        shared_fitnesses = []
        for i, genome in enumerate(species):
            sh = sum(self.sharing_function(genome, other) for j, other in enumerate(species) if i != j)
            shared_fitnesses.append(genome.fitness / (sh + 1))
        return shared_fitnesses

    def sharing_function(self, genome1, genome2, sharing_threshold=3.0):
        d = self.compatibility_distance(genome1, genome2)
        if d > sharing_threshold:
            return 0
        return 1 - (d / sharing_threshold)

    def tournament_selection(self, species, adjusted_fitnesses, tournament_size=1):
        selected = random.sample(range(len(species)), tournament_size)
        winner = max(selected, key=lambda i: adjusted_fitnesses[i])
        return species[winner]
    
    def get_best_genome(self):
        return max(self.population, key=lambda x: x.fitness)

class Policy:
    def __init__(self, genome, pops, config):
        self.genome = genome
        self.pops = pops
        self.forward = self._forward
        n_nodes = jnp.zeros((len(self.pops), 1))
        self.config = config

        for i in range(len(self.pops)):
            n_nodes = n_nodes.at[i, 0].set(self.pops[i].nodes.shape[0])

        self.n_nodes = n_nodes
        self.matrices = jnp.array(self.genome.express_all(self.pops, n_nodes))

        def _slice_activation(activations, n_nodes):
            """Helper function to slice activations based on n_nodes."""
            start_idx = n_nodes - self.config["output_num"]
            return jax.lax.dynamic_slice(activations, (jnp.int32(start_idx),), (self.config['output_num'],))

        self.slice_activation = _slice_activation

    def _forward(self, obs):
        activations = self.genome.forward_pops(self.matrices, obs)
        return jax.vmap(self.slice_activation)(activations, self.n_nodes[:, 0])