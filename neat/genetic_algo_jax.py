from genome import Genome, GenomeData

import jax
import jax.numpy as jnp
import random


class GeneticEvolution:
    def __init__(self, population_size, genome, obs_size, mutation_rate=0.1, crossover_rate=0.5):
        self.population_size = population_size
        self.genome = genome
        self.mutation_rate = mutation_rate
        self.cross_rate = crossover_rate
        self.obs_size = obs_size
        self.keys = jax.random.split(jax.random.PRNGKey(0), self.population_size)
        # self.population = genome.init_pops(self.keys)
        self.population = []
        self.rewards = []

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
            # self.population = self.speciate()
            self.population = self.evolve()
        
        return self.population

    
    def tell(self, rewards):
        self.rewards = rewards
        self.fitnesses = rewards

    def rank_population(self):
        sorted_population = [self.population[idx] for idx in jnp.argsort(self.fitnesses)[::-1]]
        return sorted_population

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
        sorted_population = self.rank_population()
        top_n = self.population_size // 5
        new_population = sorted_population[:top_n]

        while len(new_population) < self.population_size:
            parent1_idx = random.randint(0, top_n-1)
            parent2_idx = random.randint(0, top_n-1)

            parent1 = sorted_population[parent1_idx]
            parent2 = sorted_population[parent2_idx]

            if random.random() < self.cross_rate:
                child = self.genome.crossover(parent1, parent2)
            else:
                child, _ = self.genome.mutate(parent1)
            
            new_population.append(child)

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
    
