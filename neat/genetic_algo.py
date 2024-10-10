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
        self.population = genome.init_pops(self.keys)

    def run(self, genome, obs):
        # num_batch = len(obs)
        # for i in range(num_batch):
        output = genome.forward(obs)
        
        return output
        

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

    def connection_in_parent(self, connection, parent_connections):
    # Compare the connection with all parent connections using broadcasting
        return jnp.any(jnp.all(connection == parent_connections, axis=1))


    def crossover(self, parent1, parent2):
        child1 = Genome(parent1.nodes.copy(), parent1.connections.copy(), parent1.innovation_count, parent1.node_count)
        child2 = Genome(parent2.nodes.copy(), parent2.connections.copy(), parent2.innovation_count, parent2.node_count)

        if random.random() < self.cross_rate:
            for connection in parent1.connections:
                if self.connection_in_parent(connection , parent2.connections):
                    child1.add_connection(connection[0], connection[1], connection[2])
                else:
                    child1.add_connection(connection[0], connection[1], connection[2])

            for connection in parent2.connections:
                if self.connection_in_parent(connection, parent1.connections):
                    child2.add_connection(connection[0], connection[1], connection[2])
                else:
                    child2.add_connection(connection[0], connection[1], connection[2])

        return child1, child2

    def mutate(self, genome):
        if random.random() < self.mutation_rate:
            if random.random() < 0.5:
                genome.mutate_add_node()
            else:
                genome.mutate_add_connection()

    def evolve(self, env):
        fitnesses = [self.eval_fitness(genome, env=env) for genome in self.population]
        sorted_population = [x for _, x in sorted(zip(fitnesses, self.population), key=lambda pair: pair[0], reverse=True)]
        new_population = sorted_population[:self.population_size // 2]

        while len(new_population) < self.population_size:
            parent1 = random.choice(sorted_population)
            parent2 = random.choice(sorted_population)
            child1, child2 = self.crossover(parent1, parent2)
            self.mutate(child1)
            self.mutate(child2)
            new_population.extend([child1, child2])

        self.population = new_population

    def train(self, env, num_generations):
        for i in range(num_generations):
            self.evolve(env)
            print(f"Generation {i} completed")
            print(f"Best fitness: {self.eval_fitness(self.population[0], env=env)}")
            # print(f"Average fitness: {sum([self.eval_fitness(genome, env=env) for genome in self.population]) / self.population_size}")
            self.population[0].visualize()
            print("\n")

        return self.population[5]

    
    
x = Genome()
algo = GeneticEvolution(100, x, 12)
print(algo.population.nodes.shape)