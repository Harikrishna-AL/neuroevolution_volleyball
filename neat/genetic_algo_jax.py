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
        population = []
        for i in range(population.nodes.shape[0]):
            genome = GenomeData(population.nodes[i], population.connections[i], population.innovation_count[i], population.node_count[i], population.keys[i], population.matrix[i])
            population.append(genome)
        
        return population


    def ask(self):
        type_pop = type(self.population)
        # if type_pop == list:
        #     length = len(self.population)
        # else:
        #     length = self.population.nodes.shape[0]
        length = len(self.population)
        if length == 0:
            self.population = self.genome.init_pops(self.keys)
            self.population = self.convert_population_to_numpy(self.population)
            print(self.population)
            
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
        # sort the population based on fitness
        # fitnesses = [self.eval_fitness(genome, env=env) for genome in self.population]
        sorted_population = [self.population[idx] for idx in jnp.argsort(self.fitnesses)[::-1]]
        return sorted_population
    
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

    # def evolve(self, env):
    #     fitnesses = [self.eval_fitness(genome, env=env) for genome in self.population]
    #     sorted_population = [x for _, x in sorted(zip(fitnesses, self.population), key=lambda pair: pair[0], reverse=True)]
    #     new_population = sorted_population[:self.population_size // 2]

    #     while len(new_population) < self.population_size:
    #         parent1 = random.choice(sorted_population)
    #         parent2 = random.choice(sorted_population)
    #         child1, child2 = self.crossover(parent1, parent2)
    #         self.mutate(child1)
    #         self.mutate(child2)
    #         new_population.extend([child1, child2])

    #     self.population = new_population
    def evolve(self):
        # select the top 20% of the population
        new_population = self.population[:self.population_size // 5]
        while len(new_population) < self.population_size:
            parent1 = random.choice(self.population)
            parent2 = random.choice(self.population)
            # child1, child2 = self.crossover(parent1, parent2)
            if random.random() < self.cross_rate:
                child, innov = self.genome.crossover(parent1, parent2)
            else:
                child, innov = self.genome.mutate(parent1)

            new_population.extend(child)

        self.population = new_population
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

# config = {
#     "prob_enable": 0.5,
#     "prob_weight_mut": 0.8,
#     "clamp_weights": 5.0,
#     "prob_add_node": 0.5,
#     "prob_add_connection": 0.5
# }
# x = Genome(config=config)
# algo = GeneticEvolution(100, x, 12)
# pop = algo.ask()
# print(pop.nodes.shape)

class Policy:
    def __init__(self, genome, pops):
        self.genome = genome   #iloveyou
        self.pops = pops
    
    def forward(self, obs):
        return self.genome.forward_pops(self.pops, obs)
    
# obs = jax.random.uniform(jax.random.PRNGKey(0), shape=(100, 12), minval=-1.0, maxval=1.0)
# policy = Policy(x, pop)
# output = policy.forward(obs)
# print(output.shape)
