from genetic_algo_jax import GeneticEvolution, Policy
from genome import Genome, GenomeData
from utils import get_rewards

import jax
import jax.numpy as jnp
from evojax.task.slimevolley import SlimeVolley
import imageio


num_envs = 100
max_steps = 3000
num_generations = 20

env = SlimeVolley(max_steps=max_steps, test=True)

keys = jax.random.split(jax.random.PRNGKey(0), num_envs)


done = jnp.zeros((num_envs,), dtype=bool)
config = {
    "prob_enable": 0.5,
    "prob_weight_mut": 0.8,
    "clamp_weights": 5.0,
    "prob_add_node": 0.5,
    "prob_add_connection": 0.5
}
gen = Genome(config=config)

evolver = GeneticEvolution(100, gen, env)

frames_gifs = []
RENDER = False

def train():
    for i in range(num_generations):
        
        pops = evolver.ask()
        policy = Policy(gen, pops)
        total_rewards = get_rewards(keys, env, policy, num_envs=100)
        evolver.tell(total_rewards)

        print(f"Mean reward: {jnp.mean(total_rewards)}")
        print(f"Generation {i} completed")



train()
