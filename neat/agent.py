# # from genome import Genome
# # from genetic_algo import GeneticEvolution

import jax
import jax.numpy as jnp
from utils import get_rewards
# import gym
# import slimevolleygym
# import imageio
# import random

# RENDER = True

# env = gym.make("SlimeVolley-v0")

# key = jax.random.PRNGKey(0)

# num_actions = env.action_space.n
# obs_dim = env.observation_space.shape[0]

# print(num_actions)
# print(obs_dim)

# # def random_policy_params(key, obs_dim, num_actions):

# #     key1, key2 = jax.random.split(key)
# #     W = jax.random.normal(key1, (num_actions, obs_dim)) 
# #     b = jax.random.normal(key2, (num_actions,))  
# #     return W, b

# # def policy(obs, params):
# #     W, b = params
# #     logits = jnp.dot(W, obs) + b
# #     return logits  

# # params = random_policy_params(key, obs_dim, num_actions)

# obs = env.reset()
# done = False
# total_reward = 0
# frames = []
# steps = 3000
# # volley_agent = Genome()
# # volley_agent.init_population()

# # evolver = GeneticEvolution(10, volley_agent)
# # volley_agent = evolver.train(env, 10)
# idx=0
# while not done:

#     # random action array of size 3
#     #random array of size 3
#     action = jax.random.randint(key, shape=(3,), minval=0, maxval=3)
#     # action = jnp.argmax(output)
#     # print(action)
#     obs, reward, done, info = env.step(action)

#     if RENDER:
#         frame = env.render(mode='rgb_array')
#         frames.append(frame)

#     idx += 1
#     total_reward += reward
#     if idx > steps:
#         break

# if RENDER:
#     imageio.mimsave('slime_volleyball_episode.gif', frames, fps=30)

# print("Total reward for this episode:", total_reward)

# env.close()

from evojax.task.slimevolley import SlimeVolley
import imageio

from genetic_algo_jax import GeneticEvolution, Policy
from genome import Genome, GenomeData
# Number of parallel environments
num_envs = 100
max_steps = 3000

# Create the environment
env = SlimeVolley(max_steps=max_steps, test=True)

# Initialize random keys for each environment
keys = jax.random.split(jax.random.PRNGKey(0), num_envs)

# Reset all environments


# Initialize a placeholder to track which environments are done
done = jnp.zeros((num_envs,), dtype=bool)
config = {
    "prob_enable": 0.5,
    "prob_weight_mut": 0.8,
    "clamp_weights": 5.0,
    "prob_add_node": 0.5,
    "prob_add_connection": 0.5
}
gen = Genome(config=config)
# Initialize the genetic algorithm for evolving the agent
evolver = GeneticEvolution(100, gen, env)
pops = evolver.ask()

policy = Policy(gen, pops)
frames_gifs = []
RENDER = False


state = env.reset(keys)
# total_rewards = jnp.zeros((num_envs,))

total_rewards = get_rewards(keys, env, policy, num_envs=100)

evolver.tell(total_rewards)

pops_new = evolver.ask()
policy_new = Policy(gen, pops_new)
total_rewards2 = get_rewards(keys, env, policy, num_envs=100)


print(total_rewards)
print(total_rewards2)
