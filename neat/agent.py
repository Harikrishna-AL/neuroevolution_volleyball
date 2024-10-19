from genetic_algo_jax import GeneticEvolution, Policy
from genome import Genome, GenomeData
from utils import get_rewards

import jax
import jax.numpy as jnp
from evojax.task.slimevolley import SlimeVolley
import imageio
import gym
import slimevolleygym
import time

num_envs = 100
max_steps = 3000
num_generations = 50

env = SlimeVolley(max_steps=max_steps, test=False)

# keys = jax.random.split(jax.random.PRNGKey(0), num_envs)


done = jnp.zeros((num_envs,), dtype=bool)
config = {
    "prob_enable": 0.5,
    "prob_weight_mut": 0.8,
    "clamp_weights": 5.0,
    "prob_add_node": 0.5,
    "prob_add_connection": 0.5,
    "input_num": 12,
    "output_num": 3
}
gen = Genome(config=config)

evolver = GeneticEvolution(num_envs, gen, env)

frames_gifs = []
RENDER = True

def train():
    keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
    for i in range(num_generations):
        # split keys for each population using the initial key
        print(f"Generation {i+1}")
        keys = jax.random.split(keys[0], num_envs)
        pops = evolver.ask()
        policy = Policy(gen, pops)
        total_rewards = get_rewards(keys, env, policy, num_envs=num_envs)
        evolver.tell(total_rewards)
        
        print(total_rewards)
        print(f"Max reward: {jnp.max(total_rewards)}")
        print(f"Mean reward: {jnp.mean(total_rewards)}")
        print(f"Min reward: {jnp.min(total_rewards)}")
        print("-----------------------------------------")
        

    gen.visualize(pops[0].nodes, pops[0].connections)
    return pops[0]


def test(best_policy, RENDER=False):
    env = gym.make("SlimeVolley-v0")
    obs = env.reset()
    done = False
    total_reward = 0
    frames = []
    step = 0

    while not done:
        obs = obs.reshape(1, -1)
        action_values = Policy(gen, [best_policy]).forward(obs)
        action_values = action_values[0]
        obs, reward, done, info = env.step(action_values)

        # Render only every 10 steps to reduce load
        if RENDER and step % 10 == 0:
            frame = env.render(mode='rgb_array')
            if len(frames) < 100:  # Save only first 100 frames to limit memory use
                frames.append(frame)
            time.sleep(0.01)  # Optional delay for human-friendly frame rate

        total_reward += reward
        step += 1

    # Save the gif if rendering was enabled
    if RENDER:
        imageio.mimsave('slime_volleyball_episode.gif', frames, fps=30)
    
    print(f"Total reward: {total_reward}")
    return total_reward

if __name__ == "__main__":  
    best_net = train()
    test(best_net)


