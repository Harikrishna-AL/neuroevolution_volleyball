import jax
import jax.numpy as jnp
from evojax.task.slimevolley import SlimeVolley
import imageio
# Number of parallel environments
num_envs = 1
max_steps = 3000

# Create the environment
env = SlimeVolley(max_steps=max_steps, test=True)

# Initialize random keys for each environment
keys = jax.random.split(jax.random.PRNGKey(0), num_envs)

# Reset all environments
state = env.reset(keys)

# Initialize a placeholder to track which environments are done
done = jnp.zeros((num_envs,), dtype=bool)

# Function to generate random actions for each environment
def random_action_fn(key):
    return jax.random.randint(key, shape=(3,), minval=0, maxval=3)

# Vectorize the random action generation function
random_action_fn = jax.vmap(random_action_fn)
frames_gifs = []
RENDER = True

# Start a loop until all environments are done
total_rewards = jnp.zeros((num_envs,))
while not jnp.all(done):
    # Generate random actions for each environment
    actions = random_action_fn(keys)
    # print(state)
    # Step the environments
    state, reward, done = env.step(state, actions)

    total_rewards += reward

    # render all the games and store as gif

    if RENDER:
        frames = env.render(state)
        frames_gifs.append(frames)

    if jnp.all(done):
        # print(f"All environments completed after {step + 1} steps")
        print("All environments completed")
        break
# print(state.obs.shape)
# gif1 = [x[0] for x in frames_gifs]
if RENDER:
    imageio.mimsave('slime_volleyball_episode.gif', gif1, fps=30)
# If loop finishes without all environments being done, print that max steps were reached
print(total_rewards.mean())
if not jnp.all(done):
    print("Max steps reached without completing all environments.")


new_key = key = jax.random.PRNGKey(0)[None, :]
state = env.reset(new_key)

done = False
total_reward = 0
frames = []
RENDER = True

# def random_action_fn_new(key):
#     return jax.random.randint(key, shape=(3,), minval=0, maxval=3)

# def random_policy(key):
#     return jax.random.randint(key, shape=(3,), minval=0, maxval=3)

import random
import numpy as np

for i in range(1000):
    # creating a random action of 3 actions
    # action = random.random((0, 3))
    action = np.array([random.randint(0, 2), random.randint(0, 2), random.randint(0, 2)])
    action = action[None, :]
    print(action.shape)

    state, reward, done = env.step(state, action)
    total_reward += reward
    if RENDER:
        f = SlimeVolley.render(state)
        frames.append(f)

print(total_reward)
if RENDER:
    imageio.mimsave('slime_volleyball_episode.gif', frames, fps=30)
env.close()



