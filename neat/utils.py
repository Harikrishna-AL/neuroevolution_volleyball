import jax.numpy as jnp


def node_order(nodes, connections):
    pass

def get_rewards(keys, env, policy, num_envs):
    state = env.reset(keys)
    total_rewards = jnp.zeros((num_envs,))
    done = jnp.zeros((num_envs,), dtype=bool)
    while not jnp.all(done):
        # print(state.obs.shape, policy.matrices.shape)
        actions = policy.forward(state.obs)
    

        state, reward, done = env.step(state, actions)

        total_rewards += reward

        if jnp.all(done):
            print("All environments completed")
            break
    return total_rewards