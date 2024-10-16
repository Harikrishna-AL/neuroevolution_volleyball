import jax.numpy as jnp
import jax


def get_activation():
    # basic 15 differentiable activation functions
    activation_functions = (
        jnp.tanh,
        jnp.sin,
        jnp.cos,
        jnp.exp,
        jnp.log,
        jnp.abs,
        jnp.sqrt,
        lambda x: jnp.square(x),
        lambda x: jnp.maximum(x, 0),  # maximum requires two arguments, so we use it as ReLU-like function
        lambda x: jnp.minimum(x, 1),  # minimum for demonstration purposes
        jax.nn.relu,
        jax.nn.sigmoid,
        jax.nn.softplus,
        jax.nn.soft_sign,
        jax.nn.elu
    )
    return activation_functions

def node_order(nodes, connections):
    pass

def get_rewards(keys, env, policy, num_envs):
    state = env.reset(keys)
    total_rewards = jnp.zeros((num_envs,))
    done = jnp.zeros((num_envs,), dtype=bool)

    while not jnp.all(done):
        actions = policy.forward(state.obs)
        state, reward, done = env.step(state, actions)
        total_rewards += reward

        if jnp.all(done):
            print("All environments completed")
            break
    print(actions)
    return total_rewards