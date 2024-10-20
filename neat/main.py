from genetic_algo_jax import GeneticEvolution, Policy
from genome import Genome, GenomeData

import json
import numpy as np
import jax.numpy as jnp
import jax
import argparse
import os


data_name_map = {
    0: "circle",
    1: "xor",
    2: "2 gaussians",
    3: "spiral",
}

config = {
    "prob_enable": 0.5,
    "prob_weight_mut": 0.8,
    "clamp_weights": 5.0,
    "prob_add_node": 0.5,
    "prob_add_connection": 0.5,
    "input_num": 2,
    "output_num": 1,
}


def get_data(n):
    # read data from json
    train_data = json.load(open(f"../data/train_data_{data_name_map[n]}.json"))
    test_data = json.load(open(f"../data/test_data_{data_name_map[n]}.json"))

    train_input = train_data["train_input"]
    train_input = np.array([np.array([x["0"], x["1"]]) for x in train_input])
    test_input = test_data["test_input"]
    test_input = np.array([np.array([x["0"], x["1"]]) for x in test_input])

    train_target = train_data["train_target"]
    train_target = np.array([train_target[i] for i in train_target.keys()])
    test_target = test_data["test_target"]
    test_target = np.array([test_target[i] for i in test_target.keys()])
    #extent dimesion from (n,) to (n,1)
    train_target = train_target[:, np.newaxis]
    test_target = test_target[:, np.newaxis]

    print("Train Input Dimension:", train_input.shape)
    print("Test Target Dimension:", train_target.shape)

    print("Test Input Dimension:", test_input.shape)
    print("Test Target Dimension:", test_target.shape)

    print(f"Data {data_name_map[n]} loaded")

    return train_input, train_target, test_input, test_target

def validate(gen, pop, test_input, test_target):
    policy = Policy(gen, [pop], config)
    correct = 0
    for input, target in zip(test_input, test_target):
        input = input[np.newaxis, :]
        target = target[np.newaxis, :]

        output = jax.nn.sigmoid(policy.forward(input))
        correct += jnp.round(output) == target
    print("Correct: ", correct)
    accuracy = (correct / test_input.shape[0]) * 100
    # output = jax.nn.sigmoid(policy.forward(test_input))
    # accuracy = jnp.mean(jnp.round(output) == test_target)
    # accuracy = accuracy * 100
    # mean_error = jnp.mean((output - test_target) ** 2)
    print("Accuracy: ", accuracy)


def train():
    # get num_envs and max_generations from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=50)
    parser.add_argument("--max_generations", type=int, default=5)
    parser.add_argument("--data", type=int, default=0)
    args = parser.parse_args()

    num_envs = args.num_envs
    max_generations = args.max_generations

    train_input, train_target, test_input, test_target = get_data(args.data)
    # sample 100 random points from the data
    
    gen = Genome(config=config)
    evolver = GeneticEvolution(num_envs, gen, train_input.shape[1])

    for i in range(max_generations):
        print(f"Generation {i+1}")

        idx = np.random.choice(train_input.shape[0], num_envs)
        input = train_input[idx]
        target = train_target[idx]

        pops = evolver.ask()
        num_connections = jnp.array([p.connections.shape[0] for p in pops])
        connection_penalty = 0.1
        policy = Policy(gen, pops, config)
        output = jax.nn.sigmoid(policy.forward(input))
        # print(output.shape, train_target.shape)
        rewards =  - (output - target) ** 2 
        # convert rewards to 1d array
        rewards = jnp.mean(rewards, axis=1) 
        rewards = rewards * jnp.sqrt(1 + connection_penalty * num_connections)
        # print(rewards)
        evolver.tell(rewards)

        validate(gen, pops[0], test_input, test_target)
    
        # accuracy = jnp.mean(jnp.round(output) == train_target)
        # accuracy = accuracy * 100
        # mean_error = jnp.mean((output - train_target) ** 2)

        if i % 5==0:
            #visualize the best policy
            best_fitness = gen.visualize(pops[0], i)
            print("Best Fitness: ", best_fitness)

train()
