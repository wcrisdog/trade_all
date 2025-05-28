import jax
import jax.numpy as jnp
import jax.random as jrng
import numpy as np

from consumer import consumer_policy

def env_step(key, last_prices, pricing_params, consumer_params, sigma, env_params):
    num_consumers = env_params['num_consumers']
    demand_mean = env_params['demand_mean']
    demand_std = env_params['demand_std']
    true_cost = env_params['true_cost']
    adjacency = env_params['adjacency']
    communication_mode = env_params['communication_mode']
    lie_std = env_params['lie_std']

    # Sample demands
    key, subkey = jrng.split(key)
    demands = jrng.normal(subkey, (num_consumers,)) * demand_std + demand_mean

    # Compute prices
    theta = pricing_params
    base = last_prices
    price_mean = theta[0] * base + theta[1]
    key, subkey2 = jrng.split(key)
    prices = price_mean + sigma * jrng.normal(subkey2, (num_consumers,))

    # Communication (price-sharing)
    if communication_mode == 'price':
        messages = prices
    else:
        messages = jnp.zeros(num_consumers)
    # add noise (optional)
    key, subkey3 = jrng.split(key)
    communicated = messages  # + lie_std * jrng.normal(subkey3, (num_consumers,))

    # Neighbor-based fairness signal
    neighbor_avg = jnp.dot(adjacency, communicated)

    surplus = demands - prices
    penalty = jnp.maximum(0, prices - neighbor_avg)
    memory_penalty = jnp.maximum(0, prices - last_prices)
    net_util = (
            consumer_params[0] * surplus
            - consumer_params[1] * penalty
            - consumer_params[2] * memory_penalty
    )

    # Consumer decision via policy
    surplus = demands - prices
    penalty = jnp.maximum(0, prices - neighbor_avg)
    memory_penalty = jnp.maximum(0, prices - last_prices)
    net_util = (consumer_params[0] * surplus
                - consumer_params[1] * penalty
                - consumer_params[2] * memory_penalty)
    key, subkey4 = jrng.split(key)
    sales = jrng.bernoulli(subkey4, jax.nn.sigmoid(net_util))

    # Producer reward
    reward = jnp.sum(jnp.where(sales, prices - true_cost, 0.0))

    return key, prices, demands, neighbor_avg, reward, net_util


def producer_policy_linear(theta, base):
    # Compute the mean price as a linear function of the base.
    return theta[0] * base + theta[1]

def sample_prices(theta, last_demands, demand_mean, num_consumers, sigma, key):
    # If no past information, use default base.
    if last_demands is None:
        base = jnp.ones(num_consumers) * demand_mean
    else:
        base = last_demands
    price_mean = producer_policy_linear(theta, base)
    key, subkey = jrng.split(key)
    # Sample prices from a Gaussian around the mean.
    prices = price_mean + sigma * jrng.normal(subkey, shape=(num_consumers,))
    return prices, key, price_mean

def run_episode_scan(pricing_params, consumer_params, key, sigma, env_params, num_rounds=10):
    init_last_prices = jnp.ones(env_params['num_consumers']) * env_params['demand_mean']

    def body(carry, _):
        key, last_prices = carry
        key, prices, demands, neighbor_avg, reward, net_util = env_step(
            key, last_prices, pricing_params, consumer_params, sigma, env_params
        )

        # compute log‐prob as before …
        price_mean   = pricing_params[0] * last_prices + pricing_params[1]
        log_probs    = -0.5 * (((prices - price_mean)/sigma)**2 + 2*jnp.log(sigma) + jnp.log(2*jnp.pi))
        total_lprob  = jnp.sum(log_probs)

        return (key, prices), (reward, total_lprob, net_util)

    (key_final, _), (rewards, logps, net_utils) = jax.lax.scan(
        body,
        (key, init_last_prices),
        None,
        length=num_rounds
    )

    # net_utils has shape (num_rounds, num_consumers)
    return rewards, logps, net_utils, key_final



def producer_loss(theta_prod, env_params, theta_cons, key, sigma, num_rounds=10):
    rewards, logps, net_utils, key = run_episode_scan(
        theta_prod, theta_cons, key, sigma, env_params, num_rounds
    )
    baseline = jnp.mean(rewards)
    loss = -jnp.mean((rewards - baseline) * logps)
    return loss, (jnp.sum(rewards), key)

def consumer_loss(theta_cons, env_params, theta_prod, key, sigma, num_rounds=10):
    rewards, logps, net_utils, key = run_episode_scan(
        theta_prod, theta_cons, key, sigma, env_params, num_rounds
    )
    # flatten per‐round, per‐consumer utilities
    avg_utility = jnp.mean(net_utils)          # mean over all consumers & rounds
    loss        = - avg_utility                # minimize negative utility
    return loss, (avg_utility, key)
