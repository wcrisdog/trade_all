import jax.numpy as jnp
import jax.random as jrng

def producer_policy_linear(theta, last_demands, demand_mean, num_consumers):
    if last_demands is None:
        base = jnp.ones(num_consumers) * demand_mean
    else:
        base = last_demands
    # Linear policy: mean price for each consumer.
    price_mean = theta[0] * base + theta[1]
    return price_mean

def sample_prices(theta, last_demands, demand_mean, num_consumers, sigma, key):
    price_mean = producer_policy_linear(theta, last_demands, demand_mean, num_consumers)
    key, subkey = jrng.split(key)
    # Sample prices from Normal(mean, sigma^2)
    prices = price_mean + sigma * jrng.normal(subkey, shape=(num_consumers,))
    return prices, key, price_mean

def run_episode(theta, env, key, sigma, num_rounds=10):
    log_probs_list = []
    rewards_list = []
    for _ in range(num_rounds):
        prices, key, price_mean = sample_prices(theta, env.last_demands, env.demand_mean, env.num_consumers, sigma, key)
        # Compute log probability (per consumer, then sum) for a Gaussian
        log_probs = -0.5 * (((prices - price_mean) / sigma)**2 + 2*jnp.log(sigma) + jnp.log(2*jnp.pi))
        log_prob = jnp.sum(log_probs)
        result = env.step(prices)
        reward = result['producer_profit']
        log_probs_list.append(log_prob)
        rewards_list.append(reward)
    log_probs_array = jnp.array(log_probs_list)
    rewards_array = jnp.array(rewards_list)
    return rewards_array, log_probs_array, key

def reinforce_loss(theta, env, key, sigma, num_rounds=10):
    rewards_array, log_probs_array, key = run_episode(theta, env, key, sigma, num_rounds)
    baseline = jnp.mean(rewards_array)
    loss = -jnp.mean((rewards_array - baseline) * log_probs_array)
    # loss = -jnp.mean(log_probs_array * rewards_array)
    total_reward = jnp.sum(rewards_array)
    aux = (total_reward, key)
    return loss, aux