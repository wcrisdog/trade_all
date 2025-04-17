import jax
import jax.numpy as jnp
import jax.random as jrng
import numpy as np

# -------------------------------
# Producer Policy and Sampling
# -------------------------------
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

# -------------------------------
# Non-vectorized Episode Simulation (for reference)
# -------------------------------
def run_episode(theta, env, key, sigma, num_rounds=10):
    log_probs_list = []
    rewards_list = []
    for _ in range(num_rounds):
        prices, key, price_mean = sample_prices(theta, env.last_demands, env.demand_mean,
                                                env.num_consumers, sigma, key)
        # Compute log probability under the Gaussian assumption.
        log_probs = -0.5 * (((prices - price_mean) / sigma)**2 + 2*jnp.log(sigma) + jnp.log(2*jnp.pi))
        log_prob = jnp.sum(log_probs)
        result = env.step(prices)
        reward = result['producer_profit']
        log_probs_list.append(log_prob)
        rewards_list.append(reward)
    rewards_array = jnp.array(rewards_list)
    log_probs_array = jnp.array(log_probs_list)
    return rewards_array, log_probs_array, key

# -------------------------------
# Vectorized Episode Simulation using lax.scan
# -------------------------------
def run_episode_scan(env, sigma, theta, key, num_rounds=10):
    # Initialize last_demands with -1 as a flag for "no prior info"
    init_last_demands = -jnp.ones(env.num_consumers)
    carry = (theta, init_last_demands, key)

    def simulate_round(carry, _):
        theta, last_demands, key = carry
        # Sample current demands.
        key, subkey = jrng.split(key)
        current_demands = jrng.normal(subkey, shape=(env.num_consumers,)) * env.demand_std + env.demand_mean
        new_last_demands = current_demands  # update state for next round

        # If last_demands is a flag (all values < 0), use default base.
        base = jnp.where(last_demands < 0, jnp.ones(env.num_consumers) * env.demand_mean, last_demands)
        price_mean = producer_policy_linear(theta, base)
        key, subkey2 = jrng.split(key)
        exploration_noise = sigma * jrng.normal(subkey2, shape=(env.num_consumers,))
        prices = price_mean + exploration_noise

        # Compute log-probabilities for each consumer's price.
        log_probs = -0.5 * (((prices - price_mean) / sigma)**2 + 2 * jnp.log(sigma) + jnp.log(2 * jnp.pi))
        total_log_prob = jnp.sum(log_probs)
        
        # Determine sales: consumer purchases if the price does not exceed its demand.
        sales = prices <= current_demands
        profit = jnp.sum(jnp.where(sales, prices - env.true_cost, 0.0))
        reward = profit  # reward is producer's profit for the round.
        
        new_carry = (theta, new_last_demands, key)
        return new_carry, (reward, total_log_prob)

    # Use a dummy xs (here, an array of length num_rounds) to iterate.
    carry, (rewards, log_probs) = jax.lax.scan(simulate_round, carry, jnp.arange(num_rounds))
    # rewards and log_probs are arrays of shape (num_rounds,)
    return rewards, log_probs, carry[2]

# -------------------------------
# REINFORCE Loss Function
# -------------------------------
def reinforce_loss(theta, env, key, sigma, num_rounds=10):
    rewards_array, log_probs_array, key = run_episode_scan(env, sigma, theta, key, num_rounds)
    baseline = jnp.mean(rewards_array)
    # Use an advantage formulation.
    loss = -jnp.mean((rewards_array - baseline) * log_probs_array)
    total_reward = jnp.sum(rewards_array)
    aux = (total_reward, key)
    return loss, aux