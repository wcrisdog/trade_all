# consumer.py
import jax.numpy as jnp
import jax.nn as nn

import jax.numpy as jnp
import jax.nn as nn

def consumer_policy(theta_cons, offer_prices, willingness, fairness_signal, past_prices):
    """
    Consumer accepts based on surplus, fairness, and memory of past price.
    theta_cons: [weight_surplus, weight_fairness, weight_memory]
    fairness_signal: per-consumer neighbor-based average price
    past_prices: price offered in previous round
    """
    surplus = willingness - offer_prices
    penalty = jnp.maximum(0, offer_prices - fairness_signal)
    memory_penalty = jnp.maximum(0, offer_prices - past_prices)
    net_utility = (theta_cons[0] * surplus
                   - theta_cons[1] * penalty
                   - theta_cons[2] * memory_penalty)
    return nn.sigmoid(net_utility)
