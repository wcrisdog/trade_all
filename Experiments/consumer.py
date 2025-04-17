# consumer.py
import jax.numpy as jnp
import jax.nn as jnn

def consumer_policy(theta_cons, offer_prices, willingness, fairness_signal):
    """
    Compute the probability of acceptance for each consumer.
    
    Args:
      theta_cons: A parameter array of shape (2,). 
                  theta_cons[0] weights the consumer surplus (willingness - offer_price).
                  theta_cons[1] weights the fairness penalty (extra charge over peers).
      offer_prices: Array of offered prices (one per consumer).
      willingness: Array of consumers' willingness-to-pay.
      fairness_signal: A scalar (or array) representing a fairness measure (e.g. the average offered price).
    
    Returns:
      prob: Array of probabilities for each consumer, in [0, 1].
    """
    # Compute consumer surplus (willingness - price)
    surplus = willingness - offer_prices
    # When the offered price is above the fairness signal, consumers may feel a penalty.
    penalty = jnp.maximum(0, offer_prices - fairness_signal)
    # Compute a “net utility” combining surplus and a fairness penalty.
    net_utility = theta_cons[0] * surplus - theta_cons[1] * penalty
    # Pass through a sigmoid to get a probability in [0,1]
    prob = jnn.sigmoid(net_utility)
    return prob
