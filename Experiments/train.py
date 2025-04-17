import jax.numpy as jnp
import jax.random as jrng
import jax
import numpy as np

from trade_envs import PricingEnvironment
from producer import reinforce_loss
from consumer import  consumer_policy

def clip_gradients(grads, max_norm=1.0):
    grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grads)))
    clipped_grads = jax.tree.map(lambda g: g * (max_norm / (grad_norm + 1e-6)), grads)
    return clipped_grads

sigma = 0.5
learning_rate = 0.0001
theta = jnp.array([0.8, 0.0])
num_episodes = 500

# Set the initial parameters for the consumer policy.
theta_consumer = jnp.array([1.0, 0.5])

def train_step(carry, ep):
    """
    carry: Tuple containing (theta, key) — current policy parameters and a PRNG key.
    ep: Episode index (used to vary the environment seed).
    """
    theta, key = carry
    # Create a new environment instance. Here we adjust the seed per episode.
    env = PricingEnvironment(num_consumers=5, true_cost=5.0, demand_mean=10.0, demand_std=2.0,
                             communication_mode='price', lie_std=0.5, seed= 1234 + ep.astype(int))
    # Compute the loss and its gradients using REINFORCE loss.
    (loss_val, aux), grads = jax.value_and_grad(reinforce_loss, has_aux=True)(
        theta, env, theta_consumer, key, sigma, num_rounds=10)
    # Optionally clip gradients.
    grads = clip_gradients(grads, max_norm=1.0)
    # Update parameters.
    theta_new = theta - learning_rate * grads
    total_reward, key_new = aux
    # For logging, record loss and total reward.
    logs = (loss_val, total_reward)
    return (theta_new, key_new), logs

# Initialize the PRNG key and our starting state.
init_key = jrng.PRNGKey(12349)
carry_init = (theta, init_key)

# Use jax.lax.scan to iterate the training step over num_episodes.
# Here, we simply scan over an array of episode indices.
carry_final, logs = jax.lax.scan(train_step, carry_init, jnp.arange(num_episodes))
final_theta, final_key = carry_final

# Convert logs to NumPy arrays for printing.
logs_np = jax.tree.map(np.array, logs)
losses_np, rewards_np = logs_np

# Print progress every 50 episodes.
for ep in range(0, num_episodes, 50):
    print(f"Episode {ep}: Loss {losses_np[ep]:.2f}, Total Profit {rewards_np[ep]:.2f}")

print("Learned producer policy parameters:", final_theta)

# for ep in range(num_episodes):
#     # For each episode, create a new environment instance
#     env = PricingEnvironment(num_consumers=5, true_cost=5.0, demand_mean=10.0, demand_std=2.0,
#                              communication_mode='price', lie_std=0.5, seed=42)
#     key = jrng.PRNGKey(42 + ep)
#     (loss_val, aux), grads = jax.value_and_grad(reinforce_loss, has_aux=True)(theta, env, key, sigma, num_rounds=10)
#     grads = clip_gradients(grads, max_norm=1.0) 
#     (total_reward, key) = aux
#     theta = theta - learning_rate * grads
#     if ep % 50 == 0:
#         loss_val_np = float(loss_val)
#         total_reward_np = float(total_reward)
#         theta_np = np.array(theta)
#         print(f"Episode {ep}: Loss {loss_val_np:.2f}, Total Profit {total_reward_np:.2f}, Theta {theta_np}")

# print("Learned producer policy parameters:", theta)