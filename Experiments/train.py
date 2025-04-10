import jax.numpy as jnp
import jax.random as jrng
import jax

from trade_envs import PricingEnvironment
from producer import reinforce_loss

def clip_gradients(grads, max_norm=1.0):
    grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grads)))
    clipped_grads = jax.tree.map(lambda g: g * (max_norm / (grad_norm + 1e-6)), grads)
    return clipped_grads

sigma = 0.5
learning_rate = 0.0001
theta = jnp.array([0.8, 0.0])  # initial parameters: scaling and intercept

num_episodes = 1500
for ep in range(num_episodes):
    # For each episode, create a new environment instance (or reset an existing one)
    env = PricingEnvironment(num_consumers=5, true_cost=5.0, demand_mean=10.0, demand_std=2.0,
                             communication_mode='price', lie_std=0.5, seed=42)
    key = jrng.PRNGKey(42 + ep)
    (loss_val, aux), grads = jax.value_and_grad(reinforce_loss, has_aux=True)(theta, env, key, sigma, num_rounds=10)
    grads = clip_gradients(grads, max_norm=1.0) 
    (total_reward, key) = aux
    theta = theta - learning_rate * grads
    if ep % 50 == 0:
        print(f"Episode {ep}: Loss {loss_val:.2f}, Total Profit {total_reward:.2f}, Theta {theta}")

print("Learned producer policy parameters:", theta)