import jax.numpy as jnp
import jax.random as jrng
import jax
import numpy as np

from trade_envs import PricingEnvironment
from producer import producer_loss, consumer_loss

def clip_gradients(grads, max_norm=1.0):
    grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grads)))
    clipped_grads = jax.tree.map(lambda g: g * (max_norm / (grad_norm + 1e-6)), grads)
    return clipped_grads

sigma = 0.5
lr_prod = 1e-4
lr_cons = 1e-4
num_episodes = 500

# Set the initial parameters for the producer and consumer policy.
theta_prod = jnp.array([0.8, 0.0])
theta_cons = jnp.array([1.0, 0.5, 0.2])

# Static environment parameters dict
adj = jnp.ones((5,5)) - jnp.eye(5)
adj = adj.at[0, 1].set(0)
adj = adj.at[1, 0].set(0)
adj = adj / jnp.sum(adj, axis=1, keepdims=True)
env_params = {
    'num_consumers': 5,
    'demand_mean': 10.0,
    'demand_std': 2.0,
    'true_cost': 2.0,
    'adjacency': adj,
    'communication_mode': 'price',
    'lie_std': 0.5
}

def train_step(carry, ep):
    theta_p, theta_c, key = carry
    # Update producer
    (loss_p, (r_p, key)), grads_p = jax.value_and_grad(
        producer_loss, has_aux=True)(theta_p, env_params, theta_c, key, sigma)
    grads_p = clip_gradients(grads_p)
    theta_p = theta_p - lr_prod * grads_p
    # Update consumer
    (loss_c, (u_c, key)), grads_c = jax.value_and_grad(
        consumer_loss, has_aux=True)(theta_c, env_params, theta_p, key, sigma)
    grads_c = clip_gradients(grads_c)
    theta_c = theta_c - lr_cons * grads_c
    logs = (loss_p, r_p, loss_c, u_c)
    return (theta_p, theta_c, key), logs

# Initialize
init_key = jrng.PRNGKey(0)
carry_init = (theta_prod, theta_cons, init_key)

# Run training
carry_final, logs = jax.lax.scan(train_step, carry_init, jnp.arange(num_episodes))
final_theta_prod, final_theta_cons, _ = carry_final

# Convert logs to numpy for printing
logs_np = jax.tree.map(np.array, logs)
p_loss, p_reward, c_loss, c_util = logs_np

# Print progress
after = range(0, num_episodes, 50)
for ep in after:
    print(f"Episode {ep}: Prod Loss {p_loss[ep]:.2f}, Profit {p_reward[ep]:.2f}, "
          f"Cons Loss {c_loss[ep]:.2f}, Util {c_util[ep]:.2f}")

print("Final producer params:", final_theta_prod)
print("Final consumer params:", final_theta_cons)

