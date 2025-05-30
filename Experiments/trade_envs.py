import jax.numpy as jnp
import jax.random as jrng

class PricingEnvironment:
    def __init__(
            self, 
            num_consumers=5, 
            true_cost=5.0, 
            demand_mean=10.0, 
            demand_std=2.0,
            communication_mode='price', # cost
            lie_std=0.0,
            seed = 1234,
            adjacency = None
        ):  # standard deviation of noise added when communicating (0.0 = truthful)
        """
        num_consumers: Number of consumers in the simulation.
        true_cost: The hidden production cost for the producer.
        demand_mean, demand_std: Distribution parameters for consumer demands.
        communication_mode: 'price' if consumers share the price they were offered,
                            'cost' if consumers share their current estimate of producer cost.
        lie_std: Standard deviation of noise added when sharing info (to simulate lying or misinformation).
        """
        self.seed = seed
        self.num_consumers = num_consumers
        self.true_cost = true_cost
        self.communication_mode = communication_mode
        self.lie_std = lie_std

        self.demand_mean = demand_mean
        self.demand_std = demand_std

        self.key = jrng.PRNGKey(seed)

        # Consumer Network structure, with adjacency matrix
        if adjacency is None:
            # fully connected (excluding self)
            adj = jnp.ones((num_consumers, num_consumers)) - jnp.eye(num_consumers)
        else:
            adj = jnp.array(adjacency, dtype=float)

        degrees = jnp.sum(adj, axis=1, keepdims=True)
        self.adjacency = adj / degrees  # each row sums to 1

        self.cost_estimates = jnp.ones(num_consumers) * (demand_mean / 2)
        
        # Store information
        self.history = {
            "prices": [],
            "sales": [],
            "producer_profit": [],
            "consumer_gains": [],
            "communications": [],
            "demands": [],
            "neighbor_avg": []
        }
        self.last_demands = None
    
    def step(self, producer_prices):
        """
        Run one simulation round:
          - producer_prices: jnp.array of shape (num_consumers,) containing the price offered to each consumer.
          
        Returns:
          A dictionary containing:
            'sales': binary vector indicating whether each consumer accepted the offer.
            'producer_profit': scalar, sum over consumers of (price - true_cost) for accepted sales.
            'consumer_gains': jnp.array of shape (num_consumers,) with each consumer's gain (demand - price if accepted, 0 otherwise).
            'communications': simulated messages broadcasted among consumers.
        """
        # Sample the current round's consumer willingness-to-pay
        self.key, subkey = jrng.split(self.key)
        current_demands = jrng.normal(subkey, shape=(self.num_consumers,)) * self.demand_std + self.demand_mean
        self.last_demands = current_demands
        self.history["demands"].append(jnp.array(current_demands))

        # Each consumer accepts if the offered price is <= their demand.
        sales = producer_prices <= current_demands
        
        # Producer's profit for each accepted sale is (price - true_cost)
        individual_profit = jnp.where(sales, producer_prices - self.true_cost, 0.0)
        producer_profit = jnp.sum(individual_profit)
        
        # Consumer gain: if they accept, their gain is (demand - price), else 0.
        consumer_gains = jnp.where(sales, current_demands - producer_prices, 0.0)
        
        # Communication: Each consumer sends a message to its neighbors.
        # For simplicity, assume full communication (everyone hears from everyone else).
        # Depending on the mode, the shared message is either the received price or the consumer's current cost estimate.
        if self.communication_mode == 'price':
            messages = producer_prices
        elif self.communication_mode == 'cost':
            messages = self.cost_estimates
        else:
            messages = jnp.zeros(self.num_consumers)
        
        # Add noise to messages to simulate lying/misinformation
        self.key, subkey2 = jrng.split(self.key)
        # noise = jrng.normal(subkey2, (self.num_consumers,)) * self.lie_std
        communicated = messages

        neighbor_avg = jnp.dot(self.adjacency, communicated)
        
        # Save round history
        self.history["prices"].append(jnp.array(producer_prices))
        self.history["sales"].append(jnp.array(sales))
        self.history["producer_profit"].append(producer_profit.astype(float))
        self.history["consumer_gains"].append(jnp.array(consumer_gains))
        self.history["communications"].append(jnp.array(communicated))
        
        return {
            "sales": sales,
            "producer_profit": producer_profit,
            "consumer_gains": consumer_gains,
            "communications": communicated,
            "demands": current_demands,
            'neighbor_avg': neighbor_avg
        }
    
    def reset(self):
        """Reset the environment for a new simulation run."""
        self.history = {
            "prices": [],
            "sales": [],
            "producer_profit": [],
            "consumer_gains": [],
            "communications": [],
            "demands": []
        }
        # Reset consumer cost estimates to their initial prior if communication mode is 'cost'
        if self.communication_mode == 'cost':
            self.cost_estimates = jnp.ones(self.num_consumers) * (jnp.mean(self.demand_mean) / 2)
        self.last_demands = None
        self.key = jrng.PRNGKey(self.seed)
        

# Example usage and testing:
def run_simulation(seed = 1234):
    # key = jrng.PRNGKey(seed)
    num_rounds = 10
    # Create an environment with 5 consumers, true production cost 5, and consumers' demands drawn from N(10,2)
    env = PricingEnvironment(num_consumers=5, true_cost=5.0, demand_mean=10.0, demand_std=2.0,
                               communication_mode='price', lie_std=0.5)
    
    # For test needs, let the producer use a fixed policy that offers a price that is 80% of each consumer's demand.
    for round_idx in range(num_rounds):
        if env.last_demands is not None:
            base_price = 0.8 * env.last_demands
        else:
            base_price = jnp.ones(env.num_consumers) * 0.8 * env.demand_mean
        
        # Compute producer prices as 0.8 * demand + some exploration noise.
        env.key, subkey = jrng.split(env.key)
        exploration_noise = jrng.normal(subkey, shape=(env.num_consumers,)) * 0.5
        producer_prices = base_price + exploration_noise
        
        # Run the simulation step
        result = env.step(producer_prices)
        
        print(f"Round {round_idx+1}")
        print(" Offered Prices:", jnp.array(producer_prices))
        print(" Realized Demands:", jnp.array(result["demands"]))
        print(" Sales (accepted):", jnp.array(result["sales"]))
        print(" Producer Profit:", float(result["producer_profit"]))
        print(" Consumer Gains:", jnp.array(result["consumer_gains"]))
        print(" Communications:", jnp.array(result["communications"]))
        print("-"*40)
        
if __name__ == "__main__":
    run_simulation(seed = 12)
