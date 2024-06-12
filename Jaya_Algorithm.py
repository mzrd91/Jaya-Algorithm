import numpy as np
import pandas as pd
from scipy.special import gammaln
import math

# Function to calculate expected number of updates based on the provided equation on paper
def expected_updates(n):
    sum_series = 0
    for k in range(1, n+1):
        for m in range(1, k+1):
            # Using gammaln for more efficient computation of log-factorials
            log_factorial_k_minus_1 = gammaln(k)
            log_factorial_m_minus_1 = gammaln(m)
            log_factorial_k_minus_m = gammaln(k - m + 1)

            log_term = log_factorial_k_minus_1 - log_factorial_m_minus_1 - log_factorial_k_minus_m + m * math.log(1/n)
            term = math.exp(log_term - math.log(n))  # Adjusting the term to prevent overflow

            sum_series += term

    return (1 + sum_series) - 1


# Placeholder for the actual fitness function.
def fitness_function(solution):
    return np.sum(np.square(solution))  # Example fitness function

# Update rule for the Jaya algorithm
def update_solution(solution, best_solution, worst_solution, p=1):
    # Since p=1, the update is always performed.
    r1, r2 = np.random.rand(), np.random.rand()
    new_solution = solution + r1 * (best_solution - abs(solution)) - r2 * (worst_solution - abs(solution))
    return new_solution

# semi-steady-state Jaya algorithm implementation.
def semi_steady_state_Jaya(population_size, dimensions, lower_bound, upper_bound, max_iterations, p=1):
    population = np.random.uniform(low=lower_bound, high=upper_bound, size=(population_size, dimensions))
    fitness = np.apply_along_axis(fitness_function, 1, population)
    worst_index_updates = 0  # Track the number of worst-index updates

    for iteration in range(max_iterations):
        best_solution = population[np.argmin(fitness)]
        worst_solution = population[np.argmax(fitness)]

        for i in range(population_size):
            if p == 1 or np.random.rand() < p:  # Update always happens if p=1
                population[i] = update_solution(population[i], best_solution, worst_solution)
                population[i] = np.clip(population[i], lower_bound, upper_bound)

                new_fitness = fitness_function(population[i])
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                else:
                    population[i] = best_solution if new_fitness > fitness[i] else population[i]

                # If the worst index was updated, increment the counter
                if np.argmax(fitness) != worst_index_updates:
                    worst_index_updates += 1

    # Calculate the theoretical expected updates for comparison
    theoretical_updates = expected_updates(population_size)

    return best_solution, fitness_function(best_solution), worst_index_updates, theoretical_updates

population_sizes = [10, 50, 100, 500, 1500, 2500, 3500, 4500, 10000]
results = []

# Run the semi-steady-state Jaya algorithm for each population size and calculate the expected updates
for n in population_sizes:
    best_solution, best_fitness, empirical_worst_index_updates, theoretical_worst_index_updates = semi_steady_state_Jaya(
        population_size=n,
        dimensions=5,
        lower_bound=-10,
        upper_bound=10,
        max_iterations=100
    )
    results.append([n, theoretical_worst_index_updates])

max_updates = max(result[1] for result in results)

# Create a DataFrame to display the results in a table format
df = pd.DataFrame(results, columns=['Population Size', 'Theoretical Worst-Index Updates'])

# Display the DataFrame table and results
print("Table: Maximum value (theoretical) of E(X|n):\n")
print(df)
print("\nbest fitness", best_fitness)
print("\nBest Solution:", best_solution)
print("\nTheoretical Worst-Index Updates:", theoretical_worst_index_updates)
print(f"\nMaximum value: {max_updates}")
