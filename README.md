# Jaya Algorithm 
## Implementing and analyzing the semi-steady-state Jaya optimization algorithm with theoretical and empirical comparison of worst-index updates

This repository contains the implementation and analysis of the [Semi-Steady-State Jaya Optimization Algorithm](https://irl.umsl.edu/cmpsci-faculty/42/), with a theoretical and empirical comparison of worst-index updates.

## Overview

The Jaya algorithm is a simple and effective optimization algorithm that does not require specific parameters like crossover or mutation rates. This implementation focuses on a semi-steady-state variant of the Jaya algorithm, aiming to optimize a given fitness function and compare the empirical and theoretical expected updates of the worst index in the population.

## Features

- **Implementation of Semi-Steady-State Jaya Algorithm**: Optimizes a given fitness function.
- **Theoretical Analysis**: Computes the expected number of worst-index updates using a theoretical equation.
- **Empirical Comparison**: Tracks and compares the empirical worst-index updates with theoretical values.
- **Population Size Analysis**: Evaluates the algorithm's performance across different population sizes.
- **Data Presentation**: Uses pandas to display results in a clear, tabular format.

## Assumptions/Approximations

- **Population Size Independence**: It is assumed that the expected number of worst-index updates per generation is independent of the population size.
- **Stochastic Modeling**: The algorithm's behavior is modeled as a stochastic process, taking into account the random nature of the individuals' replacement and the update of their index within the population.
- **Function expected_updates**: Calculates the expected number of worst-index updates based on a provided equation on the paper. I have used "gammaln" for efficient factorial computation. In this code, "gammaln" from "scipy.special" is used to help the algorithm implement faster.
  - I first applied factorial on the equation, which took a long time for the entire program to implement (I waited around 55 minutes and it was still working). So I switched to a more efficient way and used "gammaln".
- This pattern suggests that in larger populations, the likelihood of repeatedly updating the same worst index diminishes, which is a reflection of the algorithm's efficiency and effectiveness in dealing with larger groups.
- According to the output, it suggests that the algorithm becomes more effective in avoiding repetitive updates of the worst index in larger groups, highlighting its scalability and efficiency in handling diverse population sizes. The results demonstrate a characteristic of the Semi-Steady-State Jaya algorithm in balancing exploration and exploitation as the population size varies.

## Code Explanation: 
### Functions

1. **expected_updates(n)**
    - Computes the expected number of updates for the worst index using log-factorials for computational efficiency.
    - Iterates through possible values of \( k \) and \( m \) to compute the series sum.

2. **fitness_function(solution)**
    - Placeholder for the actual fitness function, calculates the sum of the squares of the solution's elements.

3. **update_solution(solution, best_solution, worst_solution, p=1)**
    - Updates the given solution based on the best and worst solutions in the population.
    - Uses random factors \( r1 \) and \( r2 \) to perturb the solution towards the best solution and away from the worst solution.

4. **semi_steady_state_Jaya(population_size, dimensions, lower_bound, upper_bound, max_iterations, p=1)**
    - Implements the semi-steady-state Jaya algorithm.
    - Initializes a population of solutions within the specified bounds.
    - Iteratively updates the population, applying the update rule and fitness function.
    - Tracks the number of worst-index updates and compares them with the theoretical expected updates.

### Main Script

1. **population_sizes**
    - Defines a list of population sizes to test.

2. **results**
    - Stores the results of the algorithm for different population sizes.

3. **Loop through population sizes**
    - Runs the `semi_steady_state_Jaya` function for each population size.
    - Appends the population size and theoretical worst-index updates to the results list.

4. **max_updates**
    - Finds the maximum value of theoretical expected updates from the results.

5. **Create a DataFrame**
    - Uses `pandas` to create a DataFrame from the results list, making it easier to display in a table format.

6. **Print Results**
    - Prints the DataFrame, best fitness, best solution, theoretical worst-index updates, and the maximum value of theoretical updates.

## Usage

To run the algorithm and view the results, execute the script:
[Jaya_Algorithm.py](Jaya_Algorithm.py)
