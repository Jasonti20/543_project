import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Function to read traffic data from a CSV file
def read_traffic_data(csv_file):
    df = pd.read_csv(csv_file)
    return df

# Custom waiting time model function
def calculate_waiting_time(signal_timings, coefficients):
    # Quadratic model: a * x^2 + b * x + c
    a, b, c = coefficients
    waiting_time = a * np.square(signal_timings) + b * signal_timings + c
    return waiting_time

# Objective function to minimize congestion
def objective_function(signal_timings, data, coefficients):
    # Filter data for 'Onstreet' == 'Clayton Rd'
    clayton_data = data[data['Onstreet'] == 'Clayton Rd']

    # Use 'PkHrVol' for traffic volume
    traffic_volume = clayton_data['PkHrVol'].values

    # Number of intersections
    num_intersections = len(signal_timings)

    # Calculate the total waiting time at intersections based on signal timings and model coefficients
    waiting_time = calculate_waiting_time(signal_timings, coefficients)

    # Check if the dimensions are compatible for multiplication
    if len(waiting_time) != len(traffic_volume):
        # Replicate or expand traffic_volume to match the length of waiting_time
        traffic_volume = np.tile(traffic_volume, len(waiting_time) // len(traffic_volume) + 1)[:len(waiting_time)]

    return np.sum(waiting_time[:num_intersections] * traffic_volume)



# Constraint function (if any)
def constraint_function(signal_timings):
    # Ensure that signal timings satisfy constraints, for example, min and max green times
    min_green_time = 10
    max_green_time = 60
    constraint_value = np.sum(np.maximum(signal_timings - max_green_time, 0)) + np.sum(np.maximum(min_green_time - signal_timings, 0))
    return constraint_value

# Optimization function
def optimize_traffic_flow(data, initial_guess, coefficients_guess):
    constraints = [{'type': 'eq', 'fun': constraint_function}]
    result = minimize(objective_function, initial_guess, args=(data, coefficients_guess), constraints=constraints)
    return result.x, result.fun

# Main function
def main():
    # Replace 'your_traffic_data.csv' with the actual CSV file name/path
    csv_file = 'Traffic_Counts.csv'

    # Read data from CSV file
    traffic_data = read_traffic_data(csv_file)

    # Filter data for 'OnStreet' == 'Clayton Rd'
    clayton_data = traffic_data[traffic_data['Onstreet'] == 'Clayton Rd']

    # Display the most common value
    print("Traffic data for 'Clayton Rd':\n", clayton_data)

    # Number of intersections
    num_intersections = 151

    # Store results for analysis and visualization
    all_optimized_timings = []
    all_objective_values = []

    # Iterate through different initial guesses (scenarios)
    for scenario in range(1, 6):
        # Generate random initial guess for signal timings
        initial_guess = np.random.uniform(10, 60, size=num_intersections)

        # Generate random initial guess for model coefficients (a, b, c)
        coefficients_guess = np.random.uniform(-1, 1, size=3)

        # Perform optimization
        optimized_timings, objective_value = optimize_traffic_flow(clayton_data, initial_guess, coefficients_guess)

        # Store results
        all_optimized_timings.append(optimized_timings)
        all_objective_values.append(objective_value)

        # Display the optimization results for each scenario
        print(f"\nScenario {scenario} - Optimized Signal Timings: {optimized_timings}")
        print(f"Objective Function Value: {objective_value}")

    # Visualize results
    visualize_results(all_optimized_timings, all_objective_values)

# Visualization function (unchanged)
def visualize_results(optimized_timings_list, objective_values_list):
    """
    Visualize the optimization results.

    Parameters:
    - optimized_timings_list (list): List of optimized signal timings for different scenarios.
    - objective_values_list (list): List of objective function values for different scenarios.
    """
    # Bar plot for visualizing objective function values
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(objective_values_list)), objective_values_list, color='blue')
    plt.xlabel('Scenario')
    plt.ylabel('Objective Function Value')
    plt.title('Objective Function Values for Different Scenarios')
    plt.show()

    # Line plot for visualizing optimized signal timings
    plt.figure(figsize=(10, 6))
    for i, timings in enumerate(optimized_timings_list):
        plt.plot(range(1, len(timings) + 1), timings, label=f'Scenario {i + 1}')

    plt.xlabel('Intersection')
    plt.ylabel('Optimized Signal Timings')
    plt.title('Optimized Signal Timings for Different Scenarios')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
