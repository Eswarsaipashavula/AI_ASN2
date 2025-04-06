import gym
import gym_TSP
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import imageio.v2 as imageio
import time
from termcolor import colored
import math

env = gym.make("tsp-v0", num_cities=100)

def total_distance(tour, cities):
    return sum(np.linalg.norm(cities[tour[i]] - cities[tour[i - 1]]) for i in range(len(tour)))

def get_cities_and_visited(obs):
    if isinstance(obs, tuple) and len(obs) == 2:
        return obs
    elif isinstance(obs, dict):
        return obs.get('current_city', 0), obs.get('visited', [])
    else:
        raise ValueError(f"Unexpected observation format: {type(obs)}")

def generate_image(cities, tour_indices, filename):
    fig, ax = plt.subplots(figsize=(8, 8))
    cities = np.array(cities)
    ax.scatter(cities[:, 0], cities[:, 1], c='red', s=100, label='Cities')
    tour_indices.append(tour_indices[0])
    for i in range(len(tour_indices) - 1):
        ax.plot([cities[tour_indices[i], 0], cities[tour_indices[i+1], 0]],
                [cities[tour_indices[i], 1], cities[tour_indices[i+1], 1]], 'b-')
    for i, (x, y) in enumerate(cities):
        ax.text(x, y, str(i), color='black', ha='center', va='center')
    ax.set_title(f'TSP Tour (Cities: {len(cities)})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, format='png', dpi=100)
    plt.close(fig)

def calculate_tour_distance(cities, tour):
    return sum(np.linalg.norm(cities[tour[i]] - cities[tour[i+1]])
               for i in range(len(tour) - 1)) + np.linalg.norm(cities[tour[-1]] - cities[tour[0]])

def log_results(method_name, run_id, best_tour, best_distance, min_iteration, time_taken):
    log_filename = f"log_{method_name.lower().replace(' ', '_')}_{run_id}.txt"
    with open(log_filename, "w") as f:
        f.write(f"Best Tour: {best_tour}\n")
        f.write(f"Best Distance: {best_distance:.2f}\n")
        f.write(f"Minimum found at iteration: {min_iteration}\n")
        f.write(f"Time Taken: {time_taken:.2f} seconds\n")

def save_final_tour_image(cities, best_tour, method_name, run_id):
    filename = f"{method_name.lower().replace(' ', '_')}_tour_run{run_id}.png"
    generate_image(cities, list(best_tour) + [best_tour[0]], filename)

def hill_climbing(env, max_time=300, run_id=0, patience=50):
    print(colored(f"Starting Hill Climbing Run {run_id+1}...", "green"))
    start_time = time.time()
    obs, _ = env.reset()
    _, visited = get_cities_and_visited(obs)
    num_cities = len(visited)
    cities = getattr(env, 'cities', np.random.rand(num_cities, 2))

    best_tour = list(range(num_cities))
    best_distance = calculate_tour_distance(cities, best_tour)
    rewards = [best_distance]
    min_iteration = 0
    images = []
    iteration = 0
    no_improve_counter = 0

    while time.time() - start_time < max_time:
        i, j = random.sample(range(num_cities), 2)
        new_tour = best_tour[:]
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        new_distance = calculate_tour_distance(cities, new_tour)

        if new_distance < best_distance:
            best_tour, best_distance = new_tour, new_distance
            min_iteration = iteration
            no_improve_counter = 0
        else:
            no_improve_counter += 1

        rewards.append(best_distance)

        if iteration % 5 == 0:
            img_filename = f"temp_hc_{run_id}_{iteration}.png"
            generate_image(cities, best_tour[:], img_filename)
            if os.path.exists(img_filename):
                images.append(imageio.imread(img_filename))
                os.remove(img_filename)

        if no_improve_counter >= patience:
            break
        iteration += 1

    time_taken = time.time() - start_time
    if images:
        gif_filename = f"hill_climbing_{run_id}.gif"
        imageio.mimsave(gif_filename, images, duration=0.1)

    save_final_tour_image(cities, best_tour, "Hill Climbing", run_id)
    log_results("Hill Climbing", run_id, best_tour, best_distance, min_iteration, time_taken)

    print(colored(f"HC Run {run_id+1}: Min distance {best_distance:.2f} | Time: {time_taken:.2f}s", "green"))
    return rewards, min_iteration, best_distance, time_taken

def simulated_annealing(env, max_time=600, initial_temp=1000, cooling_rate=0.995, run_id=0, patience=50):
    print(colored(f"Starting Simulated Annealing Run {run_id+1}...", "blue"))
    start_time = time.time()
    obs, _ = env.reset()
    _, visited = get_cities_and_visited(obs)
    num_cities = len(visited)
    cities = getattr(env, 'cities', np.random.rand(num_cities, 2))

    current_tour = list(range(num_cities))
    random.shuffle(current_tour)
    current_distance = calculate_tour_distance(cities, current_tour)
    best_tour = current_tour[:]
    best_distance = current_distance
    temperature = float(initial_temp)
    rewards = [best_distance]
    min_iteration = 0
    images = []
    iteration = 0
    no_improve_counter = 0

    while time.time() - start_time < max_time and temperature > 1:
        i, j = random.sample(range(num_cities), 2)
        new_tour = current_tour[:]
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        new_distance = calculate_tour_distance(cities, new_tour)

        accept = new_distance < current_distance or random.random() < math.exp((current_distance - new_distance) / temperature)

        if accept:
            current_tour, current_distance = new_tour, new_distance
            
            no_improve_counter = 0
            if new_distance < best_distance:
                best_distance = new_distance
                best_tour = new_tour[:]
                min_iteration = iteration
        else:
            no_improve_counter += 1
        rewards.append(current_distance)
        

        if iteration % 5 == 0:
            img_filename = f"temp_sa_{run_id}_{iteration}.png"
            generate_image(cities, best_tour[:], img_filename)
            if os.path.exists(img_filename):
                images.append(imageio.imread(img_filename))
                os.remove(img_filename)

        if no_improve_counter >= patience:
            break

        iteration += 1
        temperature *= cooling_rate

    time_taken = time.time() - start_time
    if images:
        gif_filename = f"simulated_annealing_{run_id}.gif"
        imageio.mimsave(gif_filename, images, duration=0.1)

    save_final_tour_image(cities, best_tour, "Simulated Annealing", run_id)
    log_results("Simulated Annealing", run_id, best_tour, best_distance, min_iteration, time_taken)

    print(colored(f"SA Run {run_id+1}: Min distance {best_distance:.2f} | Time: {time_taken:.2f}s", "blue"))
    return rewards, min_iteration, best_distance, time_taken

def plot_results(all_rewards, min_iterations, method_name, best_distances, times):
    plt.figure(figsize=(12, 5))
    for run_id, rewards in enumerate(all_rewards):
        plt.plot(rewards, label=f'Run {run_id+1}')
    plt.xlabel("Iterations")
    plt.ylabel("Distance")
    plt.title(f"Distance vs Iterations for {method_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"distance_vs_iterations_{method_name.lower().replace(' ', '_')}.png", dpi=300)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(min_iterations) + 1), min_iterations, color='blue')
    plt.xlabel("Run")
    plt.ylabel("Iteration where Minimum Found")
    plt.title(f"Iteration of Minimum Distance per Run ({method_name})")
    plt.xticks(range(1, len(min_iterations) + 1))
    plt.grid(axis="y")
    plt.savefig(f"min_iteration_per_run_{method_name.lower().replace(' ', '_')}.png", dpi=300)
    plt.show()

    # Plot Best Distance per Run
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(best_distances) + 1), best_distances, marker='o', linestyle='-', color='purple')
    plt.xlabel("Run")
    plt.ylabel("Best Distance")
    plt.title(f"Best Distance per Run ({method_name})")
    plt.grid(True)
    plt.savefig(f"best_distance_per_run_{method_name.lower().replace(' ', '_')}.png", dpi=300)
    plt.show()

    # Plot Time Taken per Run
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(times) + 1), times, marker='o', linestyle='-', color='darkorange')
    plt.xlabel("Run")
    plt.ylabel("Time Taken (s)")
    plt.title(f"Time Taken per Run ({method_name})")
    plt.grid(True)
    plt.savefig(f"time_per_run_{method_name.lower().replace(' ', '_')}.png", dpi=300)
    plt.show()

def run_hill_climbing_experiment(env, runs=5, max_time=600):
    all_rewards = []
    min_iterations = []
    best_distances = []
    times = []
    for run_id in range(runs):
        rewards, min_iteration, best_distance, time_taken = hill_climbing(env, max_time=max_time, run_id=run_id)
        all_rewards.append(rewards)
        min_iterations.append(min_iteration)
        best_distances.append(best_distance)
        times.append(time_taken)
    plot_results(all_rewards, min_iterations, "Hill Climbing", best_distances, times)

def run_simulated_annealing_experiment(env, runs=5, max_time=600):
    all_rewards = []
    min_iterations = []
    best_distances = []
    times = []
    for run_id in range(runs):
        rewards, min_iteration, best_distance, time_taken = simulated_annealing(env, max_time=max_time, run_id=run_id)
        all_rewards.append(rewards)
        min_iterations.append(min_iteration)
        best_distances.append(best_distance)
        times.append(time_taken)
    plot_results(all_rewards, min_iterations, "Simulated Annealing", best_distances, times)

# Run the experiments
run_hill_climbing_experiment(env)
run_simulated_annealing_experiment(env)
