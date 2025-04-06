# AI Assignment 2: Search and Optimization

This project implements four classic search and optimization algorithms to solve problems in various environments.

## 👥 Team Members

- **Pashaula Eswar Sai** - CS24M109
- **Gooty Bharadwaj** - CS24M123

## 📌 Algorithms Implemented

- **Branch and Bound (BnB)**
- **Iterative Deepening A\***
- **Hill Climbing (HC)**
- **Simulated Annealing (SA)**

## 🧪 Environments Used

### For **Branch and Bound** & **Iterative Deepening A\***
- Frozen Lake Environment: https://gymnasium.farama.org/environments/toy_text/frozen_lake/

### For **Hill Climbing** & **Simulated Annealing**
- Traveling Salesman Problem (TSP): https://github.com/g-dendiev/gym_TSP

## 💡 Problem Descriptions

- **Frozen Lake / Ant Maze**: Navigation tasks for BnB and IDA*, aiming to reach the goal with optimal path-finding under stochasticity.
- **TSP**: An optimization problem where the agent must visit all cities exactly once with minimum travel distance.

## 🚀 How to Run
Frozen Lake
Running these files is really simple—just upload the `.ipynb` files to Google Colab and execute them directly.

**TSP**
- clone into the HC and SA folder
- Run
- pip install -e .
- Install any python libraries that are missing (matplotlib numpy gymnasium)
- Execute the experiments.py in the HC and SA folder 
## 📈 Evaluation Criteria

- **Performance Metrics**: Reward, time, and point of convergence.
- **Repetitions**: Each algorithm is tested at least 5 times.
- **Timeout Handling**: Execution is terminated if the algorithm does not converge within 10 minutes.
- **Visualizations**: Execution gifs are included for presentation purposes.
- **Heuristics**: Each search algorithm includes a clearly defined heuristic function.

## 🔗 Useful Links

- Branch and Bound: https://en.wikipedia.org/wiki/Branch_and_bound
- Iterative Deepening A\*: https://en.wikipedia.org/wiki/Iterative_deepening_A*
- Hill Climbing: https://en.wikipedia.org/wiki/Hill_climbing
- Simulated Annealing: https://en.wikipedia.org/wiki/Simulated_annealing
- Frozen Lake Environment: https://gymnasium.farama.org/environments/toy_text/frozen_lake/
- TSP Environment: https://github.com/g-dendiev/gym_TSP
  
**Simulated Annealing**
![simulated_annealing_0](https://github.com/user-attachments/assets/745c2b14-f2eb-43f2-9f9a-80656a5991e6)

**Hill Climbing**
![hill_climbing_0](https://github.com/user-attachments/assets/0e310776-ec27-4f2d-a0ea-86e03bf66e97)
