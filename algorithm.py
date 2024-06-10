import numpy as np
import random
import sqlite3
import datetime
import csv

# Set up database
db_file = 'aco_results.db'

def create_table():
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS results (
                    iteration_id INTEGER,
                    ant_id INTEGER,
                    visited_cities TEXT,
                    path_length REAL,
                    alpha REAL,
                    beta REAL,
                    evaporation_rate REAL,
                    pheromone_deposit REAL,
                    timestamp TEXT
                )''')
    conn.commit()
    conn.close()

def save_to_db(iteration_id, ant_id, visited_cities, path_length, alpha, beta, evaporation_rate, pheromone_deposit):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    timestamp = datetime.datetime.now().isoformat()
    c.execute('''INSERT INTO results (iteration_id, ant_id, visited_cities, path_length, alpha, beta, evaporation_rate, pheromone_deposit, timestamp)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (iteration_id, ant_id, str(visited_cities), path_length, alpha, beta, evaporation_rate, pheromone_deposit, timestamp))
    conn.commit()
    conn.close()

# Load city map data from file
def load_city_map(file_path):
    city_map = {}
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            city_id = int(row['city_id'])
            latitude = float(row['latitude'])
            longitude = float(row['longitude'])
            city_map[city_id] = (latitude, longitude)
    return city_map

# Load city map
city_map = load_city_map('cities.csv')

# Define distance function using city map
def distance(city1, city2):
    lat1, lon1 = city_map[city1]
    lat2, lon2 = city_map[city2]
    # Implement the Haversine formula to calculate distance between two points on Earth
    # ...
    return np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)  # Placeholder distance calculation

# Parameters
num_cities = len(city_map)  # Number of cities in the map
num_ants = 50
num_iterations = 10
alpha = 1.0
beta = 5.0
evaporation_rate = 0.5
pheromone_deposit = 100.0

# Initialize pheromone matrix
pheromones = np.ones((num_cities, num_cities))

# Define function to select next city based on probabilities
def select_next_city(current_city, visited):
    probabilities = []
    for i in range(num_cities):
        if (i + 1) not in visited:
            pheromone = pheromones[current_city - 1, i] ** alpha
            heuristic = (1.0 / distance(current_city, i + 1)) ** beta
            probabilities.append(pheromone * heuristic)
        else:
            probabilities.append(0)
    probabilities = probabilities / np.sum(probabilities)
    return np.random.choice(range(1, num_cities + 1), p=probabilities)

# Create table in the database
create_table()

# Ant Colony Optimization algorithm
best_path = None
best_path_length = float('inf')

for iteration in range(num_iterations):
    all_paths = []
    all_paths_lengths = []

    for ant in range(num_ants):
        path = []
        visited = set()
        current_city = random.randint(1, num_cities)
        path.append(current_city)
        visited.add(current_city)
        
        # Log the visit of the initial city
        print(f'Iteration {iteration + 1}, Ant {ant + 1}: Visited City = {current_city}')
        
        for _ in range(num_cities - 1):
            next_city = select_next_city(current_city, visited)
            path.append(next_city)
            visited.add(next_city)
            current_city = next_city

            # Log the visit of subsequent cities
            print(f'Iteration {iteration + 1}, Ant {ant + 1}: Visited City = {current_city}')

        path_length = sum([distance(path[i], path[i + 1]) for i in range(num_cities - 1)]) + distance(path[-1], path[0])
        all_paths.append(path)
        all_paths_lengths.append(path_length)

        if path_length < best_path_length:
            best_path_length = path_length
            best_path = path

        # Save data to the database
        save_to_db(iteration, ant, path, path_length, alpha, beta, evaporation_rate, pheromone_deposit)
    
    # Update pheromones
    pheromones = pheromones * (1 - evaporation_rate)
    for i, path in enumerate(all_paths):
        for j in range(num_cities - 1):
            pheromones[path[j] - 1, path[j + 1] - 1] += pheromone_deposit / all_paths_lengths[i]
        pheromones[path[-1] - 1, path[0] - 1] += pheromone_deposit / all_paths_lengths[i]

print(f'Best Path: {best_path}')
print(f'Best Path Length: {best_path_length:.2f}')
