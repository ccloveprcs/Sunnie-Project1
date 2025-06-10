import json
import math
import random
import numpy as np
from typing import List, Tuple, Dict
import os

class AntColonyTSP:
    def __init__(self, distance_matrix: np.ndarray, city_names: List[str], 
                 start_city: int = 0, end_city: int = None,
                 n_ants: int = 50, n_iterations: int = 100,
                 alpha: float = 1.0, beta: float = 2.0,
                 evaporation_rate: float = 0.5, q: float = 100.0):
        """
        Ant Colony Optimization for TSP
        
        Parameters:
        - distance_matrix: NxN matrix of distances between cities
        - city_names: List of city names
        - start_city: Index of starting city
        - end_city: Index of ending city (None for standard TSP)
        - n_ants: Number of ants in the colony
        - n_iterations: Number of iterations to run
        - alpha: Pheromone importance factor
        - beta: Heuristic importance factor (1/distance)
        - evaporation_rate: Pheromone evaporation rate (0-1)
        - q: Pheromone deposit factor
        """
        self.distance_matrix = distance_matrix
        self.city_names = city_names
        self.n_cities = len(distance_matrix)
        self.start_city = start_city
        self.end_city = end_city
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q = q
        
        # Initialize pheromone matrix
        self.pheromone = np.ones((self.n_cities, self.n_cities))
        
        # Heuristic information (1/distance)
        self.heuristic = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                if i != j:
                    self.heuristic[i][j] = 1.0 / distance_matrix[i][j]
        
        # Best solution tracking
        self.best_route = None
        self.best_distance = float('inf')
        self.distance_history = []
    
    def calculate_route_distance(self, route: List[int]) -> float:
        """Calculate total distance of a route"""
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += self.distance_matrix[route[i]][route[i + 1]]
        return total_distance
    
    def get_next_city_probabilities(self, current_city: int, unvisited: List[int]) -> np.ndarray:
        """Calculate probability of choosing each unvisited city"""
        if not unvisited:
            return np.array([])
        
        probabilities = np.zeros(len(unvisited))
        
        for i, city in enumerate(unvisited):
            # Probability = (pheromone^alpha) * (heuristic^beta)
            probabilities[i] = (self.pheromone[current_city][city] ** self.alpha) * \
                             (self.heuristic[current_city][city] ** self.beta)
        
        # Normalize probabilities
        total = probabilities.sum()
        if total > 0:
            probabilities /= total
        else:
            # If all probabilities are 0, choose randomly
            probabilities = np.ones(len(unvisited)) / len(unvisited)
        
        return probabilities
    
    def select_next_city(self, current_city: int, unvisited: List[int]) -> int:
        """Select next city based on probabilities"""
        probabilities = self.get_next_city_probabilities(current_city, unvisited)
        
        # Roulette wheel selection
        rand = random.random()
        cumulative_prob = 0
        
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if rand <= cumulative_prob:
                return unvisited[i]
        
        # Fallback: return last city
        return unvisited[-1]
    
    def construct_ant_solution(self) -> List[int]:
        """Construct a solution for one ant"""
        # Start at the specified start city
        route = [self.start_city]
        current_city = self.start_city
        
        # Cities available for visiting (exclude start and end)
        if self.end_city is not None:
            unvisited = [i for i in range(self.n_cities) if i not in {self.start_city, self.end_city}]
        else:
            unvisited = [i for i in range(self.n_cities) if i != self.start_city]
        
        # Visit all available cities
        while unvisited:
            next_city = self.select_next_city(current_city, unvisited)
            route.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city
        
        # Add end city if specified
        if self.end_city is not None:
            route.append(self.end_city)
        
        return route
    
    def update_pheromones(self, all_routes: List[List[int]], all_distances: List[float]):
        """Update pheromone levels based on ant solutions"""
        # Evaporation
        self.pheromone *= (1 - self.evaporation_rate)
        
        # Deposit pheromones
        for route, distance in zip(all_routes, all_distances):
            pheromone_deposit = self.q / distance
            
            for i in range(len(route) - 1):
                city_a, city_b = route[i], route[i + 1]
                self.pheromone[city_a][city_b] += pheromone_deposit
                # For symmetric TSP, also update reverse direction
                self.pheromone[city_b][city_a] += pheromone_deposit
    
    def solve(self, verbose: bool = True) -> Tuple[List[int], float]:
        """Run the ant colony optimization algorithm"""
        if verbose:
            print(f"Starting Ant Colony Optimization...")
            print(f"Cities: {self.n_cities}, Ants: {self.n_ants}, Iterations: {self.n_iterations}")
            print(f"Start: {self.city_names[self.start_city]}")
            if self.end_city is not None:
                print(f"End: {self.city_names[self.end_city]}")
            print()
        
        for iteration in range(self.n_iterations):
            # Generate solutions for all ants
            all_routes = []
            all_distances = []
            
            for ant in range(self.n_ants):
                route = self.construct_ant_solution()
                distance = self.calculate_route_distance(route)
                
                all_routes.append(route)
                all_distances.append(distance)
                
                # Update best solution
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_route = route[:]
            
            # Update pheromones
            self.update_pheromones(all_routes, all_distances)
            
            # Track progress
            avg_distance = sum(all_distances) / len(all_distances)
            self.distance_history.append(avg_distance)
            
            if verbose and (iteration + 1) % 20 == 0:
                print(f"Iteration {iteration + 1:3d}: Best = {self.best_distance:.2f} miles, "
                      f"Avg = {avg_distance:.2f} miles")
        
        if verbose:
            print(f"\nOptimization completed!")
            print(f"Best distance: {self.best_distance:.2f} miles")
            print(f"Best route: {' -> '.join([self.city_names[i] for i in self.best_route])}")
        
        return self.best_route, self.best_distance

def haversine(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate great-circle distance between two points on Earth (in miles)"""
    lat1, lon1, lat2, lon2 = map(math.radians, (*p1, *p2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2) ** 2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2) ** 2
    return 3959.0 * 2 * math.asin(math.sqrt(a))

def find_state_by_capital(data: Dict, capital_name: str) -> str:
    """Find state name by capital city name"""
    for state, info in data.items():
        if info.get("capital", "").lower() == capital_name.lower():
            return state
    return None

def load_state_capitals(json_file: str) -> Tuple[Dict, List[str], np.ndarray]:
    """Load state capitals data and create distance matrix"""
    
    # Load JSON data
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"{json_file} not found in current directory")
    
    with open(json_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    # Extract the states data from the nested structure
    if "states" in json_data:
        data = json_data["states"]
        print(f"Found {len(data)} states in JSON file")
    else:
        # Fallback: assume data is directly at top level
        data = json_data
        print(f"Found {len(data)} entries in JSON file")
    
    print("Sample states:", list(data.keys())[:5])
    
    # Add Washington DC if not present
    if "Washington DC" not in data:
        data["Washington DC"] = {
            "capital": "Washington",
            "address": "1 First St SE, Washington, DC 20004",
            "latitude": 38.907192,
            "longitude": -77.036873
        }
    
    # Find Iowa by looking for Des Moines as capital, or use first available state
    iowa_state = find_state_by_capital(data, "Des Moines")
    if iowa_state is None:
        # If Iowa/Des Moines not found, use the first state alphabetically
        iowa_state = sorted(data.keys())[0]
        print(f"Warning: Iowa not found, using {iowa_state} as start state")
    
    START_STATE = iowa_state
    END_CITY = "Washington DC"
    
    # Create ordered list: Iowa/start first, DC last, others alphabetical
    other_states = sorted(s for s in data if s not in {START_STATE, END_CITY})
    ordered_states = [START_STATE] + other_states + [END_CITY]
    
    print(f"Start city: {START_STATE}")
    print(f"End city: {END_CITY}")
    print(f"Total cities: {len(ordered_states)}")
    
    # Extract coordinates with error handling
    coords = []
    valid_states = []
    
    for state in ordered_states:
        if state in data:
            state_data = data[state]
            if "latitude" in state_data and "longitude" in state_data:
                lat = float(state_data["latitude"])
                lon = float(state_data["longitude"])
                coords.append((lat, lon))
                valid_states.append(state)
            else:
                print(f"Warning: Missing coordinates for {state}")
        else:
            print(f"Warning: {state} not found in data")
    
    # Create distance matrix
    n = len(coords)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = haversine(coords[i], coords[j])
    
    return data, valid_states, distance_matrix

def main():
    """Main function to run ACO on state capitals TSP"""
    
    # Path to your JSON file in Downloads folder
    import os
    home_dir = os.path.expanduser("~")
    JSON_FILE = os.path.join(home_dir, "Downloads", "state_capitals_json.json")
    
    try:
        data, city_names, distance_matrix = load_state_capitals(JSON_FILE)
    except FileNotFoundError:
        print(f"Error: {JSON_FILE} not found!")
        print("Please make sure the JSON file is in the current directory.")
        print("Current directory files:", os.listdir("."))
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    if len(city_names) < 2:
        print("Error: Need at least 2 cities with valid coordinates")
        return
    
    print("=== Ant Colony Optimization for State Capitals TSP ===")
    print(f"Loaded {len(city_names)} cities with valid coordinates")
    
    # Set up ACO parameters
    start_idx = 0  # First city (Iowa or first available)
    end_idx = len(city_names) - 1  # Last city (Washington DC)
    
    # Create ACO solver
    aco = AntColonyTSP(
        distance_matrix=distance_matrix,
        city_names=city_names,
        start_city=start_idx,
        end_city=end_idx,
        n_ants=50,           # Number of ants
        n_iterations=100,    # Number of iterations
        alpha=1.0,           # Pheromone importance
        beta=2.0,            # Heuristic importance
        evaporation_rate=0.5, # Pheromone evaporation
        q=100.0              # Pheromone deposit factor
    )
    
    # Solve the problem
    best_route, best_distance = aco.solve(verbose=True)
    
    # Display detailed results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Total distance: {best_distance:.2f} miles")
    print(f"Number of cities: {len(best_route)}")
    print("\nOptimal Route:")
    for i, city_idx in enumerate(best_route, 1):
        city_name = city_names[city_idx]
        if city_name == "Washington DC":
            capital_name = "Washington"
        else:
            capital_name = data[city_name].get("capital", "Unknown") if city_name in data else "Unknown"
        print(f"{i:2d}. {city_name} ({capital_name})")
    
    # Compare with a simple nearest neighbor for reference
    print(f"\n" + "-"*40)
    print("COMPARISON WITH NEAREST NEIGHBOR:")
    
    def nearest_neighbor(dist_matrix, start, end):
        n = len(dist_matrix)
        visited = [False] * n
        route = [start]
        visited[start] = True
        current = start
        
        # Visit all cities except end
        for _ in range(n - 2):
            nearest = -1
            nearest_dist = float('inf')
            for city in range(n):
                if not visited[city] and city != end:
                    if dist_matrix[current][city] < nearest_dist:
                        nearest_dist = dist_matrix[current][city]
                        nearest = city
            
            if nearest != -1:
                route.append(nearest)
                visited[nearest] = True
                current = nearest
        
        route.append(end)
        total_dist = sum(dist_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
        return route, total_dist
    
    nn_route, nn_distance = nearest_neighbor(distance_matrix, start_idx, end_idx)
    improvement = ((nn_distance - best_distance) / nn_distance) * 100
    
    print(f"Nearest Neighbor distance: {nn_distance:.2f} miles")
    print(f"ACO distance: {best_distance:.2f} miles")
    print(f"Improvement: {improvement:.1f}%")

if __name__ == "__main__":
    main()
   
