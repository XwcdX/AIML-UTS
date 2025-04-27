import streamlit as st
import math
import random
import pydeck as pdk
import pandas as pd
import json
import os
import numpy as np
import itertools
from itertools import combinations, permutations # Ditambahkan permutations
import plotly.graph_objects as go
from functools import lru_cache

random.seed(30)
np.random.seed(30)

# --- Constants ---
OVERLAP_PENALTY_FACTOR = 2.0
OUTBOUND_PENALTY_FACTOR = 2.0
W_MAX = 0.9
W_MIN = 0.4
# -----------------

def quick_feasible(truck_dims, items):
    L, W, H = truck_dims
    total_vol = sum(l*w*h for l, w, h in (it['dims'] for it in items))
    if total_vol > L*W*H:
        return False
    # Quick check base area might be less reliable with rotations,
    # but can still serve as a fast initial filter.
    # Consider the minimum base area for each item.
    min_total_base = sum(min(l*w, l*h, w*h) for l, w, h in (it['dims'] for it in items))
    if min_total_base > L*W: # If even the minimum possible base area sum is too large
        return False
    return True

@lru_cache(maxsize=128)
def packing_penalty_cache(truck_dims, items_tuple):
    # Convert tuple back to list of dicts for PackingPSO
    items_list = [ {'id':i[0], 'name':i[1], 'dims':i[2]} for i in items_tuple ]
    # Use a fixed seed for deterministic packing penalty calculation within the cache
    # This improves consistency when the same set of items is evaluated multiple times.
    current_np_state = np.random.get_state()
    np.random.seed(42) # Seed for packing consistency
    packer = PackingPSO(truck_dims, items_list)
    _, pen = packer.optimize(max_iters=100) # Inner loop iterations
    np.random.set_state(current_np_state) # Restore numpy random state
    return pen

class PackingPSO:
    """
    Optimizes 3D packing of items into a container using PSO.
    Considers 6 possible item orientations.
    """
    def __init__(self, truck_dims, items,
                 compaction_weight=1e-3,
                 num_particles=30, # Particles for inner packing
                 c1=1.5, c2=1.5):
        """
        truck_dims: (L, W, H) of the container.
        items: list of {'id', 'name', 'dims':(lx,ly,lz)}
        compaction_weight: penalty weight for unused bounding‐box volume.
        """
        self.L, self.W, self.H = truck_dims
        self.items = items
        self.n = len(items)
        self.dim = self.n * 4 # x, y, z, orientation_index for each item

        # 6 Standard orientations for a cuboid
        self.orientations = list(permutations([0, 1, 2])) # Indices for dims (l, w, h)
        self.K = 6 # Number of orientations

        self.num_particles = num_particles
        self.c1 = c1
        self.c2 = c2

        self.comp_w = compaction_weight

        self.particles = []
        self.velocities = []
        self.pbest_pos = []
        self.pbest_score = []
        self.gbest_pos = None
        self.gbest_score = float('inf')

        # Initialize particles
        for _ in range(self.num_particles):
            pos = np.zeros(self.dim)
            vel = np.zeros(self.dim)
            for i in range(self.n):
                # Random initial position within truck bounds
                pos[4*i+0] = np.random.uniform(0, self.L)
                pos[4*i+1] = np.random.uniform(0, self.W)
                pos[4*i+2] = np.random.uniform(0, self.H)
                # Random initial orientation (index 0-5)
                pos[4*i+3] = np.random.uniform(0, self.K)
                # Random initial velocity
                vel[4*i:4*i+4] = np.random.uniform(-1, 1, 4) # Smaller initial velocity range?

            self._clamp(pos)
            score = self._penalty(pos)
            self.particles.append(pos.copy())
            self.velocities.append(vel)
            self.pbest_pos.append(pos.copy())
            self.pbest_score.append(score)
            if score < self.gbest_score:
                self.gbest_score = score
                self.gbest_pos = pos.copy()

    @staticmethod
    def _get_rotated_dims(original_dims, orientation_index):
        """
        Returns the dimensions (width, depth, height) relative to truck axes
        for a given orientation index.
        orientation_index: 0 to 5, corresponding to permutations of (l, w, h).
        Permutations:
        0: (l, w, h) -> w=l, d=w, h=h
        1: (l, h, w) -> w=l, d=h, h=w
        2: (w, l, h) -> w=w, d=l, h=h
        3: (w, h, l) -> w=w, d=h, h=l
        4: (h, l, w) -> w=h, d=l, h=w
        5: (h, w, l) -> w=h, d=w, h=l
        """
        l, w, h = original_dims
        dims_map = [
            (l, w, h), (l, h, w),
            (w, l, h), (w, h, l),
            (h, l, w), (h, w, l)
        ]
        if 0 <= orientation_index < 6:
            return dims_map[orientation_index]
        else:
            # Fallback or error? Return original for now.
            return original_dims # Should not happen if clamped

    def _clamp(self, pos):
        """Clamp x,y,z within [0,dim] and orientation index within [0, K-epsilon]."""
        for i in range(self.n):
            pos[4*i+0] = np.clip(pos[4*i+0], 0, self.L)
            pos[4*i+1] = np.clip(pos[4*i+1], 0, self.W)
            pos[4*i+2] = np.clip(pos[4*i+2], 0, self.H)
            # Clamp orientation index
            idx = pos[4*i+3]
            pos[4*i+3] = min(max(idx, 0), self.K - 1e-3) # Clamp between 0 and 5.999...

    def _penalty(self, pos):
        """Calculates the penalty for a given particle position (packing layout)."""
        placement = []
        total_item_vol = 0
        for i in range(self.n):
            x, y, z = pos[4*i : 4*i+3]
            ori_idx = int(pos[4*i+3]) # Orientation index 0-5
            original_dims = self.items[i]['dims']
            # Get actual dimensions based on orientation
            w_rot, d_rot, h_rot = self._get_rotated_dims(original_dims, ori_idx)

            placement.append({
                'id': self.items[i]['id'], # Keep track of item id
                'x': x, 'y': y, 'z': z,   # Position of the corner
                'w': w_rot, 'd': d_rot, 'h': h_rot # Dimensions in this orientation
            })
            total_item_vol += original_dims[0] * original_dims[1] * original_dims[2]

        pen = 0.0

        # 1. Out-of-bounds Penalty
        for it in placement:
            # Amount protruding outside the negative boundaries
            x_under = max(0, -it['x'])
            y_under = max(0, -it['y'])
            z_under = max(0, -it['z'])
            pen += OUTBOUND_PENALTY_FACTOR * (x_under * it['d'] * it['h'] +
                                             y_under * it['w'] * it['h'] +
                                             z_under * it['w'] * it['d'])

            # Amount protruding outside the positive boundaries
            x_over = max(0, (it['x'] + it['w']) - self.L)
            y_over = max(0, (it['y'] + it['d']) - self.W)
            z_over = max(0, (it['z'] + it['h']) - self.H)
            pen += OUTBOUND_PENALTY_FACTOR * (x_over * it['d'] * it['h'] +
                                             y_over * it['w'] * it['h'] +
                                             z_over * it['w'] * it['d'])

        # 2. Overlap Penalty
        for i, A in enumerate(placement):
            for j, B in enumerate(placement):
                if i >= j: continue # Avoid self-comparison and double counting

                # Calculate overlap volume (if any)
                ox = max(0, min(A['x'] + A['w'], B['x'] + B['w']) - max(A['x'], B['x']))
                oy = max(0, min(A['y'] + A['d'], B['y'] + B['d']) - max(A['y'], B['y']))
                oz = max(0, min(A['z'] + A['h'], B['z'] + B['h']) - max(A['z'], B['z']))

                overlap_vol = ox * oy * oz
                if overlap_vol > 1e-6: # Use a small tolerance
                    pen += OVERLAP_PENALTY_FACTOR * overlap_vol

        # 3. Support Penalty (Simplified: Check if center of base is supported)
        # More robust support checks can be complex. This is a basic version.
        # for i, it in enumerate(placement):
        #     if it['z'] < 1e-6: continue # Item is on the floor

        #     is_supported = False
        #     it_center_x = it['x'] + it['w'] / 2
        #     it_center_y = it['y'] + it['d'] / 2
        #     it_base_z = it['z']

        #     for j, base in enumerate(placement):
        #         if i == j: continue # Don't check against self

        #         # Check if 'base' is directly below 'it'
        #         if abs((base['z'] + base['h']) - it_base_z) < 1e-6:
        #             # Check if the center of 'it' is within the base's x-y projection
        #             if (base['x'] <= it_center_x <= base['x'] + base['w'] and
        #                 base['y'] <= it_center_y <= base['y'] + base['d']):
        #                 is_supported = True
        #                 break # Found support

        #     if not is_supported:
        #         # Penalize based on item volume? Or a fixed large penalty?
        #         # Penalizing by volume might favor dropping smaller items without support.
        #         pen += 10.0 * (it['w'] * it['d'] * it['h']) # Example: Support penalty scaled by volume

        # 4. Compaction Penalty (Minimize bounding box volume)
        if placement: # Avoid errors if placement is empty
            max_x = max(it['x'] + it['w'] for it in placement)
            max_y = max(it['y'] + it['d'] for it in placement)
            max_z = max(it['z'] + it['h'] for it in placement)
            # Consider min bounds as well? For now, just max relative to origin.
            # min_x = min(it['x'] for it in placement)
            # min_y = min(it['y'] for it in placement)
            # min_z = min(it['z'] for it in placement)
            # used_bbox_vol = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
            used_bbox_vol = max_x * max_y * max_z # Volume of the overall bounding box
            # Penalize empty space within the bounding box
            pen += self.comp_w * max(0, used_bbox_vol - total_item_vol)
        else:
            pen += 0 # No penalty if no items are placed

        return pen

    def optimize(self, max_iters=100):
        """Runs the PSO algorithm."""
        best_iter_score = float('inf')
        no_improvement_iters = 0
        patience = max(10, max_iters // 10) # Stop if no improvement for a while

        for t in range(max_iters):
            # Update inertia weight (optional, can be constant)
            w_inertia = W_MAX - (W_MAX - W_MIN) * (t / float(max_iters - 1))

            for p in range(self.num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                # Update velocity
                vel = ( w_inertia * self.velocities[p]
                      + self.c1 * r1 * (self.pbest_pos[p] - self.particles[p])
                      + self.c2 * r2 * (self.gbest_pos - self.particles[p]) )
                self.velocities[p] = vel # Should velocity also be clamped? Maybe.

                # Update position
                self.particles[p] += self.velocities[p]
                self._clamp(self.particles[p]) # Ensure position is valid

                # Evaluate new position
                sc = self._penalty(self.particles[p])

                # Update personal best
                if sc < self.pbest_score[p]:
                    self.pbest_score[p] = sc
                    self.pbest_pos[p]   = self.particles[p].copy()

                    # Update global best
                    if sc < self.gbest_score:
                        self.gbest_score = sc
                        self.gbest_pos   = self.particles[p].copy()

            # Early stopping check
            if self.gbest_score < best_iter_score - 1e-4: # Check for meaningful improvement
                 best_iter_score = self.gbest_score
                 no_improvement_iters = 0
            else:
                 no_improvement_iters += 1

            if no_improvement_iters >= patience:
                 # print(f"PackingPSO early stopping at iter {t+1}, score: {self.gbest_score:.4f}")
                 break
            if self.gbest_score < 1e-6: # Found a near-perfect packing
                # print(f"PackingPSO found solution at iter {t+1}, score: {self.gbest_score:.4f}")
                break

        # Extract final layout from the best position found
        layout = []
        if self.gbest_pos is not None:
            pos = self.gbest_pos
            for i in range(self.n):
                ori_idx = int(pos[4*i+3])
                original_dims = self.items[i]['dims']
                # Get the dimensions corresponding to the best orientation found
                w_rot, d_rot, h_rot = self._get_rotated_dims(original_dims, ori_idx)
                layout.append({
                    'id': self.items[i]['id'],
                    'name': self.items[i]['name'],
                    'x': pos[4*i], 'y': pos[4*i+1], 'z': pos[4*i+2],
                    'w': w_rot, 'd': d_rot, 'h': h_rot, # Use rotated dimensions
                    'orientation': ori_idx # Store the chosen orientation index
                })
        # Return the layout and the final penalty score
        return layout, self.gbest_score

###############################
# Load Distance Data
###############################
# Assume city_to_city_polygon.csv exists and is correct
csv_filename = "city_to_city_polygon.csv"
distance_dict = {}
polygons = {}
if os.path.exists(csv_filename):
    df_distance = pd.read_csv(csv_filename)
    if "CityA" in df_distance.columns and "CityB" in df_distance.columns and "Polygon" in df_distance.columns:
        for _, row in df_distance.iterrows():
            key = tuple(sorted((row["CityA"], row["CityB"]))) # Use sorted tuple for key
            distance_dict[key] = row["Distance (meters)"] / 1000
            try:
                polygon_coords = json.loads(row["Polygon"])
                # Store polygon under the same sorted key, but check direction later if needed
                polygons[key] = [[p["lng"], p["lat"]] for p in polygon_coords]
            except json.JSONDecodeError:
                polygons[key] = [] # Or handle error appropriately
    else:
        st.error("CSV file is missing required columns: 'CityA', 'CityB', 'Polygon'")
else:
    st.warning(f"Distance file '{csv_filename}' not found. Using Haversine distance.")


# (Data items, n_trucks, truck constraints, city_coords remain the same)
items = [
    {"id": "Item1", "name": "TV", "weight": 120, "dims": (40, 50, 30), "city": "Jakarta"},
    {"id": "Item2", "name": "Kulkas", "weight": 300, "dims": (70, 60, 90), "city": "Bandung"},
    {"id": "Item3", "name": "AC", "weight": 250, "dims": (80, 50, 60), "city": "Semarang"},
    {"id": "Item4", "name": "Buku", "weight": 50, "dims": (30, 30, 20), "city": "Jakarta"},
    {"id": "Item5", "name": "Sofa", "weight": 500, "dims": (150, 80, 100), "city": "Yogyakarta"},
    {"id": "Item6", "name": "Meja", "weight": 150, "dims": (120, 100, 40), "city": "Semarang"},
    {"id": "Item7", "name": "Ranjang", "weight": 400, "dims": (200, 160, 50), "city": "Malang"}, # Oversized width
    {"id": "Item8", "name": "Kipas Angin", "weight": 30, "dims": (20, 20, 40), "city": "Bandung"},
    {"id": "Item9",  "name": "WashingMachine","weight":350, "dims": (60,60,85), "city": "Jakarta"},
    {"id": "Item10", "name": "Bookshelf", "weight":100, "dims": (80,30,180), "city": "Surabaya"}, # Oversized height
    {"id": "Item11", "name": "Mattress", "weight":200, "dims": (200,90,30), "city": "Bandung"},
    {"id": "Item12", "name": "Wardrobe", "weight":450, "dims": (100,60,200), "city": "Yogyakarta"}, # Oversized height
    {"id": "Item13", "name": "DiningTable", "weight":250, "dims": (160,90,75), "city": "Semarang"},
    {"id": "Item14", "name": "DeskLamp", "weight":10,  "dims": (15,15,40), "city": "Malang"},
    {"id": "Item15", "name": "Microwave", "weight":40,  "dims": (50,40,35), "city": "Jakarta"},
    {"id": "Item16", "name": "Printer", "weight":25,  "dims": (45,40,30), "city": "Surabaya"},
    {"id": "Item17", "name": "FloorLamp", "weight":20,  "dims": (30,30,160), "city": "Bandung"}, # Oversized height
    {"id": "Item18", "name": "AirPurifier", "weight":15,  "dims": (25,25,60), "city": "Yogyakarta"},
    {"id": "Item19", "name": "WaterHeater", "weight":80,  "dims": (50,50,100), "city": "Semarang"},
    {"id": "Item20", "name": "CoffeeTable", "weight":80,  "dims": (120,60,45), "city": "Malang"},
]

n_trucks = 4
truck_max_weight = 1000 # Example weight limit
truck_max_length = 200
truck_max_width  = 150
truck_max_height = 150

city_coords = {
    "Surabaya": (-7.2575, 112.7521),
    "Jakarta": (-6.2088, 106.8456),
    "Bandung": (-6.9175, 107.6191),
    "Semarang": (-6.9667, 110.4167),
    "Yogyakarta": (-7.7956, 110.3695),
    "Malang": (-7.9824, 112.6304)
}

# --- Pre-process items ---
threshold_small = 50*50*50
threshold_medium = 100*100*100

def get_dimension_category(l, w, h):
    volume = l * w * h
    if volume < threshold_small: return "Kecil", 50
    elif volume < threshold_medium: return "Sedang", 75
    else: return "Besar", 100

for item in items:
    l, w, h = item["dims"]
    cat, factor = get_dimension_category(l, w, h)
    item["dim_category"] = cat
    item["cat_factor"] = factor
    # Check if item cannot fit in *any* orientation
    item["is_oversized"] = not any(
        p[0] <= truck_max_length and p[1] <= truck_max_width and p[2] <= truck_max_height
        for p in permutations((l, w, h))
    )
    if item["is_oversized"]:
        print(f"Warning: Item {item['id']} ({item['name']}) dimensions {item['dims']} is oversized for truck {truck_max_length}x{truck_max_width}x{truck_max_height} in all orientations.")

# --- Distance Functions ---
@st.cache_data(show_spinner=False)
def haversine(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

@st.cache_data(show_spinner=False)
def distance(city1, city2):
    if city1 == city2: return 0.0
    key = tuple(sorted((city1, city2)))
    if key in distance_dict:
        return distance_dict[key]
    elif city1 in city_coords and city2 in city_coords:
        return haversine(city_coords[city1], city_coords[city2])
    else:
        st.error(f"Cannot find distance between {city1} and {city2}")
        return float('inf') # Or handle error appropriately

@st.cache_data(show_spinner=False)
def route_distance(cities):
    """Nearest-neighbor heuristic for route distance from Surabaya -> cities -> Surabaya."""
    if not cities: return 0.0
    unique_cities = list(set(cities)) # Work with unique cities for routing
    if not unique_cities: return 0.0

    current = "Surabaya"
    unvisited = unique_cities[:]
    total_dist = 0.0
    route_path = ["Surabaya"]

    while unvisited:
        nearest = min(unvisited, key=lambda c: distance(current, c))
        total_dist += distance(current, nearest)
        current = nearest
        unvisited.remove(nearest)
        route_path.append(current)

    total_dist += distance(current, "Surabaya") # Return to origin
    route_path.append("Surabaya")
    # print(f"Route for {cities}: {route_path} -> {total_dist:.2f} km") # Debugging
    return total_dist

###############################
# Fitness Function (Outer PSO)
###############################
def compute_fitness(assignment):
    """
    Evaluates the quality of an assignment of items to trucks.
    Returns: (total_revenue - total_fuel_cost) - total_packing_penalty
    """
    truck_details = {
        t: {"items": [], "cities": set(), "weight": 0, "revenue": 0.0}
        for t in range(1, n_trucks + 1)
    }
    total_revenue = 0.0
    total_packing_penalty = 0.0 # Penalty from PackingPSO
    fitness_penalty = 0.0 # Penalty for constraint violations (weight, oversize)

    # 1. Assign items and calculate basic revenue/weight
    for idx, truck_idx in enumerate(assignment):
        if truck_idx == 0: # Item not assigned
            continue
        if not (1 <= truck_idx <= n_trucks): # Invalid assignment index
            fitness_penalty += 10000 # Penalize invalid assignments
            continue

        item = items[idx]

        # Constraint Check: Item Oversized (already checked, but double-check)
        if item["is_oversized"]:
           # This should ideally not happen if decode_position handles it
           fitness_penalty += 5000 * (item["dims"][0]*item["dims"][1]*item["dims"][2]) # Heavy penalty
           continue # Don't assign oversized items

        # Add item to truck
        truck_details[truck_idx]["items"].append(item)
        truck_details[truck_idx]["cities"].add(item["city"])
        truck_details[truck_idx]["weight"] += item["weight"]

        # Calculate revenue for this item
        dist = distance("Surabaya", item["city"])
        rev = item["weight"] * dist * item["cat_factor"]
        truck_details[truck_idx]["revenue"] += rev
        total_revenue += rev

    # 2. Calculate Costs and Packing Penalties per Truck
    FUEL_PRICE_PER_L = 9000
    TRUCK_CONSUMPTION_KM_L = 4 # km per liter
    cost_per_km = FUEL_PRICE_PER_L / TRUCK_CONSUMPTION_KM_L
    total_cost = 0.0

    truck_dims = (truck_max_length, truck_max_width, truck_max_height)

    for t in range(1, n_trucks + 1):
        info = truck_details[t]
        truck_items = info["items"]

        if not truck_items:
            continue # Skip empty trucks

        # Constraint Check: Truck Weight Limit
        if info["weight"] > truck_max_weight:
            overweight = info["weight"] - truck_max_weight
            fitness_penalty += 10.0 * overweight # Penalize overweight trucks

        # Calculate Fuel Cost for this truck's route
        if info["cities"]:
            dist = route_distance(list(info["cities"]))
            total_cost += cost_per_km * dist

        # Packing Feasibility and Penalty
        if not quick_feasible(truck_dims, truck_items):
            # If quick check fails, assign a large penalty, avoid calling detailed packing
            total_packing_penalty += 1000.0 + 10 * len(truck_items) # Large penalty + per item
        else:
            # Prepare items for caching (needs hashable representation)
            items_tuple = tuple(sorted((it["id"], it["name"], it["dims"]) for it in truck_items))
            packing_pen = packing_penalty_cache(truck_dims, items_tuple)
            total_packing_penalty += packing_pen

    # Final Fitness Calculation
    profit = total_revenue - total_cost
    fitness = profit - total_packing_penalty - fitness_penalty

    # We want to MAXIMIZE fitness, so PSO's default minimization needs target negated
    # However, since we check `if fit > gbest_fitness`, we work with maximization directly.
    return fitness


def decode_position(position):
    """Converts continuous PSO particle position to discrete truck assignments."""
    assignment = []
    for i, val in enumerate(position):
        item = items[i]
        if item["is_oversized"]:
            assignment.append(0) # Cannot be assigned
        else:
            # Clamp value between 0.5 and n_trucks + 0.5 before rounding
            # Ensures rounding maps to 1..n_trucks correctly.
            # A value slightly below 1 rounds to 1, slightly above n rounds to n.
            # Assign 0 if value is very low? No, let fitness penalize bad assignments.
            # We map value range [0, n_trucks+1] roughly to assignments [0..n_trucks]
            # Let's try simple rounding from 1 to n_trucks first.
            # Values < 1 might map to 1, values > n might map to n.
            clamped_val = max(0.51, min(n_trucks + 0.49, val)) # Map closer to [1, n] range
            assigned_truck = int(round(clamped_val))

            # Re-check bounds just in case, although clamping should handle it
            assigned_truck = max(1, min(n_trucks, assigned_truck))
            assignment.append(assigned_truck)

            # Alternative: Map based on intervals
            # step = (n_trucks + 1) / (n_trucks + 1) # if range is 0 to n_trucks+1
            # assigned_truck = int(val / step) # 0 = unassigned, 1..n = truck 1..n

    return assignment

###############################
# Outer PSO: Initialization and Iteration (Item Assignment)
###############################
num_particles_assign = 30 # Particles for assignment PSO
max_iter_assign = 500     # Iterations for assignment PSO
patience_assign = 50      # Early stopping patience

# PSO parameters for assignment
assign_w_max = 0.9
assign_w_min = 0.4
assign_c1 = 2.0
assign_c2 = 2.0

improvement_threshold = 1.0 # Minimum improvement to reset patience counter
no_improvement_count = 0

particles_assign = []
velocities_assign = []
pbest_positions_assign = []
pbest_fitness_assign = []
gbest_position_assign = None
gbest_fitness_assign = -float('inf') # Maximize fitness
prev_gbest_assign = -float('inf')

print("Initializing Assignment PSO...")
# Initialize assignment particles
for p_idx in range(num_particles_assign):
    # Each dimension corresponds to an item. Value suggests truck assignment.
    # Range [0, n_trucks+1] might be better to allow unassigned (0) possibility?
    # Let's try range [1, n_trucks] for assignable items, handle oversized separately.
    position = np.zeros(len(items))
    velocity = np.zeros(len(items))
    for i in range(len(items)):
        if items[i]["is_oversized"]:
            position[i] = 0 # Fixed to unassigned
            velocity[i] = 0
        else:
            # Start with random assignment 1 to n_trucks
            position[i] = random.uniform(1, n_trucks + 1e-9) # Range slightly less than n_trucks+1
            # Initial velocity range? Needs consideration based on position range.
            # If position is 1 to 4, velocity maybe -1 to 1? or -3 to 3?
            velocity[i] = random.uniform(-(n_trucks/2), n_trucks/2)

    particles_assign.append(position)
    velocities_assign.append(velocity)

    # Evaluate initial position
    assignment = decode_position(position)
    fit = compute_fitness(assignment)

    pbest_positions_assign.append(position.copy())
    pbest_fitness_assign.append(fit)

    # Update global best
    if fit > gbest_fitness_assign:
        gbest_fitness_assign = fit
        gbest_position_assign = position.copy()
print(f"Initial Global Best Fitness: {gbest_fitness_assign:.2f}")

# --- Assignment PSO Loop ---
fitness_history_assign = []
print(f"Running Assignment PSO for {max_iter_assign} iterations...")
for it in range(1, max_iter_assign + 1):
    w_inertia = assign_w_max - (assign_w_max - assign_w_min) * (it / max_iter_assign)

    for i in range(num_particles_assign):
        # Update velocity and position only for assignable items
        r1 = np.random.rand(len(items))
        r2 = np.random.rand(len(items))

        for d in range(len(items)):
            if items[d]["is_oversized"]: continue # Skip oversized items

            velocities_assign[i][d] = (w_inertia * velocities_assign[i][d] +
                                       assign_c1 * r1[d] * (pbest_positions_assign[i][d] - particles_assign[i][d]) +
                                       assign_c2 * r2[d] * (gbest_position_assign[d] - particles_assign[i][d]))

            # Clamp velocity? Optional. Can prevent explosion.
            # max_vel = n_trucks / 2
            # velocities_assign[i][d] = np.clip(velocities_assign[i][d], -max_vel, max_vel)

            particles_assign[i][d] += velocities_assign[i][d]

            # Clamp position to valid range (e.g., 0.5 to n_trucks + 0.5 for rounding)
            particles_assign[i][d] = np.clip(particles_assign[i][d], 0.51, n_trucks + 0.49)


        # Decode and Evaluate
        assignment = decode_position(particles_assign[i])
        fit = compute_fitness(assignment)

        # Update pbest
        if fit > pbest_fitness_assign[i]:
            pbest_fitness_assign[i] = fit
            pbest_positions_assign[i] = particles_assign[i].copy()

            # Update gbest
            if fit > gbest_fitness_assign:
                gbest_fitness_assign = fit
                gbest_position_assign = particles_assign[i].copy()

    # Track fitness history and check early stopping
    fitness_history_assign.append(gbest_fitness_assign)
    improvement = gbest_fitness_assign - prev_gbest_assign

    if it % 20 == 0: # Print progress periodically
         print(f"Iter {it}/{max_iter_assign}, Best Fitness: {gbest_fitness_assign:.2f}")

    if improvement < improvement_threshold and it > patience_assign : # Allow some initial iterations
        no_improvement_count += 1
        if no_improvement_count >= patience_assign:
            print(f"\nEarly stopping at iteration {it} due to no improvement.")
            break
    else:
        no_improvement_count = 0 # Reset counter if improvement found
        prev_gbest_assign = gbest_fitness_assign # Update previous best only on improvement

print(f"\nAssignment PSO finished. Final Best Fitness: {gbest_fitness_assign:.2f}")
best_assignment = decode_position(gbest_position_assign)

# Clear cache before final packing run if desired
# packing_penalty_cache.cache_clear()

###############################
# Mendapatkan Info Detail per Truk (Post-Assignment)
###############################
def get_truck_info(assignment):
    truck_info = {
        t: {"items": [], "weight": 0, "volume": 0, "cities": [], "revenue": 0.0}
        for t in range(1, n_trucks + 1)
    }
    assigned_item_details = [] # For summary table

    for i, truck_idx in enumerate(assignment):
        item = items[i]
        if truck_idx == 0:
             assigned_item_details.append({**item, "assigned_truck": "Unassigned/Oversized"})
             continue # Skip unassigned/oversized

        truck_info[truck_idx]["items"].append(item)
        truck_info[truck_idx]["weight"] += item["weight"]
        l, w, h = item["dims"]
        truck_info[truck_idx]["volume"] += (l * w * h)
        dist = distance("Surabaya", item["city"])
        rev = item["weight"] * dist * item["cat_factor"]
        truck_info[truck_idx]["revenue"] += rev
        if item["city"] not in truck_info[truck_idx]["cities"]:
            truck_info[truck_idx]["cities"].append(item["city"])

        assigned_item_details.append({**item, "assigned_truck": f"Truk {truck_idx}"})

    # Calculate route distance and cost per truck AFTER assignments are final
    FUEL_PRICE_PER_L = 9000
    TRUCK_CONSUMPTION_KM_L = 4
    cost_per_km = FUEL_PRICE_PER_L / TRUCK_CONSUMPTION_KM_L
    for t in range(1, n_trucks + 1):
        route_dist = route_distance(truck_info[t]["cities"])
        truck_info[t]["route_distance"] = route_dist
        truck_info[t]["fuel_cost"] = cost_per_km * route_dist
        truck_info[t]["profit"] = truck_info[t]["revenue"] - truck_info[t]["fuel_cost"]

    return truck_info, pd.DataFrame(assigned_item_details)

final_truck_info, assigned_items_df = get_truck_info(best_assignment)

###############################
# Fungsi Rute dan Path untuk Visualisasi Peta
###############################
def get_route_sequence(cities):
    """Returns the sequence of cities visited for a given list."""
    if not cities: return ["Surabaya"]
    unique_cities = list(set(cities))
    if not unique_cities: return ["Surabaya"]

    route = ["Surabaya"]
    current = "Surabaya"
    unvisited = unique_cities[:]
    while unvisited:
        nearest = min(unvisited, key=lambda c: distance(current, c))
        route.append(nearest)
        current = nearest
        unvisited.remove(nearest)
    route.append("Surabaya")
    return route

def get_segment_path(city_a, city_b):
    """Gets the path segment (list of coords) between two cities."""
    key = tuple(sorted((city_a, city_b)))
    original_key = (city_a, city_b) # To check direction

    path = []
    if key in polygons and len(polygons[key]) > 0:
        path = polygons[key]
        # Check if the stored path needs reversing based on the requested order
        # Heuristic: Check if first point of stored path matches city_a's coords approximately
        start_coord_stored = (round(path[0][1], 4), round(path[0][0], 4)) # lat, lon
        start_coord_city_a = (round(city_coords[city_a][0], 4), round(city_coords[city_a][1], 4))
        if start_coord_stored != start_coord_city_a:
            path = path[::-1] # Reverse if direction is wrong
    # Fallback to straight line if no polygon data or empty polygon
    if not path:
        start = [city_coords[city_a][1], city_coords[city_a][0]] # lon, lat
        end = [city_coords[city_b][1], city_coords[city_b][0]]   # lon, lat
        path = [start, end]
    return path

def get_full_route_path(route_sequence):
    """Builds the complete coordinate path for a route sequence."""
    full_path = []
    if not route_sequence or len(route_sequence) < 2:
        return [] # Need at least two points for a path

    # Initial point
    # full_path.append([city_coords[route_sequence[0]][1], city_coords[route_sequence[0]][0]])

    for i in range(len(route_sequence) - 1):
        city_a = route_sequence[i]
        city_b = route_sequence[i+1]
        segment = get_segment_path(city_a, city_b)

        # Add segment, avoiding duplicate points at junctions
        if full_path and segment:
             # Check if last point of full_path is same as first point of segment
             if (abs(full_path[-1][0] - segment[0][0]) < 1e-6 and
                 abs(full_path[-1][1] - segment[0][1]) < 1e-6):
                 full_path.extend(segment[1:])
             else:
                 full_path.extend(segment) # Add whole segment if no overlap detected
        elif segment:
             full_path.extend(segment) # Add first segment

    return full_path

# --- Prepare Route Data for Pydeck ---
routes_data_pydeck = []
colors = [
    [255, 0, 0, 200],   # Red
    [0, 255, 0, 200],   # Green
    [0, 0, 255, 200],   # Blue
    [255, 165, 0, 200], # Orange
    # Add more colors if n_trucks > 4
    [128, 0, 128, 200], # Purple
    [0, 255, 255, 200]  # Cyan
]

print("Generating route paths for visualization...")
for t in range(1, n_trucks + 1):
    info = final_truck_info[t]
    if info["cities"]:
        route_seq = get_route_sequence(info["cities"])
        full_path = get_full_route_path(route_seq)

        if full_path: # Only add if path is generated
             # Tooltip info: remove start/end Surabaya for display
             display_route = " → ".join([c for c in route_seq if c != "Surabaya"])
             if not display_route : display_route = "Base Only" # If only Surabaya->Surabaya

             routes_data_pydeck.append({
                 "truck": f"Truk {t}",
                 "path": full_path,
                 "color": colors[(t - 1) % len(colors)],
                 "route_info": display_route,
                 "distance_km": info.get("route_distance", 0) # Get calculated distance
             })
        else:
             print(f"Warning: Could not generate path for Truck {t} with cities {info['cities']}")


###############################
# Final Packing and Visualization
###############################
# Run PackingPSO one last time for each truck to get the layout for visualization
# Use more iterations for the final visualization packing
final_layouts = {}
print("Running final packing optimization for visualization...")
for t in range(1, n_trucks + 1):
    items_for_truck = final_truck_info[t]["items"]
    if not items_for_truck:
        final_layouts[t] = ([], 0) # Empty layout, zero penalty
        continue

    print(f"  Optimizing packing for Truck {t} ({len(items_for_truck)} items)...")
    truck_dims = (truck_max_length, truck_max_width, truck_max_height)

    # Try multiple random seeds for packing and take the best result
    best_penalty_viz = float('inf')
    best_layout_viz = None
    num_packing_attempts = 3 # Number of attempts with different seeds
    packing_iters_viz = 300 # More iterations for final packing

    for attempt in range(num_packing_attempts):
        seed = 42 + attempt * 101 # Different seed for each attempt
        # print(f"    Attempt {attempt+1} with seed {seed}")
        current_np_state = np.random.get_state() # Save state
        np.random.seed(seed)

        packer = PackingPSO(truck_dims, items_for_truck, num_particles=40) # Maybe more particles?
        layout, penalty = packer.optimize(max_iters=packing_iters_viz)

        np.random.set_state(current_np_state) # Restore state

        # print(f"      Penalty: {penalty:.4f}")
        if penalty < best_penalty_viz:
            best_penalty_viz = penalty
            best_layout_viz = layout
            # print(f"      New best layout found for Truck {t}")

        if best_penalty_viz < 1e-6: # Stop if a (near) perfect packing is found
            break

    print(f"  Truck {t} - Best Packing Penalty: {best_penalty_viz:.4f}")
    final_layouts[t] = (best_layout_viz, best_penalty_viz)


# --- Visualization Function ---
def create_truck_figure(truck_dims, packed_items):
    """Creates a Plotly 3D figure visualizing packed items in a truck outline."""
    L, W, H = truck_dims
    fig = go.Figure()

    if not packed_items: # Handle case with no items
         pass # Just draw the truck outline

    # Define colors for items (cycle through them)
    color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                     "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    # Add packed items as Mesh3d objects
    for idx, item in enumerate(packed_items):
        # Position (corner)
        x, y, z = item['x'], item['y'], item['z']
        # Dimensions in the chosen orientation
        w_rot, d_rot, h_rot = item['w'], item['d'], item['h']

        # Calculate the 8 vertices of the item box
        x_verts = [x, x, x + w_rot, x + w_rot, x, x, x + w_rot, x + w_rot]
        y_verts = [y, y + d_rot, y + d_rot, y, y, y + d_rot, y + d_rot, y]
        z_verts = [z, z, z, z, z + h_rot, z + h_rot, z + h_rot, z + h_rot]

        # Define faces using vertex indices (standard cube triangulation)
        faces_i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 1]
        faces_j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 2]
        faces_k = [0, 7, 2, 3, 6, 7, 1, 5, 4, 5, 7, 6]

        item_color = color_palette[idx % len(color_palette)]

        fig.add_trace(go.Mesh3d(
            x=x_verts, y=y_verts, z=z_verts,
            i=faces_i, j=faces_j, k=faces_k,
            color=item_color, opacity=0.8,
            name=item['name'],
            # Show dimensions and orientation in hover text
            hovertext=f"{item['name']}<br>Dims: {w_rot:.1f}x{d_rot:.1f}x{h_rot:.1f}<br>Pos: ({x:.1f},{y:.1f},{z:.1f})<br>Ori: {item.get('orientation', 'N/A')}",
            hoverinfo="text"
        ))

    # Add truck outline (wireframe)
    corners = [(0,0,0), (L,0,0), (L,W,0), (0,W,0), (0,0,H), (L,0,H), (L,W,H), (0,W,H)]
    edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4),
             (0,4), (1,5), (2,6), (3,7)]
    edge_x, edge_y, edge_z = [], [], []
    for (u, v) in edges:
        edge_x += [corners[u][0], corners[v][0], None] # None separates lines
        edge_y += [corners[u][1], corners[v][1], None]
        edge_z += [corners[u][2], corners[v][2], None]

    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines', line=dict(color='black', width=2),
        hoverinfo='none', showlegend=False
    ))

    # Configure scene layout
    fig.update_layout(
        title_text=f"Truck Load ({L}x{W}x{H})",
        scene=dict(
            xaxis=dict(range=[0, L], title='Length (X)', backgroundcolor="rgb(230, 230, 230)"),
            yaxis=dict(range=[0, W], title='Width (Y)', backgroundcolor="rgb(230, 230, 230)"),
            zaxis=dict(range=[0, H], title='Height (Z)', backgroundcolor="rgb(230, 230, 230)"),
            aspectratio=dict(x=L / max(L, W, H), y=W / max(L, W, H), z=H / max(L, W, H)), # Adjust aspect ratio based on dims
            aspectmode='manual' # Use calculated aspect ratio
            # aspectmode='data' # Tries to make units equal, can distort view
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=True # Show item names in legend
    )
    return fig

###############################
# Streamlit UI Layout
###############################
st.set_page_config(layout="wide")
st.title("Optimasi Penugasan dan Pemuatan Truk dengan PSO")

# --- Row 1: Summary Metrics ---
col1, col2, col3 = st.columns(3)
total_profit_final = sum(info['profit'] for info in final_truck_info.values())
total_revenue_final = sum(info['revenue'] for info in final_truck_info.values())
total_cost_final = sum(info['fuel_cost'] for info in final_truck_info.values())

col1.metric("Total Estimasi Profit", f"Rp {total_profit_final:,.0f}")
col2.metric("Total Revenue", f"Rp {total_revenue_final:,.0f}")
col3.metric("Total Fuel Cost", f"Rp {total_cost_final:,.0f}")
st.caption("Profit = Revenue - Fuel Cost. Penalti packing tidak termasuk dalam profit ini.")


# --- Row 2: Map and Fitness Chart ---
col1_map, col2_chart = st.columns([3, 1]) # Map takes more space

with col1_map:
    st.subheader("Rute Pengiriman per Truk")
    if routes_data_pydeck:
        # Add city markers
        city_points = [{"name": city, "coordinates": [coord[1], coord[0]]} # lon, lat
                       for city, coord in city_coords.items()]
        marker_layer = pdk.Layer(
            "ScatterplotLayer",
            data=city_points,
            get_position="coordinates",
            get_fill_color=[0, 0, 0, 180], # Black markers
            get_radius=6000, # Radius in meters
            radius_min_pixels=5,
            pickable=True,
            auto_highlight=True
        )
        text_layer = pdk.Layer(
            "TextLayer",
            data=city_points,
            get_position="coordinates",
            get_text="name",
            get_color=[0, 0, 0, 200],
            get_size=14,
            get_alignment_baseline="'bottom'",
            get_pixel_offset=[0, -15] # Offset text above marker
        )

        path_layers = [
            pdk.Layer(
                "PathLayer",
                data=[route], # Pydeck expects a list of data objects
                get_path="path",
                get_color="color",
                get_width=5,
                width_scale=1,
                width_min_pixels=3,
                pickable=True,
                auto_highlight=True
            ) for route in routes_data_pydeck
        ]

        view_state = pdk.ViewState(
            latitude=city_coords["Surabaya"][0],
            longitude=city_coords["Surabaya"][1],
            zoom=6,
            pitch=45 # Angled view
        )

        tooltip = {
            "html": """
                <b>{truck}</b><br/>
                Route: {route_info}<br/>
                Dist: {distance_km:.1f} km
            """,
            "style": {"backgroundColor": "steelblue", "color": "white", "fontSize": "12px"}
        }

        st.pydeck_chart(pdk.Deck(
            layers=path_layers + [marker_layer, text_layer],
            initial_view_state=view_state,
            map_style='mapbox://styles/mapbox/light-v9', # Use a light map style
            tooltip=tooltip
        ))
    else:
        st.write("Tidak ada rute untuk ditampilkan.")

with col2_chart:
    st.subheader("Fitness (Assignment PSO)")
    if fitness_history_assign:
        st.line_chart(pd.DataFrame({'Fitness': fitness_history_assign}), height=300)
    else:
        st.write("Data fitness tidak tersedia.")

# --- Row 3: Assignment Details ---
st.subheader("Detail Penugasan Item ke Truk")
# Display dataframe with assigned truck info
st.dataframe(assigned_items_df[['id', 'name', 'weight', 'dims', 'city', 'dim_category', 'assigned_truck']])


# --- Row 4 onwards: Truck Details and Visualization ---
st.subheader("Detail Muatan dan Visualisasi per Truk")

# Use columns for layout if needed, e.g., 2 columns for trucks
cols_per_row = 2
num_rows = (n_trucks + cols_per_row - 1) // cols_per_row

truck_idx = 1
for r in range(num_rows):
    cols = st.columns(cols_per_row)
    for c in range(cols_per_row):
        if truck_idx <= n_trucks:
            with cols[c]:
                st.markdown(f"### Truk {truck_idx}")
                info = final_truck_info[truck_idx]
                layout, penalty = final_layouts.get(truck_idx, (None, float('inf')))

                if not info["items"]:
                    st.write("Tidak ada barang yang diangkut.")
                else:
                    # Display summary table for the truck
                    df_truck = pd.DataFrame(info["items"])[["id", "name", "weight", "dims", "city"]]
                    st.dataframe(df_truck, height=150) # Limit height

                    st.metric("Total Berat", f"{info['weight']} / {truck_max_weight} kg",
                              delta=f"{truck_max_weight - info['weight']:.0f} kg available",
                              delta_color="off" if info['weight'] > truck_max_weight else "normal")
                    st.metric("Estimasi Profit Truk", f"Rp {info['profit']:,.0f}")
                    st.write(f"**Kota Tujuan:** {', '.join(info['cities'])}")
                    st.write(f"**Jarak Rute:** {info['route_distance']:.1f} km")

                    # Display Packing Visualization
                    if layout is None:
                         st.error(f"⚠️ Truk {truck_idx}: Packing gagal menghasilkan layout.")
                    elif penalty > 1e-3: # Use a tolerance for floating point
                         st.warning(f"⚠️ Truk {truck_idx}: Packing belum optimal (penalty={penalty:.4f}). Visualisasi mungkin tidak valid.")
                         # Optionally show the best attempt anyway
                         fig = create_truck_figure(
                             (truck_max_length, truck_max_width, truck_max_height),
                             layout # Show the layout even with penalty
                         )
                         st.plotly_chart(fig, use_container_width=True)
                    else:
                         st.success(f"Truk {truck_idx}: Muatan terpack dengan baik (penalty={penalty:.4f}).")
                         fig = create_truck_figure(
                             (truck_max_length, truck_max_width, truck_max_height),
                             layout
                         )
                         st.plotly_chart(fig, use_container_width=True)

                truck_idx += 1
        else:
             # If odd number of trucks, the last column might be empty
             pass

print("\nStreamlit App Ready.")