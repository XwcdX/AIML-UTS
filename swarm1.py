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

st.set_page_config(layout="wide")

random.seed(30)
np.random.seed(30)

# --- Constants ---
OVERLAP_PENALTY_FACTOR = 2.0
OUTBOUND_PENALTY_FACTOR = 2.0
# --- PENALTI BARU ---
OVERWEIGHT_CONSTANT_PENALTY = 1e8 # Penalti konstan sangat besar jika overweight
OVERWEIGHT_FACTOR_PENALTY = 10000.0 # Penalti proporsional dg kuadrat kelebihan berat
UNSUPPORTED_PENALTY_BASE = 10000.0  # Penalti dasar jika item tidak didukung
UNSUPPORTED_PENALTY_HEIGHT_FACTOR = 200.0 # Penalti tambahan berdasarkan tinggi item
# -----------------
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
    items_list = [ {'id':i[0], 'name':i[1], 'dims':i[2], 'weight':i[3]} for i in items_tuple ] # Tambahkan weight jika perlu
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
    Includes basic support penalty.
    """
    def __init__(self, truck_dims, items,
                 compaction_weight=1e-3,
                 num_particles=30, # Particles for inner packing
                 c1=1.5, c2=1.5):
        """
        truck_dims: (L, W, H) of the container.
        items: list of {'id', 'name', 'dims':(lx,ly,lz), 'weight'}
        compaction_weight: penalty weight for unused bounding‐box volume.
        """
        self.L, self.W, self.H = truck_dims
        self.items = items
        self.n = len(items)
        self.dim = self.n * 4 # x, y, z, orientation_index for each item

        # 6 Standard orientations for a cuboid (permutations of axes)
        # Ini MEREPRESENTASIKAN rotasi 90 derajat, BUKAN rotasi arbitrer.
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
                pos[4*i+2] = np.random.uniform(0, self.H) # Start anywhere, let penalty push down
                # Random initial orientation (index 0-5)
                pos[4*i+3] = np.random.uniform(0, self.K)
                # Random initial velocity
                vel[4*i:4*i+4] = np.random.uniform(-1, 1, 4)

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
        for a given orientation index (0-5). Represents 90-degree rotations.
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
            return original_dims # Fallback

    def _clamp(self, pos):
        """Clamp x,y,z within [0,dim] and orientation index within [0, K-epsilon]."""
        for i in range(self.n):
            # Clamp positions - Allow items to start slightly below 0 initially,
            # penalty will push them up if needed, or gravity penalty push down if unsupported.
            # Let's clamp strictly within bounds for now to avoid complexity.
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
            w_rot, d_rot, h_rot = self._get_rotated_dims(original_dims, ori_idx)

            placement.append({
                'id': self.items[i]['id'],
                'name': self.items[i]['name'], # Added name for potential debugging
                'x': x, 'y': y, 'z': z,
                'w': w_rot, 'd': d_rot, 'h': h_rot
            })
            total_item_vol += original_dims[0] * original_dims[1] * original_dims[2]

        pen = 0.0

        # 1. Out-of-bounds Penalty
        for it in placement:
            x_under = max(0, -it['x'])
            y_under = max(0, -it['y'])
            z_under = max(0, -it['z'])
            pen += OUTBOUND_PENALTY_FACTOR * (x_under * it['d'] * it['h'] +
                                             y_under * it['w'] * it['h'] +
                                             z_under * it['w'] * it['d'])

            x_over = max(0, (it['x'] + it['w']) - self.L)
            y_over = max(0, (it['y'] + it['d']) - self.W)
            z_over = max(0, (it['z'] + it['h']) - self.H)
            pen += OUTBOUND_PENALTY_FACTOR * (x_over * it['d'] * it['h'] +
                                             y_over * it['w'] * it['h'] +
                                             z_over * it['w'] * it['d'])

        # 2. Overlap Penalty
        for i, A in enumerate(placement):
            for j, B in enumerate(placement):
                if i >= j: continue

                ox = max(0, min(A['x'] + A['w'], B['x'] + B['w']) - max(A['x'], B['x']))
                oy = max(0, min(A['y'] + A['d'], B['y'] + B['d']) - max(A['y'], B['y']))
                oz = max(0, min(A['z'] + A['h'], B['z'] + B['h']) - max(A['z'], B['z']))

                overlap_vol = ox * oy * oz
                if overlap_vol > 1e-6:
                    pen += OVERLAP_PENALTY_FACTOR * overlap_vol

        # --- MODIFIED/ACTIVATED: Support Penalty (Gravity Logic) ---
        # Checks if the center of the item's base is reasonably supported.
        # Items on the floor (z near 0) are considered supported by default.
        support_tolerance = 1e-4 # Small tolerance for floating point comparisons
        for i, it in enumerate(placement):
            if it['z'] < support_tolerance: continue # Item is on the floor, considered supported

            is_supported = False
            it_center_x = it['x'] + it['w'] / 2.0
            it_center_y = it['y'] + it['d'] / 2.0
            it_base_z = it['z']

            for j, base in enumerate(placement):
                if i == j: continue # Don't check against self

                # Check if 'base' is directly below 'it' within tolerance
                if abs((base['z'] + base['h']) - it_base_z) < support_tolerance:
                    # Check if the center of 'it's base projection is within the 'base' item's x-y projection
                    if (base['x'] - support_tolerance <= it_center_x <= base['x'] + base['w'] + support_tolerance and
                        base['y'] - support_tolerance <= it_center_y <= base['y'] + base['d'] + support_tolerance):
                        is_supported = True
                        break # Found sufficient support (based on center point)

            if not is_supported:
                # Apply HARSH penalty if not supported and not on the floor
                # The penalty increases the higher the item is "floating"
                pen += UNSUPPORTED_PENALTY_BASE + UNSUPPORTED_PENALTY_HEIGHT_FACTOR * it['h']
        # --- End of Support Penalty Modification ---

        # 4. Compaction Penalty (Minimize bounding box volume)
        if placement:
            max_x = max(it['x'] + it['w'] for it in placement) if placement else 0
            max_y = max(it['y'] + it['d'] for it in placement) if placement else 0
            max_z = max(it['z'] + it['h'] for it in placement) if placement else 0
            used_bbox_vol = max_x * max_y * max_z
            pen += self.comp_w * max(0, used_bbox_vol - total_item_vol)
        else:
            pen += 0

        return pen

    def optimize(self, max_iters=100):
        """Runs the PSO algorithm."""
        best_iter_score = float('inf')
        no_improvement_iters = 0
        # Lower patience slightly as inner loop runs many times
        patience = max(8, max_iters // 12)

        # Ensure gbest_pos is initialized if it's None (should be done in __init__)
        if self.gbest_pos is None and self.pbest_pos:
             # Find the best initial particle if gbest wasn't set properly
             initial_best_idx = np.argmin(self.pbest_score)
             if self.pbest_score[initial_best_idx] < self.gbest_score:
                 self.gbest_score = self.pbest_score[initial_best_idx]
                 self.gbest_pos = self.pbest_pos[initial_best_idx].copy()

        # Check if gbest_pos is still None (e.g., no particles)
        if self.gbest_pos is None:
            print("Warning: gbest_pos is None in optimize start. No particles?")
            return [], float('inf') # Cannot optimize

        for t in range(max_iters):
            w_inertia = W_MAX - (W_MAX - W_MIN) * (t / float(max_iters - 1))

            current_gbest_score_iter = self.gbest_score # Track best score within this iteration

            for p in range(self.num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                # Update velocity
                vel = ( w_inertia * self.velocities[p]
                        + self.c1 * r1 * (self.pbest_pos[p] - self.particles[p])
                        + self.c2 * r2 * (self.gbest_pos - self.particles[p]) )
                # Clamp velocity? Optional but can prevent explosion
                # max_vel = 1.0 # Example max velocity component
                # self.velocities[p] = np.clip(vel, -max_vel, max_vel)
                self.velocities[p] = vel


                # Update position
                self.particles[p] += self.velocities[p]
                self._clamp(self.particles[p]) # Ensure position is valid

                # Evaluate new position
                sc = self._penalty(self.particles[p])

                # Update personal best
                if sc < self.pbest_score[p]:
                    self.pbest_score[p] = sc
                    self.pbest_pos[p] = self.particles[p].copy()

                    # --- PERBAIKAN UTAMA DISINI ---
                    # Update global best
                    if sc < self.gbest_score:
                        self.gbest_score = sc
                        # Assign the entire position vector of the current particle
                        self.gbest_pos = self.particles[p].copy()
                    # --- Akhir Perbaikan ---


            # Early stopping check (compare with the best found IN THIS iteration cycle)
            # Use current_gbest_score_iter for comparison within the outer loop logic
            if self.gbest_score < best_iter_score - 1e-5: # Use a slightly larger tolerance
                best_iter_score = self.gbest_score
                no_improvement_iters = 0
            else:
                no_improvement_iters += 1

            if no_improvement_iters >= patience:
                # print(f"  PackingPSO early stopping iter {t+1}") # Debugging
                break
            # Optimization: if penalty is effectively zero, stop early
            if self.gbest_score < 1e-6:
                # print(f"  PackingPSO found near-zero penalty solution iter {t+1}") # Debugging
                break

        # Extract final layout
        layout = []
        final_penalty = float('inf')
        # Ensure gbest_pos is valid before using it
        if self.gbest_pos is not None and isinstance(self.gbest_pos, np.ndarray):
            # Re-calculate penalty for the absolute best position found
            final_penalty = self._penalty(self.gbest_pos)
            pos = self.gbest_pos
            for i in range(self.n):
                 # Protect against potential floating point issues in index
                ori_idx_float = pos[4*i+3]
                ori_idx = int(np.clip(ori_idx_float, 0, self.K - 1)) # Clip and convert safely

                original_dims = self.items[i]['dims']
                w_rot, d_rot, h_rot = self._get_rotated_dims(original_dims, ori_idx)
                layout.append({
                    'id': self.items[i]['id'],
                    'name': self.items[i]['name'],
                    'x': pos[4*i], 'y': pos[4*i+1], 'z': pos[4*i+2],
                    'w': w_rot, 'd': d_rot, 'h': h_rot,
                    'orientation': ori_idx
                })
        else:
             # Handle case where gbest_pos might still be invalid if optimization failed early
             print(f"Warning: PackingPSO optimize finished but gbest_pos is invalid. Using best pbest found.")
             # Fallback: find the best pbest as the result
             if self.pbest_score: # Check if pbest_score is not empty
                best_p_idx = np.argmin(self.pbest_score)
                final_penalty = self.pbest_score[best_p_idx]
                if final_penalty < float('inf'): # Ensure we found some valid pbest
                    pos = self.pbest_pos[best_p_idx]
                    # Reconstruct layout from best pbest
                    for i in range(self.n):
                        ori_idx_float = pos[4*i+3]
                        ori_idx = int(np.clip(ori_idx_float, 0, self.K - 1))
                        original_dims = self.items[i]['dims']
                        w_rot, d_rot, h_rot = self._get_rotated_dims(original_dims, ori_idx)
                        layout.append({
                            'id': self.items[i]['id'], 'name': self.items[i]['name'],
                            'x': pos[4*i], 'y': pos[4*i+1], 'z': pos[4*i+2],
                            'w': w_rot, 'd': d_rot, 'h': h_rot, 'orientation': ori_idx
                        })
                else:
                     layout = [] # No valid pbest found either
                     final_penalty = float('inf')

             else: # No pbest_score either (e.g., num_particles = 0)
                layout = []
                final_penalty = float('inf')


        # Return the layout and the final penalty score associated with that layout
        return layout, final_penalty
# --- (Load Distance Data - unchanged) ---
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


# --- (Data items, n_trucks, truck constraints, city_coords - unchanged) ---
# (Keep your items data here)
items = [
    {"id": "Item1", "name": "TV", "weight": 120, "dims": (40, 50, 30), "city": "Jakarta"},
    {"id": "Item2", "name": "Kulkas", "weight": 300, "dims": (70, 60, 90), "city": "Bandung"},
    {"id": "Item3", "name": "AC", "weight": 250, "dims": (80, 50, 60), "city": "Semarang"},
    {"id": "Item4", "name": "Buku", "weight": 50, "dims": (30, 30, 20), "city": "Jakarta"},
    {"id": "Item5", "name": "Sofa", "weight": 500, "dims": (150, 80, 100), "city": "Yogyakarta"},
    {"id": "Item6", "name": "Meja", "weight": 150, "dims": (120, 100, 40), "city": "Semarang"},
    {"id": "Item7", "name": "Ranjang", "weight": 400, "dims": (200, 160, 50), "city": "Malang"}, # Oversized width? Check truck dims
    {"id": "Item8", "name": "Kipas Angin", "weight": 30, "dims": (20, 20, 40), "city": "Bandung"},
    {"id": "Item9",  "name": "WashingMachine","weight":350, "dims": (60,60,85), "city": "Jakarta"},
    {"id": "Item10", "name": "Bookshelf", "weight":100, "dims": (80,30,180), "city": "Surabaya"}, # Oversized height? Check truck dims
    {"id": "Item11", "name": "Mattress", "weight":200, "dims": (200,90,30), "city": "Bandung"},
    {"id": "Item12", "name": "Wardrobe", "weight":450, "dims": (100,60,200), "city": "Yogyakarta"}, # Oversized height? Check truck dims
    {"id": "Item13", "name": "DiningTable", "weight":250, "dims": (160,90,75), "city": "Semarang"},
    {"id": "Item14", "name": "DeskLamp", "weight":10,  "dims": (15,15,40), "city": "Malang"},
    {"id": "Item15", "name": "Microwave", "weight":40,  "dims": (50,40,35), "city": "Jakarta"},
    {"id": "Item16", "name": "Printer", "weight":25,  "dims": (45,40,30), "city": "Surabaya"},
    {"id": "Item17", "name": "FloorLamp", "weight":20,  "dims": (30,30,160), "city": "Bandung"}, # Oversized height? Check truck dims
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

# --- (Pre-process items - unchanged, make sure oversized check is correct) ---
# --- (Pre-process items - MODIFIED oversized check logic) ---
threshold_small = 50*50*50
threshold_medium = 100*100*100

def get_dimension_category(l, w, h):
    volume = l * w * h
    if volume < threshold_small: return "Kecil", 50
    elif volume < threshold_medium: return "Sedang", 75
    else: return "Besar", 100

print("Checking item dimensions against truck...")
for item in items:
    l, w, h = item["dims"]
    cat, factor = get_dimension_category(l, w, h)
    item["dim_category"] = cat
    item["cat_factor"] = factor

    # --- PERBAIKAN LOGIKA is_oversized ---
    # Periksa apakah ADA orientasi (dari 6 kemungkinan) yang dimensinya MUAT di dalam truk
    can_fit_any_orientation = False
    for i in range(6): # Iterasi melalui setiap index orientasi (0 sampai 5)
        # Dapatkan dimensi (p) untuk orientasi ke-i
        p = PackingPSO._get_rotated_dims((l, w, h), i)
        # Periksa apakah dimensi p muat dalam batas truk
        if p[0] <= truck_max_length and p[1] <= truck_max_width and p[2] <= truck_max_height:
            can_fit_any_orientation = True # Jika ketemu satu orientasi yang muat, set flag True
            break # Tidak perlu cek orientasi lain, keluar dari loop

    # Item dianggap oversized jika TIDAK ADA satupun orientasi yang muat
    item["is_oversized"] = not can_fit_any_orientation
    # --- Akhir Perbaikan ---

    if item["is_oversized"]:
        print(f"  -> Warning: Item {item['id']} ({item['name']}) dims {item['dims']} IS OVERSIZED for truck {truck_max_length}x{truck_max_width}x{truck_max_height} in all 6 orientations.")
print("Dimension check complete.")

# --- (Lanjutan kode Anda setelah blok ini...) ---
# --- (Distance Functions - unchanged) ---
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
        return float('inf')

@st.cache_data(show_spinner=False)
def route_distance(cities):
    """Nearest-neighbor heuristic for route distance from Surabaya -> cities -> Surabaya."""
    if not cities: return 0.0
    unique_cities = list(set(cities))
    if not unique_cities: return 0.0

    current = "Surabaya"
    unvisited = unique_cities[:]
    total_dist = 0.0
    # route_path = ["Surabaya"] # Keep path if needed for debugging

    while unvisited:
        nearest = min(unvisited, key=lambda c: distance(current, c))
        total_dist += distance(current, nearest)
        current = nearest
        unvisited.remove(nearest)
        # route_path.append(current)

    total_dist += distance(current, "Surabaya") # Return to origin
    # route_path.append("Surabaya")
    return total_dist

###############################
# Fitness Function (Outer PSO) - MODIFIED PENALTIES
###############################
def compute_fitness(assignment):
    """
    Evaluates the quality of an assignment of items to trucks.
    Returns: (total_revenue - total_fuel_cost) - total_packing_penalty - CONSTRAINT_PENALTIES
    Constraint penalties (overweight, oversized) are now much harsher.
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
        if truck_idx == 0: # Item not assigned (could be oversized or intentionally unassigned)
            continue
        # Basic validation, although decode should handle this
        if not (1 <= truck_idx <= n_trucks):
            fitness_penalty += 1e9 # Penalize invalid truck indices heavily
            continue

        item = items[idx]

        # Constraint Check: Item Oversized (Should already be handled by decode)
        if item["is_oversized"]:
            # This indicates a potential flaw in decode or assignment logic if it happens
            print(f"ERROR: Oversized item {item['id']} assigned to truck {truck_idx}!")
            fitness_penalty += 1e9 # Extremely heavy penalty for assigning oversized
            continue # Do not assign oversized items

        # Add item to truck
        truck_details[truck_idx]["items"].append(item)
        truck_details[truck_idx]["cities"].add(item["city"])
        truck_details[truck_idx]["weight"] += item["weight"]

        # Calculate revenue for this item
        dist = distance("Surabaya", item["city"])
        rev = item["weight"] * dist * item["cat_factor"]
        truck_details[truck_idx]["revenue"] += rev
        total_revenue += rev

    # 2. Calculate Costs and Penalties per Truck
    FUEL_PRICE_PER_L = 9000
    TRUCK_CONSUMPTION_KM_L = 4
    cost_per_km = FUEL_PRICE_PER_L / TRUCK_CONSUMPTION_KM_L
    total_cost = 0.0

    truck_dims = (truck_max_length, truck_max_width, truck_max_height)

    for t in range(1, n_trucks + 1):
        info = truck_details[t]
        truck_items = info["items"]

        if not truck_items:
            continue # Skip empty trucks

        # --- MODIFIED: HARSH Overweight Penalty ---
        if info["weight"] > truck_max_weight:
            overweight = info["weight"] - truck_max_weight
            # Apply a massive constant penalty AND a penalty proportional to the square of the overweight amount
            fitness_penalty += OVERWEIGHT_CONSTANT_PENALTY
            fitness_penalty += OVERWEIGHT_FACTOR_PENALTY * (overweight ** 2)
            # Optional: Add message for debugging PSO behaviour
            # print(f"      Truck {t} OVERWEIGHT by {overweight} kg! Applying penalty.")
        # --- End Overweight Penalty Modification ---

        # Calculate Fuel Cost for this truck's route
        if info["cities"]:
            dist = route_distance(list(info["cities"]))
            total_cost += cost_per_km * dist

        # Packing Feasibility and Penalty
        if not quick_feasible(truck_dims, truck_items):
            # Quick check fails: assign large penalty, avoid detailed packing
            total_packing_penalty += 10000.0 + 100 * len(truck_items) # Large penalty
        else:
            # Prepare items for caching (needs hashable representation)
            # Include weight in tuple if PackingPSO needs it later (doesn't currently, but good practice)
            items_tuple = tuple(sorted((it["id"], it["name"], it["dims"], it["weight"]) for it in truck_items))
            packing_pen = packing_penalty_cache(truck_dims, items_tuple)
            # Cap the packing penalty contribution to avoid huge values masking other penalties
            total_packing_penalty += min(packing_pen, 50000) # Cap packing penalty contribution


    # Final Fitness Calculation
    profit = total_revenue - total_cost
    # Subtract ALL penalties from the profit. Because we MAXIMIZE fitness.
    fitness = profit - total_packing_penalty - fitness_penalty

    # We want to MAXIMIZE fitness. PSO libraries often minimize, but our loop uses > comparison.
    return fitness


def decode_position(position):
    """Converts continuous PSO particle position to discrete truck assignments."""
    assignment = []
    for i, val in enumerate(position):
        item = items[i]
        if item["is_oversized"]:
            assignment.append(0) # Cannot be assigned
        else:
            # Map continuous value to discrete truck index [1, n_trucks]
            # Clamp value between 0.51 and n_trucks + 0.49 ensures rounding maps correctly
            clamped_val = max(0.51, min(n_trucks + 0.49, val))
            assigned_truck = int(round(clamped_val))

            # Re-check bounds just in case, safety measure
            assigned_truck = max(1, min(n_trucks, assigned_truck))
            assignment.append(assigned_truck)

    return assignment

# --- (Outer PSO Initialization and Iteration - unchanged, but will now use modified fitness) ---
num_particles_assign = 30 # Particles for assignment PSO
max_iter_assign = 500     # Iterations for assignment PSO
patience_assign = 50      # Early stopping patience

# PSO parameters for assignment
assign_w_max = 0.9
assign_w_min = 0.4
assign_c1 = 2.0
assign_c2 = 2.0

improvement_threshold = 10.0 # Minimum improvement to reset patience counter (adjust based on fitness scale)
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
    position = np.zeros(len(items))
    velocity = np.zeros(len(items))
    for i in range(len(items)):
        if items[i]["is_oversized"]:
            position[i] = 0 # Fixed to unassigned
            velocity[i] = 0
        else:
            # Start with random assignment 1 to n_trucks
            position[i] = random.uniform(1, n_trucks + 1e-9) # Range slightly less than n_trucks+1
            velocity[i] = random.uniform(-(n_trucks/2.0), n_trucks/2.0)

    particles_assign.append(position.copy()) # Use copy
    velocities_assign.append(velocity.copy()) # Use copy

    # Evaluate initial position
    assignment = decode_position(position)
    fit = compute_fitness(assignment)

    pbest_positions_assign.append(position.copy())
    pbest_fitness_assign.append(fit)

    # Update global best
    if fit > gbest_fitness_assign:
        gbest_fitness_assign = fit
        gbest_position_assign = position.copy()
print(f"Initial Global Best Fitness: {gbest_fitness_assign:,.0f}") # Format for readability

# --- Assignment PSO Loop ---
fitness_history_assign = []
print(f"Running Assignment PSO for {max_iter_assign} iterations...")
progress_bar = st.progress(0) # Add progress bar
status_text = st.empty()

for it in range(1, max_iter_assign + 1):
    w_inertia = assign_w_max - (assign_w_max - assign_w_min) * (it / max_iter_assign)

    for i in range(num_particles_assign):
        # Update velocity and position only for assignable items
        r1 = np.random.rand(len(items))
        r2 = np.random.rand(len(items))

        current_particle_pos = particles_assign[i] # Reference, not copy
        current_particle_vel = velocities_assign[i] # Reference, not copy

        # Update velocity and position (vectorized)
        is_assignable = np.array([not item["is_oversized"] for item in items])

        new_vel = (w_inertia * current_particle_vel +
                   assign_c1 * r1 * (pbest_positions_assign[i] - current_particle_pos) +
                   assign_c2 * r2 * (gbest_position_assign - current_particle_pos))

        # Apply velocity update only to assignable items
        velocities_assign[i][is_assignable] = new_vel[is_assignable]

        # Update position
        particles_assign[i] += velocities_assign[i]

        # Clamp position to valid range only for assignable items
        particles_assign[i][is_assignable] = np.clip(particles_assign[i][is_assignable], 0.51, n_trucks + 0.49)

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
                # No need to reset improvement count here, handled below

    # Track fitness history and check early stopping
    fitness_history_assign.append(gbest_fitness_assign)
    improvement = gbest_fitness_assign - prev_gbest_assign

    status_text.text(f"Iter {it}/{max_iter_assign}, Best Fitness: {gbest_fitness_assign:,.0f}")
    progress_bar.progress(it / max_iter_assign)

    # Check for improvement to reset patience counter
    if improvement >= improvement_threshold:
        no_improvement_count = 0
        prev_gbest_assign = gbest_fitness_assign # Update previous best only on significant improvement
    else:
        # Only increment counter if improvement is below threshold AND after some initial iterations
        if it > patience_assign :
             no_improvement_count += 1

    # Check if patience is exceeded
    if no_improvement_count >= patience_assign:
        print(f"\nEarly stopping at iteration {it} due to no significant improvement for {patience_assign} iterations.")
        status_text.text(f"Early stopping at iteration {it}. Final Best Fitness: {gbest_fitness_assign:,.0f}")
        break

progress_bar.empty() # Remove progress bar after completion
print(f"\nAssignment PSO finished. Final Best Fitness: {gbest_fitness_assign:,.0f}")
best_assignment = decode_position(gbest_position_assign)

# Clear cache before final packing run if desired
# packing_penalty_cache.cache_clear()

# --- (Get Truck Info - unchanged, but added check for overweight display) ---
def get_truck_info(assignment):
    truck_info = {
        t: {"items": [], "weight": 0, "volume": 0, "cities": [], "revenue": 0.0, "is_overweight": False} # Add overweight flag
        for t in range(1, n_trucks + 1)
    }
    assigned_item_details = []

    for i, truck_idx in enumerate(assignment):
        item = items[i]
        if truck_idx == 0:
             assigned_item_details.append({**item, "assigned_truck": "Unassigned/Oversized"})
             continue

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

    # Calculate route distance, cost, profit, and check overweight AFTER assignments
    FUEL_PRICE_PER_L = 9000
    TRUCK_CONSUMPTION_KM_L = 4
    cost_per_km = FUEL_PRICE_PER_L / TRUCK_CONSUMPTION_KM_L
    for t in range(1, n_trucks + 1):
        route_dist = route_distance(truck_info[t]["cities"])
        truck_info[t]["route_distance"] = route_dist
        truck_info[t]["fuel_cost"] = cost_per_km * route_dist
        truck_info[t]["profit"] = truck_info[t]["revenue"] - truck_info[t]["fuel_cost"]
        # Check final weight status
        if truck_info[t]["weight"] > truck_max_weight:
            truck_info[t]["is_overweight"] = True
            print(f"WARNING: Final assignment resulted in Truck {t} being OVERWEIGHT ({truck_info[t]['weight']} kg). PSO penalty might need adjustment.")

    return truck_info, pd.DataFrame(assigned_item_details)

final_truck_info, assigned_items_df = get_truck_info(best_assignment)

# --- (Route/Path Functions for Visualization - unchanged) ---
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
    # original_key = (city_a, city_b) # To check direction

    path = []
    if key in polygons and len(polygons.get(key, [])) > 1: # Ensure polygon has at least 2 points
        path = polygons[key]
        # Simple direction check: is the first point closer to city_a or city_b?
        coord_a = city_coords[city_a]
        coord_b = city_coords[city_b]
        path_start = (path[0][1], path[0][0]) # lat, lon from stored lng, lat

        dist_start_a = haversine(coord_a, path_start)
        dist_start_b = haversine(coord_b, path_start)

        if dist_start_b < dist_start_a: # If path starts closer to city_b, reverse it
             path = path[::-1]
    # Fallback to straight line if no polygon data or invalid polygon
    if not path or len(path) < 2:
        start = [city_coords[city_a][1], city_coords[city_a][0]] # lon, lat
        end = [city_coords[city_b][1], city_coords[city_b][0]]   # lon, lat
        path = [start, end]
    return path

def get_full_route_path(route_sequence):
    """Builds the complete coordinate path for a route sequence."""
    full_path = []
    if not route_sequence or len(route_sequence) < 2:
        return [] # Need at least two points for a path

    # Add starting point explicitly if needed, though extend should handle it
    # full_path.append([city_coords[route_sequence[0]][1], city_coords[route_sequence[0]][0]])

    for i in range(len(route_sequence) - 1):
        city_a = route_sequence[i]
        city_b = route_sequence[i+1]
        segment = get_segment_path(city_a, city_b)

        # Add segment, handle potential duplicate points at junctions carefully
        if not full_path:
            full_path.extend(segment)
        elif segment:
            # Check if last point of full_path is close to first point of segment
            if (abs(full_path[-1][0] - segment[0][0]) < 1e-6 and
                abs(full_path[-1][1] - segment[0][1]) < 1e-6):
                full_path.extend(segment[1:]) # Add from second point onwards
            else:
                 # Unusual case: gap or different start point? Add whole segment.
                 # print(f"Warning: Gap detected or segment start mismatch between {city_a} and {city_b}. Appending full segment.")
                 full_path.extend(segment)

    return full_path

# --- (Prepare Route Data for Pydeck - unchanged) ---
routes_data_pydeck = []
colors = [
    [255, 0, 0, 200],   # Red
    [0, 255, 0, 200],   # Green
    [0, 0, 255, 200],   # Blue
    [255, 165, 0, 200], # Orange
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
            display_route = " → ".join([c for c in route_seq if c != "Surabaya"])
            if not display_route : display_route = "Base Only"

            routes_data_pydeck.append({
                "truck": f"Truk {t}",
                "path": full_path,
                "color": colors[(t - 1) % len(colors)],
                "route_info": display_route,
                "distance_km": info.get("route_distance", 0)
            })
        else:
             print(f"Warning: Could not generate path for Truck {t} with cities {info['cities']}")

# --- (Final Packing and Visualization Function - slightly modified packer call) ---
final_layouts = {}
print("Running final packing optimization for visualization...")
packing_progress = st.progress(0)
packing_status = st.empty()

truck_dims_tuple = (truck_max_length, truck_max_width, truck_max_height) # Define once

for idx, t in enumerate(range(1, n_trucks + 1)):
    items_for_truck = final_truck_info[t]["items"]
    packing_status.text(f"Optimizing packing for Truck {t} ({len(items_for_truck)} items)...")

    if not items_for_truck:
        final_layouts[t] = ([], 0) # Empty layout, zero penalty
        packing_progress.progress((idx + 1) / n_trucks)
        continue

    # Prepare items tuple for caching (if cache is used)
    items_tuple = tuple(sorted((it["id"], it["name"], it["dims"], it["weight"]) for it in items_for_truck))

    best_penalty_viz = float('inf')
    best_layout_viz = None
    num_packing_attempts = 3
    packing_iters_viz = 300 # More iterations for final visualization

    for attempt in range(num_packing_attempts):
        seed = 42 + attempt * 101 # Different seed
        current_np_state = np.random.get_state()
        np.random.seed(seed)

        # Use more particles for final packing attempt
        packer = PackingPSO(truck_dims_tuple, items_for_truck, num_particles=50) # Increased particles
        layout, penalty = packer.optimize(max_iters=packing_iters_viz)

        np.random.set_state(current_np_state)

        if penalty < best_penalty_viz:
            best_penalty_viz = penalty
            best_layout_viz = layout

        if best_penalty_viz < 1e-6: # Stop if perfect packing found
            break

    print(f"  Truck {t} - Best Final Packing Penalty: {best_penalty_viz:.4f}")
    final_layouts[t] = (best_layout_viz, best_penalty_viz)
    packing_progress.progress((idx + 1) / n_trucks)

packing_status.text("Final packing optimizations complete.")
packing_progress.empty()

# --- (Visualization Function `create_truck_figure` - unchanged) ---
def create_truck_figure(truck_dims, packed_items):
    """Creates a Plotly 3D figure visualizing packed items in a truck outline."""
    L, W, H = truck_dims
    fig = go.Figure()

    if not packed_items:
        pass

    color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                     "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    for idx, item in enumerate(packed_items):
        x, y, z = item['x'], item['y'], item['z']
        w_rot, d_rot, h_rot = item['w'], item['d'], item['h']

        x_verts = [x, x, x + w_rot, x + w_rot, x, x, x + w_rot, x + w_rot]
        y_verts = [y, y + d_rot, y + d_rot, y, y, y + d_rot, y + d_rot, y]
        z_verts = [z, z, z, z, z + h_rot, z + h_rot, z + h_rot, z + h_rot]

        faces_i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 1]
        faces_j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 2]
        faces_k = [0, 7, 2, 3, 6, 7, 1, 5, 4, 5, 7, 6]

        item_color = color_palette[idx % len(color_palette)]

        fig.add_trace(go.Mesh3d(
            x=x_verts, y=y_verts, z=z_verts,
            i=faces_i, j=faces_j, k=faces_k,
            color=item_color, opacity=0.8,
            name=item['name'],
            hovertext=f"{item['name']}<br>Dims: {w_rot:.1f}x{d_rot:.1f}x{h_rot:.1f}<br>Pos: ({x:.1f},{y:.1f},{z:.1f})<br>Ori: {item.get('orientation', 'N/A')}",
            hoverinfo="text"
        ))

    # Add truck outline (wireframe)
    corners = [(0,0,0), (L,0,0), (L,W,0), (0,W,0), (0,0,H), (L,0,H), (L,W,H), (0,W,H)]
    edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4),
             (0,4), (1,5), (2,6), (3,7)]
    edge_x, edge_y, edge_z = [], [], []
    for (u, v) in edges:
        edge_x += [corners[u][0], corners[v][0], None]
        edge_y += [corners[u][1], corners[v][1], None]
        edge_z += [corners[u][2], corners[v][2], None]

    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines', line=dict(color='black', width=2),
        hoverinfo='none', showlegend=False
    ))

    # Configure scene layout
    fig.update_layout(
        # title_text=f"Truck Load ({L}x{W}x{H})", # Title added in Streamlit layout
        scene=dict(
            xaxis=dict(range=[0, L], title='Length (X)', backgroundcolor="rgb(230, 230, 230)"),
            yaxis=dict(range=[0, W], title='Width (Y)', backgroundcolor="rgb(230, 230, 230)"),
            zaxis=dict(range=[0, H], title='Height (Z)', backgroundcolor="rgb(230, 230, 230)"),
            aspectratio=dict(x=1, y=W/L, z=H/L), # Adjust aspect ratio based on dims
            aspectmode='manual',
            camera_eye=dict(x=1.5, y=1.5, z=0.8) # Adjust initial camera angle if needed
        ),
        margin=dict(l=10, r=10, t=10, b=10), # Reduced top margin
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="right", x=1) # Legend at bottom
    )
    return fig

# --- (Streamlit UI Layout - modified to show overweight warning) ---
st.title("Optimasi Penugasan dan Pemuatan Truk dengan PSO")

# --- Row 1: Summary Metrics ---
col1, col2, col3 = st.columns(3)
total_profit_final = sum(info['profit'] for info in final_truck_info.values() if not info['is_overweight']) # Exclude overweight truck profit? Or show total?
total_revenue_final = sum(info['revenue'] for info in final_truck_info.values())
total_cost_final = sum(info['fuel_cost'] for info in final_truck_info.values())
num_overweight = sum(1 for info in final_truck_info.values() if info['is_overweight'])

col1.metric("Total Estimasi Profit*", f"Rp {total_profit_final:,.0f}")
col2.metric("Total Revenue", f"Rp {total_revenue_final:,.0f}")
col3.metric("Total Fuel Cost", f"Rp {total_cost_final:,.0f}")
st.caption("*Profit = Revenue - Fuel Cost. Penalti packing/overweight tidak mengurangi metrik ini secara langsung. Truk overweight mungkin terindikasi.")
if num_overweight > 0:
    st.warning(f"🚨 PERHATIAN: {num_overweight} truk terdeteksi MELEBIHI BATAS BERAT MAKSIMUM! Solusi ini mungkin tidak valid.")


# --- Row 2: Map and Fitness Chart ---
col1_map, col2_chart = st.columns([3, 2]) # Adjusted ratio

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
            get_radius=7000, # Radius in meters
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
            map_style='mapbox://styles/mapbox/light-v9',
            tooltip=tooltip
        ))
    else:
        st.write("Tidak ada rute untuk ditampilkan.")

with col2_chart:
    st.subheader("Fitness (Assignment PSO)")
    if fitness_history_assign:
        # st.line_chart(pd.DataFrame({'Fitness': fitness_history_assign}), height=350)
        fig_fitness = go.Figure()
        fig_fitness.add_trace(go.Scatter(y=fitness_history_assign, mode='lines', name='Best Fitness'))
        fig_fitness.update_layout(
            title="Perkembangan Fitness Terbaik Assignment PSO",
            xaxis_title="Iterasi",
            yaxis_title="Fitness Value",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_fitness, use_container_width=True)

    else:
        st.write("Data fitness tidak tersedia.")

# --- Row 3: Assignment Details ---
st.subheader("Detail Penugasan Item ke Truk")
st.dataframe(assigned_items_df[['id', 'name', 'weight', 'dims', 'city', 'dim_category', 'assigned_truck']])


# --- Row 4 onwards: Truck Details and Visualization ---
st.subheader("Detail Muatan dan Visualisasi per Truk")

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

                # Display Overweight Warning FIRST if applicable
                if info["is_overweight"]:
                     st.error(f"⛔️ OVERWEIGHT: {info['weight']} / {truck_max_weight} kg")
                else:
                     st.metric("Total Berat", f"{info['weight']} / {truck_max_weight} kg",
                               delta=f"{truck_max_weight - info['weight']:.0f} kg Sisa", delta_color="normal")

                if not info["items"]:
                    st.write("Tidak ada barang yang diangkut.")
                else:
                    # Display summary table for the truck
                    df_truck = pd.DataFrame(info["items"])[["id", "name", "weight", "dims", "city"]]
                    # st.dataframe(df_truck, height=150) # Limit height

                    st.metric("Estimasi Profit Truk", f"Rp {info['profit']:,.0f}")
                    st.write(f"**Kota Tujuan:** {', '.join(info['cities'])}")
                    st.write(f"**Jarak Rute:** {info['route_distance']:.1f} km")

                    # Display Packing Visualization
                    if layout is None and info["items"]: # If items exist but layout failed
                         st.error(f"⚠️ Truk {truck_idx}: Packing gagal menghasilkan layout.")
                    elif penalty > 1e-3 and info["items"]: # High penalty
                         st.warning(f"⚠️ Truk {truck_idx}: Packing belum optimal (penalty={penalty:.4f}). Visualisasi mungkin tumpang tindih atau tidak stabil.")
                         # Show the layout anyway
                         fig = create_truck_figure(truck_dims_tuple, layout)
                         st.plotly_chart(fig, use_container_width=True)
                    elif info["items"]: # Good packing
                         st.success(f"Truk {truck_idx}: Muatan terpack (penalty={penalty:.4f}).")
                         fig = create_truck_figure(truck_dims_tuple, layout)
                         st.plotly_chart(fig, use_container_width=True)
                    # else: # No items case is handled above

                st.markdown("---") # Separator between trucks
                truck_idx += 1
        else:
            pass # Empty column

print("\nStreamlit App Ready.")