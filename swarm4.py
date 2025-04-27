# -*- coding: utf-8 -*-
import streamlit as st
import math
import random
import pydeck as pdk
import pandas as pd
import json
import os
import numpy as np
import itertools
from itertools import combinations, permutations
import plotly.graph_objects as go
from functools import lru_cache

# --- Konfigurasi Halaman Streamlit (HARUS JADI PERINTAH st PERTAMA) ---
st.set_page_config(layout="wide", page_title="Optimasi Muatan Truk PSO")
# --------------------------------------------------------------------

# --- Seed untuk Reproducibility ---
random.seed(30)
np.random.seed(30)

# --- Constants ---
OVERLAP_PENALTY_FACTOR = 2.0
OUTBOUND_PENALTY_FACTOR = 2.0
# --- PENALTI KENDALA ---
OVERWEIGHT_CONSTANT_PENALTY = 1e9 # Penalti konstan sangat besar jika overweight (lebih besar dari profit manapun)
OVERWEIGHT_FACTOR_PENALTY = 10000.0 # Penalti proporsional dg kuadrat kelebihan berat
UNSUPPORTED_PENALTY_BASE = 10000.0  # Penalti dasar jika item tidak didukung
UNSUPPORTED_PENALTY_HEIGHT_FACTOR = 200.0 # Penalti tambahan berdasarkan tinggi item
# --- Penalti Prioritas Item Besar di Bawah ---
HEIGHT_VOLUME_PENALTY_FACTOR = 0.005 # Faktor penalti: Volume * Tinggi Posisi Z (PERLU TUNING)
# ---------------------------------------------

# --- Optimasi: Parameter untuk Packing Cepat (Estimasi Fitness di PSO Luar) ---
PACKING_CACHE_MAX_ITERS = 25     # Iterasi packing dikurangi drastis untuk cache
PACKING_CACHE_NUM_PARTICLES = 15 # Partikel packing dikurangi drastis untuk cache
# --------------------------------------------------------------------

# --- Parameter untuk Packing Final (Visualisasi Kualitas Tinggi) ---
FINAL_PACKING_MAX_ITERS = 300    # Iterasi lebih banyak untuk hasil akhir
FINAL_PACKING_NUM_PARTICLES = 50 # Partikel lebih banyak untuk hasil akhir
# --------------------------------------------------------------------

# --- Parameter PSO (Umum) ---
W_MAX = 0.9  # Inertia weight max
W_MIN = 0.4  # Inertia weight min
# -----------------

def quick_feasible(truck_dims, items):
    """Pengecekan cepat kelayakan packing berdasarkan volume total dan luas alas minimum."""
    L, W, H = truck_dims
    if not items: return True # Truk kosong pasti feasible

    # 1. Cek Volume Total
    total_vol = sum(it['dims'][0] * it['dims'][1] * it['dims'][2] for it in items)
    # Beri sedikit toleransi untuk floating point error? Atau tetap ketat?
    if total_vol > L * W * H:
         return False # Jelas tidak muat jika volume total > volume truk

    # 2. Cek Luas Alas Minimum Total (Heuristik kasar)
    min_total_base = sum(min(l*w, l*h, w*h) for l, w, h in (it['dims'] for it in items))
    if min_total_base > L*W: # Jika total luas alas minimum > luas alas truk
        return False # Kemungkinan besar tidak muat di dasar

    return True

# Cache untuk hasil packing (dengan parameter ringan)
@lru_cache(maxsize=256)
def packing_penalty_cache(truck_dims, items_tuple):
    """
    Menjalankan PackingPSO dengan parameter CEPAT (iterasi/partikel sedikit)
    untuk mendapatkan estimasi penalti packing. Hasilnya di-cache.
    """
    items_list = [ {'id':i[0], 'name':i[1], 'dims':i[2], 'weight':i[3]} for i in items_tuple ]
    if not items_list:
        return 0.0

    current_np_state = np.random.get_state()
    np.random.seed(42) # Seed konsisten untuk cache

    # --- OPTIMASI UTAMA: Gunakan parameter ringan ---
    packer = PackingPSO(truck_dims, items_list,
                        num_particles=PACKING_CACHE_NUM_PARTICLES)
    _, pen = packer.optimize(max_iters=PACKING_CACHE_MAX_ITERS)
    # --- Akhir Optimasi ---

    np.random.set_state(current_np_state)
    return pen

class PackingPSO:
    """
    Optimasi packing 3D menggunakan PSO. Mempertimbangkan 6 orientasi standar.
    Termasuk penalti support dasar dan penalti ketinggian item besar.
    """
    def __init__(self, truck_dims, items,
                 compaction_weight=1e-3,
                 num_particles=30, # Default, bisa di-override
                 c1=1.5, c2=1.5):
        self.L, self.W, self.H = truck_dims
        self.items = items
        self.n = len(items)
        if self.n == 0:
             self.dim = 0; self.particles = []; self.velocities = []
             self.pbest_pos = []; self.pbest_score = []; self.gbest_pos = None
             self.gbest_score = 0.0
             return

        self.dim = self.n * 4
        self.orientations = list(permutations([0, 1, 2]))
        self.K = 6
        self.num_particles = num_particles
        self.c1 = c1; self.c2 = c2
        self.comp_w = compaction_weight

        # Simpan volume asli untuk penalti ketinggian
        self.item_original_volumes = [it['dims'][0] * it['dims'][1] * it['dims'][2] for it in items]

        self.particles = []
        self.velocities = []
        self.pbest_pos = []
        self.pbest_score = []
        self.gbest_pos = None
        self.gbest_score = float('inf')

        # Inisialisasi partikel
        for _ in range(self.num_particles):
            pos = np.zeros(self.dim)
            vel = np.zeros(self.dim)
            for i in range(self.n):
                pos[4*i+0] = np.random.uniform(0, self.L)
                pos[4*i+1] = np.random.uniform(0, self.W)
                pos[4*i+2] = np.random.uniform(0, self.H) # Posisi Z awal acak
                pos[4*i+3] = np.random.uniform(0, self.K)
                vel[4*i:4*i+4] = np.random.uniform(-1, 1, 4) * 0.1

            self._clamp(pos)
            score = self._penalty(pos)
            self.particles.append(pos.copy())
            self.velocities.append(vel.copy())
            self.pbest_pos.append(pos.copy())
            self.pbest_score.append(score)
            if score < self.gbest_score:
                self.gbest_score = score
                self.gbest_pos = pos.copy()

    @staticmethod
    def _get_rotated_dims(original_dims, orientation_index):
        l, w, h = original_dims
        dims_map = [
            (l, w, h), (l, h, w), (w, l, h),
            (w, h, l), (h, l, w), (h, w, l)
        ]
        safe_index = int(np.clip(orientation_index, 0, 5))
        return dims_map[safe_index]

    def _clamp(self, pos):
        for i in range(self.n):
            pos[4*i+0] = np.clip(pos[4*i+0], 0, self.L)
            pos[4*i+1] = np.clip(pos[4*i+1], 0, self.W)
            pos[4*i+2] = np.clip(pos[4*i+2], 0, self.H)
            pos[4*i+3] = np.clip(pos[4*i+3], 0, self.K - 1e-9)

    def _penalty(self, pos):
        """Menghitung penalti: Out-of-bounds, Overlap, Support, Ketinggian Item Besar, Kompaksi."""
        if self.n == 0: return 0.0

        placement = []
        total_item_vol = 0.0
        for i in range(self.n):
            x, y, z = pos[4*i : 4*i+3]
            ori_idx = int(np.clip(pos[4*i+3], 0, self.K - 1))
            original_dims = self.items[i]['dims']
            w_rot, d_rot, h_rot = self._get_rotated_dims(original_dims, ori_idx)
            placement.append({
                'id': self.items[i]['id'], 'name': self.items[i]['name'],
                'x': x, 'y': y, 'z': z,
                'w': w_rot, 'd': d_rot, 'h': h_rot
            })
            total_item_vol += self.item_original_volumes[i] # Gunakan volume asli tersimpan

        pen = 0.0

        # 1. Penalti Out-of-bounds
        for it in placement:
            x_under = max(0, -it['x']); y_under = max(0, -it['y']); z_under = max(0, -it['z'])
            pen += OUTBOUND_PENALTY_FACTOR * (x_under*it['d']*it['h'] + y_under*it['w']*it['h'] + z_under*it['w']*it['d'])
            x_over = max(0, (it['x'] + it['w']) - self.L); y_over = max(0, (it['y'] + it['d']) - self.W); z_over = max(0, (it['z'] + it['h']) - self.H)
            pen += OUTBOUND_PENALTY_FACTOR * (x_over*it['d']*it['h'] + y_over*it['w']*it['h'] + z_over*it['w']*it['d'])

        # 2. Penalti Overlap
        for i, A in enumerate(placement):
            for j, B in enumerate(placement):
                if i >= j: continue
                ox = max(0, min(A['x'] + A['w'], B['x'] + B['w']) - max(A['x'], B['x']))
                oy = max(0, min(A['y'] + A['d'], B['y'] + B['d']) - max(A['y'], B['y']))
                oz = max(0, min(A['z'] + A['h'], B['z'] + B['h']) - max(A['z'], B['z']))
                overlap_vol = ox * oy * oz
                if overlap_vol > 1e-6:
                    pen += OVERLAP_PENALTY_FACTOR * overlap_vol

        # 3. Penalti Support (Gravitasi)
        support_tolerance = 1e-4
        for i, it in enumerate(placement):
            if it['z'] < support_tolerance: continue # Di lantai = didukung
            is_supported = False
            it_center_x = it['x'] + it['w'] / 2.0; it_center_y = it['y'] + it['d'] / 2.0
            it_base_z = it['z']
            for j, base in enumerate(placement):
                if i == j: continue
                if abs((base['z'] + base['h']) - it_base_z) < support_tolerance:
                    if (base['x'] - support_tolerance <= it_center_x <= base['x'] + base['w'] + support_tolerance and
                        base['y'] - support_tolerance <= it_center_y <= base['y'] + base['d'] + support_tolerance):
                        is_supported = True; break
            if not is_supported:
                pen += UNSUPPORTED_PENALTY_BASE + UNSUPPORTED_PENALTY_HEIGHT_FACTOR * it['h']

        # 4. Penalti Ketinggian untuk Item Besar (Volume * Z)
        for i, it in enumerate(placement):
            # Penalti = Faktor * Volume_Asli_Item * Posisi_Z_Item
            pen += HEIGHT_VOLUME_PENALTY_FACTOR * self.item_original_volumes[i] * it['z']

        # 5. Penalti Kompaksi
        if placement:
            max_x = max(it['x'] + it['w'] for it in placement); max_y = max(it['y'] + it['d'] for it in placement)
            max_z = max(it['z'] + it['h'] for it in placement)
            used_bbox_vol = max_x * max_y * max_z
            pen += self.comp_w * max(0, used_bbox_vol - total_item_vol)

        return pen

    def optimize(self, max_iters=100):
        """Menjalankan algoritma PSO packing."""
        if self.n == 0: return [], 0.0

        best_iter_score = float('inf')
        no_improvement_iters = 0
        patience = max(10, max_iters // 10)

        if self.gbest_pos is None:
            if self.pbest_pos:
                 initial_best_idx = np.argmin(self.pbest_score)
                 if self.pbest_score[initial_best_idx] < self.gbest_score:
                     self.gbest_score = self.pbest_score[initial_best_idx]
                     self.gbest_pos = self.pbest_pos[initial_best_idx].copy()
            else:
                 print("Error: Cannot optimize PackingPSO with no particles initialized.")
                 return [], float('inf')

        # --- Iterasi PSO ---
        for t in range(max_iters):
            w_inertia = W_MAX - (W_MAX - W_MIN) * (t / float(max_iters - 1))
            for p in range(self.num_particles):
                r1 = np.random.rand(self.dim); r2 = np.random.rand(self.dim)
                vel = ( w_inertia * self.velocities[p]
                        + self.c1 * r1 * (self.pbest_pos[p] - self.particles[p])
                        + self.c2 * r2 * (self.gbest_pos - self.particles[p]) )
                self.velocities[p] = vel
                self.particles[p] += self.velocities[p]
                self._clamp(self.particles[p])
                sc = self._penalty(self.particles[p])
                if sc < self.pbest_score[p]:
                    self.pbest_score[p] = sc
                    self.pbest_pos[p] = self.particles[p].copy()
                    if sc < self.gbest_score:
                        self.gbest_score = sc
                        self.gbest_pos = self.particles[p].copy()

            # Early Stopping
            if self.gbest_score < best_iter_score - 1e-5:
                best_iter_score = self.gbest_score; no_improvement_iters = 0
            else: no_improvement_iters += 1
            if no_improvement_iters >= patience: break
            if self.gbest_score < 1e-6: break
        # --- Akhir Iterasi PSO ---

        # Ekstrak layout final
        layout = []; final_penalty = float('inf')
        if self.gbest_pos is not None and isinstance(self.gbest_pos, np.ndarray):
            final_penalty = self._penalty(self.gbest_pos) # Hitung penalti final yg akurat
            pos = self.gbest_pos
            for i in range(self.n):
                ori_idx = int(np.clip(pos[4*i+3], 0, self.K - 1))
                w_rot, d_rot, h_rot = self._get_rotated_dims(self.items[i]['dims'], ori_idx)
                layout.append({
                    'id': self.items[i]['id'], 'name': self.items[i]['name'],
                    'x': pos[4*i], 'y': pos[4*i+1], 'z': pos[4*i+2],
                    'w': w_rot, 'd': d_rot, 'h': h_rot, 'orientation': ori_idx
                })
        else: # Fallback
            print(f"Warning: PackingPSO optimize finished but gbest_pos is invalid.")
            if self.pbest_score:
                 best_p_idx = np.argmin(self.pbest_score); final_penalty = self.pbest_score[best_p_idx]
                 if final_penalty < float('inf'):
                     pos = self.pbest_pos[best_p_idx]
                     for i in range(self.n): # Rekonstruksi layout
                          ori_idx = int(np.clip(pos[4*i+3], 0, self.K - 1))
                          w_rot, d_rot, h_rot = self._get_rotated_dims(self.items[i]['dims'], ori_idx)
                          layout.append({'id': self.items[i]['id'], 'name': self.items[i]['name'],'x': pos[4*i], 'y': pos[4*i+1], 'z': pos[4*i+2],'w': w_rot, 'd': d_rot, 'h': h_rot, 'orientation': ori_idx})

        return layout, final_penalty

# --- Load Data Jarak & Polygon ---
csv_filename = "city_to_city_polygon.csv"
distance_dict = {}; polygons = {}
if os.path.exists(csv_filename):
    try:
        df_distance = pd.read_csv(csv_filename)
        if all(col in df_distance.columns for col in ["CityA", "CityB", "Distance (meters)", "Polygon"]):
            for _, row in df_distance.iterrows():
                key = tuple(sorted((row["CityA"], row["CityB"])))
                distance_dict[key] = row["Distance (meters)"] / 1000.0
                try:
                    polygon_coords = json.loads(row["Polygon"])
                    if isinstance(polygon_coords, list) and all(isinstance(p, dict) and 'lng' in p and 'lat' in p for p in polygon_coords):
                         polygons[key] = [[p["lng"], p["lat"]] for p in polygon_coords]
                    elif isinstance(polygon_coords, list) and all(isinstance(p, list) and len(p)==2 for p in polygon_coords):
                         polygons[key] = polygon_coords
                    else: polygons[key] = []
                except (json.JSONDecodeError, TypeError): polygons[key] = []
        else: st.error("CSV file missing required columns.")
    except Exception as e: st.error(f"Failed to read CSV: {e}")
else: st.warning(f"Distance file '{csv_filename}' not found. Using Haversine distance.")

# --- Data Input ---
items = [
    {"id": "Item1", "name": "TV", "weight": 120, "dims": (40, 50, 30), "city": "Jakarta"},
    {"id": "Item2", "name": "Kulkas", "weight": 300, "dims": (70, 60, 90), "city": "Bandung"}, # Besar
    {"id": "Item3", "name": "AC", "weight": 250, "dims": (80, 50, 60), "city": "Semarang"},  # Sedang
    {"id": "Item4", "name": "Buku", "weight": 50, "dims": (30, 30, 20), "city": "Jakarta"}, # Kecil
    {"id": "Item5", "name": "Sofa", "weight": 500, "dims": (150, 80, 100), "city": "Yogyakarta"}, # Besar
    {"id": "Item6", "name": "Meja", "weight": 150, "dims": (120, 100, 40), "city": "Semarang"}, # Besar
    {"id": "Item7", "name": "Ranjang", "weight": 400, "dims": (200, 160, 50), "city": "Malang"}, # Oversized?
    {"id": "Item8", "name": "Kipas Angin", "weight": 30, "dims": (20, 20, 40), "city": "Bandung"}, # Kecil
    {"id": "Item9",  "name": "WashingMachine","weight":350, "dims": (60,60,85), "city": "Jakarta"}, # Sedang
    {"id": "Item10", "name": "Bookshelf", "weight":100, "dims": (80,30,180), "city": "Surabaya"}, # Tinggi
    {"id": "Item11", "name": "Mattress", "weight":200, "dims": (200,90,30), "city": "Bandung"}, # Besar
    {"id": "Item12", "name": "Wardrobe", "weight":450, "dims": (100,60,200), "city": "Yogyakarta"}, # Tinggi & Berat
    {"id": "Item13", "name": "DiningTable", "weight":250, "dims": (160,90,75), "city": "Semarang"}, # Besar
    {"id": "Item14", "name": "DeskLamp", "weight":10,  "dims": (15,15,40), "city": "Malang"}, # Kecil
    {"id": "Item15", "name": "Microwave", "weight":40,  "dims": (50,40,35), "city": "Jakarta"}, # Kecil
    {"id": "Item16", "name": "Printer", "weight":25,  "dims": (45,40,30), "city": "Surabaya"}, # Kecil
    {"id": "Item17", "name": "FloorLamp", "weight":20,  "dims": (30,30,160), "city": "Bandung"}, # Tinggi
    {"id": "Item18", "name": "AirPurifier", "weight":15,  "dims": (25,25,60), "city": "Yogyakarta"}, # Kecil
    {"id": "Item19", "name": "WaterHeater", "weight":80,  "dims": (50,50,100), "city": "Semarang"}, # Sedang
    {"id": "Item20", "name": "CoffeeTable", "weight":80,  "dims": (120,60,45), "city": "Malang"} # Sedang
]
n_trucks = 4
truck_max_weight = 1000; truck_max_length = 200
truck_max_width  = 150; truck_max_height = 150
truck_dims_tuple = (truck_max_length, truck_max_width, truck_max_height)
city_coords = {
    "Surabaya": (-7.2575, 112.7521), "Jakarta": (-6.2088, 106.8456),
    "Bandung": (-6.9175, 107.6191), "Semarang": (-6.9667, 110.4167),
    "Yogyakarta": (-7.7956, 110.3695), "Malang": (-7.9824, 112.6304)
}

# --- Pre-process Items ---
threshold_small = 50*50*50; threshold_medium = 100*100*100
def get_dimension_category(l, w, h):
    vol = l * w * h
    if vol < threshold_small: return "Kecil", 50
    elif vol < threshold_medium: return "Sedang", 75
    else: return "Besar", 100

print("Checking item dimensions against truck...")
for item in items:
    l, w, h = item["dims"]; cat, factor = get_dimension_category(l, w, h)
    item["dim_category"] = cat; item["cat_factor"] = factor
    can_fit = any(p[0] <= truck_max_length and p[1] <= truck_max_width and p[2] <= truck_max_height
                  for i in range(6) for p in [PackingPSO._get_rotated_dims((l, w, h), i)])
    item["is_oversized"] = not can_fit
    if item["is_oversized"]: print(f"  -> Warning: Item {item['id']} ({item['name']}) oversized.")
print("Dimension check complete.")

# --- Fungsi Jarak ---
@st.cache_data(show_spinner=False)
def haversine(coord1, coord2):
    lat1, lon1 = coord1; lat2, lon2 = coord2; R = 6371
    phi1, phi2 = map(math.radians, [lat1, lat2])
    d_phi = math.radians(lat2 - lat1); d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

@st.cache_data(show_spinner=False)
def distance(city1, city2):
    if city1 == city2: return 0.0
    key = tuple(sorted((city1, city2)))
    if key in distance_dict: return distance_dict[key]
    if city1 in city_coords and city2 in city_coords: return haversine(city_coords[city1], city_coords[city2])
    st.error(f"Cannot find distance between {city1} and {city2}."); return float('inf')

@st.cache_data(show_spinner=False)
def route_distance(cities):
    if not cities: return 0.0
    unique_cities = list(dict.fromkeys(cities))
    if not unique_cities: return 0.0
    current = "Surabaya"; unvisited = unique_cities[:]; total_dist = 0.0
    while unvisited:
        nearest = min(unvisited, key=lambda c: distance(current, c))
        dist_to_nearest = distance(current, nearest)
        if dist_to_nearest == float('inf'): return float('inf') # Jika ada jarak tak hingga
        total_dist += dist_to_nearest
        current = nearest; unvisited.remove(nearest)
    dist_back = distance(current, "Surabaya")
    if dist_back == float('inf'): return float('inf')
    return total_dist + dist_back

# --- Fitness Function (Outer PSO) ---
def compute_fitness(assignment):
    truck_details = {t: {"items": [], "cities": set(), "weight": 0, "revenue": 0.0} for t in range(1, n_trucks + 1)}
    total_revenue = 0.0; total_packing_penalty = 0.0; constraint_penalty = 0.0

    for idx, truck_idx in enumerate(assignment):
        if truck_idx == 0: continue
        if not (1 <= truck_idx <= n_trucks): constraint_penalty += 1e10; continue
        item = items[idx]
        if item["is_oversized"]: constraint_penalty += 1e10; continue

        truck_details[truck_idx]["items"].append(item)
        truck_details[truck_idx]["cities"].add(item["city"])
        truck_details[truck_idx]["weight"] += item["weight"]
        dist = distance("Surabaya", item["city"])
        if dist == float('inf'): constraint_penalty += 1e9; continue
        total_revenue += item["weight"] * dist * item["cat_factor"]
        truck_details[truck_idx]["revenue"] += item["weight"] * dist * item["cat_factor"] # Revenue per truk juga dihitung

    FUEL_PRICE_PER_L = 9000; TRUCK_CONSUMPTION_KM_L = 4
    cost_per_km = FUEL_PRICE_PER_L / TRUCK_CONSUMPTION_KM_L; total_cost = 0.0

    for t in range(1, n_trucks + 1):
        info = truck_details[t]; truck_items = info["items"]
        if not truck_items: continue

        current_weight = info["weight"]
        if current_weight > truck_max_weight:
            overweight = current_weight - truck_max_weight
            constraint_penalty += OVERWEIGHT_CONSTANT_PENALTY + OVERWEIGHT_FACTOR_PENALTY * (overweight ** 2)

        route_dist = route_distance(list(info["cities"]))
        if route_dist == float('inf'): constraint_penalty += 1e9; continue # Penalti jika rute tak bisa dihitung
        total_cost += cost_per_km * route_dist

        if not quick_feasible(truck_dims_tuple, truck_items):
            total_packing_penalty += 50000.0 + 100 * len(truck_items)
        else:
            items_tuple_key = tuple(sorted((it["id"], it["name"], it["dims"], it["weight"]) for it in truck_items))
            packing_pen = packing_penalty_cache(truck_dims_tuple, items_tuple_key)
            total_packing_penalty += packing_pen

    profit = total_revenue - total_cost
    fitness = profit - total_packing_penalty - constraint_penalty
    return fitness

# --- Fungsi Decode Posisi PSO Luar ---
def decode_position(position):
    assignment = np.zeros(len(items), dtype=int)
    for i, val in enumerate(position):
        if not items[i]["is_oversized"]:
            assignment[i] = max(1, min(n_trucks, int(round(np.clip(val, 0.51, n_trucks + 0.49)))))
    return assignment

# --- Outer PSO: Assignment ---
num_particles_assign = 30; max_iter_assign = 200; patience_assign = 30
assign_w_max = 0.9; assign_w_min = 0.4; assign_c1 = 2.0; assign_c2 = 2.0
improvement_threshold = 10.0; no_improvement_count = 0
particles_assign = []; velocities_assign = []; pbest_positions_assign = []
pbest_fitness_assign = [-float('inf')] * num_particles_assign # Inisialisasi pbest fitness
gbest_position_assign = None; gbest_fitness_assign = -float('inf'); prev_gbest_assign = -float('inf')

print("Initializing Assignment PSO...")
assignable_mask = np.array([not item["is_oversized"] for item in items])
for p_idx in range(num_particles_assign):
    position = np.zeros(len(items)); velocity = np.zeros(len(items))
    # Inisialisasi hanya untuk item yg bisa diassign
    position[assignable_mask] = np.random.uniform(1, n_trucks + 1e-9, size=assignable_mask.sum())
    velocity[assignable_mask] = np.random.uniform(-(n_trucks/2.0), n_trucks/2.0, size=assignable_mask.sum()) * 0.1

    particles_assign.append(position.copy())
    velocities_assign.append(velocity.copy())
    assignment = decode_position(position)
    fit = compute_fitness(assignment)
    pbest_positions_assign.append(position.copy()) # Simpan pbest pos awal
    pbest_fitness_assign[p_idx] = fit # Simpan pbest fit awal
    if fit > gbest_fitness_assign:
        gbest_fitness_assign = fit; gbest_position_assign = position.copy()
print(f"Initial Global Best Fitness: {gbest_fitness_assign:,.0f}")

# --- Loop Utama PSO Assignment ---
fitness_history_assign = []
print(f"Running Assignment PSO for max {max_iter_assign} iterations...")
progress_bar = st.progress(0); status_text = st.empty()

for it in range(1, max_iter_assign + 1):
    w_inertia = assign_w_max - (assign_w_max - assign_w_min) * (it / max_iter_assign)
    for i in range(num_particles_assign):
        r1 = np.random.rand(len(items)); r2 = np.random.rand(len(items))
        current_pos = particles_assign[i]; current_vel = velocities_assign[i]
        pbest_pos = pbest_positions_assign[i]

        # --- Update Velocity (vectorized) ---
        cognitive_comp = assign_c1 * r1 * (pbest_pos - current_pos)
        social_comp = np.zeros_like(current_pos) # Default jika gbest belum ada
        if gbest_position_assign is not None:
             social_comp = assign_c2 * r2 * (gbest_position_assign - current_pos)
        new_vel = w_inertia * current_vel + cognitive_comp + social_comp

        # Terapkan hanya pada yg assignable
        velocities_assign[i][assignable_mask] = new_vel[assignable_mask]
        # --- Update Position (vectorized) ---
        particles_assign[i][assignable_mask] += velocities_assign[i][assignable_mask]
        # --- Clamp Position (vectorized) ---
        particles_assign[i][assignable_mask] = np.clip(particles_assign[i][assignable_mask], 0.51, n_trucks + 0.49)

        # Decode & Evaluate
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

    # Tracking & Early stopping
    fitness_history_assign.append(gbest_fitness_assign)
    improvement = gbest_fitness_assign - prev_gbest_assign
    status_text.text(f"Iter {it}/{max_iter_assign}, Best Fitness: {gbest_fitness_assign:,.0f}")
    progress_bar.progress(it / max_iter_assign)
    if improvement >= improvement_threshold:
        no_improvement_count = 0; prev_gbest_assign = gbest_fitness_assign
    elif it > patience_assign: no_improvement_count += 1
    if no_improvement_count >= patience_assign:
        print(f"\nEarly stopping at iteration {it}."); status_text.text(f"Early stopping iter {it}. Final Best: {gbest_fitness_assign:,.0f}"); break

progress_bar.empty(); print(f"\nAssignment PSO finished. Final Best Fitness: {gbest_fitness_assign:,.0f}")

# --- Dapatkan Assignment Terbaik ---
if gbest_position_assign is not None: best_assignment = decode_position(gbest_position_assign)
else: # Fallback
    print("Error: gbest_position_assign is None. Using best pbest."); best_assignment = np.zeros(len(items), dtype=int)
    if pbest_fitness_assign and max(pbest_fitness_assign) > -float('inf'):
        best_pbest_idx = np.argmax(pbest_fitness_assign); best_assignment = decode_position(pbest_positions_assign[best_pbest_idx])

# --- Proses Hasil Assignment ---
def get_truck_info(assignment):
    truck_info = {t: {"items": [], "weight": 0, "volume": 0, "cities": [], "revenue": 0.0, "is_overweight": False} for t in range(1, n_trucks + 1)}
    assigned_item_details = []
    for i, truck_idx in enumerate(assignment):
        item = items[i]
        if truck_idx == 0: assigned_item_details.append({**item, "assigned_truck": "Unassigned/Oversized"}); continue
        truck_info[truck_idx]["items"].append(item); truck_info[truck_idx]["weight"] += item["weight"]
        l, w, h = item["dims"]; truck_info[truck_idx]["volume"] += (l * w * h)
        dist = distance("Surabaya", item["city"])
        if dist != float('inf'): truck_info[truck_idx]["revenue"] += item["weight"] * dist * item["cat_factor"]
        if item["city"] not in truck_info[truck_idx]["cities"]: truck_info[truck_idx]["cities"].append(item["city"])
        assigned_item_details.append({**item, "assigned_truck": f"Truk {truck_idx}"})

    FUEL_PRICE_PER_L = 9000; TRUCK_CONSUMPTION_KM_L = 4; cost_per_km = FUEL_PRICE_PER_L / TRUCK_CONSUMPTION_KM_L
    for t in range(1, n_trucks + 1):
        route_dist = route_distance(truck_info[t]["cities"])
        truck_info[t]["route_distance"] = route_dist if route_dist != float('inf') else 0
        truck_info[t]["fuel_cost"] = cost_per_km * truck_info[t]["route_distance"]
        truck_info[t]["profit"] = truck_info[t]["revenue"] - truck_info[t]["fuel_cost"]
        if truck_info[t]["weight"] > truck_max_weight:
            truck_info[t]["is_overweight"] = True; print(f"!!! FINAL WARNING: Truk {t} OVERWEIGHT ({truck_info[t]['weight']} kg) !!!")
    return truck_info, pd.DataFrame(assigned_item_details)
final_truck_info, assigned_items_df = get_truck_info(best_assignment)

# --- Fungsi Bantuan Rute ---
def get_route_sequence(cities):
    if not cities: return ["Surabaya"]; unique_cities = list(dict.fromkeys(cities))
    if not unique_cities: return ["Surabaya"]
    route = ["Surabaya"]; current = "Surabaya"; unvisited = unique_cities[:]
    while unvisited:
        nearest = min(unvisited, key=lambda c: distance(current, c)); route.append(nearest)
        current = nearest; unvisited.remove(nearest)
    route.append("Surabaya"); return route

def get_segment_path(city_a, city_b):
    if city_a == city_b: return []
    key = tuple(sorted((city_a, city_b))); path = []
    if key in polygons and isinstance(polygons.get(key), list) and len(polygons[key]) >= 2:
        path_data = polygons[key]; coord_a = city_coords[city_a]; coord_b = city_coords[city_b]
        path_start_coord = (path_data[0][1], path_data[0][0])
        dist_start_a = haversine(coord_a, path_start_coord); dist_start_b = haversine(coord_b, path_start_coord)
        path = path_data[::-1] if dist_start_b < dist_start_a - 1e-3 else path_data
    if not path: path = [[city_coords[city_a][1], city_coords[city_a][0]], [city_coords[city_b][1], city_coords[city_b][0]]]
    return path

def get_full_route_path(route_sequence):
    full_path = []
    if not route_sequence or len(route_sequence) < 2: return []
    for i in range(len(route_sequence) - 1):
        segment = get_segment_path(route_sequence[i], route_sequence[i+1])
        if not segment: continue
        if not full_path: full_path.extend(segment)
        elif abs(full_path[-1][0] - segment[0][0]) < 1e-6 and abs(full_path[-1][1] - segment[0][1]) < 1e-6: full_path.extend(segment[1:])
        else: full_path.extend(segment)
    return full_path

# --- Siapkan Data Rute Pydeck ---
routes_data_pydeck = []
colors = [[255,0,0,200],[0,180,0,200],[0,0,255,200],[255,165,0,200],[128,0,128,200],[0,200,200,200]]
print("Generating route paths for visualization...")
for t in range(1, n_trucks + 1):
    info = final_truck_info[t]
    if info["cities"]:
        route_seq = get_route_sequence(info["cities"]); full_path = get_full_route_path(route_seq)
        if full_path:
            display_route = " → ".join([c for c in route_seq if c != "Surabaya"]) or "Hanya Base"
            routes_data_pydeck.append({"truck": f"Truk {t}", "path": full_path, "color": colors[(t-1)%len(colors)], "route_info": display_route, "distance_km": info.get("route_distance", 0)})
        else: print(f"Warning: Failed to create path for Truck {t}")

# --- Packing Final Kualitas Tinggi ---
final_layouts = {}
print("Running final (high-quality) packing optimization...")
packing_progress = st.progress(0); packing_status = st.empty()
for idx, t in enumerate(range(1, n_trucks + 1)):
    items_for_truck = final_truck_info[t]["items"]
    packing_status.text(f"Optimizing packing for Truck {t} ({len(items_for_truck)} items)...")
    if not items_for_truck: final_layouts[t] = ([], 0.0); packing_progress.progress((idx + 1) / n_trucks); continue
    best_penalty_viz = float('inf'); best_layout_viz = None; num_packing_attempts = 3
    for attempt in range(num_packing_attempts):
        seed = 42 + attempt * 101; current_np_state = np.random.get_state(); np.random.seed(seed)
        packer = PackingPSO(truck_dims_tuple, items_for_truck, num_particles=FINAL_PACKING_NUM_PARTICLES)
        layout, penalty = packer.optimize(max_iters=FINAL_PACKING_MAX_ITERS)
        np.random.set_state(current_np_state)
        if penalty < best_penalty_viz: best_penalty_viz = penalty; best_layout_viz = layout
        if best_penalty_viz < 1e-5: break
    print(f"  Truck {t} - Best Final Packing Penalty: {best_penalty_viz:.4f}")
    final_layouts[t] = (best_layout_viz, best_penalty_viz); packing_progress.progress((idx + 1) / n_trucks)
packing_status.text("Final packing optimizations complete."); packing_progress.empty()

# --- Fungsi Visualisasi Plotly 3D ---
def create_truck_figure(truck_dims, packed_items):
    L, W, H = truck_dims; fig = go.Figure()
    if not packed_items: pass # Outline tetap digambar
    color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    for idx, item in enumerate(packed_items):
        x, y, z = item['x'], item['y'], item['z']; w_rot, d_rot, h_rot = item['w'], item['d'], item['h']
        x_v = [x,x,x+w_rot,x+w_rot,x,x,x+w_rot,x+w_rot]; y_v = [y,y+d_rot,y+d_rot,y,y,y+d_rot,y+d_rot,y]; z_v = [z,z,z,z,z+h_rot,z+h_rot,z+h_rot,z+h_rot]
        faces_i = [7,0,0,0,4,4,6,6,4,0,3,1]; faces_j = [3,4,1,2,5,6,5,2,0,1,6,2]; faces_k = [0,7,2,3,6,7,1,5,4,5,7,6]
        item_color = color_palette[idx % len(color_palette)]
        fig.add_trace(go.Mesh3d(x=x_v, y=y_v, z=z_v, i=faces_i, j=faces_j, k=faces_k, color=item_color, opacity=0.85, name=item['name'],
            hovertext=f"<b>{item['name']}</b><br>ID: {item['id']}<br>Dims: {w_rot:.1f}x{d_rot:.1f}x{h_rot:.1f}<br>Pos: ({x:.1f},{y:.1f},{z:.1f})<br>Ori: {item.get('orientation', 'N/A')}", hoverinfo="text"))
    corners = [(0,0,0),(L,0,0),(L,W,0),(0,W,0),(0,0,H),(L,0,H),(L,W,H),(0,W,H)]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    edge_x, edge_y, edge_z = [], [], []
    for u, v in edges: edge_x.extend([corners[u][0],corners[v][0],None]); edge_y.extend([corners[u][1],corners[v][1],None]); edge_z.extend([corners[u][2],corners[v][2],None])
    fig.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(color='black', width=3), hoverinfo='none', showlegend=False))
    fig.update_layout(scene=dict(xaxis=dict(title='Panjang (X)',range=[0,L],backgroundcolor="rgb(240,240,240)"), yaxis=dict(title='Lebar (Y)',range=[0,W],backgroundcolor="rgb(235,235,235)"), zaxis=dict(title='Tinggi (Z)',range=[0,H],backgroundcolor="rgb(240,240,240)"),
                        aspectratio=dict(x=1,y=W/L,z=H/L) if L>0 else dict(x=1,y=1,z=1), aspectmode='manual', camera_eye=dict(x=1.8,y=1.8,z=0.9)),
                      margin=dict(l=5,r=5,t=5,b=5), legend=dict(orientation="h",yanchor="bottom",y=0.01,xanchor="right",x=1))
    return fig

# --- Layout Aplikasi Streamlit ---
st.title("🚚 Optimasi Penugasan & Pemuatan Truk dengan PSO"); st.markdown("---")
# Baris 1: Metrik Ringkasan
col1, col2, col3, col4 = st.columns(4)
total_revenue_final = sum(info['revenue'] for info in final_truck_info.values())
total_cost_final = sum(info['fuel_cost'] for info in final_truck_info.values())
total_profit_final = total_revenue_final - total_cost_final
num_assigned_items = sum(len(info['items']) for info in final_truck_info.values())
num_overweight = sum(1 for info in final_truck_info.values() if info['is_overweight'])
col1.metric("Total Estimasi Profit*", f"Rp {total_profit_final:,.0f}", help="Profit = Total Revenue - Total Fuel Cost.")
col2.metric("Total Revenue", f"Rp {total_revenue_final:,.0f}"); col3.metric("Total Fuel Cost", f"Rp {total_cost_final:,.0f}")
col4.metric("Item Ter-assign", f"{num_assigned_items} / {len(items)}")
if num_overweight > 0: st.warning(f"🚨 **PERHATIAN:** {num_overweight} truk MELEBIHI BATAS BERAT!")
st.markdown("---")
# Baris 2: Peta & Grafik Fitness
col1_map, col2_chart = st.columns([3, 2])
with col1_map:
    st.subheader("🗺️ Peta Rute Pengiriman")
    if routes_data_pydeck:
        city_points=[{"name": city,"coordinates": [coord[1],coord[0]]} for city,coord in city_coords.items()]
        marker=pdk.Layer("ScatterplotLayer",data=city_points,get_position="coordinates",get_fill_color=[0,0,0,180],get_radius=8000,radius_min_pixels=6,pickable=True,auto_highlight=True)
        text=pdk.Layer("TextLayer",data=city_points,get_position="coordinates",get_text="name",get_color=[0,0,0,200],get_size=15,get_alignment_baseline="'bottom'",get_pixel_offset=[0,-18])
        paths=[pdk.Layer("PathLayer",data=[r],get_path="path",get_color="color",get_width=5,width_scale=1,width_min_pixels=3.5,pickable=True,auto_highlight=True) for r in routes_data_pydeck]
        view=pdk.ViewState(latitude=city_coords["Surabaya"][0],longitude=city_coords["Surabaya"][1],zoom=5.8,pitch=45)
        tooltip={"html":"<b>{truck}</b><br/>Rute: {route_info}<br/>Jarak: {distance_km:.1f} km","style": {"backgroundColor":"rgba(0,0,0,0.7)","color":"white","fontSize":"12px"}}
        st.pydeck_chart(pdk.Deck(layers=paths+[marker,text],initial_view_state=view,map_style='mapbox://styles/mapbox/outdoors-v11',tooltip=tooltip))
    else: st.info("Tidak ada rute.")
with col2_chart:
    st.subheader("📈 Grafik Fitness Assignment PSO")
    if fitness_history_assign:
        fig_fit=go.Figure(go.Scatter(y=fitness_history_assign,mode='lines',name='Best Fitness'))
        fig_fit.update_layout(title="Perkembangan Fitness",xaxis_title="Iterasi",yaxis_title="Fitness",height=400,margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig_fit,use_container_width=True)
    else: st.info("Data fitness tidak tersedia.")
st.markdown("---")
# Baris 3: Detail Assignment
st.subheader("📦 Detail Penugasan Item"); st.dataframe(assigned_items_df[['id','name','weight','dims','city','dim_category','assigned_truck']])
st.markdown("---")
# Baris 4+: Detail Truk & Visualisasi
st.subheader("🚛 Detail Muatan & Visualisasi per Truk")
cols_per_row = 2; num_rows = (n_trucks + cols_per_row - 1) // cols_per_row; truck_idx = 1
for r in range(num_rows):
    cols = st.columns(cols_per_row)
    for c in range(cols_per_row):
        if truck_idx <= n_trucks:
            with cols[c]:
                st.markdown(f"#### Truk {truck_idx}"); info = final_truck_info[truck_idx]; layout, penalty = final_layouts.get(truck_idx,(None,float('inf')))
                if info["is_overweight"]: st.error(f"⛔️ OVERWEIGHT: {info['weight']}/{truck_max_weight} kg")
                else: st.metric("Berat Muatan",f"{info['weight']}/{truck_max_weight} kg",delta=f"{truck_max_weight-info['weight']:.0f} kg Sisa",delta_color="normal")
                st.metric("Profit Truk Ini*",f"Rp {info['profit']:,.0f}",help="Revenue Truk - Fuel Cost Truk")
                if not info["items"]: st.info("Truk kosong.")
                else:
                    st.caption(f"**Tujuan:** {', '.join(info['cities'])} | **Jarak:** {info['route_distance']:.1f} km")
                    with st.expander("Lihat item di truk ini"): st.dataframe(pd.DataFrame(info["items"])[["id","name","weight","dims","city"]])
                    st.markdown("**Visualisasi Muatan:**")
                    if layout is None and info["items"]: st.error("Packing gagal.")
                    elif penalty > 1e-3 and info["items"]:
                        st.warning(f"Packing belum optimal (Penalty: {penalty:.4f}).")
                        fig = create_truck_figure(truck_dims_tuple, layout); st.plotly_chart(fig, use_container_width=True)
                    elif info["items"]:
                        st.success(f"Muatan terpack (Penalty: {penalty:.4f}).")
                        fig = create_truck_figure(truck_dims_tuple, layout); st.plotly_chart(fig, use_container_width=True)
                st.markdown("---"); truck_idx += 1

print("\nStreamlit App Ready.")