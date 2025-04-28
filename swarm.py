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

random.seed(30)
np.random.seed(30)

def quick_feasible(truck_dims, items):
    L, W, H = truck_dims

    for it in items:
        dims = it['dims']
        fits_somehow = False
        for (l, w, h) in set(permutations(dims, 3)):
            if l <= L and w <= W and h <= H:
                fits_somehow = True
                break
        if not fits_somehow:
            return False

    total_vol = sum(l*w*h for l, w, h in (it['dims'] for it in items))
    if total_vol > L*W*H:
        return False
    total_base = sum(l*w for l, w, _ in (it['dims'] for it in items))
    if total_base > L*W:
        return False
    return True

@lru_cache(maxsize=256)
def packing_penalty_cache(truck_dims, items_tuple):
    items = [ {'id':i[0], 'name':i[1], 'dims':i[2]} for i in items_tuple ]
    packer = PackingPSO(truck_dims, items)
    layout, pen = packer.optimize(max_iters=100)
    return layout, pen

class PackingPSO:
    def __init__(self, truck_dims, items,
                 angles=(0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi),
                 compaction_weight=1e-3):
        """
        truck_dims: (L, W, H)
        items: list of {'name', 'dims':(lx,ly,lz)}
        angles: allowed yaw rotations around vertical axis
        compaction_weight: penalty weight for unused bounding‐box volume
        """
        self.L, self.W, self.H = truck_dims
        self.items = items
        self.n = len(items)
        self.dim = self.n * 4

        self.angles = angles
        self.K = len(angles)

        self.num_particles = 30
        self.c1 = 1.5
        self.c2 = 1.5

        self.comp_w = compaction_weight

        self.particles = []
        self.velocities = []
        self.pbest_pos = []
        self.pbest_score = []
        self.gbest_pos = None
        self.gbest_score = float('inf')

        for _ in range(self.num_particles):
            pos = np.zeros(self.dim)
            vel = np.zeros(self.dim)
            for i in range(self.n):
                lx, ly, lz = items[i]['dims']
                pos[4*i+0] = np.random.uniform(0, self.L)
                pos[4*i+1] = np.random.uniform(0, self.W)
                pos[4*i+2] = np.random.uniform(0, self.H)
                pos[4*i+3] = np.random.uniform(0, self.K)
                vel[4*i:4*i+4] = np.random.uniform(-1, 1, 4)
            self._clamp(pos)
            score = self._penalty(pos)
            self.particles.append(pos.copy())
            self.velocities.append(vel)
            self.pbest_pos.append(pos.copy())
            self.pbest_score.append(score)
            if score < self.gbest_score:
                self.gbest_score, self.gbest_pos = score, pos.copy()

    def _clamp(self, pos):
        """Clamp x,y,z within [0,dim] and orientation index within [0,K-ε]."""
        for i in range(self.n):
            pos[4*i+0] = np.clip(pos[4*i+0], 0, self.L)
            pos[4*i+1] = np.clip(pos[4*i+1], 0, self.W)
            pos[4*i+2] = np.clip(pos[4*i+2], 0, self.H)
            idx = pos[4*i+3]
            pos[4*i+3] = min(max(idx, 0), self.K - 1e-3)

    def _penalty(self, pos):
        placement = []
        total_vol = 0
        for i in range(self.n):
            x, y, z = pos[4*i:4*i+3]
            ori = int(pos[4*i+3])
            θ = self.angles[ori]
            lx, ly, lz = self.items[i]['dims']
            w = abs(lx*math.cos(θ)) + abs(ly*math.sin(θ))
            d = abs(lx*math.sin(θ)) + abs(ly*math.cos(θ))
            placement.append({'x':x,'y':y,'z':z,'w':w,'d':d,'h':lz})
            total_vol += lx*ly*lz

        pen = 0.0
        for it in placement:
            if it['x'] < 0:
                pen += OUTBOUND_PENALTY_FACTOR * (-it['x'] * it['d'] * it['h'])
            if it['y'] < 0:
                pen += OUTBOUND_PENALTY_FACTOR * (-it['y'] * it['w'] * it['h'])
            if it['z'] < 0:
                pen += OUTBOUND_PENALTY_FACTOR * (-it['z'] * it['w'] * it['d'])

            x_over = max(it['x']+it['w'] - self.L, 0)
            y_over = max(it['y']+it['d'] - self.W, 0)
            z_over = max(it['z']+it['h'] - self.H, 0)
            pen += OVERLAP_PENALTY_FACTOR * (
                x_over * it['d'] * it['h'] +
                y_over * it['w'] * it['h'] +
                z_over * it['w'] * it['d']
            )

        for A,B in combinations(placement,2):
            ox = min(A['x']+A['w'], B['x']+B['w']) - max(A['x'], B['x'])
            oy = min(A['y']+A['d'], B['y']+B['d']) - max(A['y'], B['y'])
            oz = min(A['z']+A['h'], B['z']+B['h']) - max(A['z'], B['z'])
            if ox>0 and oy>0 and oz>0:
                pen += OVERLAP_PENALTY_FACTOR * (ox * oy * oz)

        for i, it in enumerate(placement):
            if it['z'] < 1e-6: continue
            supported = False
            base_z = it['z']
            for j, base in enumerate(placement):
                if j==i: continue
                if abs(base['z']+base['h'] - base_z) < 1e-6:
                    if (base['x']<=it['x']+1e-6
                        and base['x']+base['w']>=it['x']+it['w']-1e-6
                        and base['y']<=it['y']+1e-6
                        and base['y']+base['d']>=it['y']+it['d']-1e-6):
                        supported = True
                        break
            if not supported:
                pen += OVERLAP_PENALTY_FACTOR * (self.L * self.W * self.H)

        max_x = max(it['x']+it['w'] for it in placement)
        max_y = max(it['y']+it['d'] for it in placement)
        max_z = max(it['z']+it['h'] for it in placement)
        used_vol = max_x * max_y * max_z
        pen += self.comp_w * (used_vol - total_vol)

        return pen

    def optimize(self, max_iters=100):
        for t in range(max_iters):
            w = W_MAX - (W_MAX - W_MIN) * (t / float(max_iters - 1))
            for p in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                vel = ( w * self.velocities[p]
                      + self.c1 * r1 * (self.pbest_pos[p] - self.particles[p])
                      + self.c2 * r2 * (self.gbest_pos - self.particles[p]) )
                self.velocities[p] = vel
                self.particles[p] += vel
                self._clamp(self.particles[p])

                sc = self._penalty(self.particles[p])
                if sc < self.pbest_score[p]:
                    self.pbest_score[p] = sc
                    self.pbest_pos[p]   = self.particles[p].copy()
                if sc < self.gbest_score:
                    self.gbest_score = sc
                    self.gbest_pos   = self.particles[p].copy()

            if self.gbest_score <= 0:
                break

        layout = []
        pos = self.gbest_pos
        for i in range(self.n):
            ori = int(pos[4*i+3]); θ = self.angles[ori]
            lx, ly, lz = self.items[i]['dims']
            w = abs(lx*math.cos(θ)) + abs(ly*math.sin(θ))
            d = abs(lx*math.sin(θ)) + abs(ly*math.cos(θ))
            layout.append({
                'name': self.items[i]['name'],
                'x': pos[4*i], 'y': pos[4*i+1], 'z': pos[4*i+2],
                'w': w, 'd': d, 'h': lz,
                'angle_deg': round(math.degrees(θ),1)
            })
        return layout, self.gbest_score
    
###############################
# Load Distance Data
###############################
csv_filename = "city_to_city_polygon.csv"

if os.path.exists(csv_filename):
    df_distance = pd.read_csv(csv_filename)

    if "CityA" in df_distance.columns and "CityB" in df_distance.columns and "Polygon" in df_distance.columns:
        distance_dict = {}
        polygons = {}

        for _, row in df_distance.iterrows():
            key = (row["CityA"], row["CityB"])
            distance_dict[key] = row["Distance (meters)"] / 1000 
            
            try:
                polygon_coords = json.loads(row["Polygon"])
                polygons[key] = [[p["lng"], p["lat"]] for p in polygon_coords]
            except json.JSONDecodeError:
                polygons[key] = []

    else:
        st.error("CSV file is missing required columns: 'CityA', 'CityB', 'Polygon'")
else:
    distance_dict = {}
    polygons = {}
    

items = [
    {"id": "Item1", "name": "TV", "weight": 120, "dims": (40, 50, 30), "city": "Jakarta"},
    {"id": "Item2", "name": "Kulkas", "weight": 300, "dims": (70, 60, 90), "city": "Bandung"},
    {"id": "Item3", "name": "AC", "weight": 250, "dims": (80, 50, 60), "city": "Semarang"},
    {"id": "Item4", "name": "Buku", "weight": 50, "dims": (30, 30, 20), "city": "Jakarta"},
    {"id": "Item5", "name": "Sofa", "weight": 500, "dims": (150, 80, 100), "city": "Yogyakarta"},
    {"id": "Item6", "name": "Meja", "weight": 150, "dims": (120, 100, 40), "city": "Semarang"},
    {"id": "Item7", "name": "Ranjang", "weight": 400, "dims": (200, 160, 50), "city": "Malang"},
    {"id": "Item8", "name": "Kipas Angin", "weight": 30, "dims": (20, 20, 40), "city": "Bandung"},
    {"id": "Item9",  "name": "WashingMachine","weight":350, "dims": (60,60,85), "city": "Jakarta"},
    {"id": "Item10", "name": "Bookshelf", "weight":100, "dims": (80,30,180), "city": "Surabaya"},
    {"id": "Item11", "name": "Mattress", "weight":200, "dims": (200,90,30), "city": "Bandung"},
    {"id": "Item12", "name": "Wardrobe", "weight":450, "dims": (100,60,200), "city": "Yogyakarta"},
    {"id": "Item13", "name": "DiningTable", "weight":250, "dims": (160,90,75), "city": "Semarang"},
    {"id": "Item14", "name": "DeskLamp", "weight":10,  "dims": (15,15,40), "city": "Malang"},
    {"id": "Item15", "name": "Microwave", "weight":40,  "dims": (50,40,35), "city": "Jakarta"},
    {"id": "Item16", "name": "Printer", "weight":25,  "dims": (45,40,30), "city": "Surabaya"},
    {"id": "Item17", "name": "FloorLamp", "weight":20,  "dims": (30,30,160), "city": "Bandung"},
    {"id": "Item18", "name": "AirPurifier", "weight":15,  "dims": (25,25,60), "city": "Yogyakarta"},
    {"id": "Item19", "name": "WaterHeater", "weight":80,  "dims": (50,50,100), "city": "Semarang"},
    {"id": "Item20", "name": "CoffeeTable", "weight":80,  "dims": (120,60,45), "city": "Malang"},
]

n_trucks = 4
truck_max_weight = 1000
truck_max_length = 200
truck_max_width  = 150
truck_max_height = 150
floor_diag = math.hypot(truck_max_length, truck_max_width)

city_coords = {
    "Surabaya": (-7.2575, 112.7521), 
    "Jakarta": (-6.2088, 106.8456),
    "Bandung": (-6.9175, 107.6191),
    "Semarang": (-6.9667, 110.4167),
    "Yogyakarta": (-7.7956, 110.3695),
    "Malang": (-7.9824, 112.6304)
}

threshold_small = 50*50*50 
threshold_medium = 100*100*100

def get_dimension_category(l, w, h):
    """Menentukan kategori dimensi dan faktor biaya berdasarkan volume."""
    volume = l * w * h
    if volume < threshold_small:
        return "Kecil", 50
    elif volume < threshold_medium:
        return "Sedang", 75
    else:
        return "Besar", 100

for item in items:
    l, w, h = item["dims"]
    cat, factor = get_dimension_category(l, w, h)
    item["dim_category"] = cat
    item["cat_factor"] = factor
    # can_axis_fit = any(
    #     (dx <= truck_max_length and dy <= truck_max_weight and dz <= truck_max_height)
    #     for dx, dy, dz in permutations((l, w, h))
    # )
    # can_diag_fit = any(
    #     (max(dx, dy) <= floor_diag and dz <= truck_max_height)
    #     for dx, dy, dz in permutations((l, w, h))
    # )
    # item["is_oversized"] = not (can_axis_fit or can_diag_fit)

###############################
# Fungsi Perhitungan Jarak (Haversine)
###############################
@st.cache_data(show_spinner=False)
def haversine(coord1, coord2):
    """
    Menghitung jarak antara dua titik (lat, lon) dengan rumus Haversine.
    Hasil dalam kilometer.
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

@st.cache_data(show_spinner=False)
def distance(city1, city2):
    """Get distance from CSV if available, otherwise use Haversine formula."""
    if (city1, city2) in distance_dict:
        return distance_dict[(city1, city2)]
    elif (city2, city1) in distance_dict:
        return distance_dict[(city2, city1)]
    else:
        return haversine(city_coords[city1], city_coords[city2])

@st.cache_data(show_spinner=False)
def route_distance(cities):
    """
    Menghitung jarak rute pulang-pergi dari Surabaya melewati semua kota dalam list
    menggunakan heuristik nearest-neighbor.
    """
    if not cities:
        return 0.0
    unvisited = cities[:]
    current = "Surabaya"
    total = 0.0
    while unvisited:
        nearest = min(unvisited, key=lambda c: distance(current, c))
        total += distance(current, nearest)
        current = nearest
        unvisited.remove(nearest)
    total += distance(current, "Surabaya")
    return total

###############################
# Fungsi Fitness dan PSO
###############################
def compute_fitness(assignment):
    """
    Soft‐penalty version of fitness:
      - Accumulates penalties for items that don't fit any orientation
        (OUTBOUND_PENALTY_FACTOR × item_volume).
      - Adds a large penalty if quick_feasible fails.
      - Adds the inner packing-penalty from packing_penalty_cache.
    Returns: (revenue - fuel_cost) - total_penalty
    """
    truck_info = {
        t: {"items": [], "cities": [], "revenue": 0.0}
        for t in range(1, n_trucks+1)
    }
    total_revenue = 0.0
    total_penalty = 0.0
    layouts = {}
    pack_penalties = {}

    for idx, truck in enumerate(assignment):
        if truck == 0:
            continue
        item = items[idx]
        l, w, h = item["dims"]
        item_vol = l * w * h

        if not any(
            (dx <= truck_max_length and dy <= truck_max_width and dz <= truck_max_height)
            for dx,dy,dz in itertools.permutations((l,w,h))
        ):
            total_penalty += OUTBOUND_PENALTY_FACTOR * item_vol
            continue

        d = distance("Surabaya", item["city"])
        rev = item["weight"] * d * item["cat_factor"]
        truck_info[truck]["items"].append(item)
        truck_info[truck]["revenue"] += rev
        total_revenue += rev
        truck_info[truck]["cities"].append(item["city"])

    for t, info in truck_info.items():
        its = info["items"]
        if not its:
            continue

        if not quick_feasible((truck_max_length, truck_max_width, truck_max_height), its):
            total_penalty += 10000000.0      

        tpl = tuple((it["id"], it["name"], it["dims"]) for it in its)
        layout, pen = packing_penalty_cache((truck_max_length, truck_max_width, truck_max_height), tpl)
        total_penalty += pen
        layouts[t] = layout
        pack_penalties[t] = pen

    FUEL_PRICE_PER_L       = 9000
    TRUCK_CONSUMPTION_KM_L = 4
    cost_per_km = FUEL_PRICE_PER_L / TRUCK_CONSUMPTION_KM_L

    total_cost = sum(
        cost_per_km * route_distance(info["cities"])
        for info in truck_info.values() if info["cities"]
    )

    profit = total_revenue - total_cost
    fitness = profit - total_penalty
    return fitness, layouts, pack_penalties



def decode_position(position):
    """Konversi posisi continuous ke assignment diskrit (0-4) dengan pembulatan terdekat."""
    assignment = []
    for i, val in enumerate(position):
        # item = items[i]
        clamped = max(1, min(n_trucks, val))
        assign = int(round(clamped))
        assignment.append(assign)
        # if item["is_oversized"]:
        #     assignment.append(0)
        # else:
        #     clamped = max(1, min(n_trucks, val))
        #     assign = int(round(clamped))
        #     assignment.append(assign)
    return assignment

###############################
# PSO: Inisialisasi dan Iterasi
###############################
num_particles = 30
max_iter = 1000
patience = 50
W_MAX = 0.9
W_MIN = 0.4
c1 = 2.0
c2 = 2.0
OVERLAP_PENALTY_FACTOR  = 2.0
OUTBOUND_PENALTY_FACTOR = 2.0

improvement_threshold = 1.0
patience = 10
no_improvement_count = 0

particles = []
velocities = []
pbest_positions = []
pbest_fitness = []
gbest_position = None
gbest_fitness = -float('inf')
prev_gbest = -float('inf')

for _ in range(num_particles):
    position = []
    for item in items:
        position.append(random.uniform(1, n_trucks))
        # if item["is_oversized"]:
        #     position.append(0.0)
        # else:
        #     position.append(random.uniform(1, n_trucks))
    velocity = [random.uniform(-n_trucks, n_trucks) for _ in range(len(items))]
    particles.append(position)
    velocities.append(velocity)
    assignment = decode_position(position)
    fit, _, _ = compute_fitness(assignment)
    pbest_positions.append(position[:])
    pbest_fitness.append(fit)
    if fit > gbest_fitness:
        gbest_fitness = fit
        gbest_position = position[:]
        
fitness_history = []
for it in range(1, max_iter+1):
    w = W_MAX - (W_MAX - W_MIN) * ((it-1)/(max_iter-1))
    for i in range(num_particles):
        for d in range(len(items)):
            r1 = random.random()
            r2 = random.random()
            velocities[i][d] = (w * velocities[i][d] +
                                c1 * r1 * (pbest_positions[i][d] - particles[i][d]) +
                                c2 * r2 * (gbest_position[d] - particles[i][d]))
            particles[i][d] += velocities[i][d]
        # for d, item in enumerate(items):
        #     if item["is_oversized"]:
        #         velocities[i][d] = 0.0
        #         particles[i][d]  = 0.0
        assignment = decode_position(particles[i])
        fit, _, _ = compute_fitness(assignment)
        if fit > pbest_fitness[i]:
            pbest_fitness[i] = fit
            pbest_positions[i] = particles[i][:]
        if fit > gbest_fitness:
            gbest_fitness = fit
            gbest_position = particles[i][:]
            
    improvement = gbest_fitness - prev_gbest
    fitness_history.append(gbest_fitness)
    if improvement <= improvement_threshold:
        no_improvement_count += 1
        if no_improvement_count >= patience:
            print(f"Early stopping at iteration {it}")
            break
    else:
        no_improvement_count = 0
    print(f"current gbest: {gbest_fitness}, prev gbest: {prev_gbest}")
    prev_gbest = gbest_fitness

best_assignment = decode_position(gbest_position)
best_fitness, best_layouts, pack_penalties = compute_fitness(best_assignment)

###############################
# Mendapatkan Info Detail per Truk
###############################
def get_truck_info(assignment):
    """
    Mengembalikan informasi per truk:
    - Daftar item
    - Total berat, volume, revenue
    - List kota tujuan (unik)
    """
    truck_info = {t: {"items": [], "weight": 0, "volume": 0, "cities": [], "revenue": 0} for t in range(1, n_trucks+1)}
    for i, truck in enumerate(assignment):
        if truck == 0:
            continue
        item = items[i]
        truck_info[truck]["items"].append(item)
        truck_info[truck]["weight"] += item["weight"]
        l, w, h = item["dims"]
        truck_info[truck]["volume"] += (l * w * h)
        truck_info[truck]["revenue"] += item["weight"] * distance("Surabaya", item["city"]) * item["cat_factor"]
        if item["city"] not in truck_info[truck]["cities"]:
            truck_info[truck]["cities"].append(item["city"])
    return truck_info

truck_info = get_truck_info(best_assignment)

###############################
# Fungsi Menentukan Rute (Nearest Neighbor) untuk Visualisasi
###############################
def get_route(cities):
    """
    Mengembalikan urutan kota (dalam list) dari Surabaya -> kota-kota tujuan (berdasarkan nearest neighbor) -> Surabaya.
    Jika tidak ada kota, mengembalikan [Surabaya].
    """
    if not cities:
        return ["Surabaya"]
    unvisited = cities[:]
    route = ["Surabaya"]
    current = "Surabaya"
    while unvisited:
        nearest = min(unvisited, key=lambda c: distance(current, c))
        route.append(nearest)
        current = nearest
        unvisited.remove(nearest)
    route.append("Surabaya")
    return route

def get_segment_path(city_a, city_b):
    """Mendapatkan path antara dua kota dari data CSV atau garis lurus"""
    key = (city_a, city_b)
    reverse_key = (city_b, city_a)
    
    if key in polygons and len(polygons[key]) > 0:
        return polygons[key]
    elif reverse_key in polygons and len(polygons[reverse_key]) > 0:
        return polygons[reverse_key][::-1]
    else:
        start = [city_coords[city_a][1], city_coords[city_a][0]]
        end = [city_coords[city_b][1], city_coords[city_b][0]]
        return [start, end]

def get_full_route_path(cities):
    """Membangun path lengkap untuk rute dengan menggabungkan segmen antar kota"""
    full_path = []
    for i in range(len(cities)-1):
        city_a = cities[i]
        city_b = cities[i+1]
        segment = get_segment_path(city_a, city_b)
        
        if full_path and segment:
            if full_path[-1] == segment[0]:
                full_path.extend(segment[1:])
            else:
                full_path.extend(segment)
        else:
            full_path.extend(segment)
    return full_path

routes_data = []
colors = [
    [255, 0, 0, 200],    # merah
    [0, 255, 0, 200],    # hijau
    [0, 0, 255, 200],    # biru
    [255, 165, 0, 200]   # oranye
]

for t in range(1, n_trucks+1):
    info = truck_info[t]
    if info["cities"]:
        route_order = get_route(info["cities"])
        full_path = get_full_route_path(route_order)
        
        routes_data.append({
            "truck": f"Truk {t}",
            "path": full_path,
            "color": colors[t-1],
            "cities": route_order
        })

###############################
# Visualisasi dengan Streamlit (Updated)
###############################
def create_truck_figure(truck_dims, packed_items):
    L, W, H = truck_dims
    fig = go.Figure()
    for idx, item in enumerate(packed_items):
        xmin, ymin, zmin = item['x'], item['y'], item['z']
        xmax = xmin + item['w']
        ymax = ymin + item['d']
        zmax = zmin + item['h']
        xverts = [xmin, xmin, xmax, xmax, xmin, xmin, xmax, xmax]
        yverts = [ymin, ymax, ymax, ymin, ymin, ymax, ymax, ymin]
        zverts = [zmin, zmin, zmin, zmin, zmax, zmax, zmax, zmax]
        faces_i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 1]
        faces_j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 2]
        faces_k = [0, 7, 2, 3, 6, 7, 1, 5, 4, 5, 7, 6]
        color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                         "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        color = color_palette[idx % len(color_palette)]
        fig.add_trace(go.Mesh3d(
            x=xverts, y=yverts, z=zverts,
            i=faces_i, j=faces_j, k=faces_k,
            color=color, opacity=0.8,
            name=item['name'],
            hovertext=f"{item['name']}: {item['w']}x{item['d']}x{item['h']}",
            hoverinfo="text"
        ))
    corners = [(0,0,0), (L,0,0), (L,W,0), (0,W,0), (0,0,H), (L,0,H), (L,W,H), (0,W,H)]
    edges = [(0,1), (1,2), (2,3), (3,0),
             (4,5), (5,6), (6,7), (7,4),
             (0,4), (1,5), (2,6), (3,7)]
    edge_x = []; edge_y = []; edge_z = []
    for (u, v) in edges:
        edge_x += [corners[u][0], corners[v][0], None]
        edge_y += [corners[u][1], corners[v][1], None]
        edge_z += [corners[u][2], corners[v][2], None]
    fig.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z,
                               mode='lines', line_color='black', line_width=3,
                               hoverinfo='skip', showlegend=False))
    fig.update_layout(
        scene = dict(xaxis=dict(range=[0, L], title='Length'),
                     yaxis=dict(range=[0, W], title='Width'),
                     zaxis=dict(range=[0, H], title='Height'),
                     aspectmode='data'),
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False
    )
    return fig

st.line_chart(fitness_history, height=200, width=400)
st.title("Optimasi Penempatan Barang dengan PSO")
st.write(f"**Total Profit:** {gbest_fitness:,.2f} (selama ada penalty berarti bukan profit asli)")

st.subheader("Rute Pengiriman per Truk")
if routes_data:
    layers = []
    
    for route in routes_data:
        cleaned_route = [c for c in route["cities"] if c != "Surabaya"]
        
        route_data = {
            "path": route["path"],
            "color": route["color"],
            "truck_name": route["truck"],
            "route_info": " → ".join(cleaned_route)
        }
        
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=[route_data],
                get_path="path",
                get_color="color",
                get_width=5,
                width_scale=1,
                width_min_pixels=3,
                pickable=True,
                auto_highlight=True
            )
        )
    
    city_points = []
    for city, coord in city_coords.items():
        city_points.append({
            "name": city,
            "coordinates": [coord[1], coord[0]]
        })
    
    marker_layer = pdk.Layer(
        "ScatterplotLayer",
        data=city_points,
        get_position="coordinates",
        get_color=[0, 0, 255, 200],
        get_radius=5000,
        pickable=True
    )
    
    layers.append(marker_layer)
    
    tooltip = {
        "html": """
            <b>{truck_name}</b><br/>
            <b>Route: {route_info}</b>
        """,
        "style": {
            "backgroundColor": "steelblue",
            "color": "white",
            "fontSize": "14px"
        }
    }
    
    view_state = pdk.ViewState(
        latitude=city_coords["Surabaya"][0],
        longitude=city_coords["Surabaya"][1],
        zoom=6,
        pitch=0
    )

    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip=tooltip
    ))
else:
    st.write("Tidak ada rute untuk ditampilkan.")

st.subheader("Detail Muatan per Truk")
for t in range(1, n_trucks+1):
    info = truck_info[t]
    st.markdown(f"### Truk {t}")
    if info["items"]:
        df = pd.DataFrame(info["items"])
        df = df[["id", "name", "weight", "dims", "city", "dim_category"]]
        st.table(df)
        st.write(f"**Total Berat:** {info['weight']} kg")
        st.write(f"**Total Revenue:** {info['revenue']:,.2f}")
        st.write(f"**Kota Tujuan:** {', '.join(info['cities'])}")
    else:
        st.write("Tidak ada barang yang diangkut.")

profit_data = {"Truk": [], "Profit": []}
for t in range(1, n_trucks+1):
    info = truck_info[t]
    profit_data["Truk"].append(f"Truk {t}")
    profit_data["Profit"].append(info["revenue"])
profit_df = pd.DataFrame(profit_data).set_index("Truk")
st.subheader("Distribusi Profit per Truk")
st.bar_chart(profit_df)

for t in range(1, n_trucks+1):
    entry = best_layouts.get(t)
    penalty = pack_penalties.get(t)
    if entry is None:
        continue

    if (isinstance(entry, tuple)
        and len(entry) >= 2
        and isinstance(entry[1], (int, float))):
        layout, pen = entry[0], entry[1]
    else:
        layout, pen = entry, 0.0

    truck_info[t]['layout']  = entry
    truck_info[t]['penalty'] = penalty

st.subheader("Visualisasi Muatan per Truk")
for t, info in truck_info.items():
    items   = info.get('items', [])
    layout  = info.get('layout',  [])
    penalty = info.get('penalty', 0.0)

    if not items:
        st.write(f"Truk {t}: tidak ada muatan.")
        continue

    if penalty > 0:
        st.error(f"⚠️ Truk {t} gagal dipacking (penalty={penalty:.0f}).")
        fig = create_truck_figure(
            (truck_max_length, truck_max_width, truck_max_height),
            layout
        )
    else:
        st.write(f"**Truk {t}** muatan terpack dengan baik:")
        fig = create_truck_figure(
            (truck_max_length, truck_max_width, truck_max_height),
            layout
        )

    st.plotly_chart(fig, use_container_width=True)