import streamlit as st
import math
import random
import pydeck as pdk
import pandas as pd
import json
import os

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
    {"id": "Item8", "name": "Kipas Angin", "weight": 30, "dims": (20, 20, 40), "city": "Bandung"}
]

n_trucks = 4
truck_max_weight = 1000
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

threshold_small = 50*50*50 
threshold_medium = 100*100*100

def get_dimension_category(l, w, h):
    """Menentukan kategori dimensi dan faktor biaya berdasarkan volume."""
    volume = l * w * h
    if volume < threshold_small:
        return "Kecil", 10
    elif volume < threshold_medium:
        return "Sedang", 15
    else:
        return "Besar", 20

for item in items:
    l, w, h = item["dims"]
    cat, factor = get_dimension_category(l, w, h)
    item["dim_category"] = cat
    item["cat_factor"] = factor
    item["is_oversized"] = (l > truck_max_length or 
                        w > truck_max_width or 
                        h > truck_max_height)

###############################
# Fungsi Perhitungan Jarak (Haversine)
###############################
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

def distance(city1, city2):
    """Get distance from CSV if available, otherwise use Haversine formula."""
    if (city1, city2) in distance_dict:
        return distance_dict[(city1, city2)]
    elif (city2, city1) in distance_dict:
        return distance_dict[(city2, city1)]
    else:
        return haversine(city_coords[city1], city_coords[city2])

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
    Menghitung total profit (fitness) dari satu solusi assignment.
    assignment: list dengan panjang = jumlah barang, berisi nomor truk (0-4) untuk tiap barang.
    """
    truck_info = {t: {"weight": 0, "volume": 0, "cities": [], "revenue": 0} for t in range(1, n_trucks+1)}
    total_revenue = 0
    for i, truck in enumerate(assignment):
        if truck == 0:
            continue
        item = items[i]
        l, w, h = item["dims"]
        if l > truck_max_length or w > truck_max_width or h > truck_max_height:
            return -1e9
        truck_info[truck]["weight"] += item["weight"]
        truck_info[truck]["volume"] += (l * w * h)
        truck_info[truck]["cities"].append(item["city"])
        
        d = distance("Surabaya", item["city"])
        revenue = item["weight"] * d * item["cat_factor"]
        truck_info[truck]["revenue"] += revenue
        total_revenue += revenue
    
    for t, info in truck_info.items():
        if info["weight"] > truck_max_weight or info["volume"] > (truck_max_length*truck_max_width*truck_max_height):
            return -1e9
   
    fuel_cost_per_km = 3500.0
    total_cost = 0.0
    for t, info in truck_info.items():
        if info["cities"]:
            total_cost += fuel_cost_per_km * route_distance(info["cities"])
    profit = total_revenue - total_cost
    return profit

def decode_position(position):
    """Konversi posisi continuous ke assignment diskrit (0-4) dengan pembulatan terdekat."""
    assignment = []
    for i, val in enumerate(position):
        item = items[i]
        if item["is_oversized"]:
            assignment.append(0)
        else:
            clamped = max(1, min(n_trucks, val))
            assign = int(round(clamped))
            assignment.append(assign)
    return assignment

###############################
# PSO: Inisialisasi dan Iterasi
###############################
num_particles = 10
max_iter = 50
w_inertia = 0.7
c1 = 1.5
c2 = 1.5

improvement_threshold = 1.0
patience = 10

particles = []
velocities = []
pbest_positions = []
pbest_fitness = []
gbest_position = None
gbest_fitness = -1e9
prev_gbest = -1e9

random.seed(30)

for _ in range(num_particles):
    position = []
    for item in items:
        if item["is_oversized"]:
            position.append(0.0)
        else:
            position.append(random.uniform(1, n_trucks))
    velocity = [random.uniform(-n_trucks, n_trucks) for _ in range(len(items))]
    particles.append(position)
    velocities.append(velocity)
    assignment = decode_position(position)
    fit = compute_fitness(assignment)
    pbest_positions.append(position[:])
    pbest_fitness.append(fit)
    if fit > gbest_fitness:
        gbest_fitness = fit
        gbest_position = position[:]

for it in range(1, max_iter+1):
    for i in range(num_particles):
        for d in range(len(items)):
            r1 = random.random()
            r2 = random.random()
            velocities[i][d] = (w_inertia * velocities[i][d] +
                                c1 * r1 * (pbest_positions[i][d] - particles[i][d]) +
                                c2 * r2 * (gbest_position[d] - particles[i][d]))
            particles[i][d] += velocities[i][d]
        assignment = decode_position(particles[i])
        fit = compute_fitness(assignment)
        if fit > pbest_fitness[i]:
            pbest_fitness[i] = fit
            pbest_positions[i] = particles[i][:]
        if fit > gbest_fitness:
            gbest_fitness = fit
            gbest_position = particles[i][:]
            
    improvement = gbest_fitness - prev_gbest
    print(f"Iter {it}: Profit = {gbest_fitness:.2f}, Improvement = {improvement:.2f}")
    if improvement <= improvement_threshold:
        no_improvement_count += 1
        if no_improvement_count >= patience:
            print(f"Early stopping at iteration {it}")
            break
    else:
        no_improvement_count = 0
    prev_gbest = gbest_fitness

best_assignment = decode_position(gbest_position)

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
st.title("Optimasi Penempatan Barang dengan PSO")
st.write(f"**Total Profit:** {gbest_fitness:,.2f}")

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
