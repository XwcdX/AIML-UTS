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

st.set_page_config(layout="wide", page_title="Optimasi Muatan Truk PSO")

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
    # 1. Cek Volume Total
    total_vol = sum(it['dims'][0] * it['dims'][1] * it['dims'][2] for it in items)
    if total_vol > L * W * H * 1.05: # Beri sedikit toleransi volume? Atau tidak? (Mari kita ketat)
         if total_vol > L * W * H:
              return False # Jelas tidak muat jika volume total > volume truk

    # 2. Cek Luas Alas Minimum Total (Heuristik kasar)
    # Ini kurang bisa diandalkan karena rotasi, tapi bisa jadi filter awal
    min_total_base = sum(min(l*w, l*h, w*h) for l, w, h in (it['dims'] for it in items))
    if min_total_base > L*W: # Jika total luas alas minimum > luas alas truk
        return False # Kemungkinan besar tidak muat di dasar

    # Jika kedua cek lolos, anggap 'mungkin' layak untuk dicoba packing lebih detail
    return True

# Cache untuk hasil packing (dengan parameter ringan)
# Ukuran cache bisa disesuaikan, 256 atau 512 mungkin cukup
@lru_cache(maxsize=256)
def packing_penalty_cache(truck_dims, items_tuple):
    """
    Menjalankan PackingPSO dengan parameter CEPAT (iterasi/partikel sedikit)
    untuk mendapatkan estimasi penalti packing. Hasilnya di-cache.
    """
    # Konversi tuple kembali ke list of dicts
    items_list = [ {'id':i[0], 'name':i[1], 'dims':i[2], 'weight':i[3]} for i in items_tuple ]
    if not items_list: # Jika tidak ada item, tidak ada penalti
        return 0.0

    # Simpan state random numpy agar konsisten dalam cache
    current_np_state = np.random.get_state()
    np.random.seed(42) # Gunakan seed tetap untuk hasil deterministik dalam cache

    # --- OPTIMASI UTAMA: Gunakan parameter yang jauh lebih ringan ---
    packer = PackingPSO(truck_dims, items_list,
                        num_particles=PACKING_CACHE_NUM_PARTICLES) # Partikel LEBIH SEDIKIT
    _, pen = packer.optimize(max_iters=PACKING_CACHE_MAX_ITERS) # Iterasi LEBIH SEDIKIT
    # --- Akhir Optimasi ---

    # Kembalikan state random numpy
    np.random.set_state(current_np_state)

    # Kembalikan penalti hasil optimasi cepat
    return pen

class PackingPSO:
    """
    Optimasi packing 3D menggunakan PSO. Mempertimbangkan 6 orientasi standar.
    Termasuk penalti support dasar.
    """
    def __init__(self, truck_dims, items,
                 compaction_weight=1e-3,
                 num_particles=30, # Default, bisa di-override
                 c1=1.5, c2=1.5):
        """
        Args:
            truck_dims (tuple): Dimensi kontainer (L, W, H).
            items (list): List item, {'id', 'name', 'dims':(l,w,h), 'weight'}.
            compaction_weight (float): Bobot penalti untuk volume bounding box tak terpakai.
            num_particles (int): Jumlah partikel untuk PSO packing ini.
            c1, c2 (float): Koefisien kognitif dan sosial PSO.
        """
        self.L, self.W, self.H = truck_dims
        self.items = items
        self.n = len(items)
        if self.n == 0: # Handle jika tidak ada item
             self.dim = 0
             self.particles = []
             self.velocities = []
             self.pbest_pos = []
             self.pbest_score = []
             self.gbest_pos = None
             self.gbest_score = 0.0 # Tidak ada item, tidak ada penalti
             return # Langsung keluar dari init

        self.dim = self.n * 4 # x, y, z, orientation_index

        # 6 Orientasi standar (permutasi sumbu l, w, h) -> rotasi 90 derajat
        self.orientations = list(permutations([0, 1, 2]))
        self.K = 6 # Jumlah orientasi

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

        # Inisialisasi partikel
        for _ in range(self.num_particles):
            pos = np.zeros(self.dim)
            vel = np.zeros(self.dim)
            for i in range(self.n):
                # Posisi acak di dalam truk
                pos[4*i+0] = np.random.uniform(0, self.L)
                pos[4*i+1] = np.random.uniform(0, self.W)
                pos[4*i+2] = np.random.uniform(0, self.H) # Biarkan penalti mendorong ke bawah
                # Orientasi acak (index 0-5)
                pos[4*i+3] = np.random.uniform(0, self.K)
                # Kecepatan awal acak
                vel[4*i:4*i+4] = np.random.uniform(-1, 1, 4) * 0.1 # Kecepatan awal lebih kecil?

            self._clamp(pos) # Pastikan posisi awal valid
            score = self._penalty(pos) # Hitung skor awal

            self.particles.append(pos.copy())
            self.velocities.append(vel.copy()) # Gunakan copy
            self.pbest_pos.append(pos.copy())
            self.pbest_score.append(score)

            # Update global best awal
            if score < self.gbest_score:
                self.gbest_score = score
                self.gbest_pos = pos.copy()

    @staticmethod
    def _get_rotated_dims(original_dims, orientation_index):
        """Mengembalikan dimensi (lebar, kedalaman, tinggi) relatif thd sumbu truk."""
        l, w, h = original_dims
        # 0: (l, w, h), 1: (l, h, w), 2: (w, l, h), 3: (w, h, l), 4: (h, l, w), 5: (h, w, l)
        dims_map = [
            (l, w, h), (l, h, w),
            (w, l, h), (w, h, l),
            (h, l, w), (h, w, l)
        ]
        # Pastikan index valid sebelum mengakses map
        safe_index = int(np.clip(orientation_index, 0, 5))
        return dims_map[safe_index]


    def _clamp(self, pos):
        """Memastikan posisi (x,y,z) dan index orientasi dalam batas valid."""
        for i in range(self.n):
            # Batasi posisi x, y, z di dalam kontainer [0, Dimensi]
            pos[4*i+0] = np.clip(pos[4*i+0], 0, self.L)
            pos[4*i+1] = np.clip(pos[4*i+1], 0, self.W)
            pos[4*i+2] = np.clip(pos[4*i+2], 0, self.H)
            # Batasi index orientasi antara 0 dan K-epsilon (misal 0 - 5.999...)
            pos[4*i+3] = np.clip(pos[4*i+3], 0, self.K - 1e-9)

    def _penalty(self, pos):
        """Menghitung penalti untuk layout packing tertentu."""
        if self.n == 0: return 0.0 # Tidak ada penalti jika tidak ada item

        placement = []
        total_item_vol = 0.0
        for i in range(self.n):
            x, y, z = pos[4*i : 4*i+3]
             # Konversi index orientasi float ke int dengan aman
            ori_idx = int(np.clip(pos[4*i+3], 0, self.K - 1))
            original_dims = self.items[i]['dims']
            w_rot, d_rot, h_rot = self._get_rotated_dims(original_dims, ori_idx)

            placement.append({
                'id': self.items[i]['id'], 'name': self.items[i]['name'],
                'x': x, 'y': y, 'z': z,
                'w': w_rot, 'd': d_rot, 'h': h_rot
            })
            # Volume asli item (bukan volume terotasi)
            total_item_vol += original_dims[0] * original_dims[1] * original_dims[2]

        pen = 0.0

        # 1. Penalti Keluar Batas (Out-of-bounds)
        for it in placement:
            # Protrusi negatif (di bawah 0)
            x_under = max(0, -it['x'])
            y_under = max(0, -it['y'])
            z_under = max(0, -it['z'])
            pen += OUTBOUND_PENALTY_FACTOR * (x_under*it['d']*it['h'] + y_under*it['w']*it['h'] + z_under*it['w']*it['d'])
            # Protrusi positif (melebihi L, W, H)
            x_over = max(0, (it['x'] + it['w']) - self.L)
            y_over = max(0, (it['y'] + it['d']) - self.W)
            z_over = max(0, (it['z'] + it['h']) - self.H)
            pen += OUTBOUND_PENALTY_FACTOR * (x_over*it['d']*it['h'] + y_over*it['w']*it['h'] + z_over*it['w']*it['d'])

        # 2. Penalti Tumpang Tindih (Overlap)
        for i, A in enumerate(placement):
            for j, B in enumerate(placement):
                if i >= j: continue # Hindari cek diri sendiri & duplikasi
                # Hitung volume tumpang tindih
                ox = max(0, min(A['x'] + A['w'], B['x'] + B['w']) - max(A['x'], B['x']))
                oy = max(0, min(A['y'] + A['d'], B['y'] + B['d']) - max(A['y'], B['y']))
                oz = max(0, min(A['z'] + A['h'], B['z'] + B['h']) - max(A['z'], B['z']))
                overlap_vol = ox * oy * oz
                if overlap_vol > 1e-6: # Gunakan toleransi kecil
                    pen += OVERLAP_PENALTY_FACTOR * overlap_vol

        # 3. Penalti Support (Gravitasi) - Pengecekan sederhana pada titik tengah dasar
        support_tolerance = 1e-4 # Toleransi untuk perbandingan float
        for i, it in enumerate(placement):
            if it['z'] < support_tolerance: continue # Item di lantai dianggap didukung

            is_supported = False
            it_center_x = it['x'] + it['w'] / 2.0
            it_center_y = it['y'] + it['d'] / 2.0
            it_base_z = it['z']

            for j, base in enumerate(placement):
                if i == j: continue # Jangan cek dengan diri sendiri
                # Cek apakah 'base' tepat di bawah 'it' (secara vertikal)
                if abs((base['z'] + base['h']) - it_base_z) < support_tolerance:
                    # Cek apakah pusat proyeksi dasar 'it' berada di dalam proyeksi x-y 'base'
                    if (base['x'] - support_tolerance <= it_center_x <= base['x'] + base['w'] + support_tolerance and
                        base['y'] - support_tolerance <= it_center_y <= base['y'] + base['d'] + support_tolerance):
                        is_supported = True
                        break # Sudah ditemukan support

            if not is_supported:
                # Penalti BESAR jika tidak didukung (tidak di lantai & tidak ada item di bawah)
                pen += UNSUPPORTED_PENALTY_BASE + UNSUPPORTED_PENALTY_HEIGHT_FACTOR * it['h']

        # 4. Penalti Kompaksi (Minimize volume bounding box terpakai)
        if placement: # Hanya jika ada item
            max_x = max(it['x'] + it['w'] for it in placement)
            max_y = max(it['y'] + it['d'] for it in placement)
            max_z = max(it['z'] + it['h'] for it in placement)
            # Volume BBox = L * W * H dari item terjauh dari origin
            used_bbox_vol = max_x * max_y * max_z
            # Penalti = selisih antara BBox terpakai dan volume total item aktual
            pen += self.comp_w * max(0, used_bbox_vol - total_item_vol)
        # else: pen += 0 # Tidak ada penalti kompaksi jika tidak ada item

        return pen

    def optimize(self, max_iters=100):
        """Menjalankan algoritma PSO packing."""
        if self.n == 0: # Jika tidak ada item, langsung return
             return [], 0.0

        best_iter_score = float('inf')
        no_improvement_iters = 0
        # Sesuaikan patience berdasarkan max_iters
        patience = max(10, max_iters // 10)

        # Pastikan gbest_pos terinisialisasi
        if self.gbest_pos is None:
            if self.pbest_pos: # Jika pbest ada, gunakan yang terbaik dari pbest
                 initial_best_idx = np.argmin(self.pbest_score)
                 if self.pbest_score[initial_best_idx] < self.gbest_score:
                     self.gbest_score = self.pbest_score[initial_best_idx]
                     self.gbest_pos = self.pbest_pos[initial_best_idx].copy()
            else: # Jika pbest juga tidak ada (misal num_particles=0)
                 print("Error: Cannot optimize PackingPSO with no particles initialized.")
                 return [], float('inf') # Tidak bisa optimasi

        # --- Iterasi PSO ---
        for t in range(max_iters):
            w_inertia = W_MAX - (W_MAX - W_MIN) * (t / float(max_iters - 1))

            for p in range(self.num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                # Update velocity
                vel = ( w_inertia * self.velocities[p]
                        + self.c1 * r1 * (self.pbest_pos[p] - self.particles[p])
                        + self.c2 * r2 * (self.gbest_pos - self.particles[p]) )
                self.velocities[p] = vel # Bisa ditambahkan clamping velocity jika perlu

                # Update position
                self.particles[p] += self.velocities[p]
                self._clamp(self.particles[p]) # Pastikan posisi tetap valid

                # Evaluate new position
                sc = self._penalty(self.particles[p])

                # Update personal best (pbest)
                if sc < self.pbest_score[p]:
                    self.pbest_score[p] = sc
                    self.pbest_pos[p] = self.particles[p].copy()

                    # Update global best (gbest)
                    if sc < self.gbest_score:
                        self.gbest_score = sc
                        self.gbest_pos = self.particles[p].copy() # Salin seluruh posisi partikel p

            # Cek Early Stopping
            if self.gbest_score < best_iter_score - 1e-5: # Cek perbaikan signifikan
                best_iter_score = self.gbest_score
                no_improvement_iters = 0
            else:
                no_improvement_iters += 1

            if no_improvement_iters >= patience:
                break # Berhenti jika tidak ada perbaikan dalam 'patience' iterasi
            if self.gbest_score < 1e-6: # Berhenti jika solusi sudah (hampir) sempurna
                break
        # --- Akhir Iterasi PSO ---

        # Ekstrak layout final dari gbest_pos
        layout = []
        final_penalty = float('inf')
        if self.gbest_pos is not None and isinstance(self.gbest_pos, np.ndarray):
            # Hitung ulang penalti untuk gbest_pos TERBAIK yang ditemukan
            final_penalty = self._penalty(self.gbest_pos)
            pos = self.gbest_pos
            for i in range(self.n):
                # Ambil index orientasi dengan aman
                ori_idx_float = pos[4*i+3]
                ori_idx = int(np.clip(ori_idx_float, 0, self.K - 1))

                original_dims = self.items[i]['dims']
                w_rot, d_rot, h_rot = self._get_rotated_dims(original_dims, ori_idx)
                layout.append({
                    'id': self.items[i]['id'], 'name': self.items[i]['name'],
                    'x': pos[4*i], 'y': pos[4*i+1], 'z': pos[4*i+2],
                    'w': w_rot, 'd': d_rot, 'h': h_rot,
                    'orientation': ori_idx
                })
        else:
            # Fallback jika gbest_pos tidak valid (seharusnya tidak terjadi jika init benar)
            print(f"Warning: PackingPSO optimize finished but gbest_pos is invalid.")
            # Coba gunakan pbest terbaik sebagai fallback
            if self.pbest_score:
                 best_p_idx = np.argmin(self.pbest_score)
                 final_penalty = self.pbest_score[best_p_idx]
                 if final_penalty < float('inf'):
                     pos = self.pbest_pos[best_p_idx]
                     # Rekonstruksi layout dari pbest terbaik
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
                 else: layout = [] # pbest juga tidak valid
            else: layout = [] # Tidak ada pbest

        # Return layout TERBAIK yang ditemukan dan penaltinya
        return layout, final_penalty

# --- Load Data Jarak & Polygon (Jika Ada) ---
csv_filename = "city_to_city_polygon.csv"
distance_dict = {}
polygons = {}
if os.path.exists(csv_filename):
    try:
        df_distance = pd.read_csv(csv_filename)
        if all(col in df_distance.columns for col in ["CityA", "CityB", "Distance (meters)", "Polygon"]):
            for _, row in df_distance.iterrows():
                key = tuple(sorted((row["CityA"], row["CityB"])))
                distance_dict[key] = row["Distance (meters)"] / 1000.0 # Konversi ke KM
                try:
                    polygon_coords = json.loads(row["Polygon"])
                    # Pastikan format polygon [[lng, lat], [lng, lat], ...]
                    if isinstance(polygon_coords, list) and all(isinstance(p, dict) and 'lng' in p and 'lat' in p for p in polygon_coords):
                         polygons[key] = [[p["lng"], p["lat"]] for p in polygon_coords]
                    elif isinstance(polygon_coords, list) and all(isinstance(p, list) and len(p)==2 for p in polygon_coords):
                         polygons[key] = polygon_coords # Asumsi sudah [lng, lat]
                    else:
                         polygons[key] = [] # Format tidak dikenali
                except (json.JSONDecodeError, TypeError):
                    polygons[key] = [] # Gagal parse atau tipe salah
        else:
            st.error("File CSV 'city_to_city_polygon.csv' tidak memiliki kolom 'CityA', 'CityB', 'Distance (meters)', 'Polygon'.")
    except Exception as e:
        st.error(f"Gagal membaca file CSV: {e}")
else:
    st.warning(f"File jarak '{csv_filename}' tidak ditemukan. Akan digunakan jarak Haversine.")


# --- Data Input: Items, Truk, Kota ---
# (Gunakan data item Anda di sini)
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
truck_max_weight = 1000 # kg
truck_max_length = 200  # cm (asumsi satuan sama dengan item)
truck_max_width  = 150  # cm
truck_max_height = 150  # cm
truck_dims_tuple = (truck_max_length, truck_max_width, truck_max_height) # Definisikan sekali

city_coords = {
    "Surabaya": (-7.2575, 112.7521),
    "Jakarta": (-6.2088, 106.8456),
    "Bandung": (-6.9175, 107.6191),
    "Semarang": (-6.9667, 110.4167),
    "Yogyakarta": (-7.7956, 110.3695),
    "Malang": (-7.9824, 112.6304)
}

# --- Pre-process Items (Kategori Dimensi & Cek Oversized) ---
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

    # Cek apakah item bisa muat dalam *salah satu* dari 6 orientasi
    can_fit_any_orientation = False
    for i in range(6): # 0..5
        p = PackingPSO._get_rotated_dims((l, w, h), i)
        if p[0] <= truck_max_length and p[1] <= truck_max_width and p[2] <= truck_max_height:
            can_fit_any_orientation = True
            break
    item["is_oversized"] = not can_fit_any_orientation

    if item["is_oversized"]:
        print(f"  -> Warning: Item {item['id']} ({item['name']}) dims {item['dims']} IS OVERSIZED for truck {truck_max_length}x{truck_max_width}x{truck_max_height} in all 6 orientations.")
print("Dimension check complete.")


# --- Fungsi Jarak ---
@st.cache_data(show_spinner=False)
def haversine(coord1, coord2):
    """Menghitung jarak Haversine antara dua koordinat (lat, lon)."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371 # Radius bumi dalam km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

@st.cache_data(show_spinner=False)
def distance(city1, city2):
    """Mendapatkan jarak antara dua kota, utamakan dari data CSV, fallback ke Haversine."""
    if city1 == city2: return 0.0
    key = tuple(sorted((city1, city2)))
    if key in distance_dict:
        return distance_dict[key] # Jarak dari CSV (sudah dalam KM)
    elif city1 in city_coords and city2 in city_coords:
        return haversine(city_coords[city1], city_coords[city2]) # Jarak Haversine
    else:
        st.error(f"Tidak dapat menemukan jarak antara {city1} dan {city2} (koordinat tidak ada?).")
        return float('inf') # Return infinite jika tidak bisa hitung jarak

@st.cache_data(show_spinner=False)
def route_distance(cities):
    """Menghitung estimasi jarak rute menggunakan Nearest Neighbor heuristic."""
    if not cities: return 0.0
    # Gunakan list unik untuk routing, tapi input asli bisa punya duplikat kota
    unique_cities = list(dict.fromkeys(cities)) # Ambil unik tapi pertahankan urutan relatif (tidak terlalu penting untuk NN)
    if not unique_cities: return 0.0

    current = "Surabaya" # Asumsi selalu mulai dari Surabaya
    unvisited = unique_cities[:]
    total_dist = 0.0

    while unvisited:
        # Cari kota terdekat yang belum dikunjungi dari kota saat ini
        nearest = min(unvisited, key=lambda c: distance(current, c))
        total_dist += distance(current, nearest)
        current = nearest
        unvisited.remove(nearest)

    # Tambah jarak kembali ke Surabaya dari kota terakhir
    total_dist += distance(current, "Surabaya")
    return total_dist


# --- Fitness Function (Outer PSO) ---
def compute_fitness(assignment):
    """
    Mengevaluasi kualitas assignment item ke truk.
    Mempertimbangkan revenue, biaya bahan bakar, dan penalti (overweight, packing).
    Penalti overweight SANGAT TINGGI. Penalti packing menggunakan estimasi cepat.
    """
    truck_details = {
        t: {"items": [], "cities": set(), "weight": 0, "revenue": 0.0}
        for t in range(1, n_trucks + 1)
    }
    total_revenue = 0.0
    total_packing_penalty = 0.0 # Akumulasi penalti packing (estimasi)
    constraint_penalty = 0.0 # Akumulasi penalti constraint (overweight, dll)

    # 1. Proses Assignment & Hitung Revenue Awal
    for idx, truck_idx in enumerate(assignment):
        if truck_idx == 0: continue # Item tidak di-assign / oversized
        if not (1 <= truck_idx <= n_trucks): # Safety check
            constraint_penalty += 1e10 # Penalti besar untuk index truk invalid
            continue

        item = items[idx]
        # Seharusnya tidak terjadi jika decode benar, tapi cek lagi
        if item["is_oversized"]:
            # print(f"ERROR: Oversized item {item['id']} assigned to truck {truck_idx}!")
            constraint_penalty += 1e10 # Penalti sangat besar
            continue

        # Tambahkan item ke truk
        truck_details[truck_idx]["items"].append(item)
        truck_details[truck_idx]["cities"].add(item["city"])
        truck_details[truck_idx]["weight"] += item["weight"]

        # Hitung revenue item
        dist = distance("Surabaya", item["city"])
        if dist == float('inf'): # Jika jarak tidak bisa dihitung
             constraint_penalty += 1e9 # Penalti besar
             continue # Jangan proses item ini lebih lanjut
        rev = item["weight"] * dist * item["cat_factor"]
        truck_details[truck_idx]["revenue"] += rev
        total_revenue += rev

    # 2. Hitung Biaya & Penalti per Truk
    FUEL_PRICE_PER_L = 9000 # Rp per Liter
    TRUCK_CONSUMPTION_KM_L = 4 # km per liter
    cost_per_km = FUEL_PRICE_PER_L / TRUCK_CONSUMPTION_KM_L
    total_cost = 0.0

    for t in range(1, n_trucks + 1):
        info = truck_details[t]
        truck_items = info["items"]

        if not truck_items: continue # Lewati truk kosong

        # --- PENALTI OVERWEIGHT SANGAT TINGGI ---
        current_weight = info["weight"]
        if current_weight > truck_max_weight:
            overweight = current_weight - truck_max_weight
            constraint_penalty += OVERWEIGHT_CONSTANT_PENALTY # Penalti konstan besar
            constraint_penalty += OVERWEIGHT_FACTOR_PENALTY * (overweight ** 2) # Penalti proporsional kuadratik
        # --- Akhir Penalti Overweight ---

        # Hitung Biaya Bahan Bakar Rute
        if info["cities"]:
            dist = route_distance(list(info["cities"]))
            total_cost += cost_per_km * dist

        # Estimasi Penalti Packing (Gunakan hasil cache dari optimasi cepat)
        if not quick_feasible(truck_dims_tuple, truck_items):
            # Jika cek cepat gagal, beri penalti tinggi tanpa panggil packing detail
            total_packing_penalty += 50000.0 + 100 * len(truck_items) # Penalti tinggi tetap
        else:
            # Siapkan tuple item untuk kunci cache (harus hashable & urutan konsisten)
            items_tuple_key = tuple(sorted((it["id"], it["name"], it["dims"], it["weight"]) for it in truck_items))
            # Panggil fungsi cache yg menjalankan PackingPSO (parameter ringan)
            packing_pen = packing_penalty_cache(truck_dims_tuple, items_tuple_key)
            # Tambahkan penalti packing (mungkin perlu di-cap nilainya agar tidak terlalu dominan?)
            # total_packing_penalty += min(packing_pen, 75000) # Coba capping
            total_packing_penalty += packing_pen # Atau tanpa capping


    # 3. Hitung Fitness Total
    profit = total_revenue - total_cost
    # Fitness = Profit dikurangi SEMUA penalti. Kita ingin MAXIMIZE fitness.
    fitness = profit - total_packing_penalty - constraint_penalty

    return fitness


# --- Fungsi Decode Posisi PSO Luar ---
def decode_position(position):
    """Mengubah posisi kontinu partikel PSO menjadi assignment truk diskrit."""
    assignment = np.zeros(len(items), dtype=int) # Inisialisasi array integer
    for i, val in enumerate(position):
        if items[i]["is_oversized"]:
            assignment[i] = 0 # Tetap 0 (tidak di-assign)
        else:
            # Map nilai kontinu ke index truk [1, n_trucks]
            clamped_val = np.clip(val, 0.51, n_trucks + 0.49) # Batasi rentang sebelum pembulatan
            assigned_truck = int(round(clamped_val))
            # Pastikan hasil dalam rentang [1, n_trucks]
            assignment[i] = max(1, min(n_trucks, assigned_truck))
    return assignment


# --- Outer PSO: Assignment Item ke Truk ---
num_particles_assign = 30 # Jumlah partikel PSO assignment
max_iter_assign = 200     # Maks iterasi PSO assignment (bisa dikurangi karena evaluasi lebih cepat)
patience_assign = 30      # Patience untuk early stopping

# Parameter PSO Assignment
assign_w_max = 0.9
assign_w_min = 0.4
assign_c1 = 2.0
assign_c2 = 2.0

improvement_threshold = 10.0 # Minimal perbaikan fitness untuk reset patience
no_improvement_count = 0

particles_assign = []
velocities_assign = []
pbest_positions_assign = []
pbest_fitness_assign = []
gbest_position_assign = None
gbest_fitness_assign = -float('inf') # Maksimalkan fitness
prev_gbest_assign = -float('inf')

print("Initializing Assignment PSO...")
# Inisialisasi partikel assignment
for p_idx in range(num_particles_assign):
    # Posisi: Nilai kontinu, tiap dimensi merepresentasikan 1 item
    # Kecepatan: Perubahan posisi per iterasi
    position = np.zeros(len(items))
    velocity = np.zeros(len(items))
    for i in range(len(items)):
        if items[i]["is_oversized"]:
            position[i] = 0 # Tetap 0
            velocity[i] = 0
        else:
            # Posisi awal acak antara 1 dan n_trucks
            position[i] = random.uniform(1, n_trucks + 1e-9)
            # Kecepatan awal acak
            velocity[i] = random.uniform(-(n_trucks/2.0), n_trucks/2.0) * 0.1 # Kecepatan awal kecil

    particles_assign.append(position.copy())
    velocities_assign.append(velocity.copy())

    # Evaluasi posisi awal
    assignment = decode_position(position)
    fit = compute_fitness(assignment)

    pbest_positions_assign.append(position.copy())
    pbest_fitness_assign.append(fit)

    # Update gbest awal
    if fit > gbest_fitness_assign:
        gbest_fitness_assign = fit
        gbest_position_assign = position.copy()

print(f"Initial Global Best Fitness: {gbest_fitness_assign:,.0f}")

# --- Loop Utama PSO Assignment ---
fitness_history_assign = []
print(f"Running Assignment PSO for max {max_iter_assign} iterations...")
progress_bar = st.progress(0)
status_text = st.empty()

for it in range(1, max_iter_assign + 1):
    w_inertia = assign_w_max - (assign_w_max - assign_w_min) * (it / max_iter_assign)

    for i in range(num_particles_assign):
        r1 = np.random.rand(len(items))
        r2 = np.random.rand(len(items))

        current_particle_pos = particles_assign[i]
        current_particle_vel = velocities_assign[i]

        # Mask untuk item yang bisa di-assign (tidak oversized)
        is_assignable = np.array([not item["is_oversized"] for item in items])

        # --- Update Velocity (Hanya untuk item assignable) ---
        # Komponen kognitif & sosial hanya dihitung jika pbest/gbest valid
        cognitive_comp = np.zeros_like(current_particle_pos)
        social_comp = np.zeros_like(current_particle_pos)

        if pbest_positions_assign[i] is not None:
             cognitive_comp = assign_c1 * r1 * (pbest_positions_assign[i] - current_particle_pos)
        if gbest_position_assign is not None:
             social_comp = assign_c2 * r2 * (gbest_position_assign - current_particle_pos)

        new_vel = (w_inertia * current_particle_vel + cognitive_comp + social_comp)

        # Terapkan velocity baru HANYA pada item assignable
        velocities_assign[i][is_assignable] = new_vel[is_assignable]

        # --- Update Position (Hanya untuk item assignable) ---
        particles_assign[i][is_assignable] += velocities_assign[i][is_assignable]

        # --- Clamp Position (Hanya untuk item assignable) ---
        particles_assign[i][is_assignable] = np.clip(particles_assign[i][is_assignable], 0.51, n_trucks + 0.49)

        # Decode dan Evaluasi Fitness
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

    # Catat history & cek early stopping
    fitness_history_assign.append(gbest_fitness_assign)
    improvement = gbest_fitness_assign - prev_gbest_assign

    status_text.text(f"Iter {it}/{max_iter_assign}, Best Fitness: {gbest_fitness_assign:,.0f}")
    progress_bar.progress(it / max_iter_assign)

    # Cek perbaikan untuk reset patience
    if improvement >= improvement_threshold:
        no_improvement_count = 0
        prev_gbest_assign = gbest_fitness_assign
    else:
        if it > patience_assign : # Mulai hitung setelah iterasi awal
             no_improvement_count += 1

    # Hentikan jika patience terlampaui
    if no_improvement_count >= patience_assign:
        print(f"\nEarly stopping at iteration {it} (no improvement for {patience_assign} iters).")
        status_text.text(f"Early stopping at iteration {it}. Final Best Fitness: {gbest_fitness_assign:,.0f}")
        break

progress_bar.empty() # Hapus progress bar
print(f"\nAssignment PSO finished. Final Best Fitness: {gbest_fitness_assign:,.0f}")

# --- Dapatkan Assignment Terbaik ---
if gbest_position_assign is not None:
    best_assignment = decode_position(gbest_position_assign)
else:
    # Fallback jika gbest tidak pernah ter-set (seharusnya tidak terjadi)
    print("Error: gbest_position_assign is None after PSO. Using initial best guess.")
    # Cari pbest terbaik sebagai fallback
    if pbest_fitness_assign:
         best_pbest_idx = np.argmax(pbest_fitness_assign) # Cari max fitness
         best_assignment = decode_position(pbest_positions_assign[best_pbest_idx])
    else: # Jika pbest juga kosong
         best_assignment = np.zeros(len(items), dtype=int) # Semua unassigned

# Hapus cache jika perlu (misal jika input berubah drastis antar run)
# packing_penalty_cache.cache_clear()

# --- Proses Hasil Assignment: Dapatkan Info Detail per Truk ---
def get_truck_info(assignment):
    """Mengumpulkan detail item, berat, volume, kota, revenue, dll. per truk."""
    truck_info = {
        t: {"items": [], "weight": 0, "volume": 0, "cities": [], "revenue": 0.0, "is_overweight": False}
        for t in range(1, n_trucks + 1)
    }
    assigned_item_details = [] # Untuk tabel ringkasan item

    for i, truck_idx in enumerate(assignment):
        item = items[i]
        if truck_idx == 0: # Item tidak di-assign atau oversized
             assigned_item_details.append({**item, "assigned_truck": "Unassigned/Oversized"})
             continue

        # Tambahkan item ke truk yang sesuai
        truck_info[truck_idx]["items"].append(item)
        truck_info[truck_idx]["weight"] += item["weight"]
        l, w, h = item["dims"]
        truck_info[truck_idx]["volume"] += (l * w * h) # Volume item aktual
        # Hitung revenue (lagi? Sebenarnya sudah di fitness, tapi ok untuk info final)
        dist = distance("Surabaya", item["city"])
        if dist != float('inf'):
             rev = item["weight"] * dist * item["cat_factor"]
             truck_info[truck_idx]["revenue"] += rev
        # Tambahkan kota tujuan (unik)
        if item["city"] not in truck_info[truck_idx]["cities"]:
            truck_info[truck_idx]["cities"].append(item["city"]) # Gunakan list agar urutan bisa dijaga (meski NN tidak perlu)

        assigned_item_details.append({**item, "assigned_truck": f"Truk {truck_idx}"})

    # Hitung biaya & profit final per truk & cek overweight final
    FUEL_PRICE_PER_L = 9000
    TRUCK_CONSUMPTION_KM_L = 4
    cost_per_km = FUEL_PRICE_PER_L / TRUCK_CONSUMPTION_KM_L
    for t in range(1, n_trucks + 1):
        # Hitung Jarak Rute
        route_dist = route_distance(truck_info[t]["cities"]) # Gunakan list kota unik dari truk t
        truck_info[t]["route_distance"] = route_dist
        # Hitung Biaya Bahan Bakar
        truck_info[t]["fuel_cost"] = cost_per_km * route_dist
        # Hitung Profit Truk
        truck_info[t]["profit"] = truck_info[t]["revenue"] - truck_info[t]["fuel_cost"]
        # Cek Status Overweight Final
        if truck_info[t]["weight"] > truck_max_weight:
            truck_info[t]["is_overweight"] = True
            print(f"!!! PERINGATAN FINAL: Truk {t} MELEBIHI BERAT ({truck_info[t]['weight']} kg) !!!")

    return truck_info, pd.DataFrame(assigned_item_details)

final_truck_info, assigned_items_df = get_truck_info(best_assignment)

# --- Fungsi Bantuan untuk Rute Visualisasi Peta ---
def get_route_sequence(cities):
    """Menentukan urutan kota yang dikunjungi (Nearest Neighbor dari Surabaya)."""
    if not cities: return ["Surabaya"] # Hanya base jika tidak ada tujuan
    unique_cities = list(dict.fromkeys(cities)) # Ambil unik
    if not unique_cities: return ["Surabaya"]

    route = ["Surabaya"]
    current = "Surabaya"
    unvisited = unique_cities[:]
    while unvisited:
        nearest = min(unvisited, key=lambda c: distance(current, c))
        route.append(nearest)
        current = nearest
        unvisited.remove(nearest)
    route.append("Surabaya") # Kembali ke base
    return route

def get_segment_path(city_a, city_b):
    """Mendapatkan segmen path (list koordinat [lng, lat]) antara dua kota."""
    if city_a == city_b: return [] # Tidak ada path jika kota sama
    key = tuple(sorted((city_a, city_b)))

    path = []
    # Coba ambil dari data polygon jika ada dan valid
    if key in polygons and isinstance(polygons.get(key), list) and len(polygons[key]) >= 2:
        path_data = polygons[key]
        # Cek arah sederhana: apakah titik awal polygon lebih dekat ke city_a?
        coord_a = city_coords[city_a]
        coord_b = city_coords[city_b]
        # Ambil koordinat titik awal polygon (asumsi [lng, lat])
        path_start_coord = (path_data[0][1], path_data[0][0]) # (lat, lon)
        dist_start_a = haversine(coord_a, path_start_coord)
        dist_start_b = haversine(coord_b, path_start_coord)

        if dist_start_b < dist_start_a - 1e-3: # Jika lebih dekat ke city_b (beri toleransi), balik arah
             path = path_data[::-1]
        else:
             path = path_data # Arah sudah benar atau sama jauh

    # Fallback ke garis lurus jika tidak ada polygon atau polygon tidak valid
    if not path:
        start = [city_coords[city_a][1], city_coords[city_a][0]] # [lon, lat]
        end = [city_coords[city_b][1], city_coords[city_b][0]]   # [lon, lat]
        path = [start, end]
    return path

def get_full_route_path(route_sequence):
    """Membangun path koordinat lengkap untuk visualisasi rute."""
    full_path = []
    if not route_sequence or len(route_sequence) < 2: return []

    for i in range(len(route_sequence) - 1):
        city_a = route_sequence[i]
        city_b = route_sequence[i+1]
        segment = get_segment_path(city_a, city_b)

        if not segment: continue # Lewati jika segmen kosong

        # Tambahkan segmen, hindari duplikasi titik sambungan
        if not full_path:
            full_path.extend(segment)
        else:
            # Cek apakah titik akhir path sama dengan titik awal segmen
            if (abs(full_path[-1][0] - segment[0][0]) < 1e-6 and
                abs(full_path[-1][1] - segment[0][1]) < 1e-6):
                full_path.extend(segment[1:]) # Tambah mulai dari titik kedua segmen
            else:
                 full_path.extend(segment) # Tambah seluruh segmen jika tidak nyambung

    return full_path

# --- Siapkan Data Rute untuk Pydeck ---
routes_data_pydeck = []
# Palet warna bisa diperbanyak jika truk > 6
colors = [
    [255, 0, 0, 200],   # Merah
    [0, 180, 0, 200],   # Hijau
    [0, 0, 255, 200],   # Biru
    [255, 165, 0, 200], # Oranye
    [128, 0, 128, 200], # Ungu
    [0, 200, 200, 200]  # Cyan
]

print("Generating route paths for visualization...")
for t in range(1, n_trucks + 1):
    info = final_truck_info[t]
    if info["cities"]: # Hanya jika truk punya tujuan
        route_seq = get_route_sequence(info["cities"])
        full_path = get_full_route_path(route_seq)

        if full_path: # Hanya jika path berhasil dibuat
            # Info untuk tooltip (hapus Surabaya awal/akhir)
            display_route = " → ".join([c for c in route_seq if c != "Surabaya"])
            if not display_route: display_route = "Hanya Base"

            routes_data_pydeck.append({
                "truck": f"Truk {t}",
                "path": full_path,
                "color": colors[(t - 1) % len(colors)], # Ambil warna cyclical
                "route_info": display_route,
                "distance_km": info.get("route_distance", 0) # Ambil jarak dari info truk
            })
        else:
             print(f"Warning: Gagal membuat path untuk Truk {t} dengan kota {info['cities']}")

# --- Packing Final Kualitas Tinggi untuk Visualisasi ---
final_layouts = {}
print("Running final (high-quality) packing optimization for visualization...")
packing_progress = st.progress(0)
packing_status = st.empty()

for idx, t in enumerate(range(1, n_trucks + 1)):
    items_for_truck = final_truck_info[t]["items"]
    packing_status.text(f"Optimizing packing for Truck {t} ({len(items_for_truck)} items)...")

    if not items_for_truck: # Truk kosong
        final_layouts[t] = ([], 0.0)
        packing_progress.progress((idx + 1) / n_trucks)
        continue

    # --- Packing dengan Parameter Kualitas Tinggi ---
    best_penalty_viz = float('inf')
    best_layout_viz = None
    num_packing_attempts = 3 # Berapa kali coba packing dengan seed berbeda

    for attempt in range(num_packing_attempts):
        seed = 42 + attempt * 101 # Seed berbeda tiap attempt
        current_np_state = np.random.get_state()
        np.random.seed(seed)

        # Gunakan Partikel/Iterasi FINAL_PACKING_*
        packer = PackingPSO(truck_dims_tuple, items_for_truck,
                            num_particles=FINAL_PACKING_NUM_PARTICLES)
        layout, penalty = packer.optimize(max_iters=FINAL_PACKING_MAX_ITERS)

        np.random.set_state(current_np_state) # Kembalikan state random

        if penalty < best_penalty_viz:
            best_penalty_viz = penalty
            best_layout_viz = layout

        # Jika sudah sangat bagus, tidak perlu coba lagi
        if best_penalty_viz < 1e-5:
            break

    print(f"  Truck {t} - Best Final Packing Penalty: {best_penalty_viz:.4f}")
    final_layouts[t] = (best_layout_viz, best_penalty_viz) # Simpan layout & penalti terbaik
    packing_progress.progress((idx + 1) / n_trucks) # Update progress bar

packing_status.text("Final packing optimizations complete.")
packing_progress.empty()


# --- Fungsi Visualisasi Plotly 3D ---
def create_truck_figure(truck_dims, packed_items):
    """Membuat figure Plotly 3D untuk visualisasi packing."""
    L, W, H = truck_dims
    fig = go.Figure()

    if not packed_items: # Jika tidak ada item, hanya gambar outline truk
        pass # Outline digambar di bawah

    color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                     "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"] # Palet warna Plotly default

    # Tambahkan item sebagai Mesh3d
    for idx, item in enumerate(packed_items):
        x, y, z = item['x'], item['y'], item['z'] # Posisi pojok item
        w_rot, d_rot, h_rot = item['w'], item['d'], item['h'] # Dimensi terotasi

        # Koordinat 8 titik sudut box item
        x_verts = [x, x, x + w_rot, x + w_rot, x, x, x + w_rot, x + w_rot]
        y_verts = [y, y + d_rot, y + d_rot, y, y, y + d_rot, y + d_rot, y]
        z_verts = [z, z, z, z, z + h_rot, z + h_rot, z + h_rot, z + h_rot]

        # Definisi wajah kubus menggunakan index titik sudut (triangulasi standar)
        faces_i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 1]
        faces_j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 2]
        faces_k = [0, 7, 2, 3, 6, 7, 1, 5, 4, 5, 7, 6]

        item_color = color_palette[idx % len(color_palette)] # Ambil warna cyclical

        fig.add_trace(go.Mesh3d(
            x=x_verts, y=y_verts, z=z_verts,
            i=faces_i, j=faces_j, k=faces_k, # Definisikan segitiga wajah
            color=item_color, opacity=0.85, # Warna & opasitas item
            name=item['name'], # Nama item untuk legenda
            # Info saat hover
            hovertext=f"<b>{item['name']}</b><br>ID: {item['id']}<br>Dims: {w_rot:.1f}x{d_rot:.1f}x{h_rot:.1f}<br>Pos: ({x:.1f},{y:.1f},{z:.1f})<br>Ori: {item.get('orientation', 'N/A')}",
            hoverinfo="text"
        ))

    # Tambahkan outline truk (wireframe hitam)
    corners = [(0,0,0), (L,0,0), (L,W,0), (0,W,0), (0,0,H), (L,0,H), (L,W,H), (0,W,H)]
    edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4),
             (0,4), (1,5), (2,6), (3,7)] # 12 rusuk kubus
    edge_x, edge_y, edge_z = [], [], []
    for (u, v) in edges:
        edge_x.extend([corners[u][0], corners[v][0], None]) # None untuk memutus garis
        edge_y.extend([corners[u][1], corners[v][1], None])
        edge_z.extend([corners[u][2], corners[v][2], None])

    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines', line=dict(color='black', width=3), # Garis hitam tebal
        hoverinfo='none', showlegend=False # Tidak perlu info hover/legenda untuk outline
    ))

    # Konfigurasi layout scene 3D
    fig.update_layout(
        # title_text=f"Muatan Truk ({L}x{W}x{H})", # Judul ditambahkan di Streamlit
        scene=dict(
            xaxis=dict(title='Panjang (X)', range=[0, L], backgroundcolor="rgb(240, 240, 240)"),
            yaxis=dict(title='Lebar (Y)', range=[0, W], backgroundcolor="rgb(235, 235, 235)"),
            zaxis=dict(title='Tinggi (Z)', range=[0, H], backgroundcolor="rgb(240, 240, 240)"),
            # Sesuaikan aspect ratio agar proporsional
            aspectratio=dict(x=1, y=W/L, z=H/L) if L > 0 else dict(x=1,y=1,z=1),
            aspectmode='manual', # Gunakan rasio yang dihitung
            # Atur sudut pandang kamera awal
            camera_eye=dict(x=1.8, y=1.8, z=0.9)
        ),
        margin=dict(l=5, r=5, t=5, b=5), # Margin kecil
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="right", x=1) # Legenda horizontal di bawah
    )
    return fig


# --- Layout Aplikasi Streamlit ---
st.title("🚚 Optimasi Penugasan & Pemuatan Truk dengan PSO")
st.markdown("---")

# --- Baris 1: Metrik Ringkasan ---
col1, col2, col3, col4 = st.columns(4)
# Hitung profit hanya dari truk yang tidak overweight? Atau total? Mari hitung total revenue/cost dulu.
total_revenue_final = sum(info['revenue'] for info in final_truck_info.values())
total_cost_final = sum(info['fuel_cost'] for info in final_truck_info.values())
total_profit_final = total_revenue_final - total_cost_final # Profit kotor sebelum penalti
num_assigned_items = sum(len(info['items']) for info in final_truck_info.values())
num_overweight = sum(1 for info in final_truck_info.values() if info['is_overweight'])

col1.metric("Total Estimasi Profit*", f"Rp {total_profit_final:,.0f}", help="Profit = Total Revenue - Total Fuel Cost. Belum termasuk penalti packing/overweight.")
col2.metric("Total Revenue", f"Rp {total_revenue_final:,.0f}")
col3.metric("Total Fuel Cost", f"Rp {total_cost_final:,.0f}")
col4.metric("Item Ter-assign", f"{num_assigned_items} / {len(items)}")

if num_overweight > 0:
    st.warning(f"🚨 **PERHATIAN:** {num_overweight} truk terdeteksi **MELEBIHI BATAS BERAT MAKSIMUM!** Solusi ini mungkin tidak valid secara praktis.")
st.markdown("---")

# --- Baris 2: Peta Rute & Grafik Fitness ---
col1_map, col2_chart = st.columns([3, 2]) # Peta lebih lebar

with col1_map:
    st.subheader("🗺️ Peta Rute Pengiriman")
    if routes_data_pydeck:
        city_points = [{"name": city, "coordinates": [coord[1], coord[0]]} # lon, lat
                       for city, coord in city_coords.items()]
        marker_layer = pdk.Layer(
            "ScatterplotLayer", data=city_points, get_position="coordinates",
            get_fill_color=[0, 0, 0, 180], get_radius=8000, radius_min_pixels=6,
            pickable=True, auto_highlight=True
        )
        text_layer = pdk.Layer(
            "TextLayer", data=city_points, get_position="coordinates", get_text="name",
            get_color=[0, 0, 0, 200], get_size=15, get_alignment_baseline="'bottom'",
            get_pixel_offset=[0, -18]
        )
        path_layers = [
            pdk.Layer(
                "PathLayer", data=[route], get_path="path", get_color="color",
                get_width=5, width_scale=1, width_min_pixels=3.5,
                pickable=True, auto_highlight=True
            ) for route in routes_data_pydeck
        ]
        view_state = pdk.ViewState(
            latitude=city_coords["Surabaya"][0], longitude=city_coords["Surabaya"][1],
            zoom=5.8, pitch=45 # Sesuaikan zoom & pitch
        )
        tooltip = {
            "html": "<b>{truck}</b><br/>Rute: {route_info}<br/>Jarak: {distance_km:.1f} km",
            "style": {"backgroundColor": "rgba(0,0,0,0.7)", "color": "white", "fontSize": "12px"}
        }
        st.pydeck_chart(pdk.Deck(
            layers=path_layers + [marker_layer, text_layer],
            initial_view_state=view_state,
            map_style='mapbox://styles/mapbox/outdoors-v11', # Coba style map berbeda
            tooltip=tooltip
        ))
    else:
        st.info("Tidak ada rute pengiriman untuk ditampilkan.")

with col2_chart:
    st.subheader("📈 Grafik Fitness Assignment PSO")
    if fitness_history_assign:
        fig_fitness = go.Figure()
        fig_fitness.add_trace(go.Scatter(y=fitness_history_assign, mode='lines', name='Best Fitness'))
        fig_fitness.update_layout(
            title="Perkembangan Fitness Terbaik",
            xaxis_title="Iterasi", yaxis_title="Fitness Value", height=400,
            margin=dict(l=20, r=20, t=50, b=20) # Beri ruang untuk judul
        )
        st.plotly_chart(fig_fitness, use_container_width=True)
    else:
        st.info("Data histori fitness tidak tersedia.")

st.markdown("---")

# --- Baris 3: Detail Assignment Item ---
st.subheader("📦 Detail Penugasan Item")
# Tampilkan dataframe dengan kolom yang relevan
st.dataframe(assigned_items_df[['id', 'name', 'weight', 'dims', 'city', 'dim_category', 'assigned_truck']])
st.markdown("---")

# --- Baris 4 dst.: Detail & Visualisasi per Truk ---
st.subheader("🚛 Detail Muatan & Visualisasi per Truk")

cols_per_row = 2 # Tampilkan 2 truk per baris
num_rows = (n_trucks + cols_per_row - 1) // cols_per_row

truck_idx = 1
for r in range(num_rows):
    cols = st.columns(cols_per_row)
    for c in range(cols_per_row):
        if truck_idx <= n_trucks:
            with cols[c]:
                st.markdown(f"#### Truk {truck_idx}")
                info = final_truck_info[truck_idx]
                layout, penalty = final_layouts.get(truck_idx, (None, float('inf')))

                # Tampilkan status berat & profit
                if info["is_overweight"]:
                     st.error(f"⛔️ OVERWEIGHT: {info['weight']} / {truck_max_weight} kg")
                else:
                     st.metric("Berat Muatan", f"{info['weight']} / {truck_max_weight} kg",
                               delta=f"{truck_max_weight - info['weight']:.0f} kg Sisa", delta_color="normal")
                st.metric("Profit Truk Ini*", f"Rp {info['profit']:,.0f}", help="Revenue Truk - Fuel Cost Truk")

                if not info["items"]:
                    st.info("Truk ini tidak mengangkut barang.")
                else:
                    # Tampilkan info ringkas lain
                    st.caption(f"**Tujuan:** {', '.join(info['cities'])} | **Jarak:** {info['route_distance']:.1f} km")

                    # Expander untuk detail item di truk ini
                    with st.expander("Lihat item di truk ini"):
                         df_truck = pd.DataFrame(info["items"])[["id", "name", "weight", "dims", "city"]]
                         st.dataframe(df_truck)

                    # Visualisasi Packing 3D
                    st.markdown("**Visualisasi Muatan:**")
                    if layout is None and info["items"]:
                         st.error("Packing gagal menghasilkan layout untuk truk ini.")
                    elif penalty > 1e-3 and info["items"]: # Penalti tinggi
                         st.warning(f"Packing belum optimal (Penalty: {penalty:.4f}). Visualisasi mungkin tumpang tindih/tidak stabil.")
                         fig = create_truck_figure(truck_dims_tuple, layout)
                         st.plotly_chart(fig, use_container_width=True)
                    elif info["items"]: # Packing bagus
                         st.success(f"Muatan terpack (Penalty: {penalty:.4f}).")
                         fig = create_truck_figure(truck_dims_tuple, layout)
                         st.plotly_chart(fig, use_container_width=True)

                st.markdown("---") # Pemisah antar truk dalam kolom
                truck_idx += 1
        # else: pass # Kolom kosong jika jumlah truk ganjil

print("\nStreamlit App Ready.")