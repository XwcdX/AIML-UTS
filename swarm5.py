# -*- coding: utf-8 -*-
import streamlit as st
import math
import random
import pydeck as pdk
import pandas as pd
import json
import os
import numpy as np
# import itertools # combinations, permutations -> combinations tidak terpakai
from itertools import permutations
import plotly.graph_objects as go
from functools import lru_cache
import logging # Tambahkan logging untuk debug

# --- Konfigurasi Logging ---
# Ganti level ke DEBUG jika ingin melihat log support per item, dll.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Konfigurasi Halaman Streamlit (HARUS JADI PERINTAH st PERTAMA) ---
st.set_page_config(layout="wide", page_title="Optimasi Muatan Truk PSO V2 Final")
# --------------------------------------------------------------------

# --- Seed untuk Reproducibility ---
RANDOM_SEED = 30
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# -----------------------------------

# --- Constants: Tolerances & Bounds ---
POS_LOWER_BOUND_ASSIGN = 0.51 # Batas bawah untuk decoding posisi assignment
# Batas atas dihitung dinamis: n_trucks + (1.0 - POS_LOWER_BOUND_ASSIGN)
ORIENTATION_UPPER_BOUND_PACK = 6.0 - 1e-9 # Indeks orientasi maks (0-5)
ZERO_TOLERANCE = 1e-6 # Toleransi untuk perbandingan float (misal volume overlap)
SUPPORT_TOLERANCE = 1e-4 # Toleransi untuk pengecekan support Z

# --- Constants: Penalty Factors ---
OVERLAP_PENALTY_FACTOR = 4.0 # Penalti volume overlap
OUTBOUND_PENALTY_FACTOR = 2.0 # Penalti volume keluar batas
# --- Constants: Constraint Penalties ---
OVERWEIGHT_CONSTANT_PENALTY = 1e9 # Penalti dasar jika overweight
OVERWEIGHT_FACTOR_PENALTY = 10000.0 # Penalti tambahan overweight (quadratic)
# --- PENALTI SUPPORT BARU (Contoh, perlu tuning) ---
# Coba turunkan ini jika penalti support terlalu dominan
UNSUPPORTED_AREA_PENALTY_FACTOR = 50000.0 # Penalti berdasarkan rasio area tidak didukung
# --------------------------------------------------
HEIGHT_VOLUME_PENALTY_FACTOR = 0.005 # Penalti menempatkan item besar di atas
PACKING_IMPOSSIBLE_PENALTY = 50000.0 # Penalti jika quick_feasible gagal
PACKING_IMPOSSIBLE_ITEM_FACTOR = 100.0 # Penalti tambahan per item jika quick_feasible gagal
INVALID_ROUTE_PENALTY = 1e9 # Penalti jika rute tidak bisa dihitung
INVALID_ASSIGNMENT_PENALTY = 1e10 # Penalti jika assignment ke truk invalid
OVERSIZED_ITEM_PENALTY = 1e10 # Penalti jika item oversized coba diassign

# --- Constants: Packing PSO Cache (Estimasi Cepat) ---
# Pastikan tidak ada spasi aneh/non-breaking space di akhir baris ini
PACKING_CACHE_MAX_ITERS = 35 # Naikkan sedikit (dari 30)
PACKING_CACHE_NUM_PARTICLES = 20 # Naikkan sedikit (dari 18)
PACKING_CACHE_PATIENCE_FACTOR = 0.15 # Patience = max_iters * factor
PACKING_CACHE_COMPACTION_WEIGHT = 1e-4 # Mungkin lebih kecil untuk cache?

# --- Constants: Packing PSO Final (Visualisasi Kualitas Tinggi) ---
# Pastikan tidak ada spasi aneh/non-breaking space di akhir baris ini
FINAL_PACKING_MAX_ITERS = 600 # Tetap (dari 600)
FINAL_PACKING_NUM_PARTICLES = 60 # Tetap (dari 60)
FINAL_PACKING_PATIENCE_FACTOR = 0.1 # Patience = max_iters * factor
FINAL_PACKING_COMPACTION_WEIGHT = 1e-3 # Compaction lebih penting di final
FINAL_PACKING_ATTEMPTS = 3 # Jumlah percobaan packing final

# --- Constants: General PSO Parameters ---
W_MAX = 0.9
W_MIN = 0.4
C1_PACKING = 1.5 # Kognitif untuk packing
C2_PACKING = 1.5 # Sosial untuk packing
C1_ASSIGN = 2.0 # Kognitif untuk assignment
C2_ASSIGN = 2.0 # Sosial untuk assignment

# --- Constants: Routing & Cost ---
FUEL_PRICE_PER_L = 9000 # Harga BBM per liter
TRUCK_CONSUMPTION_KM_L = 4 # Konsumsi BBM truk (km/liter)
COST_PER_KM = FUEL_PRICE_PER_L / TRUCK_CONSUMPTION_KM_L

# --- Constants: Item Categorization ---
THRESHOLD_SMALL_VOLUME = 50*50*50
THRESHOLD_MEDIUM_VOLUME = 100*100*100
CAT_FACTOR_SMALL = 50
CAT_FACTOR_MEDIUM = 75
CAT_FACTOR_LARGE = 100

# --- Constants: Lain-lain ---
DEFAULT_START_CITY = "Surabaya"

# --- Helper Function: Quick Feasibility ---
def quick_feasible(truck_dims, items):
    """Pengecekan cepat kelayakan packing berdasarkan volume total dan luas alas minimum."""
    L, W, H = truck_dims
    if not items: return True, "Empty truck" # Truk kosong pasti feasible

    # 1. Cek Volume Total
    total_item_vol = sum(it['dims'][0] * it['dims'][1] * it['dims'][2] for it in items)
    if total_item_vol > L * W * H + ZERO_TOLERANCE: # Beri toleransi kecil
        reason = f"Total volume {total_item_vol:.2f} > Truck volume {L*W*H:.2f}"
        logging.debug(f"quick_feasible failed: {reason}")
        return False, reason

    # 2. Cek Luas Alas Minimum Total (Heuristik kasar)
    min_total_base_area = sum(min(l*w, l*h, w*h) for l, w, h in (it['dims'] for it in items))
    if min_total_base_area > L*W + ZERO_TOLERANCE:
        reason = f"Min base area {min_total_base_area:.2f} > Truck base area {L*W:.2f}"
        logging.debug(f"quick_feasible failed: {reason}")
        return False, reason

    # 3. Cek Dimensi Individu (sudah di pre-processing)

    return True, "Feasible"

# --- Cache for Packing Results (Lightweight Parameters) ---
@lru_cache(maxsize=512)
def packing_penalty_cache(truck_dims, items_tuple):
    """
    Menjalankan PackingPSO dengan parameter CEPAT untuk estimasi penalti packing.
    Hasilnya di-cache. Menggunakan seed konsisten untuk reproduktifitas cache.
    """
    items_list = [ {'id':i[0], 'name':i[1], 'dims':i[2], 'weight':i[3]} for i in items_tuple ]
    if not items_list:
        return 0.0

    current_np_state = np.random.get_state()
    np.random.seed(42) # Seed konsisten untuk cache

    try:
        packer = PackingPSO(truck_dims, items_list,
                            num_particles=PACKING_CACHE_NUM_PARTICLES,
                            compaction_weight=PACKING_CACHE_COMPACTION_WEIGHT,
                            c1=C1_PACKING, c2=C2_PACKING)
        patience = max(5, int(PACKING_CACHE_MAX_ITERS * PACKING_CACHE_PATIENCE_FACTOR))
        # Ambil hanya penalti total untuk cache
        _, penalty, _ = packer.optimize(max_iters=PACKING_CACHE_MAX_ITERS, patience=patience)
    finally:
        np.random.set_state(current_np_state)

    # Kembalikan hanya penalti total untuk fitness function
    return penalty

# --- Core Class: Packing PSO ---
class PackingPSO:
    """
    Optimasi packing 3D menggunakan PSO. Mempertimbangkan 6 orientasi standar.
    Termasuk penalti support berbasis area (basic) dan penalti ketinggian item besar.
    """
    def __init__(self, truck_dims, items,
                 compaction_weight=FINAL_PACKING_COMPACTION_WEIGHT,
                 num_particles=FINAL_PACKING_NUM_PARTICLES,
                 c1=C1_PACKING, c2=C2_PACKING):
        self.L, self.W, self.H = truck_dims
        self.items = items
        self.n = len(items)
        if self.n == 0:
            self.dim = 0; self.particles = []; self.velocities = []
            self.pbest_pos = []; self.pbest_score = []; self.gbest_pos = None
            self.gbest_score = 0.0; self.gbest_details = {}
            logging.info("PackingPSO initialized with 0 items.")
            return

        self.dim = self.n * 4
        self.K = 6
        self.num_particles = num_particles
        self.c1 = c1
        self.c2 = c2
        self.comp_w = compaction_weight

        self.item_original_dims = [it['dims'] for it in items]
        self.item_original_volumes = [l*w*h for l,w,h in self.item_original_dims]

        self.particles = np.zeros((self.num_particles, self.dim))
        self.velocities = np.zeros((self.num_particles, self.dim))
        self.pbest_pos = np.zeros((self.num_particles, self.dim))
        self.pbest_score = np.full(self.num_particles, float('inf'))
        self.gbest_pos = None
        self.gbest_score = float('inf')
        self.gbest_details = {} # Simpan detail penalti gbest

        logging.info(f"Initializing PackingPSO with {self.n} items, {self.num_particles} particles.")
        for p in range(self.num_particles):
            pos = np.zeros(self.dim)
            vel = np.zeros(self.dim)
            for i in range(self.n):
                pos[4*i+0] = np.random.uniform(0, self.L)
                pos[4*i+1] = np.random.uniform(0, self.W)
                pos[4*i+2] = np.random.uniform(0, self.H)
                pos[4*i+3] = np.random.uniform(0, self.K)

                vel_range = 0.1
                vel[4*i+0] = np.random.uniform(-self.L*vel_range, self.L*vel_range)
                vel[4*i+1] = np.random.uniform(-self.W*vel_range, self.W*vel_range)
                vel[4*i+2] = np.random.uniform(-self.H*vel_range, self.H*vel_range)
                vel[4*i+3] = np.random.uniform(-self.K*vel_range, self.K*vel_range)

            self._clamp(pos)
            # Hitung skor awal dan detailnya (meskipun detail mungkin belum optimal)
            score, details = self._penalty(pos)

            self.particles[p] = pos.copy()
            self.velocities[p] = vel.copy()
            self.pbest_pos[p] = pos.copy()
            self.pbest_score[p] = score

            if score < self.gbest_score:
                self.gbest_score = score
                self.gbest_pos = pos.copy()
                self.gbest_details = details.copy() # Simpan detail awal

        if self.gbest_pos is None and self.n > 0:
            best_initial_p = np.argmin(self.pbest_score)
            self.gbest_score = self.pbest_score[best_initial_p]
            self.gbest_pos = self.pbest_pos[best_initial_p].copy()
            # Hitung ulang detail untuk gbest awal fallback
            _, self.gbest_details = self._penalty(self.gbest_pos)
            logging.warning("Initial gbest was None, setting from best pbest.")

        logging.info(f"PackingPSO Initialization Complete. Initial GBest Score: {self.gbest_score:.4f}")

    @staticmethod
    def _get_rotated_dims(original_dims, orientation_index):
        l, w, h = original_dims
        dims_map = [
            (l, w, h), (l, h, w), (w, l, h),
            (w, h, l), (h, l, w), (h, w, l),
        ]
        safe_index = int(np.clip(orientation_index, 0, 5))
        return dims_map[safe_index]

    def _clamp(self, pos):
        for i in range(self.n):
            pos[4*i+0] = np.clip(pos[4*i+0], 0, self.L)
            pos[4*i+1] = np.clip(pos[4*i+1], 0, self.W)
            pos[4*i+2] = np.clip(pos[4*i+2], 0, self.H)
            pos[4*i+3] = np.clip(pos[4*i+3], 0, ORIENTATION_UPPER_BOUND_PACK)

    def _get_placement(self, pos):
        placement = []
        total_item_vol = 0.0
        for i in range(self.n):
            x, y, z = pos[4*i : 4*i+3]
            ori_idx = int(np.clip(pos[4*i+3], 0, 5))
            original_dims = self.item_original_dims[i]
            w_rot, d_rot, h_rot = self._get_rotated_dims(original_dims, ori_idx)
            placement.append({
                'id': self.items[i]['id'], 'name': self.items[i]['name'],
                'x': x, 'y': y, 'z': z,
                'w': w_rot, 'd': d_rot, 'h': h_rot,
                'vol': self.item_original_volumes[i]
            })
            total_item_vol += self.item_original_volumes[i]
        return placement, total_item_vol

    @staticmethod
    def _calculate_overlap_volume(item_a, item_b):
        ox = max(0, min(item_a['x'] + item_a['w'], item_b['x'] + item_b['w']) - max(item_a['x'], item_b['x']))
        oy = max(0, min(item_a['y'] + item_a['d'], item_b['y'] + item_b['d']) - max(item_a['y'], item_b['y']))
        oz = max(0, min(item_a['z'] + item_a['h'], item_b['z'] + item_b['h']) - max(item_a['z'], item_b['z']))
        return ox * oy * oz

    @staticmethod
    def _calculate_2d_overlap_area(rect_a, rect_b):
        x_overlap = max(0, min(rect_a[2], rect_b[2]) - max(rect_a[0], rect_b[0]))
        y_overlap = max(0, min(rect_a[3], rect_b[3]) - max(rect_a[1], rect_b[1]))
        return x_overlap * y_overlap

    def _penalty(self, pos):
        """Menghitung total penalti dan rinciannya untuk sebuah konfigurasi packing."""
        if self.n == 0:
            return 0.0, {} # Kembalikan penalti nol dan dict kosong jika tidak ada item

        placement, total_item_vol = self._get_placement(pos)
        pen_outbound = 0.0
        pen_overlap = 0.0
        pen_support = 0.0
        pen_height = 0.0
        pen_compaction = 0.0

        # 1. Penalti Out-of-bounds
        for it in placement:
            vol_out_x_neg = max(0, -it['x']) * it['d'] * it['h']
            vol_out_y_neg = max(0, -it['y']) * it['w'] * it['h']
            vol_out_z_neg = max(0, -it['z']) * it['w'] * it['d']
            vol_out_x_pos = max(0, (it['x'] + it['w']) - self.L) * it['d'] * it['h']
            vol_out_y_pos = max(0, (it['y'] + it['d']) - self.W) * it['w'] * it['h']
            vol_out_z_pos = max(0, (it['z'] + it['h']) - self.H) * it['w'] * it['d']
            total_vol_out = vol_out_x_neg + vol_out_y_neg + vol_out_z_neg + \
                            vol_out_x_pos + vol_out_y_pos + vol_out_z_pos
            if total_vol_out > ZERO_TOLERANCE:
                pen_outbound += OUTBOUND_PENALTY_FACTOR * total_vol_out

        # 2. Penalti Overlap
        for i, item_a in enumerate(placement):
            for j, item_b in enumerate(placement):
                if i >= j: continue
                overlap_vol = self._calculate_overlap_volume(item_a, item_b)
                if overlap_vol > ZERO_TOLERANCE:
                    pen_overlap += OVERLAP_PENALTY_FACTOR * overlap_vol

        # 3. Penalti Support
        for i, it in enumerate(placement):
            item_base_z = it['z']
            if item_base_z < SUPPORT_TOLERANCE: continue
            item_base_area = it['w'] * it['d']
            if item_base_area < ZERO_TOLERANCE: continue
            total_supported_area = 0.0
            item_rect = (it['x'], it['y'], it['x'] + it['w'], it['y'] + it['d'])
            for j, base in enumerate(placement):
                if i == j: continue
                base_top_z = base['z'] + base['h']
                if abs(base_top_z - item_base_z) < SUPPORT_TOLERANCE:
                    base_rect = (base['x'], base['y'], base['x'] + base['w'], base['y'] + base['d'])
                    overlap_area = self._calculate_2d_overlap_area(item_rect, base_rect)
                    total_supported_area += overlap_area
            support_ratio = total_supported_area / item_base_area if item_base_area > 0 else 1.0
            unsupported_ratio = max(0.0, 1.0 - support_ratio)
            if unsupported_ratio > ZERO_TOLERANCE:
                 support_penalty_val = UNSUPPORTED_AREA_PENALTY_FACTOR * (unsupported_ratio ** 2)
                 pen_support += support_penalty_val
                 # Optional: Log item specific support issue if debugging needed
                 # if logging.getLogger().isEnabledFor(logging.DEBUG):
                 #    logging.debug(f"Item {it['id']}@({it['x']:.1f},{it['y']:.1f},{it['z']:.1f}) unsup_ratio: {unsupported_ratio:.3f}, sup_pen: {support_penalty_val:.2f}")


        # 4. Penalti Ketinggian Item Besar
        for i, it in enumerate(placement):
            height_penalty_val = HEIGHT_VOLUME_PENALTY_FACTOR * self.item_original_volumes[i] * it['z']
            pen_height += height_penalty_val

        # 5. Penalti Kompaksi
        if placement:
            max_x = max(it['x'] + it['w'] for it in placement) if placement else 0
            max_y = max(it['y'] + it['d'] for it in placement) if placement else 0
            max_z = max(it['z'] + it['h'] for it in placement) if placement else 0
            used_bbox_vol = max_x * max_y * max_z
            compaction_penalty_val = self.comp_w * max(0, used_bbox_vol - total_item_vol)
            pen_compaction += compaction_penalty_val

        # Hitung total penalti
        total_pen = pen_outbound + pen_overlap + pen_support + pen_height + pen_compaction

        # Buat dictionary rincian
        penalty_details = {
            "Outbound": pen_outbound,
            "Overlap": pen_overlap,
            "Support": pen_support,
            "Height": pen_height,
            "Compaction": pen_compaction,
            "Total": total_pen # Sertakan total juga untuk verifikasi
        }

        return total_pen, penalty_details # Kembalikan total dan rincian

    def optimize(self, max_iters=100, patience=10):
        """Menjalankan algoritma PSO packing. Mengembalikan layout, penalti total, dan rincian penalti."""
        if self.n == 0:
            logging.info("PackingPSO optimize called with 0 items. Returning empty layout.")
            return [], 0.0, {} # Kembalikan dict kosong untuk rincian

        if self.gbest_pos is None:
            logging.error("PackingPSO cannot optimize without initialized gbest_pos.")
            if self.pbest_pos.size > 0:
                best_initial_p = np.argmin(self.pbest_score)
                if self.pbest_score[best_initial_p] < float('inf'):
                    self.gbest_score = self.pbest_score[best_initial_p]
                    self.gbest_pos = self.pbest_pos[best_initial_p].copy()
                    # Hitung detail untuk gbest fallback
                    _, self.gbest_details = self._penalty(self.gbest_pos)
                    logging.warning("gbest_pos was None during optimize start, re-setting from best pbest.")
                else:
                    logging.error("Fallback failed, all pbest scores are infinite.")
                    return [], float('inf'), {} # Kembalikan dict kosong
            else:
                 logging.error("Fallback failed, no pbest positions available.")
                 return [], float('inf'), {} # Kembalikan dict kosong

        # Pastikan gbest_pos ada setelah fallback
        if self.gbest_pos is None:
             logging.critical("PSO cannot proceed without a valid gbest_pos.")
             return [], float('inf'), {}


        best_iter_score = self.gbest_score
        no_improvement_iters = 0

        logging.info(f"Starting PackingPSO optimization for {max_iters} iterations (patience={patience})...")

        # --- Iterasi PSO ---
        for t in range(max_iters):
            w_inertia = W_MAX - (W_MAX - W_MIN) * (t / float(max_iters - 1)) if max_iters > 1 else W_MIN
            for p in range(self.num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_comp = self.c1 * r1 * (self.pbest_pos[p] - self.particles[p])
                social_comp = self.c2 * r2 * (self.gbest_pos - self.particles[p])
                new_vel = (w_inertia * self.velocities[p] + cognitive_comp + social_comp)
                self.velocities[p] = new_vel
                self.particles[p] += self.velocities[p]
                self._clamp(self.particles[p])

                # Evaluasi - hanya butuh score di sini
                current_score, current_details = self._penalty(self.particles[p]) # Tangkap detail juga

                # Update pbest
                if current_score < self.pbest_score[p]:
                    self.pbest_score[p] = current_score
                    self.pbest_pos[p] = self.particles[p].copy()
                    # Update gbest
                    if current_score < self.gbest_score:
                        self.gbest_score = current_score
                        self.gbest_pos = self.particles[p].copy()
                        self.gbest_details = current_details.copy() # Simpan detail gbest baru
                        if self.gbest_score < best_iter_score - ZERO_TOLERANCE:
                            best_iter_score = self.gbest_score
                            no_improvement_iters = 0

            # Cek Early Stopping
            no_improvement_iters += 1
            if no_improvement_iters >= patience:
                logging.info(f"PackingPSO early stopping at iteration {t+1} due to no improvement for {patience} iterations.")
                break
            if self.gbest_score < ZERO_TOLERANCE:
                logging.info(f"PackingPSO stopping early at iteration {t+1} as near-zero penalty reached ({self.gbest_score:.2E}).")
                break

        logging.info(f"PackingPSO optimization finished. Final GBest Score: {self.gbest_score:.4f}")
        # --- Akhir Iterasi PSO ---

        # Ekstrak layout final dan gunakan penalti + DETAIL dari gbest_pos
        final_layout = []
        final_penalty = float('inf')
        final_penalty_details = {} # Inisialisasi dict kosong

        if self.gbest_pos is not None and isinstance(self.gbest_pos, np.ndarray):
            # Gunakan skor dan detail gbest yang sudah tersimpan
            final_penalty = self.gbest_score
            final_penalty_details = self.gbest_details.copy()

            # Rekonstruksi layout
            placement, _ = self._get_placement(self.gbest_pos)
            for i, item_layout in enumerate(placement):
                ori_idx = int(np.clip(self.gbest_pos[4*i+3], 0, 5))
                item_layout['orientation'] = ori_idx
            final_layout = placement
        else:
            # Fallback jika gbest_pos tidak valid (seharusnya sudah ditangani di awal)
            logging.error("Optimize finished but gbest_pos is unexpectedly None/invalid!")
            if self.pbest_score.size > 0 and np.min(self.pbest_score) < float('inf'):
                best_p_idx = np.argmin(self.pbest_score)
                fallback_pos = self.pbest_pos[best_p_idx]
                # Hitung penalti dan DETAIL dari fallback_pos
                final_penalty, final_penalty_details = self._penalty(fallback_pos)
                # Rekonstruksi layout dari fallback_pos
                placement, _ = self._get_placement(fallback_pos)
                for i, item_layout in enumerate(placement):
                    ori_idx = int(np.clip(fallback_pos[4*i+3], 0, 5))
                    item_layout['orientation'] = ori_idx
                final_layout = placement
                logging.warning(f"Using fallback pbest layout with penalty: {final_penalty:.4f}")
            else:
                # Kasus terburuk: tidak ada solusi sama sekali
                final_layout = []
                final_penalty = float('inf')
                final_penalty_details = {} # Tetap dict kosong
                logging.error("No valid solution found in PackingPSO.")

        # Kembalikan layout, total penalti, dan rincian penalti
        return final_layout, final_penalty, final_penalty_details


# --- Load Data Jarak & Polygon ---
csv_filename = "city_to_city_polygon.csv"
distance_dict = {}
polygons = {}
try:
    if os.path.exists(csv_filename):
        df_distance = pd.read_csv(csv_filename)
        required_cols = ["CityA", "CityB", "Distance (meters)", "Polygon"]
        if all(col in df_distance.columns for col in required_cols):
            for _, row in df_distance.iterrows():
                city_a = str(row["CityA"]).strip()
                city_b = str(row["CityB"]).strip()
                key = tuple(sorted((city_a, city_b)))
                distance_km = row["Distance (meters)"] / 1000.0
                if distance_km < 0: distance_km = 0
                distance_dict[key] = distance_km

                try:
                    polygon_str = row["Polygon"]
                    if pd.isna(polygon_str) or not isinstance(polygon_str, str) or not polygon_str.strip():
                        polygons[key] = []
                        continue

                    polygon_coords_raw = json.loads(polygon_str)
                    if isinstance(polygon_coords_raw, list) and polygon_coords_raw:
                        first_point = polygon_coords_raw[0]
                        if isinstance(first_point, dict) and 'lng' in first_point and 'lat' in first_point:
                            polygons[key] = [[p["lng"], p["lat"]] for p in polygon_coords_raw if isinstance(p, dict) and 'lng' in p and 'lat' in p]
                        elif isinstance(first_point, list) and len(first_point) == 2:
                            polygons[key] = [[float(p[0]), float(p[1])] for p in polygon_coords_raw if isinstance(p, list) and len(p)==2]
                        else:
                            polygons[key] = []
                    else:
                       polygons[key] = []
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    logging.warning(f"Failed to parse polygon for {key}: {e}. Polygon set to empty.")
                    polygons[key] = []
            logging.info(f"Successfully loaded {len(distance_dict)} distances and {len(polygons)} polygons from {csv_filename}.")
        else:
            st.error(f"CSV file '{csv_filename}' missing required columns: {required_cols}.")
            logging.error(f"CSV file '{csv_filename}' missing required columns.")
    else:
        st.warning(f"Distance file '{csv_filename}' not found. Using Haversine distance.")
        logging.warning(f"Distance file '{csv_filename}' not found.")
except pd.errors.EmptyDataError:
    st.error(f"CSV file '{csv_filename}' is empty.")
    logging.error(f"CSV file '{csv_filename}' is empty.")
except Exception as e:
    st.error(f"Failed to read or process CSV '{csv_filename}': {e}")
    logging.exception(f"Failed to read or process CSV '{csv_filename}'")

# --- Data Input ---
items_input = [
    {"id": "Item1", "name": "TV", "weight": 120, "dims": (40, 50, 30), "city": "Jakarta"},
    {"id": "Item2", "name": "Kulkas", "weight": 300, "dims": (70, 60, 90), "city": "Bandung"}, # Besar
    {"id": "Item3", "name": "AC", "weight": 250, "dims": (80, 50, 60), "city": "Semarang"}, # Sedang
    {"id": "Item4", "name": "Buku", "weight": 50, "dims": (30, 30, 20), "city": "Jakarta"}, # Kecil
    {"id": "Item5", "name": "Sofa", "weight": 500, "dims": (150, 80, 100), "city": "Yogyakarta"}, # Besar
    {"id": "Item6", "name": "Meja", "weight": 150, "dims": (120, 100, 40), "city": "Semarang"}, # Besar
    {"id": "Item7", "name": "Ranjang", "weight": 400, "dims": (200, 160, 50), "city": "Malang"}, # Oversized? -> Cek di pre-processing
    {"id": "Item8", "name": "Kipas Angin", "weight": 30, "dims": (20, 20, 40), "city": "Bandung"}, # Kecil
    {"id": "Item9", "name": "WashingMachine","weight":350, "dims": (60,60,85), "city": "Jakarta"}, # Sedang
    {"id": "Item10", "name": "Bookshelf", "weight":100, "dims": (80,30,180), "city": "Surabaya"}, # Tinggi -> Cek Oversized
    {"id": "Item11", "name": "Mattress", "weight":200, "dims": (200,90,30), "city": "Bandung"}, # Besar
    {"id": "Item12", "name": "Wardrobe", "weight":450, "dims": (100,60,200), "city": "Yogyakarta"}, # Tinggi & Berat -> Cek Oversized
    {"id": "Item13", "name": "DiningTable", "weight":250, "dims": (160,90,75), "city": "Semarang"}, # Besar
    {"id": "Item14", "name": "DeskLamp", "weight":10, "dims": (15,15,40), "city": "Malang"}, # Kecil
    {"id": "Item15", "name": "Microwave", "weight":40, "dims": (50,40,35), "city": "Jakarta"}, # Kecil
    {"id": "Item16", "name": "Printer", "weight":25, "dims": (45,40,30), "city": "Surabaya"}, # Kecil
    {"id": "Item17", "name": "FloorLamp", "weight":20, "dims": (30,30,160), "city": "Bandung"}, # Tinggi -> Cek Oversized
    {"id": "Item18", "name": "AirPurifier", "weight":15, "dims": (25,25,60), "city": "Yogyakarta"}, # Kecil
    {"id": "Item19", "name": "WaterHeater", "weight":80, "dims": (50,50,100), "city": "Semarang"}, # Sedang
    {"id": "Item20", "name": "CoffeeTable", "weight":80, "dims": (120,60,45), "city": "Malang"} # Sedang
]
n_trucks = 4
truck_max_weight = 1000.0 # Jadikan float
truck_max_length = 200.0 # Jadikan float
truck_max_width = 150.0 # Jadikan float
truck_max_height = 150.0 # Jadikan float
truck_dims_tuple = (truck_max_length, truck_max_width, truck_max_height)
city_coords = {
    "Surabaya": (-7.2575, 112.7521), "Jakarta": (-6.2088, 106.8456),
    "Bandung": (-6.9175, 107.6191), "Semarang": (-6.9667, 110.4167),
    "Yogyakarta": (-7.7956, 110.3695), "Malang": (-7.9824, 112.6304)
}

# --- Pre-process Items ---
def get_dimension_category(l, w, h):
    vol = l * w * h
    if vol < THRESHOLD_SMALL_VOLUME: return "Kecil", CAT_FACTOR_SMALL
    elif vol < THRESHOLD_MEDIUM_VOLUME: return "Sedang", CAT_FACTOR_MEDIUM
    else: return "Besar", CAT_FACTOR_LARGE

items = []
oversized_count = 0
logging.info("Preprocessing items...")
for item_data in items_input:
    item = item_data.copy()
    l, w, h = item["dims"]
    cat, factor = get_dimension_category(l, w, h)
    item["dim_category"] = cat
    item["cat_factor"] = factor

    can_fit = False
    for i in range(6):
        w_rot, d_rot, h_rot = PackingPSO._get_rotated_dims((l, w, h), i)
        if w_rot <= truck_max_length + ZERO_TOLERANCE and \
           d_rot <= truck_max_width + ZERO_TOLERANCE and \
           h_rot <= truck_max_height + ZERO_TOLERANCE:
            can_fit = True
            break

    item["is_oversized"] = not can_fit
    if item["is_oversized"]:
        oversized_count += 1
        logging.warning(f"Item {item['id']} ({item['name']}) is OVERSIZED for truck dimensions {truck_dims_tuple}.")

    items.append(item)

logging.info(f"Item preprocessing complete. {len(items) - oversized_count} assignable items, {oversized_count} oversized items.")


# --- Fungsi Jarak (Haversine & dari Dictionary) ---
@st.cache_data(show_spinner=False)
def haversine(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371.0

    phi1, phi2 = map(math.radians, [lat1, lat2])
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance_km = R * c
    return distance_km

@st.cache_data(show_spinner=False)
def distance(city1, city2, start_city=DEFAULT_START_CITY):
    city1_norm = str(city1).strip()
    city2_norm = str(city2).strip()

    if city1_norm == city2_norm:
        return 0.0

    key = tuple(sorted((city1_norm, city2_norm)))

    if key in distance_dict:
        return distance_dict[key]

    if city1_norm in city_coords and city2_norm in city_coords:
        dist_hv = haversine(city_coords[city1_norm], city_coords[city2_norm])
        logging.warning(f"Distance for {key} not found in CSV, using Haversine: {dist_hv:.2f} km.")
        # distance_dict[key] = dist_hv # Optional: Cache Haversine result
        return dist_hv

    logging.error(f"DISTANCE NOT FOUND: Cannot find distance between '{city1_norm}' and '{city2_norm}'. Check CSV and city_coords.")
    st.error(f"Jarak antara {city1_norm} dan {city2_norm} tidak ditemukan.")
    return float('inf')

@st.cache_data(show_spinner=False)
def route_distance(cities, start_city=DEFAULT_START_CITY):
    if not cities: return 0.0
    unique_cities = list(dict.fromkeys(c for c in cities if c != start_city))
    if not unique_cities: return 0.0

    current_city = start_city
    unvisited = unique_cities[:]
    total_dist = 0.0

    while unvisited:
        nearest_city = None
        min_dist = float('inf')
        for city in unvisited:
            dist_to_city = distance(current_city, city, start_city)
            if dist_to_city < min_dist:
                min_dist = dist_to_city
                nearest_city = city

        if min_dist == float('inf') or nearest_city is None:
            logging.error(f"Cannot find valid route from {current_city} to any of {unvisited}. Route calculation failed.")
            return float('inf')

        total_dist += min_dist
        current_city = nearest_city
        unvisited.remove(nearest_city)

    dist_back = distance(current_city, start_city, start_city)
    if dist_back == float('inf'):
        logging.error(f"Cannot find valid route back from {current_city} to {start_city}. Route calculation failed.")
        return float('inf')

    total_dist += dist_back
    return total_dist

# --- Fungsi Bantuan Rute untuk Visualisasi ---
def get_route_sequence(cities, start_city=DEFAULT_START_CITY):
    if not cities: return [start_city, start_city]
    unique_cities = list(dict.fromkeys(c for c in cities if c != start_city))
    if not unique_cities: return [start_city, start_city]

    route = [start_city]
    current = start_city
    unvisited = unique_cities[:]

    while unvisited:
        distances_to_unvisited = {city: distance(current, city, start_city) for city in unvisited}
        valid_distances = {city: dist for city, dist in distances_to_unvisited.items() if dist != float('inf')}

        if not valid_distances:
            logging.warning(f"Cannot reach remaining cities {unvisited} from {current}. Ending route sequence prematurely.")
            dist_back = distance(current, start_city, start_city)
            if dist_back != float('inf'):
                route.append(start_city)
            return route

        nearest = min(valid_distances, key=valid_distances.get)
        route.append(nearest)
        current = nearest
        unvisited.remove(nearest)

    dist_final_back = distance(current, start_city, start_city)
    if dist_final_back != float('inf'):
        route.append(start_city)
    else:
        logging.warning(f"Cannot return to {start_city} from last city {current}. Route sequence might be incomplete.")

    return route

# --- Fitness Function (Outer PSO - Assignment) ---
def compute_fitness(assignment, items, n_trucks, truck_dims, truck_max_weight):
    truck_details = {t: {"items": [], "cities": set(), "weight": 0.0}
                     for t in range(1, n_trucks + 1)}
    total_revenue = 0.0
    constraint_penalty = 0.0
    total_packing_penalty = 0.0 # Based on cache or quick_feasible fail

    # Logika untuk breakdown penalti di level assignment (jika diperlukan)
    pen_details_assignment = {"oversized": 0.0, "invalid_assign": 0.0, "invalid_route": 0.0, "overweight": 0.0, "packing_impossible": 0.0}

    for item_idx, assigned_truck_idx in enumerate(assignment):
        if assigned_truck_idx == 0: continue
        item = items[item_idx]

        if item["is_oversized"]:
            pen_details_assignment["oversized"] += OVERSIZED_ITEM_PENALTY
            constraint_penalty += OVERSIZED_ITEM_PENALTY
            continue

        if not (1 <= assigned_truck_idx <= n_trucks):
            pen_details_assignment["invalid_assign"] += INVALID_ASSIGNMENT_PENALTY
            constraint_penalty += INVALID_ASSIGNMENT_PENALTY
            continue

        truck_id = int(assigned_truck_idx)
        truck_details[truck_id]["items"].append(item)
        truck_details[truck_id]["cities"].add(item["city"])
        truck_details[truck_id]["weight"] += item["weight"]

        dist_to_dest = distance(DEFAULT_START_CITY, item["city"])
        if dist_to_dest == float('inf'):
            pen_details_assignment["invalid_route"] += INVALID_ROUTE_PENALTY # Penalti per item, mungkin lebih baik per rute?
            constraint_penalty += INVALID_ROUTE_PENALTY
            item_revenue = 0
        else:
            item_revenue = item["weight"] * dist_to_dest * item["cat_factor"]
        total_revenue += item_revenue


    total_fuel_cost = 0.0
    for truck_id in range(1, n_trucks + 1):
        info = truck_details[truck_id]
        truck_items = info["items"]
        if not truck_items: continue

        # Penalti Overweight
        current_weight = info["weight"]
        if current_weight > truck_max_weight:
            overweight_amount = current_weight - truck_max_weight
            ow_pen = OVERWEIGHT_CONSTANT_PENALTY + OVERWEIGHT_FACTOR_PENALTY * (overweight_amount ** 2)
            pen_details_assignment["overweight"] += ow_pen
            constraint_penalty += ow_pen
            # Logging Overweight saat fitness calculation
            # logging.debug(f"Fitness Eval: Truck {truck_id} overweight by {overweight_amount:.2f} kg. Penalty: {ow_pen:.2E}")


        # Hitung Biaya Rute & Cek Validitas Rute Truk
        route_dist_km = route_distance(list(info["cities"]), DEFAULT_START_CITY)
        if route_dist_km == float('inf'):
            # Tambahkan penalti rute per truk (jika belum ditambahkan per item)
            if pen_details_assignment["invalid_route"] == 0: # Hindari dobel penalti jika sudah ada dari item
                 pen_details_assignment["invalid_route"] += INVALID_ROUTE_PENALTY
                 constraint_penalty += INVALID_ROUTE_PENALTY
            # logging.debug(f"Fitness Eval: Truck {truck_id} has invalid route. Penalty added.")

        else:
            total_fuel_cost += COST_PER_KM * route_dist_km

        # Hitung Penalti Packing (Cache / Quick Feasible)
        feasible_check, feasible_reason = quick_feasible(truck_dims, truck_items)
        if not feasible_check:
            packing_pen = PACKING_IMPOSSIBLE_PENALTY + PACKING_IMPOSSIBLE_ITEM_FACTOR * len(truck_items)
            pen_details_assignment["packing_impossible"] += packing_pen
            total_packing_penalty += packing_pen
            # logging.debug(f"Fitness Eval: Truck {truck_id} quick_feasible failed ({feasible_reason}). Penalty: {packing_pen:.2E}")

        else:
            items_tuple_key = tuple(sorted(
                (it["id"], it["name"], tuple(it["dims"]), it["weight"]) for it in truck_items
            ))
            packing_pen = packing_penalty_cache(truck_dims, items_tuple_key)
            total_packing_penalty += packing_pen
            # logging.debug(f"Fitness Eval: Truck {truck_id} packing cache penalty: {packing_pen:.4f}")


    profit = total_revenue - total_fuel_cost
    fitness = profit - total_packing_penalty - constraint_penalty

    # Optional: Log fitness breakdown jika fitness sangat rendah
    # if fitness < -1e8: # Threshold for logging bad fitness
    #    logging.debug(f"Low Fitness Encountered: {fitness:.2E}, Profit: {profit:.2f}, PackingPen: {total_packing_penalty:.2f}, ConstrPen: {constraint_penalty:.2f}, Details: {pen_details_assignment}")

    return fitness

# --- Fungsi Decode Posisi PSO Luar ---
def decode_position(position, n_items, n_trucks, assignable_mask):
    assignment = np.zeros(n_items, dtype=int)
    pos_upper_bound_assign = n_trucks + (1.0 - POS_LOWER_BOUND_ASSIGN)

    for i in range(n_items):
        if assignable_mask[i]:
            clamped_val = np.clip(position[i], POS_LOWER_BOUND_ASSIGN, pos_upper_bound_assign)
            assignment[i] = int(round(clamped_val))
    return assignment

# --- Outer PSO: Assignment ---
num_particles_assign = 30
max_iter_assign = 200
patience_assign = 30
improvement_threshold = 10.0

assignable_mask = np.array([not item["is_oversized"] for item in items])
n_assignable_items = assignable_mask.sum()
n_total_items = len(items)

particles_assign = np.zeros((num_particles_assign, n_total_items))
velocities_assign = np.zeros((num_particles_assign, n_total_items))
pbest_positions_assign = np.zeros((num_particles_assign, n_total_items))
pbest_fitness_assign = np.full(num_particles_assign, -float('inf'))
gbest_position_assign = None
gbest_fitness_assign = -float('inf')
prev_gbest_assign = -float('inf')
no_improvement_count = 0

logging.info("Initializing Assignment PSO...")
for p_idx in range(num_particles_assign):
    position = np.zeros(n_total_items)
    velocity = np.zeros(n_total_items)

    if n_assignable_items > 0:
        pos_upper_bound_assign = n_trucks + (1.0 - POS_LOWER_BOUND_ASSIGN)
        position[assignable_mask] = np.random.uniform(POS_LOWER_BOUND_ASSIGN, pos_upper_bound_assign, size=n_assignable_items)
        vel_range_assign = (n_trucks / 2.0) * 0.1
        velocity[assignable_mask] = np.random.uniform(-vel_range_assign, vel_range_assign, size=n_assignable_items)

    particles_assign[p_idx] = position.copy()
    velocities_assign[p_idx] = velocity.copy()

    assignment = decode_position(position, n_total_items, n_trucks, assignable_mask)
    fit = compute_fitness(assignment, items, n_trucks, truck_dims_tuple, truck_max_weight)

    pbest_positions_assign[p_idx] = position.copy()
    pbest_fitness_assign[p_idx] = fit

    if fit > gbest_fitness_assign:
        gbest_fitness_assign = fit
        gbest_position_assign = position.copy()

if gbest_position_assign is None and n_assignable_items > 0:
    if np.any(pbest_fitness_assign > -float('inf')):
         best_initial_p_idx = np.argmax(pbest_fitness_assign)
         gbest_fitness_assign = pbest_fitness_assign[best_initial_p_idx]
         gbest_position_assign = pbest_positions_assign[best_initial_p_idx].copy()
         logging.warning("Initial gbest_assign was None, setting from best pbest.")
    else:
         logging.error("Could not initialize gbest_assign. All initial fitness values might be -infinity.")
         # Handle error or exit if necessary


logging.info(f"Assignment PSO Initialization Complete. Initial Global Best Fitness: {gbest_fitness_assign:,.2f}")

# --- Loop Utama PSO Assignment ---
fitness_history_assign = []
if gbest_position_assign is not None:
    logging.info(f"Running Assignment PSO for max {max_iter_assign} iterations...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Iter 0/{max_iter_assign}, Best Fitness: {gbest_fitness_assign:,.2f}")

    for it in range(1, max_iter_assign + 1):
        w_inertia = W_MAX - (W_MAX - W_MIN) * (it / float(max_iter_assign))

        for i in range(num_particles_assign):
            current_pos = particles_assign[i]
            current_vel = velocities_assign[i]
            pbest_pos = pbest_positions_assign[i]

            r1 = np.random.rand(n_total_items)
            r2 = np.random.rand(n_total_items)

            cognitive_comp = C1_ASSIGN * r1 * (pbest_pos - current_pos)
            social_comp = C2_ASSIGN * r2 * (gbest_position_assign - current_pos)
            new_vel = w_inertia * current_vel + cognitive_comp + social_comp

            velocities_assign[i][assignable_mask] = new_vel[assignable_mask]
            particles_assign[i][assignable_mask] += velocities_assign[i][assignable_mask]

            pos_upper_bound_assign = n_trucks + (1.0 - POS_LOWER_BOUND_ASSIGN)
            particles_assign[i][assignable_mask] = np.clip(
                particles_assign[i][assignable_mask],
                POS_LOWER_BOUND_ASSIGN,
                pos_upper_bound_assign
            )

            assignment = decode_position(particles_assign[i], n_total_items, n_trucks, assignable_mask)
            fit = compute_fitness(assignment, items, n_trucks, truck_dims_tuple, truck_max_weight)

            if fit > pbest_fitness_assign[i]:
                pbest_fitness_assign[i] = fit
                pbest_positions_assign[i] = particles_assign[i].copy()

                if fit > gbest_fitness_assign:
                    gbest_fitness_assign = fit
                    gbest_position_assign = particles_assign[i].copy()

        fitness_history_assign.append(gbest_fitness_assign)
        improvement = gbest_fitness_assign - prev_gbest_assign

        status_text.text(f"Iter {it}/{max_iter_assign}, Best Fitness: {gbest_fitness_assign:,.2f}")
        progress_bar.progress(it / max_iter_assign)

        if improvement >= improvement_threshold:
            no_improvement_count = 0
            prev_gbest_assign = gbest_fitness_assign
        elif it > patience_assign:
            no_improvement_count += 1

        if no_improvement_count >= patience_assign:
            logging.info(f"Assignment PSO early stopping at iteration {it} due to no significant improvement.")
            status_text.text(f"Early stopping at iteration {it}. Final Best Fitness: {gbest_fitness_assign:,.2f}")
            break

    progress_bar.empty()
    logging.info(f"Assignment PSO finished. Final Best Fitness: {gbest_fitness_assign:,.2f}")

else:
    st.error("Assignment PSO cannot run because no valid initial solution was found.")
    logging.error("Assignment PSO did not run due to missing initial gbest.")
    fitness_history_assign = []

# --- Dapatkan Assignment Terbaik ---
best_assignment = np.zeros(n_total_items, dtype=int)
if gbest_position_assign is not None:
    best_assignment = decode_position(gbest_position_assign, n_total_items, n_trucks, assignable_mask)
    logging.info("Using gbest assignment.")
else:
    if pbest_fitness_assign.size > 0 and np.any(pbest_fitness_assign > -float('inf')):
        best_pbest_idx = np.argmax(pbest_fitness_assign)
        best_pbest_pos = pbest_positions_assign[best_pbest_idx]
        best_assignment = decode_position(best_pbest_pos, n_total_items, n_trucks, assignable_mask)
        gbest_fitness_assign = pbest_fitness_assign[best_pbest_idx]
        logging.warning("gbest_position_assign was None after PSO loop. Using best pbest as fallback.")
        st.warning("Solusi terbaik global tidak ditemukan, menggunakan solusi partikel terbaik.")
    else:
        logging.error("No valid assignment solution found after PSO.")
        st.error("Optimasi penugasan gagal menemukan solusi yang valid.")


# --- Proses Hasil Assignment Terbaik ---
def get_truck_info(assignment, items, n_trucks, truck_max_weight, start_city=DEFAULT_START_CITY):
    truck_info = {
        t: {
            "items": [], "weight": 0.0, "volume": 0.0, "cities": [],
            "revenue": 0.0, "route_distance": 0.0, "fuel_cost": 0.0,
            "profit": 0.0, "is_overweight": False, "route_sequence": []
        } for t in range(1, n_trucks + 1)
    }
    assigned_item_details = []

    for i, truck_idx in enumerate(assignment):
        item = items[i].copy()
        assigned_truck_str = "" # Initialize

        if item["is_oversized"]:
            assigned_truck_str = "Oversized"
        elif truck_idx == 0:
            assigned_truck_str = "Unassigned"
        elif not (1 <= truck_idx <= n_trucks):
            assigned_truck_str = "Invalid Assignment"
            logging.error(f"Invalid truck index {truck_idx} found in final assignment for item {item['id']}.")
        else:
            assigned_truck_str = f"Truk {truck_idx}"
            truck_id = int(truck_idx)
            truck_info[truck_id]["items"].append(item)
            truck_info[truck_id]["weight"] += item["weight"]
            l, w, h = item["dims"]
            # Assuming dims are in cm, converting to m^3
            truck_info[truck_id]["volume"] += (l * w * h) / 1e6

            if item["city"] != start_city and item["city"] not in truck_info[truck_id]["cities"]:
                truck_info[truck_id]["cities"].append(item["city"])

            dist_to_dest = distance(start_city, item["city"])
            if dist_to_dest != float('inf'):
                item_revenue = item["weight"] * dist_to_dest * item["cat_factor"]
                truck_info[truck_id]["revenue"] += item_revenue
            # Else: revenue remains 0 if distance is infinity


        assigned_item_details.append({
            "id": item["id"], "name": item["name"], "weight": item["weight"],
            "dims": f"{item['dims'][0]}x{item['dims'][1]}x{item['dims'][2]}",
            "city": item["city"], "dim_category": item["dim_category"],
            "assigned_truck": assigned_truck_str
        })

    for t in range(1, n_trucks + 1):
        info = truck_info[t]
        if info["items"]:
            route_seq = get_route_sequence(info["cities"], start_city)
            route_dist = route_distance(info["cities"], start_city)

            info["route_sequence"] = route_seq
            if route_dist == float('inf'):
                info["route_distance"] = 0.0
                info["fuel_cost"] = 0.0
                # Profit might be negative due to revenue being 0 for unreachable items
                info["profit"] = info["revenue"] # Or set to -inf? Based on fitness definition
                logging.warning(f"Truck {t} has an invalid route in the final assignment.")
            else:
                info["route_distance"] = route_dist
                info["fuel_cost"] = COST_PER_KM * route_dist
                info["profit"] = info["revenue"] - info["fuel_cost"]

            if info["weight"] > truck_max_weight + ZERO_TOLERANCE:
                info["is_overweight"] = True
                logging.warning(f"Truck {t} is OVERWEIGHT in the final assignment: {info['weight']:.2f} kg > {truck_max_weight:.2f} kg.")
                # Consider adjusting final profit display if needed based on overweight status
                # info["profit"] -= OVERWEIGHT_CONSTANT_PENALTY # Or similar adjustment if desired for display

    return truck_info, pd.DataFrame(assigned_item_details)

# Panggil fungsi untuk mendapatkan info final
final_truck_info, assigned_items_df = get_truck_info(best_assignment, items, n_trucks, truck_max_weight, DEFAULT_START_CITY)


# --- Fungsi Bantuan Path untuk Pydeck ---
def get_segment_path(city_a, city_b):
    city_a_norm = str(city_a).strip()
    city_b_norm = str(city_b).strip()
    if city_a_norm == city_b_norm: return []

    key = tuple(sorted((city_a_norm, city_b_norm)))

    if key in polygons and isinstance(polygons.get(key), list) and len(polygons[key]) >= 2:
        path_data = polygons[key]
        if city_a_norm in city_coords and city_b_norm in city_coords:
            coord_a = city_coords[city_a_norm]
            coord_b = city_coords[city_b_norm]
            # Ensure path starts closer to city_a if possible
            try:
                 path_start_coord = (path_data[0][1], path_data[0][0]) # lat, lon
                 dist_start_a = haversine(coord_a, path_start_coord)
                 dist_start_b = haversine(coord_b, path_start_coord)
                 if dist_start_b < dist_start_a - ZERO_TOLERANCE:
                     return path_data[::-1]
                 else:
                     return path_data
            except Exception as e:
                  logging.warning(f"Error checking path orientation for {key}: {e}. Using original polygon order.")
                  return path_data
        else:
            return path_data # Cannot check orientation

    if city_a_norm in city_coords and city_b_norm in city_coords:
        logging.debug(f"No polygon for {key}. Creating straight line path.")
        coord_a = city_coords[city_a_norm]
        coord_b = city_coords[city_b_norm]
        return [[coord_a[1], coord_a[0]], [coord_b[1], coord_b[0]]] # [[lng_a, lat_a], [lng_b, lat_b]]

    logging.warning(f"Cannot get path segment between {city_a_norm} and {city_b_norm}. No polygon and missing coordinates.")
    return []

def get_full_route_path(route_sequence):
    full_path = []
    if not route_sequence or len(route_sequence) < 2: return []

    for i in range(len(route_sequence) - 1):
        city_a = route_sequence[i]
        city_b = route_sequence[i+1]
        segment = get_segment_path(city_a, city_b)

        if not segment:
            logging.warning(f"Skipping segment from {city_a} to {city_b} in full path generation.")
            continue

        if not full_path:
            full_path.extend(segment)
        else:
            last_point = full_path[-1]
            first_point_segment = segment[0]
            # Check for continuity using tolerance
            if abs(last_point[0] - first_point_segment[0]) < ZERO_TOLERANCE and \
               abs(last_point[1] - first_point_segment[1]) < ZERO_TOLERANCE:
                full_path.extend(segment[1:])
            else:
                logging.debug(f"Gap detected between segments ending at {last_point} and starting at {first_point_segment}. Appending full segment.")
                full_path.extend(segment)
    return full_path

# --- Siapkan Data Rute Pydeck ---
routes_data_pydeck = []
colors = [[255,0,0,200],[0,180,0,200],[0,0,255,200],[255,165,0,200],
          [128,0,128,200],[0,200,200,200],[255,20,147,200],[60,179,113,200]] # 8 colors
logging.info("Generating route paths for visualization...")
for t in range(1, n_trucks + 1):
    info = final_truck_info[t]
    route_seq = info.get("route_sequence", [])
    if len(route_seq) > 1:
        full_path = get_full_route_path(route_seq)
        if full_path:
            display_cities = [c for c in route_seq if c != DEFAULT_START_CITY]
            if not display_cities:
                 display_route = f"{DEFAULT_START_CITY} (Base Only)"
            else:
                 # Format route: Base -> Dest1 -> Dest2 -> Base
                 display_route = f"{route_seq[0]} → {' → '.join(route_seq[1:-1])} → {route_seq[-1]}"


            routes_data_pydeck.append({
                "truck": f"Truk {t}",
                "path": full_path,
                "color": colors[(t-1) % len(colors)],
                "route_info": display_route,
                "distance_km": info.get("route_distance", 0)
            })
        else:
            logging.warning(f"Failed to create visual path for Truck {t} route: {' -> '.join(route_seq)}")
    elif info["items"]:
        logging.info(f"Truck {t} only has items for the base city {DEFAULT_START_CITY}. No route path generated.")


# --- Packing Final Kualitas Tinggi ---
final_layouts = {} # Akan menyimpan (layout, penalty, details)
logging.info("Running final (high-quality) packing optimization for assigned trucks...")
packing_progress = st.progress(0)
packing_status = st.empty()

for idx, t in enumerate(range(1, n_trucks + 1)):
    items_for_truck = final_truck_info[t]["items"]
    num_items_truck = len(items_for_truck)

    packing_status.text(f"Optimizing packing for Truck {t} ({num_items_truck} items)...")

    if not items_for_truck:
        final_layouts[t] = ([], 0.0, {}) # Layout kosong, penalti nol, details kosong
        packing_progress.progress((idx + 1) / n_trucks)
        logging.info(f"Truck {t} is empty, skipping final packing.")
        continue

    best_penalty_final = float('inf')
    best_layout_final = None
    best_penalty_details_final = {} # Inisialisasi details terbaik
    patience_final = max(10, int(FINAL_PACKING_MAX_ITERS * FINAL_PACKING_PATIENCE_FACTOR))

    for attempt in range(FINAL_PACKING_ATTEMPTS):
        logging.info(f"Truck {t}: Starting final packing attempt {attempt + 1}/{FINAL_PACKING_ATTEMPTS}...")
        seed = RANDOM_SEED + (t * 100) + (attempt * 10)
        current_np_state = np.random.get_state()
        np.random.seed(seed)

        try:
            packer = PackingPSO(truck_dims_tuple, items_for_truck,
                                num_particles=FINAL_PACKING_NUM_PARTICLES,
                                compaction_weight=FINAL_PACKING_COMPACTION_WEIGHT,
                                c1=C1_PACKING, c2=C2_PACKING)
            # Optimasi - tangkap ketiga nilai return
            layout, penalty, details = packer.optimize(max_iters=FINAL_PACKING_MAX_ITERS, patience=patience_final)
        finally:
            np.random.set_state(current_np_state)

        logging.info(f"Truck {t} Attempt {attempt + 1}: Penalty = {penalty:.4f}")

        if penalty < best_penalty_final:
            best_penalty_final = penalty
            best_layout_final = layout
            best_penalty_details_final = details.copy() # Salin dictionary details
            logging.info(f"Truck {t}: New best final penalty found: {best_penalty_final:.4f}")
            # Log rincian penalti terbaik yang baru ditemukan
            logging.info(f"Truck {t}: Best Penalty Details: {best_penalty_details_final}")


        if best_penalty_final < ZERO_TOLERANCE:
            logging.info(f"Truck {t}: Near-perfect packing found in attempt {attempt + 1}. Stopping attempts.")
            break

    # Simpan hasil terbaik (layout, penalty, details) untuk truk ini
    final_layouts[t] = (best_layout_final, best_penalty_final, best_penalty_details_final)
    logging.info(f"Truck {t} - Best Final Packing Penalty after {FINAL_PACKING_ATTEMPTS} attempts: {best_penalty_final:.4f}")
    # Log rincian penalti final terbaik lagi untuk konfirmasi
    if best_penalty_final > ZERO_TOLERANCE:
         logging.warning(f"Truck {t} - FINAL Best Packing Penalty Details: {best_penalty_details_final}")
    else:
         logging.info(f"Truck {t} - FINAL Best Packing Penalty Details: {best_penalty_details_final}")

    packing_progress.progress((idx + 1) / n_trucks)

packing_status.text("Final packing optimizations complete.")
packing_progress.empty()


# --- Fungsi Visualisasi Plotly 3D ---
def create_truck_figure(truck_dims, packed_items, truck_id):
    L, W, H = truck_dims
    fig = go.Figure()

    # 1. Gambar Outline Truk
    corners = [(0,0,0),(L,0,0),(L,W,0),(0,W,0),(0,0,H),(L,0,H),(L,W,H),(0,W,H)]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    edge_x, edge_y, edge_z = [], [], []
    for u, v in edges:
        edge_x.extend([corners[u][0], corners[v][0], None])
        edge_y.extend([corners[u][1], corners[v][1], None])
        edge_z.extend([corners[u][2], corners[v][2], None])
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines', line=dict(color='rgba(0,0,0,0.7)', width=2),
        hoverinfo='none', showlegend=False, name='Truck Outline'
    ))

    # 2. Gambar Item yang Dikemas
    if packed_items:
        color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                         "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        for idx, item in enumerate(packed_items):
            x, y, z = item['x'], item['y'], item['z']
            w_rot, d_rot, h_rot = item['w'], item['d'], item['h']

            x_v = [x, x, x+w_rot, x+w_rot, x, x, x+w_rot, x+w_rot]
            y_v = [y, y+d_rot, y+d_rot, y, y, y+d_rot, y+d_rot, y]
            z_v = [z, z, z, z, z+h_rot, z+h_rot, z+h_rot, z+h_rot]

            faces_i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 1]
            faces_j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 2]
            faces_k = [0, 7, 2, 3, 6, 7, 1, 5, 4, 5, 7, 6]

            item_color = color_palette[idx % len(color_palette)]

            fig.add_trace(go.Mesh3d(
                x=x_v, y=y_v, z=z_v, i=faces_i, j=faces_j, k=faces_k,
                color=item_color, opacity=0.85,
                name=item.get('name', item.get('id', f'Item {idx}')),
                hovertext=(f"<b>{item.get('name', item['id'])}</b><br>"
                           f"ID: {item['id']}<br>"
                           f"Dims (WxDxH): {w_rot:.1f} x {d_rot:.1f} x {h_rot:.1f}<br>"
                           f"Pos (x,y,z): ({x:.1f}, {y:.1f}, {z:.1f})<br>"
                           f"Orientasi: {item.get('orientation', 'N/A')}"),
                hoverinfo="text"
            ))

    # 3. Konfigurasi Layout Scene 3D
    fig.update_layout(
        title=f"Visualisasi Muatan Truk {truck_id}",
        scene=dict(
            xaxis=dict(title='Panjang (X)', range=[0, L], backgroundcolor="rgb(240,240,240)", tickformat=".1f"),
            yaxis=dict(title='Lebar (Y)', range=[0, W], backgroundcolor="rgb(235,235,235)", tickformat=".1f"),
            zaxis=dict(title='Tinggi (Z)', range=[0, H], backgroundcolor="rgb(240,240,240)", tickformat=".1f"),
            aspectratio=dict(x=1, y=W/L, z=H/L) if L > 0 else dict(x=1, y=1, z=1),
            aspectmode='manual',
            camera_eye=dict(x=1.8, y=1.8, z=0.9)
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    return fig

# --- Layout Aplikasi Streamlit ---
st.title("🚚 Optimasi Penugasan & Pemuatan Truk V2 Final (PSO)")
st.markdown("---")

# --- Baris 1: Metrik Ringkasan Global ---
col1, col2, col3, col4 = st.columns(4)

total_revenue_final = sum(info['revenue'] for info in final_truck_info.values())
total_cost_final = sum(info['fuel_cost'] for info in final_truck_info.values())
total_profit_final = total_revenue_final - total_cost_final
num_assigned_items = sum(len(info['items']) for info in final_truck_info.values())
num_unassigned = n_total_items - num_assigned_items - oversized_count # Hitung ulang unassigned
num_overweight = sum(1 for info in final_truck_info.values() if info['is_overweight'])

col1.metric("Total Estimasi Profit*", f"Rp {total_profit_final:,.0f}",
            help="Profit = Total Revenue (item*jarak*faktor) - Total Fuel Cost (rute*cost/km). Belum termasuk penalti packing/constraint lain.")
col2.metric("Total Revenue", f"Rp {total_revenue_final:,.0f}")
col3.metric("Total Fuel Cost", f"Rp {total_cost_final:,.0f}")
# Tampilkan item assignable yang tidak terassign
assignable_items_count = len(items) - oversized_count
col4.metric("Item Ter-assign", f"{num_assigned_items} / {assignable_items_count}",
            help=f"{num_unassigned} tidak ter-assign (dari yg assignable), {oversized_count} oversized.")


if num_overweight > 0:
    st.warning(f"🚨 **PERHATIAN:** {num_overweight} truk MELEBIHI BATAS BERAT! Solusi mungkin tidak valid.")

st.markdown("---")

# --- Baris 2: Peta Rute & Grafik Fitness Assignment ---
col1_map, col2_chart = st.columns([3, 2])

with col1_map:
    st.subheader("🗺️ Peta Rute Pengiriman")
    if routes_data_pydeck:
        city_points_data = [{"name": city, "coordinates": [coord[1], coord[0]]}
                            for city, coord in city_coords.items()]
        city_marker_layer = pdk.Layer(
            "ScatterplotLayer",
            data=city_points_data,
            get_position="coordinates",
            get_fill_color=[0, 0, 0, 180],
            get_radius=8000,
            radius_min_pixels=6,
            pickable=True,
            auto_highlight=True
        )
        city_text_layer = pdk.Layer(
            "TextLayer",
            data=city_points_data,
            get_position="coordinates",
            get_text="name",
            get_color=[0, 0, 0, 200],
            get_size=14,
            get_alignment_baseline="'bottom'",
            get_pixel_offset=[0, -18]
        )
        path_layers = [
            pdk.Layer(
                "PathLayer",
                data=[route_data],
                get_path="path",
                get_color="color",
                get_width=5,
                width_scale=1,
                width_min_pixels=3.5,
                pickable=True,
                auto_highlight=True,
                tooltip={"html": "<b>{truck}</b><br/>Rute: {route_info}<br/>Jarak: {distance_km:.1f} km"}
            ) for route_data in routes_data_pydeck
        ]

        initial_view_state = pdk.ViewState(
            latitude=city_coords[DEFAULT_START_CITY][0],
            longitude=city_coords[DEFAULT_START_CITY][1],
            zoom=5.8,
            pitch=45
        )

        all_layers = path_layers + [city_marker_layer, city_text_layer]

        st.pydeck_chart(pdk.Deck(
            layers=all_layers,
            initial_view_state=initial_view_state,
            map_style='mapbox://styles/mapbox/light-v10',
            tooltip={"style": {"backgroundColor":"rgba(0,0,0,0.7)","color":"white","fontSize":"12px"}}
        ))
    else:
        st.info("Tidak ada rute pengiriman untuk ditampilkan.")

with col2_chart:
    st.subheader("📈 Grafik Fitness Assignment PSO")
    if fitness_history_assign:
        fig_fit = go.Figure()
        fig_fit.add_trace(go.Scatter(
            y=fitness_history_assign,
            mode='lines+markers',
            name='Best Fitness per Iterasi',
            line=dict(color='royalblue', width=2),
            marker=dict(size=4)
        ))
        fig_fit.update_layout(
            xaxis_title="Iterasi",
            yaxis_title="Fitness Value",
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            hovermode="x unified"
        )
        st.plotly_chart(fig_fit, use_container_width=True)
    else:
        st.info("Data histori fitness tidak tersedia (mungkin optimasi tidak berjalan).")

st.markdown("---")

# --- Baris 3: Detail Assignment Item ---
st.subheader("📦 Detail Penugasan Item")
if not assigned_items_df.empty:
     try:
         # Extract truck number for sorting, handle non-numeric gracefully
         assigned_items_df['truck_num'] = assigned_items_df['assigned_truck'].str.extract(r'(\d+)').astype(float).fillna(float('inf'))
         assigned_items_df = assigned_items_df.sort_values(by=['truck_num', 'id'], na_position='last').drop(columns=['truck_num'])
     except Exception as e:
         logging.warning(f"Could not sort items by truck number: {e}")
         assigned_items_df = assigned_items_df.sort_values(by=['assigned_truck', 'id']) # Fallback sort

     st.dataframe(assigned_items_df[[
         'id', 'name', 'weight', 'dims', 'city', 'dim_category', 'assigned_truck'
     ]], use_container_width=True, hide_index=True) # hide_index is cleaner
else:
     st.info("Tidak ada data assignment item.")


st.markdown("---")

# --- Baris 4+: Detail Truk & Visualisasi Muatan ---
st.subheader("🚛 Detail Muatan & Visualisasi per Truk")

cols_per_row = 2
num_rows = (n_trucks + cols_per_row - 1) // cols_per_row
truck_idx_display = 1

for r in range(num_rows):
    cols = st.columns(cols_per_row)
    for c in range(cols_per_row):
        if truck_idx_display <= n_trucks:
            with cols[c]:
                truck_id = truck_idx_display
                info = final_truck_info[truck_id]
                # Ambil layout, penalty, dan details
                layout, penalty, penalty_details = final_layouts.get(truck_id, (None, float('inf'), {}))

                st.markdown(f"#### Truk {truck_id}")

                if info["is_overweight"]:
                    st.error(f"⛔️ OVERWEIGHT: {info['weight']:.1f} / {truck_max_weight:.1f} kg")
                else:
                    sisa_kapasitas = truck_max_weight - info['weight']
                    delta_text = f"{sisa_kapasitas:.1f} kg Sisa" if sisa_kapasitas >= -ZERO_TOLERANCE else f"{abs(sisa_kapasitas):.1f} kg Lebih"
                    delta_color = "normal" if sisa_kapasitas >= -ZERO_TOLERANCE else "inverse"
                    st.metric("Berat Muatan", f"{info['weight']:.1f} / {truck_max_weight:.1f} kg",
                              delta=delta_text, delta_color=delta_color)

                st.metric("Profit Truk Ini*", f"Rp {info.get('profit', 0):,.0f}",
                          help="Estimasi Revenue Truk - Estimasi Fuel Cost Truk")

                if not info["items"]:
                    st.info("Truk ini tidak membawa item.")
                else:
                    route_str = " -> ".join(info.get('route_sequence', ['N/A']))
                    st.caption(f"**Rute:** {route_str} | **Jarak:** {info.get('route_distance', 0):.1f} km")

                    with st.expander("Lihat daftar item di truk ini"):
                        items_in_truck_df = pd.DataFrame(info["items"])
                        st.dataframe(items_in_truck_df[["id", "name", "weight", "dims", "city"]], hide_index=True)

                    st.markdown("**Visualisasi Muatan:**")
                    if layout is None and info["items"]:
                        st.error("Packing final gagal mendapatkan layout.")
                    elif layout is None and not info["items"]:
                        pass # Truk kosong, tidak perlu visualisasi
                    elif penalty > ZERO_TOLERANCE and info["items"]:
                        st.warning(f"Packing mungkin belum optimal (Total Penalty: {penalty:.4f}).")
                        # Tampilkan rincian penalti jika ada penalti
                        with st.expander("Lihat Rincian Penalti Packing"):
                            st.write("Komponen Penalti:")
                            if penalty_details: # Pastikan dictionary tidak kosong
                                for key, value in penalty_details.items():
                                    if value > ZERO_TOLERANCE: # Tampilkan hanya yang > 0
                                        st.code(f"- {key}: {value:,.4f}")
                            else:
                                st.caption("Tidak ada detail penalti tersedia.")
                        # Tampilkan visualisasi
                        fig = create_truck_figure(truck_dims_tuple, layout, truck_id)
                        st.plotly_chart(fig, use_container_width=True)
                    elif info["items"]: # Penalti <= ZERO_TOLERANCE
                        st.success(f"Muatan terpack dengan baik (Total Penalty: {penalty:.4f}).")
                        # Opsi: Tetap tampilkan rincian meskipun kecil
                        with st.expander("Lihat Rincian Penalti Packing (Detail)"):
                             st.write("Komponen Penalti:")
                             if penalty_details:
                                 for key, value in penalty_details.items():
                                      st.code(f"- {key}: {value:,.4f}") # Tampilkan semua
                             else:
                                 st.caption("Tidak ada detail penalti tersedia.")
                        # Tampilkan visualisasi
                        fig = create_truck_figure(truck_dims_tuple, layout, truck_id)
                        st.plotly_chart(fig, use_container_width=True)

                # Pemisah antar truk dalam satu baris (jika bukan kolom terakhir)
                # Atau pemisah antar baris (jika kolom terakhir)
                if c < cols_per_row - 1 and truck_idx_display + 1 <= n_trucks:
                     pass # Biarkan Streamlit handle spacing antar kolom
                elif truck_idx_display < n_trucks:
                    st.markdown("---") # Pemisah antar baris

            truck_idx_display += 1

logging.info("Streamlit App Layout Generated.")
print("\nStreamlit App Ready. Run with: streamlit run your_script_name.py")