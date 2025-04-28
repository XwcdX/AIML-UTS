# -*- coding: utf-8 -*-
import streamlit as st
import math
import random
import pydeck as pdk
import pandas as pd
import json
import os
import numpy as np
# import itertools # combinations, permutations -> combinations tidak terpakai -> permutations juga tidak dipakai setelah simplifikasi rotasi
# from itertools import permutations # Tidak lagi diperlukan
import plotly.graph_objects as go
from functools import lru_cache
import logging # Tambahkan logging untuk debug

# --- Konfigurasi Logging ---
# Ganti level ke DEBUG jika ingin melihat log support per item, dll.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Konfigurasi Halaman Streamlit (HARUS JADI PERINTAH st PERTAMA) ---
st.set_page_config(layout="wide", page_title="Optimasi Muatan Truk PSO V3 Final (3 Rotasi)")
# --------------------------------------------------------------------

# --- Seed untuk Reproducibility ---
RANDOM_SEED = 30
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# -----------------------------------

# --- Constants: Tolerances & Bounds ---
POS_LOWER_BOUND_ASSIGN = 0.51 # Batas bawah untuk decoding posisi assignment
# Batas atas dihitung dinamis: n_trucks + (1.0 - POS_LOWER_BOUND_ASSIGN)
# --- PERUBAHAN ORIENTASI ---
ORIENTATION_UPPER_BOUND_PACK = 3.0 - 1e-9 # Indeks orientasi maks (0-2) -> Dulu 6.0
# ---------------------------
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
# --- PERUBAHAN PENALTI OVERSIZED ---
OVERSIZED_ITEM_PENALTY = 1e10 # Penalti JIKA item oversized TERASSIGN (bukan mencegah assignment)
# -----------------------------------

# --- Constants: Packing PSO Cache (Estimasi Cepat) ---
PACKING_CACHE_MAX_ITERS = 35
PACKING_CACHE_NUM_PARTICLES = 20
PACKING_CACHE_PATIENCE_FACTOR = 0.15
PACKING_CACHE_COMPACTION_WEIGHT = 1e-4

# --- Constants: Packing PSO Final (Visualisasi Kualitas Tinggi) ---
FINAL_PACKING_MAX_ITERS = 600
FINAL_PACKING_NUM_PARTICLES = 60
FINAL_PACKING_PATIENCE_FACTOR = 0.1
FINAL_PACKING_COMPACTION_WEIGHT = 1e-3
FINAL_PACKING_ATTEMPTS = 3

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
    # Menggunakan dimensi asli item, asumsi item bisa diletakkan di sisi manapun
    min_total_base_area = sum(min(l*w, l*h, w*h) for l, w, h in (it['dims'] for it in items))
    if min_total_base_area > L*W + ZERO_TOLERANCE:
        reason = f"Min base area {min_total_base_area:.2f} > Truck base area {L*W:.2f}"
        logging.debug(f"quick_feasible failed: {reason}")
        return False, reason

    # 3. Cek Dimensi Individu (sudah di pre-processing) - tidak dicek di sini lagi

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
    Optimasi packing 3D menggunakan PSO. Mempertimbangkan 3 orientasi unik.
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
        # --- PERUBAHAN ORIENTASI ---
        self.K = 3 # Hanya 3 orientasi unik
        # ---------------------------
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
                # --- PERUBAHAN ORIENTASI ---
                pos[4*i+3] = np.random.uniform(0, self.K) # K=3
                # ---------------------------

                vel_range = 0.1
                vel[4*i+0] = np.random.uniform(-self.L*vel_range, self.L*vel_range)
                vel[4*i+1] = np.random.uniform(-self.W*vel_range, self.W*vel_range)
                vel[4*i+2] = np.random.uniform(-self.H*vel_range, self.H*vel_range)
                # --- PERUBAHAN ORIENTASI ---
                vel[4*i+3] = np.random.uniform(-self.K*vel_range, self.K*vel_range) # K=3
                # ---------------------------

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
        """Mengembalikan dimensi item setelah rotasi berdasarkan 3 sumbu unik."""
        l, w, h = original_dims
        # --- PERUBAHAN ORIENTASI: 3 Orientasi Unik ---
        # Index 0: Sumbu Z vertikal (l x w base)
        # Index 1: Sumbu Y vertikal (l x h base)
        # Index 2: Sumbu X vertikal (w x h base)
        dims_map = [
            (l, w, h), # Rotasi 0 (Original/Sumbu Z)
            (l, h, w), # Rotasi 1 (Sumbu Y)
            (w, h, l)  # Rotasi 2 (Sumbu X)
            # Dimensi (w, l, h), (h, l, w), (h, w, l) adalah pencerminan/rotasi ekuivalen dari 3 ini
        ]
        # Pastikan index aman antara 0 dan 2
        safe_index = int(np.clip(orientation_index, 0, 2))
        # ---------------------------------------------
        return dims_map[safe_index]

    def _clamp(self, pos):
        """Membatasi posisi dan orientasi partikel."""
        for i in range(self.n):
            pos[4*i+0] = np.clip(pos[4*i+0], 0, self.L)
            pos[4*i+1] = np.clip(pos[4*i+1], 0, self.W)
            pos[4*i+2] = np.clip(pos[4*i+2], 0, self.H)
            # --- PERUBAHAN ORIENTASI ---
            # Gunakan konstanta ORIENTATION_UPPER_BOUND_PACK yang sudah diupdate
            pos[4*i+3] = np.clip(pos[4*i+3], 0, ORIENTATION_UPPER_BOUND_PACK)
            # ---------------------------

    def _get_placement(self, pos):
        """Mendekode posisi partikel menjadi layout penempatan item."""
        placement = []
        total_item_vol = 0.0
        for i in range(self.n):
            x, y, z = pos[4*i : 4*i+3]
            # --- PERUBAHAN ORIENTASI ---
            ori_idx = int(np.clip(pos[4*i+3], 0, 2)) # Max index adalah 2
            # ---------------------------
            original_dims = self.item_original_dims[i]
            # Gunakan fungsi _get_rotated_dims yang sudah diupdate
            w_rot, d_rot, h_rot = self._get_rotated_dims(original_dims, ori_idx)
            placement.append({
                'id': self.items[i]['id'], 'name': self.items[i]['name'],
                'x': x, 'y': y, 'z': z,
                'w': w_rot, 'd': d_rot, 'h': h_rot, # Dimensi setelah rotasi
                'vol': self.item_original_volumes[i] # Volume asli tetap
            })
            total_item_vol += self.item_original_volumes[i]
        return placement, total_item_vol

    @staticmethod
    def _calculate_overlap_volume(item_a, item_b):
        """Menghitung volume overlap antara dua bounding box item."""
        ox = max(0, min(item_a['x'] + item_a['w'], item_b['x'] + item_b['w']) - max(item_a['x'], item_b['x']))
        oy = max(0, min(item_a['y'] + item_a['d'], item_b['y'] + item_b['d']) - max(item_a['y'], item_b['y']))
        oz = max(0, min(item_a['z'] + item_a['h'], item_b['z'] + item_b['h']) - max(item_a['z'], item_b['z']))
        return ox * oy * oz

    @staticmethod
    def _calculate_2d_overlap_area(rect_a, rect_b):
        """Menghitung area overlap antara dua rektangel 2D (untuk penalti support)."""
        # rect = (x_min, y_min, x_max, y_max)
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
            # Volume keluar batas di 6 sisi
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
                if i >= j: continue # Hindari perhitungan ganda dan self-overlap
                overlap_vol = self._calculate_overlap_volume(item_a, item_b)
                if overlap_vol > ZERO_TOLERANCE:
                    pen_overlap += OVERLAP_PENALTY_FACTOR * overlap_vol

        # 3. Penalti Support (Benda melayang)
        for i, it in enumerate(placement):
            item_base_z = it['z']
            # Hanya cek item yang tidak menyentuh dasar truk
            if item_base_z < SUPPORT_TOLERANCE: continue
            item_base_area = it['w'] * it['d']
            if item_base_area < ZERO_TOLERANCE: continue # Item tidak punya luas alas (mustahil?)

            total_supported_area = 0.0
            item_rect = (it['x'], it['y'], it['x'] + it['w'], it['y'] + it['d'])

            # Cek support dari item lain di bawahnya
            for j, base in enumerate(placement):
                if i == j: continue # Jangan cek dengan diri sendiri
                base_top_z = base['z'] + base['h']
                # Cek apakah puncak item 'base' berada tepat di bawah alas item 'it'
                if abs(base_top_z - item_base_z) < SUPPORT_TOLERANCE:
                    base_rect = (base['x'], base['y'], base['x'] + base['w'], base['y'] + base['d'])
                    overlap_area = self._calculate_2d_overlap_area(item_rect, base_rect)
                    total_supported_area += overlap_area

            # Hitung rasio area yang tidak didukung
            support_ratio = total_supported_area / item_base_area if item_base_area > 0 else 1.0
            unsupported_ratio = max(0.0, 1.0 - support_ratio)

            # Tambahkan penalti jika area tidak didukung signifikan
            if unsupported_ratio > ZERO_TOLERANCE:
                # Penalti bisa dibuat quadratic agar lebih 'menghukum' rasio besar
                support_penalty_val = UNSUPPORTED_AREA_PENALTY_FACTOR * (unsupported_ratio ** 2)
                pen_support += support_penalty_val
                # if logging.getLogger().isEnabledFor(logging.DEBUG):
                #     logging.debug(f"Item {it['id']}@({it['x']:.1f},{it['y']:.1f},{it['z']:.1f}) unsup_ratio: {unsupported_ratio:.3f}, sup_pen: {support_penalty_val:.2f}")


        # 4. Penalti Ketinggian Item Besar (mendorong item berat/besar ke bawah)
        for i, it in enumerate(placement):
            # Penalti = Faktor * Volume_Asli * Ketinggian_Z
            height_penalty_val = HEIGHT_VOLUME_PENALTY_FACTOR * self.item_original_volumes[i] * it['z']
            pen_height += height_penalty_val

        # 5. Penalti Kompaksi (mendorong item agar merapat)
        if placement:
            # Cari batas terjauh dari item yang ditempatkan
            max_x = max(it['x'] + it['w'] for it in placement) if placement else 0
            max_y = max(it['y'] + it['d'] for it in placement) if placement else 0
            max_z = max(it['z'] + it['h'] for it in placement) if placement else 0
            # Hitung volume bounding box yang digunakan
            used_bbox_vol = max_x * max_y * max_z
            # Penalti = Bobot_Kompaksi * (Volume_BBox - Volume_Total_Item)
            # Hanya penalti jika BBox lebih besar dari total volume item (ada ruang kosong)
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

        # --- Fallback GBest Initialization Check ---
        if self.gbest_pos is None:
            logging.error("PackingPSO cannot optimize without initialized gbest_pos.")
            if self.pbest_pos.size > 0 and np.any(self.pbest_score < float('inf')):
                 best_initial_p = np.argmin(self.pbest_score)
                 self.gbest_score = self.pbest_score[best_initial_p]
                 self.gbest_pos = self.pbest_pos[best_initial_p].copy()
                 # Hitung detail untuk gbest fallback
                 _, self.gbest_details = self._penalty(self.gbest_pos)
                 logging.warning("gbest_pos was None during optimize start, re-setting from best pbest.")
            else:
                logging.critical("Fallback failed, no valid initial pbest score/position available. Cannot optimize.")
                return [], float('inf'), {} # Tidak bisa optimasi

        # Pastikan gbest_pos ada setelah fallback
        if self.gbest_pos is None:
             logging.critical("PSO cannot proceed without a valid gbest_pos.")
             return [], float('inf'), {}
        # --- End Fallback Check ---


        best_iter_score = self.gbest_score
        no_improvement_iters = 0

        logging.info(f"Starting PackingPSO optimization for {max_iters} iterations (patience={patience})...")

        # --- Iterasi PSO ---
        for t in range(max_iters):
            # Update inertia weight (linear decay)
            w_inertia = W_MAX - (W_MAX - W_MIN) * (t / float(max_iters - 1)) if max_iters > 1 else W_MIN

            for p in range(self.num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                # Update velocity
                cognitive_comp = self.c1 * r1 * (self.pbest_pos[p] - self.particles[p])
                social_comp = self.c2 * r2 * (self.gbest_pos - self.particles[p])
                new_vel = (w_inertia * self.velocities[p] + cognitive_comp + social_comp)
                self.velocities[p] = new_vel

                # Update position
                self.particles[p] += self.velocities[p]

                # Clamp position and orientation
                self._clamp(self.particles[p])

                # Evaluasi - hitung penalti dan detailnya
                current_score, current_details = self._penalty(self.particles[p])

                # Update pbest
                if current_score < self.pbest_score[p]:
                    self.pbest_score[p] = current_score
                    self.pbest_pos[p] = self.particles[p].copy()

                    # Update gbest
                    if current_score < self.gbest_score:
                        self.gbest_score = current_score
                        self.gbest_pos = self.particles[p].copy()
                        self.gbest_details = current_details.copy() # Simpan detail gbest baru

                        # Reset counter jika ada improvement signifikan pada gbest
                        if self.gbest_score < best_iter_score - ZERO_TOLERANCE:
                            best_iter_score = self.gbest_score
                            no_improvement_iters = 0
                        # else: # Improvement kecil atau tidak ada, increment counter
                        #     no_improvement_iters += 1 # Pindahkan ke luar loop partikel

            # Cek Early Stopping setelah semua partikel dievaluasi
            no_improvement_iters += 1 # Increment counter jika tidak ada improvement sig. di iterasi ini
            if no_improvement_iters >= patience:
                logging.info(f"PackingPSO early stopping at iteration {t+1} due to no improvement for {patience} iterations.")
                break
            # Stop jika solusi sudah sangat baik (penalti mendekati nol)
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
            # Gunakan skor dan detail gbest yang sudah tersimpan selama optimasi
            final_penalty = self.gbest_score
            final_penalty_details = self.gbest_details.copy()

            # Rekonstruksi layout dari gbest_pos
            placement, _ = self._get_placement(self.gbest_pos)
            for i, item_layout in enumerate(placement):
                # --- PERUBAHAN ORIENTASI ---
                ori_idx = int(np.clip(self.gbest_pos[4*i+3], 0, 2)) # Max index 2
                # ---------------------------
                item_layout['orientation'] = ori_idx # Tambahkan info orientasi ke layout final
            final_layout = placement
        else:
            # Fallback jika gbest_pos tidak valid setelah loop selesai (seharusnya sudah ditangani di awal)
            logging.error("Optimize finished but gbest_pos is unexpectedly None/invalid!")
            # Coba gunakan pbest terbaik sebagai fallback
            if self.pbest_score.size > 0 and np.min(self.pbest_score) < float('inf'):
                best_p_idx = np.argmin(self.pbest_score)
                fallback_pos = self.pbest_pos[best_p_idx]
                # Hitung penalti dan DETAIL dari fallback_pos
                final_penalty, final_penalty_details = self._penalty(fallback_pos)
                # Rekonstruksi layout dari fallback_pos
                placement, _ = self._get_placement(fallback_pos)
                for i, item_layout in enumerate(placement):
                     # --- PERUBAHAN ORIENTASI ---
                    ori_idx = int(np.clip(fallback_pos[4*i+3], 0, 2)) # Max index 2
                     # ---------------------------
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
                    # Handle missing or non-string polygon data gracefully
                    if pd.isna(polygon_str) or not isinstance(polygon_str, str) or not polygon_str.strip():
                        polygons[key] = []
                        continue

                    polygon_coords_raw = json.loads(polygon_str)
                    # Validate structure: list of lists/dicts with 2 numbers (lng, lat)
                    if isinstance(polygon_coords_raw, list) and polygon_coords_raw:
                        first_point = polygon_coords_raw[0]
                        if isinstance(first_point, dict) and 'lng' in first_point and 'lat' in first_point:
                            # Convert list of dicts to list of [lng, lat]
                            polygons[key] = [[p["lng"], p["lat"]] for p in polygon_coords_raw if isinstance(p, dict) and 'lng' in p and 'lat' in p]
                        elif isinstance(first_point, list) and len(first_point) == 2:
                            # Assume list of [lng, lat] or [lat, lng] - use as is after float conversion
                            polygons[key] = [[float(p[0]), float(p[1])] for p in polygon_coords_raw if isinstance(p, list) and len(p)==2]
                        else:
                            polygons[key] = [] # Invalid format inside list
                    else:
                       polygons[key] = [] # Empty list or not a list
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
    # --- PERUBAHAN DIMENSI AGAR LEBIH MUNGKIN OVERSIZED DENGAN 3 ROTASI ---
    {"id": "Item7", "name": "Ranjang", "weight": 400, "dims": (210, 160, 50), "city": "Malang"}, # Cek Oversized (Panjang > truck_L)
    # ----------------------------------------------------------------------
    {"id": "Item8", "name": "Kipas Angin", "weight": 30, "dims": (20, 20, 40), "city": "Bandung"}, # Kecil
    {"id": "Item9", "name": "WashingMachine","weight":350, "dims": (60,60,85), "city": "Jakarta"}, # Sedang
    {"id": "Item10", "name": "Bookshelf", "weight":100, "dims": (80,30,180), "city": "Surabaya"}, # Cek Oversized (Tinggi > truck_H)
    {"id": "Item11", "name": "Mattress", "weight":200, "dims": (200,90,30), "city": "Bandung"}, # Besar
    {"id": "Item12", "name": "Wardrobe", "weight":450, "dims": (100,60,200), "city": "Yogyakarta"}, # Cek Oversized (Tinggi > truck_H)
    {"id": "Item13", "name": "DiningTable", "weight":250, "dims": (160,90,75), "city": "Semarang"}, # Besar
    {"id": "Item14", "name": "DeskLamp", "weight":10, "dims": (15,15,40), "city": "Malang"}, # Kecil
    {"id": "Item15", "name": "Microwave", "weight":40, "dims": (50,40,35), "city": "Jakarta"}, # Kecil
    {"id": "Item16", "name": "Printer", "weight":25, "dims": (45,40,30), "city": "Surabaya"}, # Kecil
    {"id": "Item17", "name": "FloorLamp", "weight":20, "dims": (30,30,160), "city": "Bandung"}, # Cek Oversized (Tinggi > truck_H)
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
    """Mengategorikan item berdasarkan volume."""
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

    # --- PERUBAHAN CEK OVERSIZED ---
    # Cek apakah item bisa muat dalam *salah satu* dari 3 orientasi unik
    can_fit = False
    for i in range(3): # Hanya cek 3 orientasi
        w_rot, d_rot, h_rot = PackingPSO._get_rotated_dims((l, w, h), i)
        if w_rot <= truck_max_length + ZERO_TOLERANCE and \
           d_rot <= truck_max_width + ZERO_TOLERANCE and \
           h_rot <= truck_max_height + ZERO_TOLERANCE:
            can_fit = True
            break # Cukup satu orientasi yang muat
    # ----------------------------------

    item["is_oversized"] = not can_fit
    if item["is_oversized"]:
        oversized_count += 1
        logging.warning(f"Item {item['id']} ({item['name']} - dims {l}x{w}x{h}) is OVERSIZED for truck dimensions {truck_dims_tuple} in all 3 orientations.")

    items.append(item)

logging.info(f"Item preprocessing complete. {len(items) - oversized_count} assignable items, {oversized_count} oversized items.")


# --- Fungsi Jarak (Haversine & dari Dictionary) ---
@st.cache_data(show_spinner=False)
def haversine(coord1, coord2):
    """Menghitung jarak Haversine antara dua koordinat (lat, lon)."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371.0 # Radius bumi dalam km

    phi1, phi2 = map(math.radians, [lat1, lat2])
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance_km = R * c
    return distance_km

@st.cache_data(show_spinner=False)
def distance(city1, city2, start_city=DEFAULT_START_CITY):
    """Mencari jarak antara dua kota, utamakan dari CSV, fallback ke Haversine."""
    city1_norm = str(city1).strip()
    city2_norm = str(city2).strip()

    if city1_norm == city2_norm:
        return 0.0

    key = tuple(sorted((city1_norm, city2_norm)))

    # Prioritas 1: Cek dictionary dari CSV
    if key in distance_dict:
        return distance_dict[key]

    # Prioritas 2: Cek koordinat dan hitung Haversine
    if city1_norm in city_coords and city2_norm in city_coords:
        dist_hv = haversine(city_coords[city1_norm], city_coords[city2_norm])
        logging.warning(f"Distance for {key} not found in CSV, using Haversine: {dist_hv:.2f} km.")
        # distance_dict[key] = dist_hv # Optional: Cache Haversine result jika ingin digunakan lagi
        return dist_hv

    # Jika tidak ditemukan sama sekali
    logging.error(f"DISTANCE NOT FOUND: Cannot find distance between '{city1_norm}' and '{city2_norm}'. Check CSV and city_coords.")
    st.error(f"Jarak antara {city1_norm} dan {city2_norm} tidak ditemukan.")
    return float('inf') # Kembalikan infinity jika jarak tidak valid

@st.cache_data(show_spinner=False)
def route_distance(cities, start_city=DEFAULT_START_CITY):
    """Menghitung total jarak rute (Nearest Neighbor Heuristic dari start_city)."""
    if not cities: return 0.0
    # Pastikan kota unik dan bukan start_city
    unique_cities = list(dict.fromkeys(c for c in cities if c != start_city))
    if not unique_cities: return 0.0 # Hanya tujuan ke start_city (jarak 0 jika tidak ada item lain)

    current_city = start_city
    unvisited = unique_cities[:]
    total_dist = 0.0

    while unvisited:
        nearest_city = None
        min_dist = float('inf')
        # Cari kota terdekat dari current_city di antara yang belum dikunjungi
        for city in unvisited:
            dist_to_city = distance(current_city, city, start_city)
            if dist_to_city < min_dist:
                min_dist = dist_to_city
                nearest_city = city

        # Cek apakah rute valid ditemukan
        if min_dist == float('inf') or nearest_city is None:
            logging.error(f"Cannot find valid route from {current_city} to any of {unvisited}. Route calculation failed.")
            return float('inf') # Rute tidak mungkin

        # Tambahkan jarak ke kota terdekat, update posisi, hapus dari unvisited
        total_dist += min_dist
        current_city = nearest_city
        unvisited.remove(nearest_city)

    # Hitung jarak kembali ke start_city dari kota terakhir
    dist_back = distance(current_city, start_city, start_city)
    if dist_back == float('inf'):
        logging.error(f"Cannot find valid route back from {current_city} to {start_city}. Route calculation failed.")
        return float('inf') # Rute kembali tidak mungkin

    total_dist += dist_back
    return total_dist

# --- Fungsi Bantuan Rute untuk Visualisasi ---
def get_route_sequence(cities, start_city=DEFAULT_START_CITY):
    """Menghasilkan urutan kota yang dikunjungi (Nearest Neighbor)."""
    if not cities: return [start_city, start_city] # Kembali ke base jika kosong
    unique_cities = list(dict.fromkeys(c for c in cities if c != start_city))
    if not unique_cities: return [start_city, start_city] # Kembali ke base jika hanya ada item base

    route = [start_city]
    current = start_city
    unvisited = unique_cities[:]

    while unvisited:
        # Cari kota terdekat dari 'current'
        distances_to_unvisited = {city: distance(current, city, start_city) for city in unvisited}
        # Filter kota yang bisa dijangkau (jarak != inf)
        valid_distances = {city: dist for city, dist in distances_to_unvisited.items() if dist != float('inf')}

        if not valid_distances:
            # Tidak bisa menjangkau kota tersisa, coba kembali ke start
            logging.warning(f"Cannot reach remaining cities {unvisited} from {current}. Ending route sequence prematurely.")
            dist_back = distance(current, start_city, start_city)
            if dist_back != float('inf'):
                route.append(start_city)
            return route # Kembalikan rute sejauh ini

        # Pilih kota terdekat yang valid
        nearest = min(valid_distances, key=valid_distances.get)
        route.append(nearest)
        current = nearest
        unvisited.remove(nearest)

    # Tambahkan kembali ke start city di akhir
    dist_final_back = distance(current, start_city, start_city)
    if dist_final_back != float('inf'):
        route.append(start_city)
    else:
        logging.warning(f"Cannot return to {start_city} from last city {current}. Route sequence might be incomplete.")

    return route

# --- Fitness Function (Outer PSO - Assignment) ---
def compute_fitness(assignment, items, n_trucks, truck_dims, truck_max_weight):
    """Menghitung nilai fitness untuk sebuah solusi assignment."""
    truck_details = {t: {"items": [], "cities": set(), "weight": 0.0}
                     for t in range(1, n_trucks + 1)}
    total_revenue = 0.0
    constraint_penalty = 0.0
    total_packing_penalty = 0.0 # Berdasarkan cache atau quick_feasible fail

    # Rincian penalti di level assignment (opsional untuk debug)
    pen_details_assignment = {"oversized": 0.0, "invalid_assign": 0.0, "invalid_route": 0.0, "overweight": 0.0, "packing_impossible": 0.0}

    # Iterasi melalui assignment setiap item
    for item_idx, assigned_truck_idx in enumerate(assignment):
        # Lewati item yang tidak diassign (assigned_truck_idx = 0)
        if assigned_truck_idx == 0: continue

        item = items[item_idx]

        # --- PERUBAHAN LOGIKA OVERSIZED ---
        # Penalti jika item oversized TERASSIGN ke truk manapun (idx != 0)
        if item["is_oversized"]:
            pen_details_assignment["oversized"] += OVERSIZED_ITEM_PENALTY
            constraint_penalty += OVERSIZED_ITEM_PENALTY
            # TIDAK pakai 'continue' di sini agar item oversized tetap diproses
            # untuk penalti lain jika relevan (misal overweight jika beratnya besar),
            # meskipun penalti oversized ini sudah sangat besar.
            # Jika ingin benar2 mengabaikannya setelah penalti, tambahkan 'continue' lagi.
            # logging.debug(f"Fitness Eval: Penalizing assignment of oversized item {item['id']} to truck {assigned_truck_idx}.")


        # Penalti jika assignment ke nomor truk tidak valid
        if not (1 <= assigned_truck_idx <= n_trucks):
            pen_details_assignment["invalid_assign"] += INVALID_ASSIGNMENT_PENALTY
            constraint_penalty += INVALID_ASSIGNMENT_PENALTY
            continue # Lewati item ini jika assignment invalid

        # Jika assignment valid, tambahkan item ke truk
        truck_id = int(assigned_truck_idx)
        truck_details[truck_id]["items"].append(item)
        truck_details[truck_id]["cities"].add(item["city"])
        truck_details[truck_id]["weight"] += item["weight"]

        # Hitung revenue item (hanya jika jarak valid)
        dist_to_dest = distance(DEFAULT_START_CITY, item["city"])
        if dist_to_dest == float('inf'):
            # Penalti jika tujuan item tidak bisa dijangkau dari start city
            # Mungkin lebih baik penalti per rute truk saja?
            pen_details_assignment["invalid_route"] += INVALID_ROUTE_PENALTY # Penalti per item
            constraint_penalty += INVALID_ROUTE_PENALTY
            item_revenue = 0
        else:
            # Revenue = Berat * Jarak * Faktor Kategori
            item_revenue = item["weight"] * dist_to_dest * item["cat_factor"]
        total_revenue += item_revenue


    # Hitung biaya & penalti per truk
    total_fuel_cost = 0.0
    for truck_id in range(1, n_trucks + 1):
        info = truck_details[truck_id]
        truck_items = info["items"]
        # Lewati truk kosong
        if not truck_items: continue

        # Penalti Overweight
        current_weight = info["weight"]
        if current_weight > truck_max_weight:
            overweight_amount = current_weight - truck_max_weight
            # Penalti = Konstanta + Faktor * (Kelebihan^2)
            ow_pen = OVERWEIGHT_CONSTANT_PENALTY + OVERWEIGHT_FACTOR_PENALTY * (overweight_amount ** 2)
            pen_details_assignment["overweight"] += ow_pen
            constraint_penalty += ow_pen
            # logging.debug(f"Fitness Eval: Truck {truck_id} overweight by {overweight_amount:.2f} kg. Penalty: {ow_pen:.2E}")

        # Hitung Biaya Rute & Cek Validitas Rute Truk
        route_dist_km = route_distance(list(info["cities"]), DEFAULT_START_CITY)
        if route_dist_km == float('inf'):
            # Tambahkan penalti jika rute untuk truk ini tidak valid
            # Cek agar tidak dobel penalti jika sudah ada dari item di atas
            if pen_details_assignment["invalid_route"] == 0:
                pen_details_assignment["invalid_route"] += INVALID_ROUTE_PENALTY # Penalti per rute
                constraint_penalty += INVALID_ROUTE_PENALTY
            # logging.debug(f"Fitness Eval: Truck {truck_id} has invalid route. Penalty added.")
        else:
            # Hitung biaya bahan bakar jika rute valid
            total_fuel_cost += COST_PER_KM * route_dist_km

        # Hitung Penalti Packing (Gunakan Cache / Quick Feasible)
        # 1. Cek kelayakan dasar dulu
        feasible_check, feasible_reason = quick_feasible(truck_dims, truck_items)
        if not feasible_check:
            # Penalti besar jika secara volume/luas alas sudah tidak mungkin
            packing_pen = PACKING_IMPOSSIBLE_PENALTY + PACKING_IMPOSSIBLE_ITEM_FACTOR * len(truck_items)
            pen_details_assignment["packing_impossible"] += packing_pen
            total_packing_penalty += packing_pen
            # logging.debug(f"Fitness Eval: Truck {truck_id} quick_feasible failed ({feasible_reason}). Penalty: {packing_pen:.2E}")
        else:
            # 2. Jika quick feasible lolos, estimasi penalti packing pakai cache
            # Buat tuple item yang hashable untuk key cache
            items_tuple_key = tuple(sorted(
                (it["id"], it["name"], tuple(it["dims"]), it["weight"]) for it in truck_items
            ))
            # Panggil fungsi cache
            packing_pen = packing_penalty_cache(truck_dims, items_tuple_key)
            total_packing_penalty += packing_pen
            # logging.debug(f"Fitness Eval: Truck {truck_id} packing cache penalty: {packing_pen:.4f}")


    # Fitness = Profit - Penalti Packing - Penalti Constraint Lain
    profit = total_revenue - total_fuel_cost
    # Total fitness adalah nilai yang ingin dimaksimalkan
    fitness = profit - total_packing_penalty - constraint_penalty

    # Optional: Log fitness breakdown jika fitness sangat rendah (negatif besar)
    # if fitness < -1e8: # Threshold for logging bad fitness
    #     logging.debug(f"Low Fitness Encountered: {fitness:.2E}, Profit: {profit:.2f}, PackPen: {total_packing_penalty:.2f}, ConstrPen: {constraint_penalty:.2f}, Details: {pen_details_assignment}")

    return fitness

# --- Fungsi Decode Posisi PSO Luar ---
def decode_position(position, n_items, n_trucks, assignable_mask):
    """Mengubah vektor posisi kontinu PSO menjadi assignment diskrit ke truk."""
    assignment = np.zeros(n_items, dtype=int)
    # Tentukan batas atas untuk nilai posisi yang akan di-decode
    # Nilai antara 0.51 dan n_trucks + 0.49
    pos_upper_bound_assign = n_trucks + (1.0 - POS_LOWER_BOUND_ASSIGN)

    for i in range(n_items):
        # Hanya proses item yang 'assignable' (tidak oversized)
        if assignable_mask[i]:
            # Batasi nilai posisi dalam rentang yang valid untuk assignment
            clamped_val = np.clip(position[i], POS_LOWER_BOUND_ASSIGN, pos_upper_bound_assign)
            # Bulatkan ke integer terdekat untuk mendapatkan nomor truk (1, 2, ..., n_trucks)
            # Nilai < POS_LOWER_BOUND_ASSIGN (misal 0) akan tetap 0 (tidak diassign)
            assignment[i] = int(round(clamped_val))
        # else: item oversized, assignment[i] tetap 0 (default) -> tapi bisa diubah oleh PSO jika tidak di-masking
        # Koreksi: Sebaiknya item oversized tidak diupdate posisinya di loop PSO utama
    return assignment

# --- Outer PSO: Assignment ---
num_particles_assign = 30
max_iter_assign = 200
patience_assign = 30
improvement_threshold = 10.0 # Minimal improvement agar dianggap signifikan

# Buat mask untuk item yang BISA diassign (tidak oversized)
assignable_mask = np.array([not item["is_oversized"] for item in items])
n_assignable_items = assignable_mask.sum()
n_total_items = len(items)

# Inisialisasi partikel, velocity, pbest, gbest untuk Assignment PSO
particles_assign = np.zeros((num_particles_assign, n_total_items))
velocities_assign = np.zeros((num_particles_assign, n_total_items))
pbest_positions_assign = np.zeros((num_particles_assign, n_total_items))
pbest_fitness_assign = np.full(num_particles_assign, -float('inf')) # Maksimasi fitness
gbest_position_assign = None
gbest_fitness_assign = -float('inf')
prev_gbest_assign = -float('inf') # Untuk cek improvement
no_improvement_count = 0

logging.info("Initializing Assignment PSO...")
# Inisialisasi posisi dan velocity awal partikel
for p_idx in range(num_particles_assign):
    position = np.zeros(n_total_items) # Posisi awal 0 untuk semua
    velocity = np.zeros(n_total_items)

    # Hanya inisialisasi posisi & velocity acak untuk item yang assignable
    if n_assignable_items > 0:
        pos_upper_bound_assign = n_trucks + (1.0 - POS_LOWER_BOUND_ASSIGN)
        # Posisi awal acak antara batas bawah dan atas untuk item assignable
        position[assignable_mask] = np.random.uniform(POS_LOWER_BOUND_ASSIGN, pos_upper_bound_assign, size=n_assignable_items)

        # Velocity awal acak (misal 10% dari range posisi)
        vel_range_assign = (pos_upper_bound_assign - POS_LOWER_BOUND_ASSIGN) * 0.1
        velocities_assign[p_idx][assignable_mask] = np.random.uniform(-vel_range_assign, vel_range_assign, size=n_assignable_items)

    particles_assign[p_idx] = position.copy()
    # Velocities sudah diisi di atas

    # Hitung fitness awal
    assignment = decode_position(position, n_total_items, n_trucks, assignable_mask)
    fit = compute_fitness(assignment, items, n_trucks, truck_dims_tuple, truck_max_weight)

    # Set pbest awal
    pbest_positions_assign[p_idx] = position.copy()
    pbest_fitness_assign[p_idx] = fit

    # Update gbest awal jika lebih baik
    if fit > gbest_fitness_assign:
        gbest_fitness_assign = fit
        gbest_position_assign = position.copy()

# Fallback jika gbest tidak terinisialisasi (misal semua fitness awal -inf)
if gbest_position_assign is None and n_assignable_items > 0:
    if np.any(pbest_fitness_assign > -float('inf')):
          best_initial_p_idx = np.argmax(pbest_fitness_assign)
          gbest_fitness_assign = pbest_fitness_assign[best_initial_p_idx]
          gbest_position_assign = pbest_positions_assign[best_initial_p_idx].copy()
          logging.warning("Initial gbest_assign was None, setting from best pbest.")
    else:
          logging.error("Could not initialize gbest_assign. All initial fitness values might be -infinity.")
          # Handle error atau exit jika diperlukan
          # Misalnya: st.error("Optimasi tidak dapat dimulai karena solusi awal tidak valid.") -> bisa stop eksekusi


logging.info(f"Assignment PSO Initialization Complete. Initial Global Best Fitness: {gbest_fitness_assign:,.2f}")

# --- Loop Utama PSO Assignment ---
fitness_history_assign = []
# Hanya jalankan jika gbest awal valid
if gbest_position_assign is not None:
    logging.info(f"Running Assignment PSO for max {max_iter_assign} iterations...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Iter 0/{max_iter_assign}, Best Fitness: {gbest_fitness_assign:,.2f}")

    for it in range(1, max_iter_assign + 1):
        # Update inertia weight
        w_inertia = W_MAX - (W_MAX - W_MIN) * (it / float(max_iter_assign))

        # Update tiap partikel
        for i in range(num_particles_assign):
            current_pos = particles_assign[i]
            current_vel = velocities_assign[i]
            pbest_pos = pbest_positions_assign[i]

            r1 = np.random.rand(n_total_items)
            r2 = np.random.rand(n_total_items)

            # Hitung komponen velocity baru
            cognitive_comp = C1_ASSIGN * r1 * (pbest_pos - current_pos)
            social_comp = C2_ASSIGN * r2 * (gbest_position_assign - current_pos)
            new_vel = w_inertia * current_vel + cognitive_comp + social_comp

            # Update velocity HANYA untuk item assignable
            velocities_assign[i][assignable_mask] = new_vel[assignable_mask]
            # Update posisi HANYA untuk item assignable
            particles_assign[i][assignable_mask] += velocities_assign[i][assignable_mask]

            # Clamp posisi HANYA untuk item assignable
            pos_upper_bound_assign = n_trucks + (1.0 - POS_LOWER_BOUND_ASSIGN)
            particles_assign[i][assignable_mask] = np.clip(
                particles_assign[i][assignable_mask],
                POS_LOWER_BOUND_ASSIGN,
                pos_upper_bound_assign
            )
            # Posisi item non-assignable (oversized) tidak berubah (tetap 0)

            # Decode posisi baru dan hitung fitness
            assignment = decode_position(particles_assign[i], n_total_items, n_trucks, assignable_mask)
            fit = compute_fitness(assignment, items, n_trucks, truck_dims_tuple, truck_max_weight)

            # Update pbest jika fitness membaik
            if fit > pbest_fitness_assign[i]:
                pbest_fitness_assign[i] = fit
                pbest_positions_assign[i] = particles_assign[i].copy()

                # Update gbest jika pbest baru lebih baik dari gbest saat ini
                if fit > gbest_fitness_assign:
                    gbest_fitness_assign = fit
                    gbest_position_assign = particles_assign[i].copy()
                    # Reset counter jika gbest membaik signifikan (cek di luar loop partikel)


        # Catat gbest fitness per iterasi
        fitness_history_assign.append(gbest_fitness_assign)
        # Hitung improvement dari iterasi sebelumnya
        improvement = gbest_fitness_assign - prev_gbest_assign

        # Update UI
        status_text.text(f"Iter {it}/{max_iter_assign}, Best Fitness: {gbest_fitness_assign:,.2f}")
        progress_bar.progress(it / max_iter_assign)

        # Cek early stopping berdasarkan improvement
        if improvement >= improvement_threshold:
            no_improvement_count = 0 # Reset counter jika ada improvement signifikan
            prev_gbest_assign = gbest_fitness_assign # Update nilai gbest sebelumnya
        elif it > patience_assign: # Mulai hitung no improvement setelah iterasi awal
            no_improvement_count += 1

        # Stop jika tidak ada improvement signifikan selama 'patience_assign' iterasi
        if no_improvement_count >= patience_assign:
            logging.info(f"Assignment PSO early stopping at iteration {it} due to no significant improvement for {patience_assign} iterations.")
            status_text.text(f"Early stopping at iteration {it}. Final Best Fitness: {gbest_fitness_assign:,.2f}")
            break # Keluar dari loop iterasi utama

    # Hapus progress bar setelah selesai
    progress_bar.empty()
    logging.info(f"Assignment PSO finished. Final Best Fitness: {gbest_fitness_assign:,.2f}")

else:
    # Kasus jika gbest_position_assign None di awal
    st.error("Assignment PSO cannot run because no valid initial solution was found.")
    logging.error("Assignment PSO did not run due to missing initial gbest.")
    fitness_history_assign = [] # Pastikan history kosong

# --- Dapatkan Assignment Terbaik ---
best_assignment = np.zeros(n_total_items, dtype=int) # Default: semua tidak terassign
if gbest_position_assign is not None:
    # Gunakan gbest jika valid
    best_assignment = decode_position(gbest_position_assign, n_total_items, n_trucks, assignable_mask)
    logging.info("Using gbest assignment.")
else:
    # Fallback ke pbest terbaik jika gbest tidak valid setelah loop (seharusnya tidak terjadi jika loop berjalan)
    if pbest_fitness_assign.size > 0 and np.any(pbest_fitness_assign > -float('inf')):
        best_pbest_idx = np.argmax(pbest_fitness_assign)
        best_pbest_pos = pbest_positions_assign[best_pbest_idx]
        best_assignment = decode_position(best_pbest_pos, n_total_items, n_trucks, assignable_mask)
        # Update gbest_fitness_assign ke nilai pbest terbaik untuk konsistensi laporan
        gbest_fitness_assign = pbest_fitness_assign[best_pbest_idx]
        logging.warning("gbest_position_assign was None after PSO loop. Using best pbest as fallback.")
        st.warning("Solusi terbaik global tidak ditemukan setelah optimasi, menggunakan solusi partikel terbaik.")
    else:
        # Kasus tidak ada solusi valid sama sekali
        logging.error("No valid assignment solution found after PSO.")
        st.error("Optimasi penugasan gagal menemukan solusi yang valid.")
        # best_assignment tetap [0, 0, ..., 0]

# --- Proses Hasil Assignment Terbaik ---
def get_truck_info(assignment, items, n_trucks, truck_max_weight, start_city=DEFAULT_START_CITY):
    """Mengolah hasil assignment terbaik menjadi informasi detail per truk."""
    truck_info = {
        t: {
            "items": [], "weight": 0.0, "volume": 0.0, "cities": [], # Gunakan list untuk urutan
            "revenue": 0.0, "route_distance": 0.0, "fuel_cost": 0.0,
            "profit": 0.0, "is_overweight": False, "route_sequence": []
        } for t in range(1, n_trucks + 1)
    }
    assigned_item_details = [] # List untuk DataFrame detail item

    for i, truck_idx in enumerate(assignment):
        item = items[i].copy() # Salin item agar tidak mengubah data asli
        assigned_truck_str = "" # Inisialisasi status assignment

        # Tentukan status assignment item
        if item["is_oversized"]:
            # Jika item memang oversized dari awal
             # Jika terassign (truck_idx != 0), akan dipenalti di fitness, tapi di sini tandai saja
            assigned_truck_str = f"Oversized (Assigned to Truck {truck_idx})" if truck_idx != 0 else "Oversized (Unassigned)"
        elif truck_idx == 0:
            assigned_truck_str = "Unassigned"
        elif not (1 <= truck_idx <= n_trucks):
            # Seharusnya tidak terjadi jika decode & PSO benar, tapi sebagai pengaman
            assigned_truck_str = f"Invalid Assignment ({truck_idx})"
            logging.error(f"Invalid truck index {truck_idx} found in final assignment for item {item['id']}.")
        else:
            # Assignment valid
            assigned_truck_str = f"Truk {truck_idx}"
            truck_id = int(truck_idx)

            # Tambahkan item ke truk
            truck_info[truck_id]["items"].append(item)
            truck_info[truck_id]["weight"] += item["weight"]
            l, w, h = item["dims"]
            # Hitung volume item dalam m^3 (asumsi dims cm)
            truck_info[truck_id]["volume"] += (l * w * h) / 1e6 # m^3

            # Tambahkan kota tujuan (jika bukan start city & belum ada)
            if item["city"] != start_city and item["city"] not in truck_info[truck_id]["cities"]:
                truck_info[truck_id]["cities"].append(item["city"]) # Tambahkan ke list

            # Hitung revenue item jika jarak valid
            dist_to_dest = distance(start_city, item["city"])
            if dist_to_dest != float('inf'):
                item_revenue = item["weight"] * dist_to_dest * item["cat_factor"]
                truck_info[truck_id]["revenue"] += item_revenue
            # Else: revenue tetap 0 jika jarak infinity (sudah dipenalti di fitness)

        # Tambahkan detail item ke list untuk DataFrame
        assigned_item_details.append({
            "id": item["id"], "name": item["name"], "weight": item["weight"],
            "dims": f"{item['dims'][0]}x{item['dims'][1]}x{item['dims'][2]}", # Format string dimensi
            "city": item["city"], "dim_category": item["dim_category"],
            "assigned_truck": assigned_truck_str # Status assignment (Truk X, Unassigned, Oversized)
        })

    # Hitung detail rute, biaya, profit per truk setelah semua item dialokasikan
    for t in range(1, n_trucks + 1):
        info = truck_info[t]
        if info["items"]: # Hanya proses truk yang ada isinya
            # Dapatkan urutan rute (NN)
            route_seq = get_route_sequence(info["cities"], start_city)
            # Dapatkan total jarak rute
            route_dist = route_distance(info["cities"], start_city)

            info["route_sequence"] = route_seq # Simpan urutan kota
            if route_dist == float('inf'):
                # Jika rute tidak valid
                info["route_distance"] = 0.0 # Atau tandai NaN?
                info["fuel_cost"] = 0.0
                # Profit hanya dari revenue (yang mungkin 0 jika item tak terjangkau)
                # Atau set profit ke -inf jika rute tidak valid? Tergantung definisi.
                info["profit"] = info["revenue"] # Profit = Revenue (karena cost 0 tapi rute invalid)
                logging.warning(f"Truck {t} has an invalid route in the final assignment.")
            else:
                # Jika rute valid
                info["route_distance"] = route_dist
                info["fuel_cost"] = COST_PER_KM * route_dist
                info["profit"] = info["revenue"] - info["fuel_cost"] # Profit = Revenue - Cost

            # Cek status overweight final
            if info["weight"] > truck_max_weight + ZERO_TOLERANCE: # Beri toleransi
                info["is_overweight"] = True
                logging.warning(f"Truck {t} is OVERWEIGHT in the final assignment: {info['weight']:.2f} kg > {truck_max_weight:.2f} kg.")
                # Profit tidak diubah di sini, tapi status overweight dicatat

    # Kembalikan dictionary info truk dan DataFrame detail item
    return truck_info, pd.DataFrame(assigned_item_details)

# Panggil fungsi untuk mendapatkan info final berdasarkan assignment terbaik
final_truck_info, assigned_items_df = get_truck_info(best_assignment, items, n_trucks, truck_max_weight, DEFAULT_START_CITY)


# --- Fungsi Bantuan Path untuk Pydeck ---
def get_segment_path(city_a, city_b):
    """Mendapatkan koordinat path (garis) antara dua kota, utamakan dari polygon CSV."""
    city_a_norm = str(city_a).strip()
    city_b_norm = str(city_b).strip()
    if city_a_norm == city_b_norm: return [] # Tidak ada path jika kota sama

    key = tuple(sorted((city_a_norm, city_b_norm)))

    # 1. Cek apakah ada polygon di data CSV
    if key in polygons and isinstance(polygons.get(key), list) and len(polygons[key]) >= 2:
        path_data = polygons[key] # List of [lng, lat]
        # Coba pastikan urutan path sesuai arah A -> B (jika koordinat ada)
        if city_a_norm in city_coords and city_b_norm in city_coords:
            coord_a = city_coords[city_a_norm] # (lat, lon)
            coord_b = city_coords[city_b_norm] # (lat, lon)
            try:
                # Asumsi path_data[0] adalah [lng, lat]
                path_start_coord = (path_data[0][1], path_data[0][0]) # (lat, lon)
                dist_start_a = haversine(coord_a, path_start_coord)
                dist_start_b = haversine(coord_b, path_start_coord)
                # Jika titik awal path lebih dekat ke B daripada A, balik urutannya
                if dist_start_b < dist_start_a - ZERO_TOLERANCE:
                    return path_data[::-1] # Balik list
                else:
                    return path_data # Urutan sudah benar
            except Exception as e:
                logging.warning(f"Error checking path orientation for {key}: {e}. Using original polygon order.")
                return path_data # Gunakan urutan asli jika error
        else:
            return path_data # Tidak bisa cek orientasi jika koordinat kota tidak ada

    # 2. Jika tidak ada polygon, buat garis lurus jika koordinat ada
    if city_a_norm in city_coords and city_b_norm in city_coords:
        logging.debug(f"No polygon for {key}. Creating straight line path.")
        coord_a = city_coords[city_a_norm] # (lat, lon)
        coord_b = city_coords[city_b_norm] # (lat, lon)
        # Format path Pydeck: [[lng_a, lat_a], [lng_b, lat_b]]
        return [[coord_a[1], coord_a[0]], [coord_b[1], coord_b[0]]]

    # 3. Jika tidak ada polygon dan koordinat
    logging.warning(f"Cannot get path segment between {city_a_norm} and {city_b_norm}. No polygon and missing coordinates.")
    return [] # Kembalikan list kosong

def get_full_route_path(route_sequence):
    """Menggabungkan segmen-segmen path menjadi path rute lengkap."""
    full_path = []
    if not route_sequence or len(route_sequence) < 2: return [] # Perlu minimal 2 kota untuk path

    for i in range(len(route_sequence) - 1):
        city_a = route_sequence[i]
        city_b = route_sequence[i+1]
        segment = get_segment_path(city_a, city_b)

        if not segment:
            logging.warning(f"Skipping segment from {city_a} to {city_b} in full path generation due to missing data.")
            continue # Lewati segmen yang tidak valid

        # Gabungkan segmen, hindari duplikasi titik jika ujung segmen sebelumnya == awal segmen baru
        if not full_path:
            # Segmen pertama, tambahkan semua titik
            full_path.extend(segment)
        else:
            last_point = full_path[-1] # Titik terakhir [lng, lat]
            first_point_segment = segment[0] # Titik pertama segmen baru [lng, lat]
            # Cek apakah titiknya sama (dengan toleransi)
            if abs(last_point[0] - first_point_segment[0]) < ZERO_TOLERANCE and \
               abs(last_point[1] - first_point_segment[1]) < ZERO_TOLERANCE:
                # Jika sama, tambahkan segmen baru mulai dari titik KEDUA
                full_path.extend(segment[1:])
            else:
                # Jika tidak sama (ada gap?), tambahkan seluruh segmen baru
                logging.debug(f"Gap detected between segments ending at {last_point} and starting at {first_point_segment}. Appending full segment.")
                full_path.extend(segment)
    return full_path

# --- Siapkan Data Rute Pydeck ---
routes_data_pydeck = []
# Warna unik untuk setiap truk (sampai 8 truk)
colors = [[255,0,0,200],[0,180,0,200],[0,0,255,200],[255,165,0,200],
          [128,0,128,200],[0,200,200,200],[255,20,147,200],[60,179,113,200]]
logging.info("Generating route paths for visualization...")
for t in range(1, n_trucks + 1):
    info = final_truck_info[t]
    route_seq = info.get("route_sequence", [])
    # Hanya buat path jika ada lebih dari 1 kota (minimal start -> end)
    if len(route_seq) > 1:
        full_path = get_full_route_path(route_seq)
        if full_path: # Pastikan path berhasil dibuat
            # Buat string rute untuk tooltip
            # Format: Base -> Dest1 -> Dest2 -> Base
            display_route = f"{route_seq[0]} → {' → '.join(route_seq[1:-1])} → {route_seq[-1]}" if len(route_seq) > 2 else f"{route_seq[0]} → {route_seq[-1]}"

            routes_data_pydeck.append({
                "truck": f"Truk {t}",
                "path": full_path, # List of [lng, lat] coordinates
                "color": colors[(t-1) % len(colors)], # Ambil warna sesuai nomor truk
                "route_info": display_route, # String untuk tooltip
                "distance_km": info.get("route_distance", 0) # Jarak untuk tooltip
            })
        else:
            logging.warning(f"Failed to create visual path for Truck {t} route: {' -> '.join(route_seq)}")
    elif info["items"]: # Jika truk ada item tapi hanya di base city
        logging.info(f"Truck {t} only has items for the base city {DEFAULT_START_CITY}. No route path generated.")


# --- Packing Final Kualitas Tinggi ---
final_layouts = {} # Akan menyimpan (layout, penalty, details) per truk
logging.info("Running final (high-quality) packing optimization for assigned trucks...")
packing_progress = st.progress(0)
packing_status = st.empty()

processed_trucks = 0
trucks_with_items = [t for t in range(1, n_trucks + 1) if final_truck_info[t]["items"]]
total_trucks_to_pack = len(trucks_with_items)

if total_trucks_to_pack == 0:
    packing_status.text("No trucks have items to pack.")
    packing_progress.progress(1.0)
else:
    for t in range(1, n_trucks + 1):
        items_for_truck = final_truck_info[t]["items"]
        num_items_truck = len(items_for_truck)

        if not items_for_truck:
            final_layouts[t] = ([], 0.0, {}) # Layout kosong, penalti nol, details kosong
            logging.info(f"Truck {t} is empty, skipping final packing.")
            # Update progress even for empty trucks if needed, or only for packed ones
            # processed_trucks += 1 # If counting all trucks
            # packing_progress.progress(processed_trucks / n_trucks)
            continue # Lanjut ke truk berikutnya

        # Update status hanya untuk truk yang diproses
        packing_status.text(f"Optimizing packing for Truck {t} ({num_items_truck} items)... ({processed_trucks+1}/{total_trucks_to_pack})")

        best_penalty_final = float('inf')
        best_layout_final = None
        best_penalty_details_final = {} # Inisialisasi details terbaik
        # Hitung patience untuk packing final
        patience_final = max(10, int(FINAL_PACKING_MAX_ITERS * FINAL_PACKING_PATIENCE_FACTOR))

        # Lakukan beberapa percobaan packing untuk hasil yang lebih baik
        for attempt in range(FINAL_PACKING_ATTEMPTS):
            logging.info(f"Truck {t}: Starting final packing attempt {attempt + 1}/{FINAL_PACKING_ATTEMPTS}...")
            # Gunakan seed berbeda untuk tiap attempt agar eksplorasi beragam
            seed = RANDOM_SEED + (t * 100) + (attempt * 10)
            current_np_state = np.random.get_state()
            np.random.seed(seed)

            try:
                # Buat instance PackingPSO dengan parameter FINAL
                packer = PackingPSO(truck_dims_tuple, items_for_truck,
                                    num_particles=FINAL_PACKING_NUM_PARTICLES,
                                    compaction_weight=FINAL_PACKING_COMPACTION_WEIGHT,
                                    c1=C1_PACKING, c2=C2_PACKING)
                # Jalankan optimasi packing - tangkap ketiga nilai return
                layout, penalty, details = packer.optimize(max_iters=FINAL_PACKING_MAX_ITERS, patience=patience_final)
            finally:
                # Kembalikan state random generator
                np.random.set_state(current_np_state)

            logging.info(f"Truck {t} Attempt {attempt + 1}: Penalty = {penalty:.4f}")

            # Simpan hasil jika lebih baik dari attempt sebelumnya
            if penalty < best_penalty_final:
                best_penalty_final = penalty
                best_layout_final = layout
                best_penalty_details_final = details.copy() # Salin dictionary details terbaik
                logging.info(f"Truck {t}: New best final penalty found: {best_penalty_final:.4f}")
                # Log rincian penalti terbaik yang baru ditemukan
                logging.info(f"Truck {t}: Best Penalty Details: {best_penalty_details_final}")

            # Stop attempt jika sudah ditemukan solusi (hampir) sempurna
            if best_penalty_final < ZERO_TOLERANCE:
                logging.info(f"Truck {t}: Near-perfect packing found in attempt {attempt + 1}. Stopping attempts.")
                break # Keluar dari loop attempt

        # Simpan hasil terbaik (layout, penalty, details) untuk truk ini setelah semua attempt
        final_layouts[t] = (best_layout_final, best_penalty_final, best_penalty_details_final)
        logging.info(f"Truck {t} - Best Final Packing Penalty after {FINAL_PACKING_ATTEMPTS} attempts: {best_penalty_final:.4f}")
        # Log rincian penalti final terbaik lagi untuk konfirmasi
        if best_penalty_final > ZERO_TOLERANCE:
             logging.warning(f"Truck {t} - FINAL Best Packing Penalty Details: {best_penalty_details_final}")
        else:
             logging.info(f"Truck {t} - FINAL Best Packing Penalty Details: {best_penalty_details_final}")

        # Update progress setelah satu truk selesai dipacking
        processed_trucks += 1
        packing_progress.progress(processed_trucks / total_trucks_to_pack)


packing_status.text("Final packing optimizations complete.")
packing_progress.empty() # Hapus progress bar


# --- Fungsi Visualisasi Plotly 3D ---
def create_truck_figure(truck_dims, packed_items, truck_id):
    """Membuat figure Plotly 3D untuk visualisasi muatan truk."""
    L, W, H = truck_dims
    fig = go.Figure()

    # 1. Gambar Outline Truk (kerangka transparan)
    corners = [(0,0,0),(L,0,0),(L,W,0),(0,W,0),(0,0,H),(L,0,H),(L,W,H),(0,W,H)]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    edge_x, edge_y, edge_z = [], [], []
    for u, v in edges:
        edge_x.extend([corners[u][0], corners[v][0], None]) # None untuk putus garis
        edge_y.extend([corners[u][1], corners[v][1], None])
        edge_z.extend([corners[u][2], corners[v][2], None])
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines', line=dict(color='rgba(0,0,0,0.7)', width=2),
        hoverinfo='none', # Tidak perlu hover untuk outline
        showlegend=False, name='Truck Outline'
    ))

    # 2. Gambar Item yang Dikemas (jika ada)
    if packed_items:
        # Palet warna untuk membedakan item
        color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                         "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        for idx, item in enumerate(packed_items):
            # Ambil posisi dan dimensi terotasi dari layout
            x, y, z = item['x'], item['y'], item['z']
            w_rot, d_rot, h_rot = item['w'], item['d'], item['h'] # Dimensi setelah rotasi

            # Tentukan 8 titik sudut item yang diputar
            x_v = [x, x, x+w_rot, x+w_rot, x, x, x+w_rot, x+w_rot]
            y_v = [y, y+d_rot, y+d_rot, y, y, y+d_rot, y+d_rot, y]
            z_v = [z, z, z, z, z+h_rot, z+h_rot, z+h_rot, z+h_rot]

            # Tentukan sisi-sisi mesh (menggunakan indeks titik sudut)
            faces_i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 1]
            faces_j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 2]
            faces_k = [0, 7, 2, 3, 6, 7, 1, 5, 4, 5, 7, 6]

            # Pilih warna item dari palet
            item_color = color_palette[idx % len(color_palette)]

            # Tambahkan item sebagai Mesh3d
            fig.add_trace(go.Mesh3d(
                x=x_v, y=y_v, z=z_v, i=faces_i, j=faces_j, k=faces_k,
                color=item_color, opacity=0.85, # Sedikit transparan
                name=item.get('name', item.get('id', f'Item {idx}')), # Nama untuk legenda
                # Informasi yang muncul saat hover
                hovertext=(f"<b>{item.get('name', item['id'])}</b><br>"
                           f"ID: {item['id']}<br>"
                           f"Dims (Rotated WxDxH): {w_rot:.1f} x {d_rot:.1f} x {h_rot:.1f}<br>"
                           f"Pos (x,y,z): ({x:.1f}, {y:.1f}, {z:.1f})<br>"
                           f"Orientation Index: {item.get('orientation', 'N/A')}"),
                hoverinfo="text" # Tampilkan hovertext
            ))

    # 3. Konfigurasi Layout Scene 3D
    fig.update_layout(
        title=f"Visualisasi Muatan Truk {truck_id}",
        scene=dict(
            xaxis=dict(title='Panjang (X)', range=[0, L], backgroundcolor="rgb(240,240,240)", tickformat=".1f", nticks=8),
            yaxis=dict(title='Lebar (Y)', range=[0, W], backgroundcolor="rgb(235,235,235)", tickformat=".1f", nticks=8),
            zaxis=dict(title='Tinggi (Z)', range=[0, H], backgroundcolor="rgb(240,240,240)", tickformat=".1f", nticks=8),
            # Atur aspek rasio agar proporsional dengan dimensi truk
            aspectratio=dict(x=1, y=W/L, z=H/L) if L > 0 else dict(x=1, y=1, z=1),
            aspectmode='manual', # Gunakan aspek rasio yang ditentukan
            camera_eye=dict(x=1.8, y=1.8, z=0.9) # Posisi kamera awal
        ),
        margin=dict(l=10, r=10, t=40, b=10), # Margin plot
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5) # Posisi legenda
    )
    return fig

# --- Layout Aplikasi Streamlit ---
st.title("🚚 Optimasi Penugasan & Pemuatan Truk V3 (3 Rotasi)")
st.markdown("---")

# --- Baris 1: Metrik Ringkasan Global ---
col1, col2, col3, col4 = st.columns(4)

# Hitung metrik global dari final_truck_info
total_revenue_final = sum(info['revenue'] for info in final_truck_info.values())
total_cost_final = sum(info['fuel_cost'] for info in final_truck_info.values())
total_profit_final = total_revenue_final - total_cost_final # Profit = Revenue - Cost
num_assigned_items = sum(len(info['items']) for info in final_truck_info.values())
# Hitung ulang unassigned: Total - Assigned - Oversized (yang tidak terassign)
# Cek DataFrame untuk status assignment final
unassigned_count = assigned_items_df[assigned_items_df['assigned_truck'] == 'Unassigned'].shape[0]
oversized_unassigned_count = assigned_items_df[assigned_items_df['assigned_truck'] == 'Oversized (Unassigned)'].shape[0]
num_overweight = sum(1 for info in final_truck_info.values() if info['is_overweight'])
assignable_items_count = len(items) - oversized_count # Jumlah item yang BISA diassign


col1.metric("Total Estimasi Profit*", f"Rp {total_profit_final:,.0f}",
            help="Profit = Total Revenue (item*jarak*faktor) - Total Fuel Cost (rute*cost/km). Belum termasuk penalti packing/constraint lain dari proses optimasi.")
col2.metric("Total Revenue", f"Rp {total_revenue_final:,.0f}")
col3.metric("Total Fuel Cost", f"Rp {total_cost_final:,.0f}")
col4.metric("Item Ter-assign", f"{num_assigned_items} / {assignable_items_count}",
            help=f"{unassigned_count} tidak ter-assign (dari yg assignable), {oversized_count} total oversized ({oversized_unassigned_count} tidak terassign).")


# Peringatan jika ada truk overweight di solusi final
if num_overweight > 0:
    st.warning(f"🚨 **PERHATIAN:** {num_overweight} truk MELEBIHI BATAS BERAT! Solusi mungkin tidak optimal atau valid.")

st.markdown("---")

# --- Baris 2: Peta Rute & Grafik Fitness Assignment ---
col1_map, col2_chart = st.columns([3, 2]) # Peta lebih lebar

with col1_map:
    st.subheader("🗺️ Peta Rute Pengiriman")
    if routes_data_pydeck:
        # Data untuk marker kota
        city_points_data = [{"name": city, "coordinates": [coord[1], coord[0]]} # [lng, lat]
                            for city, coord in city_coords.items()]

        # Layer untuk marker kota (lingkaran)
        city_marker_layer = pdk.Layer(
            "ScatterplotLayer",
            data=city_points_data,
            get_position="coordinates",
            get_fill_color=[0, 0, 0, 180], # Warna hitam semi-transparan
            get_radius=8000, # Radius dalam meter
            radius_min_pixels=6,
            pickable=True, # Bisa di-hover
            auto_highlight=True # Highlight saat hover
        )
        # Layer untuk nama kota
        city_text_layer = pdk.Layer(
            "TextLayer",
            data=city_points_data,
            get_position="coordinates",
            get_text="name",
            get_color=[0, 0, 0, 200], # Hitam solid
            get_size=14, # Ukuran font
            get_alignment_baseline="'bottom'", # Posisi teks relatif thd koordinat
            get_pixel_offset=[0, -18] # Geser teks ke atas marker
        )

        # Layer untuk rute (garis) per truk
        path_layers = [
            pdk.Layer(
                "PathLayer",
                data=[route_data], # Perlu list berisi satu dict rute
                get_path="path", # Ambil koordinat dari key 'path'
                get_color="color", # Ambil warna dari key 'color'
                get_width=5, # Lebar garis dasar
                width_scale=1, # Skala lebar
                width_min_pixels=3.5, # Lebar minimal di layar
                pickable=True, # Bisa di-hover
                # --- PERUBAHAN MAP RESPONSIVE ---
                auto_highlight=True, # Highlight path saat hover
                # -------------------------------
                # Tooltip saat hover di path
                tooltip={"html": "<b>{truck}</b><br/>Rute: {route_info}<br/>Jarak: {distance_km:.1f} km"}
            ) for route_data in routes_data_pydeck
        ]

        # Tampilan awal peta
        initial_view_state = pdk.ViewState(
            latitude=city_coords[DEFAULT_START_CITY][0],
            longitude=city_coords[DEFAULT_START_CITY][1],
            zoom=5.8,
            pitch=45 # Sudut pandang
        )

        # Gabungkan semua layer
        all_layers = path_layers + [city_marker_layer, city_text_layer]

        # Tampilkan peta Pydeck
        st.pydeck_chart(pdk.Deck(
            layers=all_layers,
            initial_view_state=initial_view_state,
            map_style='mapbox://styles/mapbox/light-v10', # Gaya peta dasar
            # Styling tooltip kustom
            tooltip={"style": {"backgroundColor":"rgba(0,0,0,0.7)","color":"white","fontSize":"12px"}}
        ))
    else:
        st.info("Tidak ada rute pengiriman untuk ditampilkan (mungkin tidak ada item yang diassign atau rute tidak valid).")

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
            height=400, # Sesuaikan tinggi chart
            margin=dict(l=20, r=20, t=30, b=20),
            hovermode="x unified" # Info hover lebih baik
        )
        st.plotly_chart(fig_fit, use_container_width=True)
    else:
        st.info("Data histori fitness tidak tersedia (optimasi mungkin tidak berjalan atau tidak menghasilkan solusi).")

st.markdown("---")

# --- Baris 3: Detail Assignment Item ---
st.subheader("📦 Detail Penugasan Item")
if not assigned_items_df.empty:
     try:
         # Coba ekstrak nomor truk untuk sorting (handle teks non-numerik)
         # 'Truk 1' -> 1, 'Oversized...' -> NaN, 'Unassigned' -> NaN
         assigned_items_df['truck_num'] = assigned_items_df['assigned_truck'].str.extract(r'(\d+)').astype(float)
         # Sort: Truk 1, Truk 2, ..., Unassigned/Oversized (NaNs last)
         assigned_items_df = assigned_items_df.sort_values(by=['truck_num', 'id'], na_position='last').drop(columns=['truck_num'])
     except Exception as e:
         logging.warning(f"Could not sort items by truck number: {e}")
         # Fallback sort jika parsing gagal
         assigned_items_df = assigned_items_df.sort_values(by=['assigned_truck', 'id'])

     # Tampilkan DataFrame (pilih kolom yang relevan)
     st.dataframe(assigned_items_df[[
         'id', 'name', 'weight', 'dims', 'city', 'dim_category', 'assigned_truck'
     ]], use_container_width=True, hide_index=True) # hide_index agar lebih rapi
else:
     st.info("Tidak ada data assignment item untuk ditampilkan.")


st.markdown("---")

# --- Baris 4+: Detail Truk & Visualisasi Muatan ---
st.subheader("🚛 Detail Muatan & Visualisasi per Truk")

cols_per_row = 2 # Tampilkan 2 truk per baris
num_rows = (n_trucks + cols_per_row - 1) // cols_per_row # Hitung jumlah baris
truck_idx_display = 1 # Mulai dari Truk 1

for r in range(num_rows):
    cols = st.columns(cols_per_row)
    for c in range(cols_per_row):
        # Cek apakah masih ada truk untuk ditampilkan
        if truck_idx_display <= n_trucks:
            with cols[c]: # Gunakan kolom saat ini
                truck_id = truck_idx_display
                info = final_truck_info[truck_id]
                # Ambil hasil packing final (layout, penalty, details)
                layout, penalty, penalty_details = final_layouts.get(truck_id, (None, float('inf'), {}))

                st.markdown(f"#### Truk {truck_id}")

                # Tampilkan status berat
                if info["is_overweight"]:
                    st.error(f"⛔️ OVERWEIGHT: {info['weight']:.1f} / {truck_max_weight:.1f} kg")
                else:
                    sisa_kapasitas = truck_max_weight - info['weight']
                    # Tampilkan sisa kapasitas
                    delta_text = f"{sisa_kapasitas:.1f} kg Sisa" if sisa_kapasitas >= -ZERO_TOLERANCE else f"{abs(sisa_kapasitas):.1f} kg Lebih"
                    delta_color = "normal" if sisa_kapasitas >= -ZERO_TOLERANCE else "inverse"
                    st.metric("Berat Muatan", f"{info['weight']:.1f} / {truck_max_weight:.1f} kg",
                              delta=delta_text, delta_color=delta_color)

                # Tampilkan profit truk
                st.metric("Profit Truk Ini*", f"Rp {info.get('profit', 0):,.0f}",
                          help="Estimasi Revenue Truk - Estimasi Fuel Cost Truk (berdasarkan rute NN)")

                # Tampilkan detail jika truk tidak kosong
                if not info["items"]:
                    st.info("Truk ini tidak membawa item.")
                else:
                    # Tampilkan rute
                    route_str = " -> ".join(info.get('route_sequence', ['N/A']))
                    st.caption(f"**Rute:** {route_str} | **Jarak:** {info.get('route_distance', 0):.1f} km")

                    # Expander untuk daftar item di truk
                    with st.expander("Lihat daftar item di truk ini"):
                        items_in_truck_df = pd.DataFrame(info["items"])
                        st.dataframe(items_in_truck_df[["id", "name", "weight", "dims", "city"]], hide_index=True)

                    # Visualisasi Muatan 3D
                    st.markdown("**Visualisasi Muatan:**")
                    if layout is None and info["items"]:
                        # Kasus packing final gagal tapi ada item
                        st.error("Packing final gagal mendapatkan layout untuk truk ini.")
                    elif layout is None and not info["items"]:
                        # Kasus truk kosong (sudah ditangani di atas)
                        pass
                    elif penalty > ZERO_TOLERANCE and info["items"]:
                        # Kasus packing berhasil tapi ada penalti
                        st.warning(f"Packing mungkin belum optimal (Total Penalty: {penalty:.4f}).")
                        # Tampilkan rincian penalti jika ada
                        with st.expander("Lihat Rincian Penalti Packing"):
                            st.write("Komponen Penalti:")
                            if penalty_details: # Pastikan dictionary tidak kosong
                                displayed_penalties = False
                                for key, value in penalty_details.items():
                                    if value > ZERO_TOLERANCE: # Tampilkan hanya yang > 0
                                        st.code(f"- {key}: {value:,.4f}")
                                        displayed_penalties = True
                                if not displayed_penalties:
                                     st.caption("Semua komponen penalti mendekati nol.")
                            else:
                                st.caption("Tidak ada detail penalti tersedia.")
                        # Tampilkan visualisasi 3D
                        fig = create_truck_figure(truck_dims_tuple, layout, truck_id)
                        st.plotly_chart(fig, use_container_width=True)
                    elif info["items"]: # Kasus packing berhasil dan penalti mendekati nol
                        st.success(f"Muatan terpack dengan baik (Total Penalty: {penalty:.4f}).")
                        # Opsi: Tetap tampilkan rincian penalti (meskipun kecil)
                        with st.expander("Lihat Rincian Penalti Packing (Detail)"):
                             st.write("Komponen Penalti:")
                             if penalty_details:
                                 for key, value in penalty_details.items():
                                     st.code(f"- {key}: {value:,.4f}") # Tampilkan semua
                             else:
                                 st.caption("Tidak ada detail penalti tersedia.")
                        # Tampilkan visualisasi 3D
                        fig = create_truck_figure(truck_dims_tuple, layout, truck_id)
                        st.plotly_chart(fig, use_container_width=True)

                # Pemisah antar baris (tambahkan setelah setiap baris selesai)
                # Tidak perlu pemisah antar kolom, Streamlit menangani itu
                # if c == cols_per_row - 1 and truck_idx_display < n_trucks:
                #     st.markdown("---") # Pemisah antar baris (opsional)


            truck_idx_display += 1 # Lanjut ke truk berikutnya
    # Tambahkan pemisah di akhir jika perlu
    # st.markdown("---")


logging.info("Streamlit App Layout Generated.")
# print("\nStreamlit App Ready. Run with: streamlit run your_script_name.py")