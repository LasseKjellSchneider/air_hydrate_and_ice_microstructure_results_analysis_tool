import os
import json
import pathlib
import numpy as np
import math
import traceback
from shapely.affinity import translate

# Fallback imports based on possible project structures
try:
    from core.reconstruction_3d_core import extract_vector_contours, extract_fa_data
except ImportError:
    from core_backend.data_merge.reconstruction_3d_core import extract_vector_contours, extract_fa_data

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, complex): return {"real": obj.real, "imag": obj.imag}
        return super(NumpyEncoder, self).default(obj)

class SummaryEngine:
    def __init__(self, base_dir):
        self.base_dir = pathlib.Path(base_dir)
        self.dataset_name = self.base_dir.name
        self.summary_path = None  # Set dynamically by run().

    def run(self, min_hydrate_area=0.0):
        # Generate the output filename based on the active filter.
        if min_hydrate_area > 0:
            out_name = f"{self.dataset_name}_summary_min_area_{int(min_hydrate_area)}.json"
        else:
            out_name = f"{self.dataset_name}_summary.json"
            
        self.summary_path = self.base_dir / out_name
        
        print(f"\n[SummaryEngine] Starting global aggregation for {self.dataset_name}...")
        
        # Load global metadata.
        sample_height_um = 6000.0
        total_sample_area_mm2 = 1.0
        res_path = self.base_dir / "analysis_results.json"
        if res_path.exists():
            with open(res_path, 'r', encoding='utf-8') as f:
                res_data = json.load(f)
                sample_height_um = res_data.get("sample_metadata", {}).get("total_sample_height_um", 6000.0)
                total_sample_area_mm2 = res_data.get("sample_metadata", {}).get("total_sample_area_mm2", 1.0)

        # Initialize storage for aggregated data.
        all_2d_grain_areas = []
        all_2d_grain_centroids = [] 
        all_3d_grain_volumes = []
        all_fa_polys_by_z = {}
        all_hydrates = []

        # Iterate over input files and apply the dataset-specific X shift.
        h5_files = sorted(list(self.base_dir.glob("*master*.h5")))
        for h5_file in h5_files:
            idx = self._extract_idx(h5_file.name)
            x_shift = -100000.0 * idx # -10 cm in um
            
            # A) 2D Grains
            try:
                extraction_result = extract_vector_contours(str(h5_file), mode='interior')
                # Use the first tuple element, which contains the contours by plane.
                contours_by_z = extraction_result[0] if isinstance(extraction_result, tuple) else extraction_result
                
                for z, cnts in contours_by_z.items():
                    for cnt in cnts:
                        x, y = cnt[:, 0] + x_shift, cnt[:, 1]
                        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                        cx, cy = np.mean(x), np.mean(y)
                        all_2d_grain_areas.append(area)
                        all_2d_grain_centroids.append([cx, cy])
                print(f"[SummaryEngine] Successfully loaded {len(all_2d_grain_areas)} 2D grains from {h5_file.name}")
            except Exception as e: 
                print(f"\n[CRITICAL ERROR] Failed to load 2D Grains from {h5_file.name}!")
                print(f"Error Message: {e}")
                traceback.print_exc()
                print("--------------------------------------------------\n")
            
            # B) FA Data
            try:
                fa_data = extract_fa_data(str(h5_file))
                for z, items in fa_data.items():
                    if z not in all_fa_polys_by_z: all_fa_polys_by_z[z] = []
                    for item in items:
                        item['poly'] = translate(item['poly'], xoff=x_shift)
                        all_fa_polys_by_z[z].append(item)
            except Exception as e: 
                print(f"[ERROR] Failed to load FA Data from {h5_file.name}: {e}")

        # C) 3D Meshes
        mesh_files = sorted(list(self.base_dir.glob("*3d_mesh_geometry*.json")))
        for m_file in mesh_files:
            try:
                with open(m_file, 'r', encoding='utf-8') as f:
                    meshes = json.load(f)
                    for m in meshes:
                        meta = m.get('meta', {})
                        if meta.get('border_touch_count', 0) == 0: 
                            vol = meta.get('volume_um3', 0.0)
                            if vol > 0: all_3d_grain_volumes.append(vol)
            except Exception as e: 
                print(f"[ERROR] Failed to load 3D Meshes from {m_file.name}: {e}")

        # D) Hydrates
        hyd_files = sorted(list(self.base_dir.glob("*3d_hydrates*.json")))
        for h_file in hyd_files:
            idx = self._extract_idx(h_file.name)
            x_shift = -100000.0 * idx
            try:
                with open(h_file, 'r', encoding='utf-8') as f:
                    h_data = json.load(f)
                    for h in h_data.get('hydrates', []):
                        
                        # Apply the minimum hydrate-area filter.
                        if h.get('area_um2', 0.0) < min_hydrate_area:
                            continue
                        
                        h['world_coordinates_um']['x'] += x_shift
                        all_hydrates.append(h)
            except Exception as e: 
                print(f"[ERROR] Failed to load Hydrates from {h_file.name}: {e}")

        # --- 4. Compute Grain Statistics ---
        grain_stats = {
            "2d_interior_grains": {
                "count": len(all_2d_grain_areas),
                "median_area_um2": float(np.median(all_2d_grain_areas)) if all_2d_grain_areas else 0.0,
                "sigma_area_um2": float(np.std(all_2d_grain_areas)) if all_2d_grain_areas else 0.0
            },
            "3d_interior_grains": {
                "count": len(all_3d_grain_volumes),
                "median_volume_um3": float(np.median(all_3d_grain_volumes)) if all_3d_grain_volumes else 0.0,
                "sigma_volume_um3": float(np.std(all_3d_grain_volumes)) if all_3d_grain_volumes else 0.0
            }
        }

        # --- 5. Compute C-Axis Tensors ---
        c_axis_stats = {
            "global_tensor": self._compute_tensor_stats([item for sublist in all_fa_polys_by_z.values() for item in sublist]),
            "per_plane_tensors": {}
        }
        for z_level, fa_list in all_fa_polys_by_z.items():
            c_axis_stats["per_plane_tensors"][str(z_level)] = self._compute_tensor_stats(fa_list)

        # --- 6. Compute Hydrate Relations (Local Spatial Search) ---
        global_median_area = grain_stats["2d_interior_grains"]["median_area_um2"]
        global_density_mm3 = len(all_hydrates) / (total_sample_area_mm2 * (sample_height_um / 1000.0)) if total_sample_area_mm2 > 0 else 0.0
        
        window_radius_um = 750.0 
        window_vol_mm3 = (1.5 ** 2) * (sample_height_um / 1000.0)

        hx_arr = np.array([h['world_coordinates_um']['x'] for h in all_hydrates])
        hy_arr = np.array([h['world_coordinates_um']['y'] for h in all_hydrates])
        
        gx_arr = np.array([c[0] for c in all_2d_grain_centroids]) if all_2d_grain_centroids else np.array([])
        gy_arr = np.array([c[1] for c in all_2d_grain_centroids]) if all_2d_grain_centroids else np.array([])
        garea_arr = np.array(all_2d_grain_areas)

        delta_thetas, relation_scores, vols, aspects = [], [], [], []

        print(f"[SummaryEngine] Injecting local features into {len(all_hydrates)} filtered hydrates...")
        for i, h in enumerate(all_hydrates):
            if h.get('assignment_type') == 'nearest_fallback': continue
            if len(h.get('assigned_grain_ids', [])) > 1: continue

            vol = h.get('estimated_volume_um3', 0.0)
            ar = h.get('aspect_ratio', 1.0)
            if vol > 0: vols.append(vol)
            aspects.append(ar)

            theta_h = h.get('fitted_ellipse', {}).get('orientation_angle_deg', 0.0) % 180.0
            c_matches = h.get('c_axis_matches', [])
            
            dt_min = None
            c_vec_prim = None
            if c_matches:
                best_c = max(c_matches, key=lambda m: m.get('intersection_area_um2', 0))['c_axis_vector']
                c_vec_prim = best_c
                theta_c = math.degrees(math.atan2(best_c['x'], best_c['y'])) % 180.0
                diff = abs(theta_h - theta_c)
                dt_min = diff if diff <= 90 else 180 - diff
                delta_thetas.append(dt_min)
            
            hx, hy = hx_arr[i], hy_arr[i]
            dist_hyds = np.hypot(hx_arr - hx, hy_arr - hy)
            local_density = np.sum(dist_hyds <= window_radius_um) / window_vol_mm3 if window_vol_mm3 > 0 else 0
            
            local_area = global_median_area
            if gx_arr.size > 0:
                dist_grains = np.hypot(gx_arr - hx, gy_arr - hy)
                local_grain_areas = garea_arr[dist_grains <= window_radius_um]
                if local_grain_areas.size > 0:
                    local_area = np.median(local_grain_areas)

            r_score = None
            if local_density > 0 and global_median_area > 0:
                R = (local_area / global_median_area) * (global_density_mm3 / local_density)
                if R > 1: r_score = R - 1.0
                elif R > 0: r_score = 1.0 - (1.0 / R)
            
            if r_score is not None: relation_scores.append(r_score)

            h['computed_features'] = {
                "delta_theta_min_deg": dt_min,
                "c_axis_primary_vector": c_vec_prim,
                "local_density_per_mm3": local_density,
                "local_median_grain_area_um2": local_area,
                "relation_score": r_score
            }

        # --- 7. Compile Macro Statistics ---
        macro_stats = {
            "total_hydrates": len(all_hydrates),
            "global_density_per_mm3": global_density_mm3,
            "median_volume_um3": float(np.median(vols)) if vols else 0.0,
            "sigma_volume_um3": float(np.std(vols)) if vols else 0.0,
            "median_aspect_ratio": float(np.median(aspects)) if aspects else 1.0,
            "sigma_aspect_ratio": float(np.std(aspects)) if aspects else 0.0,
            "median_delta_theta_min_deg": float(np.median(delta_thetas)) if delta_thetas else None,
            "sigma_delta_theta_min_deg": float(np.std(delta_thetas)) if delta_thetas else None,
            "median_relation_score": float(np.median(relation_scores)) if relation_scores else None,
            "sigma_relation_score": float(np.std(relation_scores)) if relation_scores else None,
        }

        # --- 8. Compile Statistical Tests (AR >= 4/3) ---
        stat_theta_h, stat_theta_c = [], []
        for h in all_hydrates:
            if h.get('assignment_type') == 'nearest_fallback': continue
            if len(h.get('assigned_grain_ids', [])) > 1: continue

            if h.get('aspect_ratio', 1.0) >= (4.0 / 3.0):
                feat = h.get('computed_features', {})
                if feat.get('c_axis_primary_vector'):
                    vec = feat['c_axis_primary_vector']
                    tc = math.degrees(math.atan2(vec['x'], vec['y'])) % 180.0
                    th = h.get('fitted_ellipse', {}).get('orientation_angle_deg', 0.0) % 180.0
                    stat_theta_h.append(th)
                    stat_theta_c.append(tc)
        
        stat_tests = {
            "aspect_ratio_threshold": 1.33333333,
            "n_hydrates_used": len(stat_theta_h),
            "permutation_test": None,
            "circular_correlation": None
        }

        if len(stat_theta_h) >= 10:
            print("[SummaryEngine] Running 100,000 Iterations for Permutation Test...")
            a_arr = np.array(stat_theta_h)
            c_arr = np.array(stat_theta_c)
            
            true_diffs = (a_arr - c_arr) % 180
            true_mu = float(np.mean(true_diffs))
            true_sigma = float(np.std(true_diffs))

            n_permutations = 100000
            sims_sigma = np.zeros(n_permutations)
            shuff = np.copy(c_arr)
            
            for i in range(n_permutations):
                np.random.shuffle(shuff)
                sim_diffs = (a_arr - shuff) % 180
                sims_sigma[i] = np.std(sim_diffs)
                
            p_val = float(np.sum(sims_sigma <= true_sigma) / n_permutations)
                
            stat_tests["permutation_test"] = {
                "iterations": n_permutations,
                "true_mu": true_mu,
                "true_sigma": true_sigma,
                "random_sigmas_mu": float(np.mean(sims_sigma)),
                "random_sigmas_sigma": float(np.std(sims_sigma)),
                "p_value": p_val
            }

            alpha2_rad = np.radians((a_arr * 2) % 360)
            beta2_rad = np.radians((c_arr * 2) % 360)
            X_a, Y_a = np.mean(np.cos(alpha2_rad)), np.mean(np.sin(alpha2_rad))
            X_b, Y_b = np.mean(np.cos(beta2_rad)), np.mean(np.sin(beta2_rad))
            mean_a_rad = np.arctan2(Y_a, X_a)
            mean_b_rad = np.arctan2(Y_b, X_b)
            R_a = float(np.sqrt(X_a**2 + Y_a**2))
            R_b = float(np.sqrt(X_b**2 + Y_b**2))
            sin_diff_a = np.sin(alpha2_rad - mean_a_rad)
            sin_diff_b = np.sin(beta2_rad - mean_b_rad)
            zaehler = np.sum(sin_diff_a * sin_diff_b)
            nenner = np.sqrt(np.sum(sin_diff_a**2) * np.sum(sin_diff_b**2))
            r_circ = float(zaehler / nenner) if nenner > 0 else 0.0

            stat_tests["circular_correlation"] = {
                "r_circ": r_circ,
                "R_theta_h": R_a,
                "R_theta_c": R_b
            }

        # --- 9. Save Summary JSON ---
        summary_dict = {
            "1_hydrate_macro_statistics": macro_stats,
            "2_grain_statistics": grain_stats,
            "3_c_axis_statistics": c_axis_stats,
            "4_statistical_tests": stat_tests,
            "5_hydrate_list": all_hydrates
        }

        with open(self.summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_dict, f, cls=NumpyEncoder, indent=4)
            
        print(f"[SummaryEngine] Done! Summary saved to {self.summary_path.name}")
        
        # Return the path to the generated summary.
        return self.summary_path

    def _extract_idx(self, filename):
        parts = filename.split("_")
        if parts[0].isdigit(): return int(parts[0])
        return 0

    def _compute_tensor_stats(self, fa_list):
        if not fa_list: return None
        tensor = np.zeros((3, 3))
        total_weight = 0.0
        
        for item in fa_list:
            w = item['poly'].area
            c = item.get('c_axis')
            if not c: continue
            
            vec = np.array([c['x'], c['y'], c['z']])
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
                tensor += w * np.outer(vec, vec)
                total_weight += w
                
        if total_weight == 0: return None
        tensor /= total_weight
        evals, evecs = np.linalg.eigh(tensor)
        
        idx = evals.argsort()[::-1]
        evals, evecs = evals[idx], evecs[:, idx]
        
        primary = evecs[:, 0]
        if primary[0] < 0: primary = -primary
            
        return {
            "primary_vector": {"x": primary[0], "y": primary[1], "z": primary[2]},
            "eigenvalues": {"e1": evals[0], "e2": evals[1], "e3": evals[2]},
            "strength_e1_ratio": float(evals[0] / np.sum(evals)) if np.sum(evals) > 0 else 0.0
        }