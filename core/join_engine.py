import json
import pathlib
import shutil
import numpy as np

class JoinEngine:
    def __init__(self, folders, output_folder, gap_mm, dataset_name):
        self.folders = [pathlib.Path(f) for f in folders]
        self.output_folder = pathlib.Path(output_folder)
        self.gap_um = gap_mm * 1000.0
        self.dataset_name = dataset_name
        
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
    def run(self):
        joined_hydrates = []
        joined_positions = []
        
        # Master statistics
        total_mask_area_mm2 = 0.0
        total_raw_contours = 0
        total_verified = 0
        
        # Shift variables (accumulated across iterations for the 2D layout)
        cumulative_max_x_um = 0.0
        cumulative_max_grid_x = -1
        target_center_y_um = None
        
        global_hydrate_id_counter = 1
        
        for idx, folder in enumerate(self.folders):
            print(f"Processing Folder {idx+1}/{len(self.folders)}: {folder.name}")
            
            # --- 1. Load JSON Files ---
            meta_path = folder / "metadata_preprocessed.json"
            res_path = folder / "analysis_results.json"
            
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            with open(res_path, "r", encoding="utf-8") as f:
                res = json.load(f)
                
            positions = meta.get("Positions", [])
            hydrates = res.get("verified_hydrates", [])
            
            if not positions: continue
            
            # Update master stats
            total_mask_area_mm2 += res.get("sample_metadata", {}).get("total_sample_area_mm2", 0.0)
            total_raw_contours += res.get("macro_statistics", {}).get("total_contours_raw", 0)
            
            # --- 2. Copy 3D Data & HDF5 Files (With Index Prefix for Correct Sorting) ---
            # By adding f"{idx:02d}_", we guarantee that Tab 2 and Tab 3 will process 
            # them in the exact order they were joined, ensuring the -10cm shift applies correctly.
            prefix = f"{idx:02d}_{folder.name}_"
            
            # 2a. Copy HDF5 Files
            h5_files = list(folder.glob("*.h5"))
            for h5_file in h5_files:
                dst_h5 = self.output_folder / f"{prefix}{h5_file.name}"
                print(f"Copying HDF5 file: {h5_file.name} -> {dst_h5.name}")
                shutil.copy2(h5_file, dst_h5)
                
            # 2b. Copy 3D Mesh Geometry JSONs
            mesh_files = list(folder.glob("*3d_mesh_geometry*.json"))
            for m_file in mesh_files:
                dst_m = self.output_folder / f"{prefix}{m_file.name}"
                print(f"Copying Mesh JSON: {m_file.name} -> {dst_m.name}")
                shutil.copy2(m_file, dst_m)
                
            # 2c. Copy 3D Hydrates JSONs
            hyd_files = list(folder.glob("*3d_hydrates*.json"))
            for h_file in hyd_files:
                dst_h = self.output_folder / f"{prefix}{h_file.name}"
                print(f"Copying 3D Hydrates JSON: {h_file.name} -> {dst_h.name}")
                shutil.copy2(h_file, dst_h)
            
            # --- 3. Calculate Folder Bounding Box (For 2D Join) ---
            xs = [p["positionX"] for p in positions]
            ys = [p["positionY"] for p in positions]
            
            min_x_um, max_x_um = min(xs), max(xs)
            min_y_um, max_y_um = min(ys), max(ys)
            
            # Estimate tile width (step size) to ensure max_x_um is accurate
            dx_list = [xs[i+1] - xs[i] for i in range(len(xs)-1) if xs[i+1] > xs[i]]
            step_x = np.median(dx_list) if dx_list else 1400.0
            
            min_grid_x = min([p["gridX"] for p in positions])
            current_center_y_um = (max_y_um + min_y_um) / 2.0
            
            # --- 4. Calculate Shifts (For 2D Layout) ---
            if idx == 0:
                shift_x_um = 0.0
                shift_y_um = 0.0
                shift_grid_x = 0
                target_center_y_um = current_center_y_um
            else:
                shift_x_um = cumulative_max_x_um + self.gap_um - min_x_um
                shift_y_um = target_center_y_um - current_center_y_um
                shift_grid_x = cumulative_max_grid_x + 1 - min_grid_x
            
            name_mapping = {}
            
            # --- 5. Update Metadata (Grid) ---
            for p in positions:
                old_grid_x = p["gridX"]
                old_grid_y = p["gridY"]
                
                new_grid_x = old_grid_x + shift_grid_x
                new_grid_y = old_grid_y # Y-grid remains unchanged as segments are joined horizontally
                
                # Shift world coordinates
                p["positionX"] += shift_x_um
                p["positionY"] += shift_y_um
                
                p["gridX"] = new_grid_x
                p["gridY"] = new_grid_y
                
                # Prepare filename mapping
                old_name = f"{folder.name}_{old_grid_x}_{old_grid_y}.tif"
                for tif_file in folder.glob(f"*_{old_grid_x}_{old_grid_y}.tif"):
                    old_name = tif_file.name
                    break
                
                new_name = f"Joined_{new_grid_x}_{new_grid_y}.tif"
                name_mapping[old_name] = new_name
                
                # Copy TIFF files
                src_path = folder / old_name
                dst_path = self.output_folder / new_name
                if src_path.exists():
                    shutil.copy2(src_path, dst_path)
                    
                joined_positions.append(p)
                
            # --- 6. Update Hydrates ---
            for hyd in hydrates:
                # Shift coordinates
                hyd["world_coordinates_um"]["x"] += shift_x_um
                hyd["world_coordinates_um"]["y"] += shift_y_um
                
                if "center_x_um_world" in hyd: hyd["center_x_um_world"] += shift_x_um
                if "center_y_um_world" in hyd: hyd["center_y_um_world"] += shift_y_um
                
                # Rename source stack references
                old_stack = hyd["source_stack"]
                if not old_stack.endswith(".tif"): old_stack += ".tif"
                
                if old_stack in name_mapping:
                    hyd["source_stack"] = name_mapping[old_stack].replace(".tif", "")
                else:
                    # Fallback rename
                    parts = hyd["source_stack"].split("_")
                    if len(parts) >= 2:
                        gx, gy = int(parts[-2]), int(parts[-1].split(".")[0])
                        hyd["source_stack"] = f"Joined_{gx + shift_grid_x}_{gy}"
                
                # Assign new continuous global ID to avoid conflicts
                hyd["global_id"] = global_hydrate_id_counter
                global_hydrate_id_counter += 1
                
                joined_hydrates.append(hyd)
                
            # --- 7. Update limits for the next iteration ---
            cumulative_max_x_um = max_x_um + shift_x_um + step_x
            cumulative_max_grid_x = max([p["gridX"] for p in positions]) + shift_grid_x

        print("Writing merged JSON files...")
        
        # --- 8. Save New JSONs (2D Unified Data) ---
        # Metadata
        meta_merged = meta.copy()
        meta_merged["Positions"] = joined_positions
        meta_merged["Analytics"]["total_mask_area_mm2"] = total_mask_area_mm2
        
        with open(self.output_folder / "metadata_preprocessed.json", "w", encoding="utf-8") as f:
            json.dump(meta_merged, f, indent=4)
            
        # Analysis Results
        res_merged = res.copy()
        res_merged["verified_hydrates"] = joined_hydrates
        res_merged["sample_metadata"]["sample_name"] = self.dataset_name
        res_merged["sample_metadata"]["total_sample_area_mm2"] = total_mask_area_mm2
        
        total_verified = len(joined_hydrates)
        res_merged["macro_statistics"]["total_contours_raw"] = total_raw_contours
        res_merged["macro_statistics"]["total_verified_hydrates"] = total_verified
        
        # Recalculate global density
        sample_height_mm = res_merged["sample_metadata"].get("total_sample_height_um", 1000.0) / 1000.0
        total_vol_mm3 = total_mask_area_mm2 * sample_height_mm
        res_merged["sample_metadata"]["total_sample_volume_mm3"] = total_vol_mm3
        res_merged["macro_statistics"]["hydrate_number_density_per_mm3"] = (total_verified / total_vol_mm3) if total_vol_mm3 > 0 else 0
        
        with open(self.output_folder / "analysis_results.json", "w", encoding="utf-8") as f:
            json.dump(res_merged, f, indent=4)
            
        print("Join successful!")