import json
import pathlib
import pandas as pd
import logging
import traceback
import h5py

class DataLoader:
    def __init__(self, json_path):
        self.json_path = pathlib.Path(json_path)
        self.base_dir = self.json_path.parent 
        self.out_dir = self.base_dir / "analysis_plots"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging.
        self._setup_logger()
        
        # Resolve the summary path only within the directory provided to this loader.
        if "summary" in self.json_path.name.lower():
            self.active_summary_path = self.json_path
        else:
            summaries = list(self.base_dir.glob("*_summary.json"))
            self.active_summary_path = summaries[0] if summaries else None
        
        if self.active_summary_path:
            self.logger.info(f"--- DataLoader starting with active Summary: {self.active_summary_path.name} ---")
        else:
            self.logger.info(f"--- DataLoader starting with default raw data: {self.json_path.name} ---")
        
        self.df = pd.DataFrame()
        self.raw_hydrates = []
        self.positions = []
        
        self.sample_height_um = 1000.0
        self.global_density = 1.0
        self.global_med_area = 0.0
        self.pixel_scale = 1.41
        self.sample_name = "Unknown Sample"
        
        try:
            self._verify_master_h5()
            self._load_data()
            self._load_metadata()
            self.logger.info("DataLoader initialization completed successfully.")
        except Exception as e:
            self.logger.error(f"CRITICAL ERROR loading data: {e}")
            self.logger.error(traceback.format_exc())
            raise e

    def _setup_logger(self):
        self.logger = logging.getLogger("HydrateAnalyzer")
        self.logger.setLevel(logging.DEBUG)
        
        if not self.logger.handlers:
            log_path = self.base_dir / "Analyzer_debug.txt"
            fh = logging.FileHandler(log_path, mode='a')
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
            self.logger.addHandler(fh)
            
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(sh)

    def _verify_master_h5(self):
        self.logger.info("==================================================")
        self.logger.info("STARTING HDF5 DIAGNOSTICS")
        self.logger.info("==================================================")
        
        h5_files = list(self.base_dir.glob("*master*.h5"))
        if not h5_files:
            self.logger.error("NO master.h5 FILE FOUND in the directory!")
            return
            
        for h5_path in h5_files:
            self.logger.debug(f"Inspecting file: {h5_path.name}")
            try:
                with h5py.File(h5_path, 'r') as f:
                    if '2D_Grains' not in f:
                        self.logger.error("-> Group '2D_Grains' IS MISSING completely!")
            except Exception as e:
                self.logger.error(f"Failed to read {h5_path.name}: {e}")
                
        self.logger.info("==================================================")

    def _load_data(self):
        target_path = self.active_summary_path if self.active_summary_path and self.active_summary_path.exists() else self.json_path
        is_summary = "summary" in target_path.name.lower()
        
        self.logger.debug(f"Reading data from {target_path.name}...")
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if is_summary:
            self.raw_hydrates = data.get("5_hydrate_list", [])
            self.logger.info(f"Found {len(self.raw_hydrates)} FILTERED hydrates from Summary JSON.")
            
            macro_stats = data.get("1_hydrate_macro_statistics", {})
            self.global_density = float(macro_stats.get("global_density_per_mm3", 1.0))
            
            res_path = self.base_dir / "analysis_results.json"
            sample_meta = {}
            if res_path.exists():
                with open(res_path, 'r', encoding='utf-8') as res_f:
                    res_data = json.load(res_f)
                    sample_meta = res_data.get("sample_metadata", {})
        else:
            self.raw_hydrates = data.get("verified_hydrates", [])
            self.logger.info(f"Found {len(self.raw_hydrates)} raw hydrates in JSON.")
            
            sample_meta = data.get("sample_metadata", {})
            macro_stats = data.get("macro_statistics", {})
            self.global_density = float(macro_stats.get("hydrate_number_density_per_mm3", 1.0))
        
        if not self.raw_hydrates:
            self.logger.warning("No verified hydrates found! DataFrame will be empty.")
            
        self.sample_name = str(sample_meta.get("sample_name", self.base_dir.name))
        
        try:
            self.sample_height_um = float(sample_meta.get("total_sample_height_um", 1000.0))
            self.pixel_scale = float(sample_meta.get("spatial_resolution_um_per_px", 1.41))
        except Exception as e:
            self.logger.warning(f"Error parsing metadata numbers, using defaults. Error: {e}")
        
        self.logger.debug("Parsing hydrate dictionary into flat DataFrame...")
        flat_data = []
        for i, h in enumerate(self.raw_hydrates):
            try:
                coords = h.get("world_coordinates_um", {})
                ellipse = h.get("fitted_ellipse", {})
                
                flat_data.append({
                    "global_id": h.get("global_id", 0),
                    "area_um2": h.get("area_um2", 0),
                    "volume_um3": h.get("estimated_volume_um3", h.get("volume_um3", 0.0)),
                    "aspect_ratio": h.get("aspect_ratio", 1),
                    
                    "x_um": coords.get("x", 0.0),
                    "y_um": coords.get("y", 0.0),
                    "z_um": coords.get("z_abs", coords.get("z", 0.0)),
                    
                    "center_x_px": h.get("center_x_px_image", 0),
                    "center_y_px": h.get("center_y_px_image", 0),
                    
                    "angle_deg": ellipse.get("orientation_angle_deg", 0.0),
                    "major_axis_um": ellipse.get("major_axis_um", 0.0),
                    "minor_axis_um": ellipse.get("minor_axis_um", 0.0),
                    
                    "source_stack": h.get("source_stack", ""),
                    "layer": h.get("layer", 0),
                    "pipeline_path": h.get("pipeline_path", "")
                })
            except Exception as e:
                self.logger.error(f"Missing mandatory keys in hydrate index {i}. Error: {e}")
                
        self.df = pd.DataFrame(flat_data)
        self.global_med_area = self.df['area_um2'].median() if not self.df.empty else 0
        self.logger.debug(f"DataFrame built successfully. Shape: {self.df.shape}")

    def _load_metadata(self):
        self.logger.debug("Looking for metadata_preprocessed.json...")
        meta_path = self.base_dir / "metadata_preprocessed.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_data = json.load(f)
            self.positions = meta_data.get("Positions", [])
            self.logger.info(f"Metadata loaded. Found {len(self.positions)} stack positions.")
        else:
            self.logger.warning(f"metadata_preprocessed.json not found in {self.base_dir}. Grid interaction disabled.")
            self.positions = []