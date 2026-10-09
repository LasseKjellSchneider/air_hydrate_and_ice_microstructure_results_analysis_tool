import customtkinter as ctk
import os
import json
import numpy as np
import math
import random
import pathlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

class CAxisTab(ctk.CTkFrame):
    def __init__(self, parent, base_dir):
        super().__init__(parent, fg_color="transparent")
        self.pack(fill="both", expand=True)
        
        self.base_dir = pathlib.Path(base_dir)
        
        self.data = None
        self.hydrates = []
        self.grains = []
        self.file_name = self.base_dir.name
        
        self.current_mode = "All" 
        self.single_grain_idx = 0
        self.current_grain_id = None

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        self.setup_sidebar()
        self.setup_plot_area()
        
        self.after(100, self.load_workspace_data)

    def setup_sidebar(self):
        sidebar = ctk.CTkScrollableFrame(self, width=350, fg_color="#1a1a1a")
        sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(sidebar, text="C-Axis & Hydrate Analysis", font=("Arial", 20, "bold")).pack(pady=(10, 10))
        
        # --- UPDATE BUTTON ---
        self.btn_update = ctk.CTkButton(sidebar, text="Update Plots", font=("Arial", 16, "bold"), 
                                        fg_color="#2ECC71", hover_color="#27AE60", height=45, 
                                        command=self.update_plots)
        self.btn_update.pack(fill="x", padx=10, pady=(0, 15))
        
        self.lbl_stats = ctk.CTkLabel(sidebar, text="Loading data...", text_color="gray", justify="left")
        self.lbl_stats.pack(pady=5, fill="x", padx=10)
        
        # --- 1. C-AXIS CALCULATION METHOD ---
        ctk.CTkLabel(sidebar, text="1. C-Axis Merging Logic", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=(20, 5))
        self.var_c_axis_mode = ctk.StringVar(value="dominant") # Default: Dominant
        ctk.CTkRadioButton(sidebar, text="Option 1: Area-Weighted Tensor", variable=self.var_c_axis_mode, value="weighted").pack(anchor="w", padx=10, pady=2)
        ctk.CTkRadioButton(sidebar, text="Option 2: Dominant (Max Area)", variable=self.var_c_axis_mode, value="dominant").pack(anchor="w", padx=10, pady=2)
        
        # --- 2. C-AXIS LATITUDE FILTER ---
        ctk.CTkLabel(sidebar, text="2. C-Axis Latitude Filter", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=(20, 5))
        self.lbl_min_dip = ctk.CTkLabel(sidebar, text="Min Latitude (0° = horizontal): 0°", anchor="w")
        self.lbl_min_dip.pack(padx=20, pady=(5, 0), anchor="w")
        self.slider_min_dip = ctk.CTkSlider(sidebar, from_=0, to=90, number_of_steps=90, command=self.on_min_dip_change)
        self.slider_min_dip.set(0)
        self.slider_min_dip.pack(fill="x", padx=20)
        
        self.lbl_max_dip = ctk.CTkLabel(sidebar, text="Max Latitude (90° = vertical): 90°", anchor="w")
        self.lbl_max_dip.pack(padx=20, pady=(5, 0), anchor="w")
        self.slider_max_dip = ctk.CTkSlider(sidebar, from_=0, to=90, number_of_steps=90, command=self.on_max_dip_change)
        self.slider_max_dip.set(90)
        self.slider_max_dip.pack(fill="x", padx=20)

        # --- 3. HYDRATE SHAPE & ANGLE SLIDERS ---
        ctk.CTkLabel(sidebar, text="3. Hydrate Geometry Filters", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=(20, 5))
        
        # DEFAULT 4/3 Aspect Ratio -> 1.33
        self.lbl_min_ar = ctk.CTkLabel(sidebar, text="Min Aspect Ratio (a/b): 1.33", anchor="w")
        self.lbl_min_ar.pack(padx=20, pady=(5, 0), anchor="w")
        self.slider_min_ar = ctk.CTkSlider(sidebar, from_=1.0, to=5.0, number_of_steps=400, command=self.on_min_ar_change)
        self.slider_min_ar.set(1.33)
        self.slider_min_ar.pack(fill="x", padx=20)

        self.lbl_min_theta_h = ctk.CTkLabel(sidebar, text="Hydrate Angle θ Filter: 0° to 180°", anchor="w")
        self.lbl_min_theta_h.pack(padx=20, pady=(5, 0), anchor="w")
        self.slider_min_theta_h = ctk.CTkSlider(sidebar, from_=0, to=90, number_of_steps=90, command=self.on_min_theta_h_change)
        self.slider_min_theta_h.set(0)
        self.slider_min_theta_h.pack(fill="x", padx=20)

        # --- 4. CHECKBOX FILTERS ---
        ctk.CTkLabel(sidebar, text="4. Data Filters", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=(20, 5))
        self.var_incl_fallback = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(sidebar, text="Include Fallback (Unassigned) Hydrates", variable=self.var_incl_fallback).pack(anchor="w", padx=20, pady=5)
        
        self.var_incl_multi = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(sidebar, text="Include Multi-Grain (Border) Hydrates", variable=self.var_incl_multi).pack(anchor="w", padx=20, pady=5)
        
        # --- 5. PERMUTATION TEST SETTINGS ---
        ctk.CTkLabel(sidebar, text="5. Permutation Test Settings", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=(20, 5))
        self.lbl_perm_iter = ctk.CTkLabel(sidebar, text="Iterations: 100000", anchor="w")
        self.lbl_perm_iter.pack(padx=20, pady=(5, 0), anchor="w")
        self.slider_perm_iter = ctk.CTkSlider(sidebar, from_=1000, to=200000, number_of_steps=199, command=self.on_perm_iter_change)
        self.slider_perm_iter.set(100000)
        self.slider_perm_iter.pack(fill="x", padx=20)

        # --- 6. SCOPE ---
        ctk.CTkLabel(sidebar, text="6. Visualization Scope", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=(20, 5))
        self.seg_scope = ctk.CTkSegmentedButton(sidebar, values=["All Grains", "Single Grain"], command=self.change_scope)
        self.seg_scope.set("All Grains")
        self.seg_scope.pack(fill="x", padx=10, pady=5)
        
        self.frame_single_grain = ctk.CTkFrame(sidebar, fg_color="transparent")
        self.btn_largest = ctk.CTkButton(self.frame_single_grain, text="Show Largest", command=lambda: self.pick_single_grain("largest"))
        self.btn_largest.pack(fill="x", padx=10, pady=2)
        self.btn_next = ctk.CTkButton(self.frame_single_grain, text="Show Next Largest", command=lambda: self.pick_single_grain("next"))
        self.btn_next.pack(fill="x", padx=10, pady=2)
        self.btn_random = ctk.CTkButton(self.frame_single_grain, text="Show Random", command=lambda: self.pick_single_grain("random"))
        self.btn_random.pack(fill="x", padx=10, pady=2)

    def setup_plot_area(self):
        self.right_panel = ctk.CTkFrame(self, fg_color="transparent")
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        # Create the compact header canvas.
        self.fig_header = Figure(figsize=(12, 1.2), dpi=100)
        self.fig_header.patch.set_facecolor('#2b2b2b')
        self.canvas_header = FigureCanvasTkAgg(self.fig_header, master=self.right_panel)
        self.canvas_header.get_tk_widget().pack(fill="x", pady=(0, 10))
        
        # 2. TABVIEW
        self.sub_tabs = ctk.CTkTabview(self.right_panel)
        self.sub_tabs.pack(fill="both", expand=True)
        
        tab_overview = self.sub_tabs.add("Overview")
        tab_perm = self.sub_tabs.add("Permutation Test")
        tab_circ = self.sub_tabs.add("Circular Correlation")
        
        # Build Overview Tab
        self.fig_overview = Figure(figsize=(12, 5.5), dpi=100)
        self.fig_overview.patch.set_facecolor('#2b2b2b')
        frame_overview = ctk.CTkFrame(tab_overview, fg_color="transparent")
        frame_overview.pack(fill="both", expand=True)
        self.canvas_overview = FigureCanvasTkAgg(self.fig_overview, master=frame_overview)
        self.canvas_overview.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas_overview, frame_overview).pack(side="bottom", fill="x")
        
        # Build Permutation Tab
        self.fig_perm = Figure(figsize=(12, 5.5), dpi=100)
        self.fig_perm.patch.set_facecolor('#2b2b2b')
        frame_perm = ctk.CTkFrame(tab_perm, fg_color="transparent")
        frame_perm.pack(fill="both", expand=True)
        self.canvas_perm = FigureCanvasTkAgg(self.fig_perm, master=frame_perm)
        self.canvas_perm.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas_perm, frame_perm).pack(side="bottom", fill="x")
        
        # Build Circular Tab
        self.fig_circ = Figure(figsize=(12, 5.5), dpi=100)
        self.fig_circ.patch.set_facecolor('#2b2b2b')
        frame_circ = ctk.CTkFrame(tab_circ, fg_color="transparent")
        frame_circ.pack(fill="both", expand=True)
        self.canvas_circ = FigureCanvasTkAgg(self.fig_circ, master=frame_circ)
        self.canvas_circ.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas_circ, frame_circ).pack(side="bottom", fill="x")

        self._draw_header_placeholders()

    def _draw_header_placeholders(self):
        self.fig_header.clf()
        ax = self.fig_header.add_subplot(111)
        ax.set_facecolor('#2b2b2b')
        ax.axis('off')
        ax.text(0.5, 0.5, "Load Data to visualize...", color="gray", fontsize=16, ha="center", va="center")
        self.canvas_header.draw()

    # --- DATA LOADING & AGGREGATION ---
    def load_workspace_data(self):
        self.hydrates = []
        self.grains = []
        
        json_files = sorted(list(self.base_dir.glob("*3d_hydrates*.json")))
        if not json_files:
            self.lbl_stats.configure(text="No '*3d_hydrates*.json' files found.", text_color="red")
            return
            
        for idx, fpath in enumerate(json_files):
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            prefix = f"{idx}_" 
            for g in data.get('grains', []):
                g['grain_id'] = prefix + str(g['grain_id'])
                self.grains.append(g)
            for h in data.get('hydrates', []):
                h['assigned_grain_ids'] = [prefix + str(gid) for gid in h.get('assigned_grain_ids', [])]
                self.hydrates.append(h)
                
        self.grains.sort(key=lambda x: x.get('total_volume_um3', 0), reverse=True)
        
        self.lbl_stats.configure(text=f"Loaded: {len(self.hydrates)} Hydrates\nLoaded: {len(self.grains)} Grains\n(From {len(json_files)} datasets)", text_color="white")
        self.single_grain_idx = 0
        self.current_grain_id = None
        self.seg_scope.set("All Grains")
        self.change_scope("All Grains")
        # Trigger initial plot
        self.update_plots()

    # --- CALLBACKS FOR SLIDERS (Only update labels, do not plot!) ---
    def on_min_dip_change(self, val):
        if val > self.slider_max_dip.get():
            self.slider_max_dip.set(val)
            self.lbl_max_dip.configure(text=f"Max Latitude (90° = vertical): {int(val)}°")
        self.lbl_min_dip.configure(text=f"Min Latitude (0° = horizontal): {int(val)}°")
        
    def on_max_dip_change(self, val):
        if val < self.slider_min_dip.get():
            self.slider_min_dip.set(val)
            self.lbl_min_dip.configure(text=f"Min Latitude (0° = horizontal): {int(val)}°")
        self.lbl_max_dip.configure(text=f"Max Latitude (90° = vertical): {int(val)}°")

    def on_min_ar_change(self, val):
        self.lbl_min_ar.configure(text=f"Min Aspect Ratio (a/b): {val:.2f}")

    def on_min_theta_h_change(self, val):
        val_int = int(val)
        self.lbl_min_theta_h.configure(text=f"Hydrate Angle θ Filter: {val_int}° to {180 - val_int}°")

    def on_perm_iter_change(self, val):
        self.lbl_perm_iter.configure(text=f"Iterations: {int(val)}")

    def change_scope(self, value):
        self.current_mode = value
        if value == "Single Grain":
            self.frame_single_grain.pack(fill="x", padx=10, pady=5)
            if not self.current_grain_id: self.pick_single_grain("largest")
        else:
            self.frame_single_grain.pack_forget()
            self.current_grain_id = None

    def compute_c_axis(self, matches):
        if not matches: return None
        if self.var_c_axis_mode.get() == "dominant":
            return max(matches, key=lambda m: m.get('intersection_area_um2', 0))['c_axis_vector']
        else:
            tensor = np.zeros((3, 3))
            total_weight = 0.0
            for m in matches:
                w = m.get('intersection_area_um2', 0)
                c = m['c_axis_vector']
                vec = np.array([c['x'], c['y'], c['z']])
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                    tensor += w * np.outer(vec, vec)
                    total_weight += w
            if total_weight == 0: return None
            eigenvalues, eigenvectors = np.linalg.eigh(tensor)
            best_axis = eigenvectors[:, np.argmax(eigenvalues)]
            return {'x': best_axis[0], 'y': best_axis[1], 'z': best_axis[2]}

    def get_filtered_grains(self):
        min_dip = self.slider_min_dip.get()
        max_dip = self.slider_max_dip.get()
        filtered_ids = []
        for g in self.grains:
            matches = g.get('c_axis_matches', [])
            if not matches: continue
            c_axis = self.compute_c_axis(matches)
            if not c_axis: continue
            dip = math.degrees(math.asin(abs(c_axis['x'])))
            if min_dip <= dip <= max_dip:
                filtered_ids.append(g['grain_id'])
        return filtered_ids

    def pick_single_grain(self, mode):
        filtered_ids = self.get_filtered_grains()
        if not filtered_ids:
            self.current_grain_id = None
            return
            
        if mode == "largest":
            self.single_grain_idx = 0
            self.current_grain_id = filtered_ids[self.single_grain_idx]
        elif mode == "next":
            if self.current_grain_id in filtered_ids:
                self.single_grain_idx = filtered_ids.index(self.current_grain_id)
            self.single_grain_idx = (self.single_grain_idx + 1) % len(filtered_ids)
            self.current_grain_id = filtered_ids[self.single_grain_idx]
        elif mode == "random":
            self.current_grain_id = random.choice(filtered_ids)
            self.single_grain_idx = filtered_ids.index(self.current_grain_id)
            
        # Refresh the plots when navigating between individual grains.
        self.update_plots()

    def process_data(self):
        angles_delta, angles_hydrates, angles_c_axes = [], [], []
        latitudes, c_axes_raw = [], []
        grain_aggregation = {} 
        
        min_dip = self.slider_min_dip.get()
        max_dip = self.slider_max_dip.get()
        min_ar = self.slider_min_ar.get()
        min_theta_h = self.slider_min_theta_h.get()
        incl_fallback = self.var_incl_fallback.get()
        incl_multi = self.var_incl_multi.get()
        active_grain_ids = set()
        
        for h in self.hydrates:
            if self.current_mode == "Single Grain" and self.current_grain_id:
                if self.current_grain_id not in h.get('assigned_grain_ids', []): continue
            if not incl_fallback and h.get('assignment_type') == 'nearest_fallback': continue
            if not incl_multi and len(h.get('assigned_grain_ids', [])) > 1: continue
            
            if h.get('aspect_ratio', 1.0) < min_ar: continue
            
            theta_h = h.get('fitted_ellipse', {}).get('orientation_angle_deg', 0.0) % 180.0
            if not (min_theta_h <= theta_h <= 180 - min_theta_h): continue
            
            c_axis = self.compute_c_axis(h.get('c_axis_matches', []))
            if not c_axis: continue
            lat = math.degrees(math.asin(abs(c_axis['x'])))
            if not (min_dip <= lat <= max_dip): continue
            
            theta_c = math.degrees(math.atan2(c_axis['x'], c_axis['y'])) % 180.0
            diff = abs(theta_h - theta_c)
            if diff > 90: diff = 180 - diff
                
            angles_delta.append(diff)
            angles_hydrates.append(theta_h)
            angles_c_axes.append(theta_c)
            latitudes.append(lat)
            
            if c_axis['x'] < 0: c_axes_raw.append([-c_axis['x'], -c_axis['y'], -c_axis['z']])
            else: c_axes_raw.append([c_axis['x'], c_axis['y'], c_axis['z']])
            
            for gid in h.get('assigned_grain_ids', []):
                if self.current_mode == "Single Grain" and gid != self.current_grain_id: continue
                active_grain_ids.add(gid)
                if gid not in grain_aggregation: grain_aggregation[gid] = {'th': [], 'tc': []}
                grain_aggregation[gid]['th'].append(theta_h)
                grain_aggregation[gid]['tc'].append(theta_c)
                
        h_per_grain = [len(vals['th']) for vals in grain_aggregation.values()]
        med_siblings = np.median(h_per_grain) if h_per_grain else 0
            
        return angles_delta, angles_hydrates, angles_c_axes, latitudes, len(active_grain_ids), med_siblings, c_axes_raw

    # --- PLOTTING LOGIC FOR TABS ---

    def plot_symmetric_rose(self, ax, angles, title, color):
        ax.set_facecolor('#1a1a1a')
        ax.set_title(title, pad=15, fontsize=11, color="white", weight="bold")
        if not angles: return
        
        sym_angles = []
        for a in angles: sym_angles.extend([a, a + 180])
            
        rads = np.radians(sym_angles)
        bins = np.linspace(0.0, 2 * np.pi, 37)
        hist, _ = np.histogram(rads, bins=bins)
        
        ax.set_theta_zero_location("N") 
        ax.set_theta_direction(-1)      
        ax.bar(bins[:-1], hist, width=2*np.pi/len(hist), bottom=0.0, color=color, edgecolor='white', alpha=0.7)
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax.set_xticklabels(['-y\n(0°)', '+x (top)\n(90°)', '+y\n(180°)', '-x (top)\n(270°)'])
        ax.set_yticks([]) 
        ax.grid(color='#444444', linestyle='--', linewidth=0.5)

    def update_plots(self):
        if not self.hydrates: return
        res = self.process_data()
        plt.rcParams.update({'text.color': 'white', 'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white'})
        
        self._draw_header(res)
        self._draw_overview_tab(res)
        self._draw_permutation_tab(res)
        self._draw_circular_tab(res)

    def _draw_header(self, res):
        angles_delta, m_grains, med_siblings = res[0], res[4], res[5]
        self.fig_header.clf()
        
        # Allocate half the header width to text and one quarter to each plot.
        gs = GridSpec(1, 3, top=0.95, bottom=0.05, left=0.02, right=0.98, wspace=0.05, width_ratios=[2, 1, 1])
        
        ax_text   = self.fig_header.add_subplot(gs[0, 0])
        ax_expl2d = self.fig_header.add_subplot(gs[0, 1])
        ax_expl3d = self.fig_header.add_subplot(gs[0, 2], projection="3d")
        
        # Add the text panel on the left.
        ax_text.set_facecolor('#2b2b2b'); ax_text.axis('off')
        title = self.file_name if self.current_mode != "Single Grain" else f"Grain: {self.current_grain_id}"
        ax_text.text(0.0, 0.75, title, color="white", fontsize=16, weight="bold", ha="left", va="center")
        
        # Align the summary statistics horizontally.
        ax_text.text(0.0, 0.35, f"n = {len(angles_delta)} Hydrates", color="white", fontsize=13, ha="left", va="center")
        ax_text.text(0.45, 0.35, f"m = {m_grains} Grains", color="white", fontsize=13, ha="left", va="center")
        
        if self.current_mode == "Single Grain":
            ax_text.text(0.0, -0.05, f"Hydrates in Grain = {len(angles_delta)}", color="white", fontsize=12, ha="left", va="center")
        else:
            ax_text.text(0.0, -0.05, f"Median Hydrates/Grain = {int(med_siblings)}", color="white", fontsize=12, ha="left", va="center")

        # Add the two-dimensional schematic in the center.
        ax_expl2d.set_facecolor('#2b2b2b')
        ax_expl2d.set_aspect('equal')
        ax_expl2d.annotate("", xy=(0, 1.0), xytext=(0, -0.2), arrowprops=dict(arrowstyle="->", color="white", lw=1.5))
        ax_expl2d.text(0, 1.15, "$y$ (0°)", color="white", ha="center", fontsize=12)
        ax_expl2d.annotate("", xy=(1.0, 0), xytext=(-0.2, 0), arrowprops=dict(arrowstyle="->", color="white", lw=1.5))
        ax_expl2d.text(1.15, 0, "$x$ (top) (90°)", color="white", va="center", fontsize=12)
        phi = np.radians(40)
        ax_expl2d.plot([0, np.sin(phi)], [0, np.cos(phi)], color="#f57c00", lw=3)
        arc_angles = np.linspace(0, phi, 50)
        ax_expl2d.plot(0.3 * np.sin(arc_angles), 0.3 * np.cos(arc_angles), color="#c2185b", lw=2)
        ax_expl2d.text(0.15, 0.4, r"$\theta$", color="#c2185b", fontsize=14, weight="bold")
        ax_expl2d.set_xlim(-0.3, 1.3); ax_expl2d.set_ylim(1.4, -0.3); ax_expl2d.axis('off')

        # --- 3D Scheme (Rechts) ---
        ax_expl3d.set_facecolor('#2b2b2b')
        center = np.array([1.5, 1.4, 0.0])
        a, ar = 1.3, 1.7
        b = a / ar
        dir_major = np.array([np.sin(phi), np.cos(phi), 0.0])
        dir_minor = np.array([np.cos(phi), -np.sin(phi), 0.0])
        c_dir = np.array([0.5, 1.3, 1.4])
        c_proj_dir = np.array([c_dir[0], c_dir[1], 0.0])
        
        ax_expl3d.plot_surface(np.array([[-0.3, 3.5], [-0.3, 3.5]]), np.array([[-0.3, -0.3], [3.5, 3.5]]), np.zeros((2,2)), alpha=0.15, color="#dee2e6", edgecolor="none")
        ax_expl3d.quiver(0, 0, 0, 1.3, 0, 0, color="white", lw=1.0, arrow_length_ratio=0.10)
        ax_expl3d.text(1.48, 0, 0, "$x$ (top)", fontsize=14, color="white", ha="center", va="center")
        ax_expl3d.quiver(0, 0, 0, 0, 1.3, 0, color="white", lw=1.0, arrow_length_ratio=0.10)
        ax_expl3d.text(0, 1.48, 0, "$y$", fontsize=14, color="white", ha="center", va="center")
        ax_expl3d.quiver(0, 0, 0, 0, 0, 1.3, color="white", lw=1.0, arrow_length_ratio=0.10)
        ax_expl3d.text(0, 0, 1.48, "$z$", fontsize=14, color="white", ha="center", va="center")
        
        t = np.linspace(0, 2 * np.pi, 250)
        ellipse_pts = center + (a * np.cos(t))[:, np.newaxis] * dir_major + (b * np.sin(t))[:, np.newaxis] * dir_minor
        ax_expl3d.plot(ellipse_pts[:, 0], ellipse_pts[:, 1], ellipse_pts[:, 2], color="#f57c00", lw=1.8, alpha=0.8)
        ax_expl3d.plot([center[0] - a * dir_major[0], center[0] + a * dir_major[0]], [center[1] - a * dir_major[1], center[1] + a * dir_major[1]], [0, 0], color="#e65100", lw=2.3)
        ax_expl3d.quiver(center[0], center[1], center[2], c_dir[0], c_dir[1], c_dir[2], color="#3498db", lw=2.3, arrow_length_ratio=0.08)
        ax_expl3d.text((center + c_dir)[0] + 0.08, (center + c_dir)[1] + 0.08, (center + c_dir)[2] + 0.08, r"$\mathbf{c}$", color="#3498db", fontsize=14, fontweight="bold")
        ax_expl3d.quiver(center[0], center[1], center[2], c_proj_dir[0], c_proj_dir[1], c_proj_dir[2], color="#2980b9", lw=2.0, linestyle="--", arrow_length_ratio=0.08)
        ax_expl3d.text((center + c_proj_dir)[0] + 0.08, (center + c_proj_dir)[1] + 0.08, 0.0, r"$\mathbf{c}_{\mathrm{proj}}$", color="#2980b9", fontsize=12)
        ax_expl3d.plot([(center + c_dir)[0], (center + c_proj_dir)[0]], [(center + c_dir)[1], (center + c_proj_dir)[1]], [(center + c_dir)[2], 0], color="#757575", linestyle=":", lw=1.4)
        
        theta_cproj = np.arctan2(c_proj_dir[0], c_proj_dir[1]) 
        arc_angles = np.linspace(min(phi, theta_cproj), max(phi, theta_cproj), 40)
        ax_expl3d.plot(center[0] + 0.5 * np.sin(arc_angles), center[1] + 0.5 * np.cos(arc_angles), np.zeros_like(arc_angles), color="#c2185b", lw=2.0)
        ax_expl3d.text(center[0] + 0.7 * np.sin((phi + theta_cproj)/2) + a/3, center[1] + 0.6 * np.cos((phi + theta_cproj)/2), 0.02, r"$\Delta\theta$", color="#c2185b", fontsize=14, fontweight="bold")
        
        ax_expl3d.set_axis_off(); ax_expl3d.set_xlim([-0.4, 3.6]); ax_expl3d.set_ylim([-0.4, 3.6]); ax_expl3d.set_zlim([0, 2.1])
        ax_expl3d.invert_yaxis()
        
        # Rotate the camera by 180 degrees from the default azimuth.
        ax_expl3d.view_init(elev=28, azim=215)
        ax_expl3d.disable_mouse_rotation() 

        self.canvas_header.draw()

    def _draw_overview_tab(self, res):
        angles_delta, angles_h, angles_c, latitudes, c_axes_raw = res[0], res[1], res[2], res[3], res[6]

        sym_delta = []
        for a in angles_delta: sym_delta.extend([a, 180-a, 180+a, 360-a])
        rads_delta = np.radians(sym_delta)
        bins_delta = np.linspace(0.0, 2 * np.pi, 37)
        hist_delta, _ = np.histogram(rads_delta, bins=bins_delta)

        stereo_az, stereo_r = [], []
        for c in c_axes_raw:
            az = math.atan2(c[1], c[2]) 
            lat = math.degrees(math.asin(abs(c[0])))
            schmidt_r = 90.0 * (math.sin(math.radians(90 - lat) / 2.0) / math.sin(math.radians(45)))
            stereo_az.append(az); stereo_r.append(schmidt_r)

        self.fig_overview.clf()
        
        # Use separate grid specifications to space the plots clearly.
        gs_top = GridSpec(1, 4, top=0.85, bottom=0.55, left=0.03, right=0.97, wspace=0.6)
        gs_bot = GridSpec(1, 3, top=0.35, bottom=0.08, left=0.05, right=0.95, wspace=0.4, width_ratios=[1, 1, 0.8])
        
        # Arrange the four polar plots in the first row with generous spacing.
        ax_r1 = self.fig_overview.add_subplot(gs_top[0, 0], polar=True)
        ax_r2 = self.fig_overview.add_subplot(gs_top[0, 1], polar=True)
        ax_r3 = self.fig_overview.add_subplot(gs_top[0, 2], polar=True)
        ax_stereo = self.fig_overview.add_subplot(gs_top[0, 3], polar=True)

        self.plot_symmetric_rose(ax_r1, angles_h, r"Absolute Hydrate Orientation $\theta_h$", "#e67e22")
        self.plot_symmetric_rose(ax_r2, angles_c, r"Projected C-Axis Orientation $\theta_c$", "#2ecc71")

        ax_r3.set_facecolor('#1a1a1a')
        ax_r3.set_title(r"Relative Angle $\Delta\theta$", pad=15, fontsize=11, color="white", weight="bold")
        ax_r3.set_theta_zero_location("N") 
        ax_r3.set_theta_direction(-1)      
        ax_r3.bar(bins_delta[:-1], hist_delta, width=2*np.pi/len(hist_delta), bottom=0.0, color='#3498db', edgecolor='white', alpha=0.8)
        ax_r3.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax_r3.set_xticklabels(['-y\n(0°)', '+x (top)\n(90°)', '+y\n(180°)', '-x (top)\n(270°)'])
        ax_r3.set_yticks([]) 
        ax_r3.grid(color='#444444', linestyle='--', linewidth=0.5)

        ax_stereo.set_facecolor('#1a1a1a')
        ax_stereo.set_title("C-Axes Stereonet", pad=15, fontsize=11, color="white", weight="bold")
        ax_stereo.set_theta_zero_location("N") 
        ax_stereo.set_theta_direction(-1)
        ax_stereo.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
        ax_stereo.set_xticklabels(['Z', 'Y', '-Z', '-Y'])
        ax_stereo.set_ylim(0, 90)
        ax_stereo.set_yticks([ 90.0 * (math.sin(math.radians(15)) / math.sin(math.radians(45))), 90.0 * (math.sin(math.radians(30)) / math.sin(math.radians(45))) ])
        ax_stereo.set_yticklabels(['60°', '30°'])
        if stereo_az: ax_stereo.scatter(stereo_az, stereo_r, color='#2ecc71', alpha=0.6, s=15, edgecolors='black')

        # Arrange the alignment plot, distribution, and formula in the second row.
        ax_s_align = self.fig_overview.add_subplot(gs_bot[0, 0])
        ax_s_dist  = self.fig_overview.add_subplot(gs_bot[0, 1])
        ax_def     = self.fig_overview.add_subplot(gs_bot[0, 2])

        ax_s_align.set_facecolor('#1a1a1a')
        ax_s_align.set_title("Alignment vs. C-Axis Latitude")
        ax_s_align.scatter(latitudes, angles_delta, alpha=0.5, c='#2ecc71', s=15)
        ax_s_align.set_xlabel("C-Axis Latitude (Dip) [°]")
        ax_s_align.set_ylabel(r"Relative Angle $\Delta\theta_{min}$ [°]")
        ax_s_align.set_xlim(-5, 95); ax_s_align.set_ylim(-5, 95)
        ax_s_align.grid(color='#444444', linestyle='--', linewidth=0.5)

        ax_s_dist.set_facecolor('#1a1a1a')
        ax_s_dist.set_title("Relative Angle Distribution")
        ax_s_dist.hist(angles_delta, bins=18, range=(0, 90), color='#9b59b6', edgecolor='black', alpha=0.8)
        ax_s_dist.set_xlabel(r"Relative Angle $\Delta\theta_{min}$ [°]")
        ax_s_dist.set_ylabel("Frequency")
        ax_s_dist.set_xlim(0, 90)
        ax_s_dist.grid(color='#444444', linestyle='--', linewidth=0.5)

        # Definition/Formula Box
        ax_def.set_facecolor('#1a1a1a')
        ax_def.axis('off')
        def_text = (
            r"$\mathbf{Definition:\ \Delta\theta_{min}}$" + "\n\n\n"
            r"$\Delta\theta_{min} = \min(|\theta_h - \theta_c|,\ 180^\circ - |\theta_h - \theta_c|)$"
        )
        ax_def.text(0.5, 0.5, def_text, color='#3498db', fontsize=16, ha='center', va='center')

        self.canvas_overview.draw()

    def _draw_permutation_tab(self, res):
        angles_h = np.array(res[1])
        angles_c = np.array(res[2])
        self.fig_perm.clf()

        if len(angles_h) < 10:
            ax = self.fig_perm.add_subplot(111)
            ax.set_facecolor('#2b2b2b'); ax.axis('off')
            ax.text(0.5, 0.5, "Not enough data for permutation test.", color="gray", fontsize=16, ha="center", va="center")
            self.canvas_perm.draw()
            return

        def angle_difference_180(a, b):
            return (a - b) % 180

        true_diffs = angle_difference_180(angles_h, angles_c)
        true_sigma = np.std(true_diffs)
        true_mu = np.mean(true_diffs)

        n_permutations = int(self.slider_perm_iter.get())
        simulated_sigmas = np.zeros(n_permutations)
        simulated_mus = np.zeros(n_permutations)
        B_shuffled = np.copy(angles_c)

        for i in range(n_permutations):
            np.random.shuffle(B_shuffled)
            sim_diffs = angle_difference_180(angles_h, B_shuffled)
            simulated_sigmas[i] = np.std(sim_diffs)
            simulated_mus[i] = np.mean(sim_diffs)

        p_value = np.sum(simulated_sigmas <= true_sigma) / n_permutations
        
        random_sigmas_mu = np.mean(simulated_sigmas)
        random_sigmas_sigma = np.std(simulated_sigmas)

        gs = GridSpec(1, 2, width_ratios=[1, 2.5], left=0.05, right=0.95, top=0.85, bottom=0.15)
        ax_text = self.fig_perm.add_subplot(gs[0, 0])
        ax_text.set_facecolor('#1a1a1a')
        ax_text.axis('off')

        info_text = (
            r"$\mathbf{Permutation\ Test}$" + "\n\n"
            f"Iterations: {n_permutations}\n\n"
            f"True Differences:\n"
            f"  μ = {true_mu:.2f}°\n"
            f"  σ = {true_sigma:.2f}°\n\n"
            f"Randomized Sigmas:\n"
            f"  μ = {random_sigmas_mu:.2f}°\n"
            f"  σ = {random_sigmas_sigma:.2f}°\n\n"
            f"p-value (σ_sim ≤ σ_true): {p_value:.4f}"
        )
        ax_text.text(0.1, 0.9, info_text, color="white", fontsize=15, va="top")

        ax_hist = self.fig_perm.add_subplot(gs[0, 1])
        ax_hist.set_facecolor('#1a1a1a')
        
        ax_hist.hist(simulated_sigmas, bins=30, color='#3498db', edgecolor='black', alpha=0.8, label='Random Simulations')
        ax_hist.axvline(true_sigma, color='#e74c3c', linestyle='dashed', linewidth=2.5, label=f'True Sigma ({true_sigma:.2f}°)')
        
        ax_hist.set_title('Distribution of Sigma (Standard Deviation) in Shuffled Data', color='white', pad=15, weight='bold', fontsize=14)
        ax_hist.set_xlabel('Standard Deviation (Sigma) of Angle Differences [°]', color='white', fontsize=12)
        ax_hist.set_ylabel('Frequency', color='white', fontsize=12)
        ax_hist.tick_params(colors='white')
        ax_hist.grid(axis='y', alpha=0.3, linestyle='--')
        
        legend = ax_hist.legend(facecolor='#2b2b2b', edgecolor='gray', fontsize=11)
        for text in legend.get_texts(): text.set_color('white')

        self.canvas_perm.draw()

    def _draw_circular_tab(self, res):
        angles_h = np.array(res[1])
        angles_c = np.array(res[2])
        self.fig_circ.clf()
        
        if len(angles_h) == 0 or len(angles_c) == 0:
            ax = self.fig_circ.add_subplot(111)
            ax.set_facecolor('#2b2b2b'); ax.axis('off')
            ax.text(0.5, 0.5, "Not enough data for circular correlation.", color="gray", fontsize=16, ha="center", va="center")
            self.canvas_circ.draw()
            return
            
        alpha2_rad = np.radians((angles_h * 2) % 360)
        beta2_rad = np.radians((angles_c * 2) % 360)

        X_a, Y_a = np.mean(np.cos(alpha2_rad)), np.mean(np.sin(alpha2_rad))
        X_b, Y_b = np.mean(np.cos(beta2_rad)), np.mean(np.sin(beta2_rad))

        mean_a_rad = np.arctan2(Y_a, X_a)
        mean_b_rad = np.arctan2(Y_b, X_b)

        R_a = np.sqrt(X_a**2 + Y_a**2)
        R_b = np.sqrt(X_b**2 + Y_b**2)

        sin_diff_a = np.sin(alpha2_rad - mean_a_rad)
        sin_diff_b = np.sin(beta2_rad - mean_b_rad)

        zaehler = np.sum(sin_diff_a * sin_diff_b)
        nenner = np.sqrt(np.sum(sin_diff_a**2) * np.sum(sin_diff_b**2))
        r_circ = zaehler / nenner if nenner > 0 else 0.0

        gs = GridSpec(2, 2, top=0.90, bottom=0.1, left=0.1, right=0.9, wspace=0.3, height_ratios=[4, 1])
        
        def format_polar(ax, title):
            ax.set_facecolor('#1a1a1a')
            ax.set_title(title, pad=15, color="white", weight="bold")
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_yticks([])
            ax.grid(color='#444444', linestyle='--', linewidth=0.5)

        ax1 = self.fig_circ.add_subplot(gs[0, 0], polar=True)
        format_polar(ax1, f"$\\theta_h$: Mean Vector (R={R_a:.2f})")
        ax1.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax1.set_xticklabels(['0°', '90° (top)', '180°', '270°'])
        ax1.scatter(alpha2_rad, np.ones_like(alpha2_rad), c='#e67e22', alpha=0.1, s=15, edgecolors='none')
        ax1.plot([0, mean_a_rad], [0, R_a], color='white', linewidth=3, marker='o', markersize=8)

        ax2 = self.fig_circ.add_subplot(gs[0, 1], polar=True)
        format_polar(ax2, f"$\\theta_c$: Mean Vector (R={R_b:.2f})")
        ax2.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax2.set_xticklabels(['0°', '90° (top)', '180°', '270°'])
        ax2.scatter(beta2_rad, np.ones_like(beta2_rad), c='#2ecc71', alpha=0.1, s=15, edgecolors='none')
        ax2.plot([0, mean_b_rad], [0, R_b], color='white', linewidth=3, marker='o', markersize=8)

        ax_text = self.fig_circ.add_subplot(gs[1, :])
        ax_text.set_facecolor('#2b2b2b'); ax_text.axis('off')
        ax_text.text(0.5, 0.5, f"Jammalamadaka Circular Correlation: $r_{{circ}}$ = {r_circ:.4f}", color="#3498db", fontsize=20, weight="bold", ha="center", va="center")

        self.canvas_circ.draw()