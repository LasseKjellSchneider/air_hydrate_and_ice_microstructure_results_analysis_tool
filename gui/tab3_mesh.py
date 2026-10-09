import customtkinter as ctk
import numpy as np
import traceback
import os
import json
import pathlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import tkinter as tk

def um_to_mm(val, pos):
    """Formatter to convert micrometer axes to millimeter."""
    return f"{val/1000.0:g}"

class MeshTab(ctk.CTkFrame):
    def __init__(self, parent, master_h5_path):
        super().__init__(parent, fg_color="transparent")
        self.pack(fill="both", expand=True)
        
        self.base_dir = pathlib.Path(master_h5_path).parent
        
        self.all_meshes = []
        self.all_hydrates = []  # Stores (x, y, z, area_um2) tuples.
        self.ax = None
        self.canvas_fig = None
        self.cbar = None 
        
        self.current_single_grain = None
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        self.setup_sidebar()
        
        # Use a white background for the plot container.
        self.plot_container = ctk.CTkFrame(self, fg_color="white")
        self.plot_container.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        self.after(100, self.load_precomputed_data)

    def setup_sidebar(self):
        sidebar = ctk.CTkScrollableFrame(self, width=350, fg_color="#1e1e1e")
        sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(sidebar, text="3D Mesh Viewer", font=("Arial", 20, "bold")).pack(pady=(10, 20))

        # --- Visualization Toggles ---
        ctk.CTkLabel(sidebar, text="Visualization Toggles", font=("Arial", 14, "bold"), text_color="gray").pack(anchor="w", padx=10, pady=(5, 5))
        
        self.btn_reset_view = ctk.CTkButton(sidebar, text="Reset to Full View", command=self.reset_view, fg_color="#7F8C8D", hover_color="#606B6C")
        self.btn_reset_view.pack(fill="x", padx=10, pady=(5, 15))
        
        self.var_show_clean = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(sidebar, text="1:1 Matches (Blue)", variable=self.var_show_clean, fg_color="#3498DB", hover_color="#2980B9", command=self.refresh_mesh_plot).pack(anchor="w", padx=10, pady=2)
        
        self.var_show_split = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(sidebar, text="Y-Splits (Pink)", variable=self.var_show_split, fg_color="#FF69B4", hover_color="#FF1493", command=self.refresh_mesh_plot).pack(anchor="w", padx=10, pady=2)
        
        self.var_show_unassigned = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(sidebar, text="Unassigned (Red)", variable=self.var_show_unassigned, fg_color="#E74C3C", hover_color="#C0392B", command=self.refresh_mesh_plot).pack(anchor="w", padx=10, pady=2)
        
        self.var_show_hydrates = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(sidebar, text="Show Hydrates (Color Coded)", variable=self.var_show_hydrates, fg_color="#00FFFF", hover_color="#00CCCC", command=self.refresh_mesh_plot).pack(anchor="w", padx=10, pady=(15, 20))

        # --- Manual Size Filtering (AREA) ---
        ctk.CTkLabel(sidebar, text="Manual Hydrate Size Filter", font=("Arial", 14, "bold"), text_color="#F1C40F").pack(anchor="w", padx=10, pady=(10, 5))
        
        self.lbl_min_area = ctk.CTkLabel(sidebar, text="Min Area (µm²): 0", anchor="w")
        self.lbl_min_area.pack(anchor="w", padx=10)
        
        self.slider_min_area = ctk.CTkSlider(sidebar, from_=0.0, to=100.0, number_of_steps=200, command=self.on_area_slider_change)
        self.slider_min_area.set(0.0)
        self.slider_min_area.pack(fill="x", padx=10, pady=5)

        # Display the number of hydrates that meet the current filter.
        self.lbl_hydrate_count = ctk.CTkLabel(sidebar, text="Remaining: 0 / 0 Hydrates", font=("Arial", 12, "italic"), text_color="#A0A0A0")
        self.lbl_hydrate_count.pack(anchor="w", padx=15, pady=(0, 10))

        self.btn_apply_filter = ctk.CTkButton(sidebar, text="Apply Global Filter\n& Recalculate Summary", 
                                              font=("Arial", 14, "bold"), fg_color="#E67E22", hover_color="#D35400",
                                              command=self.apply_global_filter)
        self.btn_apply_filter.pack(fill="x", padx=10, pady=(10, 20))

        # --- Z-Scaling ---
        ctk.CTkLabel(sidebar, text="Axis Scaling", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=(10, 5))
        self.lbl_zscale = ctk.CTkLabel(sidebar, text="Z-Axis Stretch: 1.0x", anchor="w")
        self.lbl_zscale.pack(anchor="w", padx=10)
        self.slider_zscale = ctk.CTkSlider(sidebar, from_=1.0, to=40.0, number_of_steps=390, command=self.update_z_scale)
        self.slider_zscale.set(1.0)
        self.slider_zscale.pack(fill="x", padx=10, pady=5)

    def _safe_clear_plot_container(self):
        for widget in self.plot_container.winfo_children():
            widget.destroy()

    def load_precomputed_data(self):
        self._safe_clear_plot_container()
        # Use dark loading text against the white background.
        lbl = ctk.CTkLabel(self.plot_container, text="Loading precomputed meshes and hydrates...", font=("Arial", 18), text_color="black")
        lbl.pack(expand=True)
        self.update() 
        
        try:
            self.all_meshes = []
            self.all_hydrates = []
            
            mesh_files = sorted(list(self.base_dir.glob("*3d_mesh_geometry*.json")))
            hyd_files = sorted(list(self.base_dir.glob("*3d_hydrates*.json")))
            
            if not mesh_files:
                lbl.configure(text="No '3d_mesh_geometry.json' found in workspace.", text_color="red")
                return
                
            for idx, m_file in enumerate(mesh_files):
                x_shift = -100000.0 * idx  # 10 cm shift in µm
                with open(m_file, 'r', encoding='utf-8') as f:
                    local_meshes = json.load(f)
                for m in local_meshes:
                    shifted_verts = []
                    for v in m['vertices']:
                        shifted_verts.append([v[0] + x_shift, v[1], v[2]])
                    m['vertices'] = shifted_verts
                    self.all_meshes.append(m)
                    
            max_area = 0.0
            for idx, h_file in enumerate(hyd_files):
                x_shift = -100000.0 * idx 
                with open(h_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    hyds = data.get('hydrates', [])
                    for h in hyds:
                        coords = h.get('world_coordinates_um', {})
                        hx = coords.get('x', 0) + x_shift
                        hy = coords.get('y', 0)
                        hz = coords.get('z_abs', coords.get('z', 0))
                        
                        area = h.get('area_um2', 0.0)
                        
                        if area > max_area: max_area = area
                        self.all_hydrates.append((hx, hy, hz, area))
            
            # Configure dynamic slider range based on loaded area data
            if max_area > 0:
                self.slider_min_area.configure(to=max_area, number_of_steps=200)
                if self.slider_min_area.get() > max_area:
                    self.slider_min_area.set(0.0)
            
            # Initialize the hydrate count label.
            self.update_hydrate_count_label(self.slider_min_area.get())
            self.refresh_mesh_plot()
            
        except Exception as e:
            traceback.print_exc()
            lbl.configure(text=f"Error loading files:\n{e}", text_color="red")

    def update_hydrate_count_label(self, min_area):
        """Update the label showing the number of remaining hydrates."""
        if not self.all_hydrates:
            self.lbl_hydrate_count.configure(text="Remaining: 0 / 0 Hydrates")
            return
            
        remaining = sum(1 for h in self.all_hydrates if h[3] >= min_area)
        total = len(self.all_hydrates)
        self.lbl_hydrate_count.configure(text=f"Remaining: {remaining:,} / {total:,} Hydrates")

    def on_area_slider_change(self, val):
        self.lbl_min_area.configure(text=f"Min Area (µm²): {int(val):,}")
        self.update_hydrate_count_label(val)
        self.refresh_mesh_plot()

    def reset_view(self):
        self.current_single_grain = None
        self.refresh_mesh_plot()

    def refresh_mesh_plot(self):
        if not self.all_meshes: return
        self._safe_clear_plot_container()
            
        fig = Figure(figsize=(10, 8), dpi=100)
        
        # Use a white background for the figure.
        fig.patch.set_facecolor('white')
        self.ax = fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('white')
        
        # Use black title and tick labels with slightly larger text.
        self.ax.set_title("3D Mesh & Hydrate Viewer", color='black', weight='bold', fontsize=16)
        self.ax.tick_params(colors='black', labelsize=12)
        
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        
        # Use light gray for the axis pane edges.
        for pane in [self.ax.xaxis.pane, self.ax.yaxis.pane, self.ax.zaxis.pane]:
            pane.set_edgecolor('#CCCCCC')
        self.ax.grid(color='#CCCCCC')

        all_x, all_y, all_z = [], [], []

        # --- 1. MESH PLOTTING ---
        colors = {
            'clean': (0.20, 0.60, 0.86, 0.5),      
            'split': (1.00, 0.41, 0.71, 0.5),      
            'unassigned': (0.91, 0.30, 0.24, 0.2)  
        }

        active_types = []
        if self.var_show_clean.get(): active_types.append('clean')
        if self.var_show_split.get(): active_types.append('split')
        if self.var_show_unassigned.get(): active_types.append('unassigned')

        polygons, face_colors = [], []
        
        for mesh in self.all_meshes:
            m_type = mesh.get('type', 'clean')
            if m_type not in active_types: continue
                
            verts = mesh['vertices']
            color = colors.get(m_type, (1, 1, 1, 0.5))
            
            for face in mesh['faces']:
                triangle = [verts[face[0]], verts[face[1]], verts[face[2]]]
                polygons.append(triangle)
                face_colors.append(color)
                
                all_x.extend([pt[0] for pt in triangle])
                all_y.extend([pt[1] for pt in triangle])
                all_z.extend([pt[2] for pt in triangle])
        
        if polygons:
            # Use slightly thicker black polygon edges for contrast.
            collection = Poly3DCollection(polygons, edgecolors='black', linewidths=0.2)
            collection.set_facecolors(face_colors)
            self.ax.add_collection3d(collection)
            
        # --- 2. HYDRATE PLOTTING (Filtered by AREA) ---
        if self.var_show_hydrates.get() and self.all_hydrates:
            min_area = self.slider_min_area.get()
            
            # Filter visually
            filtered_hyds = [h for h in self.all_hydrates if h[3] >= min_area]
            
            if filtered_hyds:
                hx = [p[0] for p in filtered_hyds]
                hy = [p[1] for p in filtered_hyds]
                hz = [p[2] for p in filtered_hyds]
                ha = [p[3] for p in filtered_hyds] # a for area
                
                # Scatter with color map based on Area
                scatter = self.ax.scatter(hx, hy, hz, c=ha, cmap='turbo', s=20, marker='o', alpha=0.9, edgecolors='black', linewidths=0.5)
                
                # Add a colorbar with black labels and tick marks.
                self.cbar = fig.colorbar(scatter, ax=self.ax, shrink=0.5, pad=0.1)
                self.cbar.set_label('Hydrate Area (µm²)', color='black', fontsize=15, weight='bold')
                self.cbar.ax.yaxis.set_tick_params(color='black', labelcolor='black', labelsize=13)
                self.cbar.outline.set_edgecolor('black')
                plt.setp(plt.getp(self.cbar.ax.axes, 'yticklabels'), color='black')
                
                all_x.extend(hx); all_y.extend(hy); all_z.extend(hz)
        
        # --- BOUNDS & SCALING ---
        if all_x:
            self.ax.set_xlim3d(np.min(all_x), np.max(all_x))
            self.ax.set_ylim3d(np.min(all_y), np.max(all_y))
            self.ax.set_zlim3d(np.min(all_z)-10, np.max(all_z)+10)

        self.ax.invert_zaxis()
        self.apply_aspect_ratio()
        self.ax.view_init(elev=20, azim=-45)
        
        formatter = FuncFormatter(um_to_mm)
        self.ax.xaxis.set_major_formatter(formatter)
        self.ax.yaxis.set_major_formatter(formatter)
        self.ax.zaxis.set_major_formatter(formatter)
        
        # Use larger, black axis labels.
        self.ax.set_xlabel("X [mm] -> top", color='black', weight='bold', fontsize=15)
        self.ax.set_ylabel("Y [mm]", color='black', weight='bold', fontsize=15)
        self.ax.set_zlabel("Z [mm]", color='black', weight='bold', fontsize=15)

        fig.tight_layout()
        self.canvas_fig = FigureCanvasTkAgg(fig, master=self.plot_container)
        self.canvas_fig.draw()
        canvas_widget = self.canvas_fig.get_tk_widget()
        canvas_widget.pack(side="top", fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas_fig, self.plot_container)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")

    def apply_aspect_ratio(self):
        if not self.ax: return
        x_limits = self.ax.get_xlim3d()
        y_limits = self.ax.get_ylim3d()
        z_limits = self.ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        
        max_range = max(x_range, y_range) if max(x_range, y_range) > 0 else 1.0
        z_scale = float(self.slider_zscale.get())
        
        self.ax.set_box_aspect([x_range / max_range, y_range / max_range, (z_range / max_range) * z_scale])
        if self.canvas_fig: self.canvas_fig.draw_idle()

    def update_z_scale(self, val):
        self.lbl_zscale.configure(text=f"Z-Axis Stretch: {val:.1f}x")
        self.apply_aspect_ratio()

    # --- GLOBAL FILTER LOGIC ---
    def apply_global_filter(self):
        min_area = self.slider_min_area.get()
        
        popup = tk.Toplevel()
        popup.title("Recalculating...")
        popup.geometry("400x150")
        popup.configure(bg="#2b2b2b")
        popup.geometry(f"+{self.winfo_rootx() + 400}+{self.winfo_rooty() + 300}")
        lbl = tk.Label(popup, text=f"Recalculating Global Summary...\nExcluding Hydrates < {int(min_area):,} µm²\n\nPlease wait.", 
                       fg="#38BDF8", bg="#2b2b2b", font=("Arial", 14, "bold"))
        lbl.pack(expand=True)
        popup.update() 
        
        try:
            from core.summary_engine import SummaryEngine
            engine = SummaryEngine(self.base_dir)
            new_summary_path = engine.run(min_hydrate_area=min_area)
            
            main_app = self.winfo_toplevel()
            if hasattr(main_app, 'set_active_summary'):
                main_app.set_active_summary(new_summary_path)
                
            popup.destroy()
            tk.messagebox.showinfo("Success", f"Global Summary updated!\nHydrates below {int(min_area):,} µm² are now filtered out across all tabs.")
            
        except Exception as e:
            popup.destroy()
            traceback.print_exc()
            tk.messagebox.showerror("Error", f"Failed to recalculate summary:\n{e}")