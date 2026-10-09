import customtkinter as ctk
import numpy as np
import math
import traceback
import pathlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
from shapely.affinity import translate

# Import backend logic
try:
    from core.reconstruction_3d_core import extract_vector_contours, extract_fa_data
except ImportError:
    from core_backend.data_merge.reconstruction_3d_core import extract_vector_contours, extract_fa_data

def um_to_mm(val, pos):
    """Formatter to convert micrometer axes to millimeter."""
    return f"{val/1000.0:g}"

class PlanesTab(ctk.CTkFrame):
    def __init__(self, parent, master_h5_path):
        super().__init__(parent, fg_color="transparent")
        self.pack(fill="both", expand=True)
        
        # Determine the base directory to load all segments of a joined dataset
        self.base_dir = pathlib.Path(master_h5_path).parent
        
        self.extracted_contours = None
        self.extracted_fa_data = None  
        self.ax = None
        self.canvas_fig = None
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        self.setup_sidebar()
        
        # Use a white background for the plot container.
        self.plot_container = ctk.CTkFrame(self, fg_color="white")
        self.plot_container.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        # Start extraction and plotting after the GUI has initialized.
        self.after(100, self.run_extraction)

    def setup_sidebar(self):
        sidebar = ctk.CTkScrollableFrame(self, width=350, fg_color="#1e1e1e")
        sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(sidebar, text="Plane Analysis", font=("Arial", 20, "bold")).pack(pady=(10, 20))
        
        # --- Extraction Settings ---
        ctk.CTkLabel(sidebar, text="1. Extraction Mode", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=(5, 5))
        self.seg_mode = ctk.CTkSegmentedButton(sidebar, values=["Interior Grains", "Full Sample"], command=self._on_mode_change)
        self.seg_mode.set("Interior Grains")
        self.seg_mode.pack(fill="x", padx=10, pady=(0, 15))

        # --- Visual Toggles ---
        ctk.CTkLabel(sidebar, text="2. Visualization Toggles", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=(25, 5))
        
        self.var_show_2d_grains = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(sidebar, text="Show 2D Grains (Orange Wireframe)", variable=self.var_show_2d_grains, 
                        command=self.refresh_plot).pack(anchor="w", padx=10, pady=5)
        
        self.var_show_fa_rgb = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(sidebar, text="Show FA Planes (RGB C-Axis Map)", variable=self.var_show_fa_rgb, 
                        command=self.refresh_plot).pack(anchor="w", padx=10, pady=5)
        
        self.var_show_fa_diff = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(sidebar, text="Show Misorientation Planes (Diff)", variable=self.var_show_fa_diff, 
                        command=self.refresh_plot).pack(anchor="w", padx=10, pady=5)

        # --- Z-Scaling ---
        ctk.CTkLabel(sidebar, text="3. Axis Scaling", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=(25, 5))
        self.lbl_zscale = ctk.CTkLabel(sidebar, text="Z-Axis Stretch (Detail Level): 1.0x", anchor="w")
        self.lbl_zscale.pack(anchor="w", padx=10)
        
        self.slider_zscale = ctk.CTkSlider(sidebar, from_=1.0, to=40.0, number_of_steps=390, command=self.update_z_scale)
        self.slider_zscale.set(1.0)
        self.slider_zscale.pack(fill="x", padx=10, pady=10)

    def _safe_clear_plot_container(self):
        for widget in self.plot_container.winfo_children():
            widget.destroy()

    def _on_mode_change(self, value):
        """Reload the data when the user changes the extraction mode."""
        self.run_extraction()

    def run_extraction(self):
        mode = "interior" if "Interior" in self.seg_mode.get() else "full"
        self._safe_clear_plot_container()
        
        # Use dark loading text against the white background.
        lbl_loading = ctk.CTkLabel(self.plot_container, text=f"Loading vector polygons and FA Data...\nApplying -10cm shift per dataset.", font=("Arial", 18), text_color="black")
        lbl_loading.pack(expand=True)
        self.update() 
        
        try:
            self.extracted_contours = {}
            self.extracted_fa_data = {}
            
            h5_files = sorted(list(self.base_dir.glob("*master*.h5")))
            
            if not h5_files:
                lbl_loading.configure(text="Error: No master.h5 files found in workspace.", text_color="red")
                return

            # Process each file and shift its coordinates artificially
            for idx, h5_file in enumerate(h5_files):
                x_shift = -100000.0 * idx # -10 cm in µm
                
                # Extract and shift contours
                c_data = extract_vector_contours(str(h5_file), mode=mode)
                for z, cnts in c_data.items():
                    if z not in self.extracted_contours: self.extracted_contours[z] = []
                    for c in cnts:
                        shifted_c = c.copy()
                        shifted_c[:, 0] += x_shift
                        self.extracted_contours[z].append(shifted_c)
                        
                # Extract and shift FA data
                fa_data = extract_fa_data(str(h5_file))
                for z, items in fa_data.items():
                    if z not in self.extracted_fa_data: self.extracted_fa_data[z] = []
                    for item in items:
                        # Translate the Shapely polygon on the X axis
                        item['poly'] = translate(item['poly'], xoff=x_shift)
                        self.extracted_fa_data[z].append(item)
                        
        except Exception as e:
            traceback.print_exc()
            lbl_loading.configure(text=f"Error loading vectors:\n{e}", text_color="red")
            return
            
        lbl_loading.destroy()
        self.plot_3d_vectors()

    def refresh_plot(self):
        if self.extracted_contours:
            self.plot_3d_vectors()

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
        if self.canvas_fig: 
            self.canvas_fig.draw_idle()

    def update_z_scale(self, val):
        self.lbl_zscale.configure(text=f"Z-Axis Stretch (Detail Level): {val:.1f}x")
        self.apply_aspect_ratio()

    def format_axes(self):
        formatter = FuncFormatter(um_to_mm)
        self.ax.xaxis.set_major_formatter(formatter)
        self.ax.yaxis.set_major_formatter(formatter)
        self.ax.zaxis.set_major_formatter(formatter)
        
        # Use larger, black labels for the 3D plot.
        self.ax.set_xlabel("X [mm] -> top", color='black', weight='bold', fontsize=15)
        self.ax.set_ylabel("Y [mm]", color='black', weight='bold', fontsize=15)
        self.ax.set_zlabel("Z [mm]", color='black', weight='bold', fontsize=15)

    def plot_3d_vectors(self):
        if not self.extracted_contours: return
            
        self._safe_clear_plot_container()
        fig = Figure(figsize=(12, 9), dpi=100)
        
        # Set a light background for the 3D plot.
        fig.patch.set_facecolor('white')
        self.ax = fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('white')
        
        # Use black tick labels on the 3D plot.
        self.ax.tick_params(axis='both', which='major', labelsize=12, colors='black')
        
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        
        # Use light gray for the grid and pane edges.
        for pane in [self.ax.xaxis.pane, self.ax.yaxis.pane, self.ax.zaxis.pane]:
            pane.set_edgecolor('#CCCCCC')
        self.ax.grid(color='#CCCCCC')

        all_x, all_y, all_z = [], [], []
        
        # 1. Plot 2D Grains (Orange Wireframe)
        if self.var_show_2d_grains.get():
            for z_um, contours in self.extracted_contours.items():
                for cnt in contours:
                    x = np.append(cnt[:, 0], cnt[0, 0])
                    y = np.append(cnt[:, 1], cnt[0, 1])
                    z_arr = np.full_like(x, z_um)
                    self.ax.plot(x, y, z_arr, color='#D67040', linewidth=1.0, alpha=0.8)
                    all_x.extend(x); all_y.extend(y); all_z.append(z_um)
                
        # 2. Plot FA Planes (RGB) & Pole Figures
        if self.extracted_fa_data and self.var_show_fa_rgb.get():
            fa_polygons_3d, fa_colors = [], []
            fa_z_levels = sorted(self.extracted_fa_data.keys())
            pole_figure_data = {z: {'theta': [], 'r': [], 'color': []} for z in fa_z_levels}
            
            for z_um in fa_z_levels:
                for fa_item in self.extracted_fa_data[z_um]:
                    poly = fa_item['poly']
                    if poly.is_empty: continue
                    
                    c_vec = fa_item['c_axis']
                    cx, cy, cz = c_vec['x'], c_vec['y'], c_vec['z']
                    norm = math.sqrt(cx**2 + cy**2 + cz**2)
                    
                    if norm > 0:
                        cx, cy, cz = cx/norm, cy/norm, cz/norm
                        if cx < 0: cx, cy, cz = -cx, -cy, -cz # Normalize the vector to the selected hemisphere.
                            
                        theta = math.atan2(cz, cy)
                        angle_zenith = math.degrees(math.acos(min(1.0, cx)))
                        
                        # Custom Mapping
                        if 0 <= theta < math.pi/2: r, g, b = 1 - (theta/(math.pi/2)), 0.0, theta/(math.pi/2)
                        elif math.pi/2 <= theta <= math.pi: r, g, b = 0.0, (theta-math.pi/2)/(math.pi/2), 1 - (theta-math.pi/2)/(math.pi/2)
                        elif -math.pi <= theta < -math.pi/2: r, g, b = (theta+math.pi)/(math.pi/2), 1.0, 0.0
                        else: r, g, b = 1.0, 1 - (theta+math.pi/2)/(math.pi/2), 0.0
                            
                        factor = 0.0 if angle_zenith >= 30.0 else 1.0 - (angle_zenith / 30.0)
                        r, g, b = r*(1-factor)+factor, g*(1-factor)+factor, b*(1-factor)+factor
                        face_color = (r, g, b, 0.7)
                        
                        pole_figure_data[z_um]['theta'].append(theta)
                        pole_figure_data[z_um]['r'].append(angle_zenith)
                        pole_figure_data[z_um]['color'].append((r, g, b, 0.9))
                    else:
                        face_color = (0.5, 0.5, 0.5, 0.7)
                        
                    def extract_faces(p):
                        coords = np.array(p.exterior.coords)
                        fa_polygons_3d.append([[pt[0], pt[1], z_um] for pt in coords])
                        fa_colors.append(face_color)
                        all_x.extend([pt[0] for pt in coords]); all_y.extend([pt[1] for pt in coords]); all_z.append(z_um)
                        
                    if poly.geom_type == 'Polygon': extract_faces(poly)
                    elif poly.geom_type in ['MultiPolygon', 'GeometryCollection']:
                        for geom in getattr(poly, 'geoms', []):
                            if geom.geom_type == 'Polygon': extract_faces(geom)
                                
            if fa_polygons_3d:
                        # Add dark polygon outlines for contrast.
                collection = Poly3DCollection(fa_polygons_3d, linewidths=0.5, edgecolors='black')
                collection.set_facecolors(fa_colors)
                self.ax.add_collection3d(collection)
                
            # Add the colorwheel legend in the upper-left grid position.
            self.ax_inset = fig.add_axes([0.55, 0.65, 0.21, 0.21]) 
            x_cw, y_cw = np.linspace(-1, 1, 150), np.linspace(-1, 1, 150)
            X_cw, Y_cw = np.meshgrid(x_cw, y_cw)
            self.R_cw = np.sqrt(X_cw**2 + Y_cw**2)
            self.Angle_cw_base = np.arctan2(Y_cw, X_cw)
            
            self.cw_im = self.ax_inset.imshow(np.zeros((150, 150, 3)), origin='lower', extent=[-1, 1, -1, 1])
            self.ax_inset.axis('off')
            
            # Label the colorwheel with the C-axis vector projection.
            self.ax_inset.set_title("C-Axis Vector\nY-Z Projection", color='black', fontsize=16, pad=8, weight='bold')

            def update_colorwheel(event=None):
                """Dynamically aligns the 2D colorwheel to the 3D Y-Axis projection."""
                if not hasattr(self, 'ax') or not hasattr(self, 'cw_im'): return
                if getattr(self, '_updating_cw', False): return
                
                current_azim, current_elev = self.ax.azim, self.ax.elev
                if current_azim != getattr(self, 'last_azim', None) or current_elev != getattr(self, 'last_elev', None):
                    self._updating_cw = True
                    self.last_azim, self.last_elev = current_azim, current_elev
                    
                    proj_matrix = self.ax.get_proj()
                    x0, y0, _ = proj3d.proj_transform(0, 0, 0, proj_matrix)
                    xY, yY, _ = proj3d.proj_transform(0, 1, 0, proj_matrix)
                    
                    angle_y_screen = np.arctan2(yY - y0, xY - x0)
                    rot_angle = self.Angle_cw_base - angle_y_screen
                    theta = np.arctan2(np.sin(rot_angle), np.cos(rot_angle))
                    
                    rgb_new = np.zeros((*theta.shape, 3))
                    m1 = (theta >= 0) & (theta < np.pi/2)
                    m2 = (theta >= np.pi/2) & (theta <= np.pi)
                    m3 = (theta >= -np.pi) & (theta < -np.pi/2)
                    m4 = (theta >= -np.pi/2) & (theta < 0)
                    
                    rgb_new[m1, 0], rgb_new[m1, 1], rgb_new[m1, 2] = 1 - (theta[m1]/(np.pi/2)), 0.0, theta[m1]/(np.pi/2)
                    rgb_new[m2, 0], rgb_new[m2, 1], rgb_new[m2, 2] = 0.0, (theta[m2]-np.pi/2)/(np.pi/2), 1 - ((theta[m2]-np.pi/2)/(np.pi/2))
                    rgb_new[m3, 0], rgb_new[m3, 1], rgb_new[m3, 2] = (theta[m3]+np.pi)/(np.pi/2), 1.0, 0.0
                    rgb_new[m4, 0], rgb_new[m4, 1], rgb_new[m4, 2] = 1.0, 1 - ((theta[m4]+np.pi/2)/(np.pi/2)), 0.0
                    
                    R = np.clip(self.R_cw, 0, 1)
                    factor = np.zeros_like(R)
                    mask_white = R < (30.0 / 90.0)
                    factor[mask_white] = 1.0 - (R[mask_white] / (30.0 / 90.0))
                    
                    for c in range(3): rgb_new[:,:,c] = rgb_new[:,:,c] * R + factor
                    
                    # Set the area outside the colorwheel to white.
                    rgb_new[self.R_cw > 1.0] = mcolors.to_rgb('white')
                    
                    self.cw_im.set_data(rgb_new)
                    self._updating_cw = False

            update_colorwheel()
            fig.canvas.mpl_connect('draw_event', update_colorwheel)

            # Arrange up to three stereonet plots in the remaining grid positions.
            max_pf = min(len(fa_z_levels), 3) 
            
            if max_pf > 0:
                for i in range(max_pf):
                    z_um = fa_z_levels[i]
                    pdata = pole_figure_data[z_um]
                    
                    grid_idx = i + 1 
                    col = grid_idx % 2
                    row = grid_idx // 2
                    
                    x_pos = 0.55 + (col * 0.23)
                    y_pos = 0.65 - (row * 0.30)
                    w, h = 0.21, 0.21
                        
                    ax_pf = fig.add_axes([x_pos, y_pos, w, h], polar=True)
                    
                    # Use a light background and grid for each pole figure.
                    ax_pf.set_facecolor('white')
                    ax_pf.grid(color='#CCCCCC', linestyle='--', linewidth=0.5)
                    
                    title = f"FA Layer {i+1}"
                    ax_pf.set_title(title, color='black', fontsize=16, pad=8, weight='bold') # Use a black title for contrast.
                    
                    ax_pf.set_rlim(0, 90)
                    ax_pf.set_rticks([30, 60, 90])
                    # Use black tick labels for contrast.
                    ax_pf.set_yticklabels(['30°', '60°', ''], fontsize=12, color='black')
                    
                    ax_pf.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
                    ax_pf.set_xticklabels(['Y+', 'Z+', 'Y-', 'Z-'], fontsize=12, color='black')
                    
                    if pdata['theta']:
                        # Use black markers for the pole-figure data.
                        ax_pf.scatter(pdata['theta'], pdata['r'], c='black', s=3, alpha=0.9, edgecolors='none')

        # 3. Plot Misorientation Planes & Colorbar
        if self.extracted_fa_data and self.var_show_fa_diff.get():
            ax_diff_cb = fig.add_axes([0.55, 0.15, 0.44, 0.02]) 
            cmap_diff = mcolors.LinearSegmentedColormap.from_list("green_red", [(0, 1, 0), (1, 0, 0)])
            norm_diff = mcolors.Normalize(vmin=0, vmax=90)
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm_diff, cmap=cmap_diff), cax=ax_diff_cb, orientation='horizontal')
            
            # Use black text, outline, and tick labels for the colorbar.
            cb.set_label('Misorientation [°]', color='black', fontsize=15, weight='bold')
            cb.outline.set_edgecolor('black')
            ax_diff_cb.tick_params(colors='black', labelsize=13)
            
            fa_diff_polygons_3d, fa_diff_colors = [], []
            fa_z_levels = sorted(self.extracted_fa_data.keys())
            
            for i in range(len(fa_z_levels) - 1):
                z1, z2 = fa_z_levels[i], fa_z_levels[i+1]
                z_mid = (z1 + z2) / 2.0
                
                for item1 in self.extracted_fa_data[z1]:
                    p1 = item1['poly']
                    if not p1.bounds: continue
                    v1 = np.array([item1['c_axis']['x'], item1['c_axis']['y'], item1['c_axis']['z']])
                    if np.linalg.norm(v1) == 0: continue
                    v1 = v1 / np.linalg.norm(v1)
                    
                    for item2 in self.extracted_fa_data[z2]:
                        p2 = item2['poly']
                        if not p2.bounds: continue
                        
                        if p1.intersects(p2):
                            try:
                                ix_poly = p1.intersection(p2)
                                if ix_poly.is_empty: continue
                                
                                v2 = np.array([item2['c_axis']['x'], item2['c_axis']['y'], item2['c_axis']['z']])
                                if np.linalg.norm(v2) == 0: continue
                                v2 = v2 / np.linalg.norm(v2)
                                
                                dot = max(0.0, min(1.0, abs(np.dot(v1, v2)))) 
                                angle_deg = math.degrees(math.acos(dot))
                                c_val = min(1.0, angle_deg / 90.0)
                                r_col, g_col, b_col, _ = cmap_diff(c_val)
                                face_color = (r_col, g_col, b_col, 0.85)
                                
                                def add_diff_faces(p_ix):
                                    coords = np.array(p_ix.exterior.coords)
                                    fa_diff_polygons_3d.append([[pt[0], pt[1], z_mid] for pt in coords])
                                    fa_diff_colors.append(face_color)
                                    all_x.extend([pt[0] for pt in coords]); all_y.extend([pt[1] for pt in coords]); all_z.append(z_mid)
                                    
                                if ix_poly.geom_type == 'Polygon': add_diff_faces(ix_poly)
                                elif ix_poly.geom_type in ['MultiPolygon', 'GeometryCollection']:
                                    for geom in getattr(ix_poly, 'geoms', []):
                                        if geom.geom_type == 'Polygon': add_diff_faces(geom)
                            except Exception: pass
                                
            if fa_diff_polygons_3d:
                # Use black outlines for the misorientation planes.
                collection = Poly3DCollection(fa_diff_polygons_3d, linewidths=0.2, edgecolors='black')
                collection.set_facecolors(fa_diff_colors)
                self.ax.add_collection3d(collection)

        # Fix Limits
        if not all_x: all_x, all_y, all_z = [0, 1], [0, 1], [0, 6000]
        self.ax.set_xlim3d(np.min(all_x), np.max(all_x))
        self.ax.set_ylim3d(np.min(all_y), np.max(all_y))
        self.ax.set_zlim3d(np.min(all_z)-10, np.max(all_z)+10)

        self.ax.invert_zaxis()
        self.apply_aspect_ratio()
        self.ax.view_init(elev=20, azim=-45)
        self.format_axes()

        # Reserve space for the enlarged grid alongside the 3D plot.
        fig.subplots_adjust(left=0.05, right=0.50, top=0.95, bottom=0.05)
        
        self.canvas_fig = FigureCanvasTkAgg(fig, master=self.plot_container)
        self.canvas_fig.draw()
        canvas_widget = self.canvas_fig.get_tk_widget()
        canvas_widget.pack(side="top", fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas_fig, self.plot_container)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")