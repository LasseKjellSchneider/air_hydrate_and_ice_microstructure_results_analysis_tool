import customtkinter as ctk
import tkinter.messagebox as messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import logging
import traceback

from core.plot_engine import PlotEngine
from gui.popup_tiff_viewer import TiffViewerPopup

def get_logger(base_dir):
    logger = logging.getLogger("HydrateAnalyzer")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        log_path = base_dir / "Analyzer_debug.txt"
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)
    return logger

class PlotPopup(ctk.CTkToplevel):
    def __init__(self, parent, data_loader, plot_type):
        super().__init__(parent)
        self.data_loader = data_loader
        self.plot_type = plot_type
        self.logger = get_logger(self.data_loader.base_dir)
        
        self.title(f"Interactive Plot: {plot_type}")
        self.geometry("1200x800")
        self.attributes('-topmost', True) 
        
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.canvas_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.canvas_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.control_frame = ctk.CTkScrollableFrame(self)
        self.control_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        self.current_fig = None
        self.canvas_widget = None
        self.current_params_str = "default"
        self.cids = []
        self._update_job = None
        
        self.ui_vars = {
            'dark_mode': ctk.BooleanVar(value=True),
            'show_title': ctk.BooleanVar(value=True),
            'show_grid': ctk.BooleanVar(value=False),
            'label_size': ctk.IntVar(value=12),
            'cmap': ctk.StringVar(value='lava'),
            'base_color': ctk.StringVar(value='Blue'),
            'dpi': ctk.StringVar(value='300')
        }
        
        self.setup_controls()
        self.update_plot()
        
    def _delayed_update(self):
        if self._update_job is not None:
            self.after_cancel(self._update_job)
        self._update_job = self.after(300, self.update_plot)
        
    def setup_controls(self):
        ctk.CTkLabel(self.control_frame, text="Parameters & Settings", font=("Arial", 20, "bold")).pack(pady=10)
        self.sliders = {}
        
        # ==========================================
        # 1. SPECIFIC PARAMS
        # ==========================================
        lbl_spec = ctk.CTkLabel(self.control_frame, text="Plot Specifics", font=("Arial", 16, "bold"), text_color="#3498db")
        lbl_spec.pack(pady=(10, 5), anchor="w", padx=10)
        
        if "Dist" in self.plot_type:
            lbl = ctk.CTkLabel(self.control_frame, text="Bins (Resolution): 50")
            lbl.pack(pady=(5, 0))
            slider = ctk.CTkSlider(self.control_frame, from_=10, to=200, number_of_steps=190, command=lambda v: [lbl.configure(text=f"Bins: {int(v)}"), self._delayed_update()])
            slider.set(50); slider.pack(pady=5, padx=20, fill="x")
            self.sliders['bins'] = slider
            
        elif "Heatmap" in self.plot_type and "Tile" not in self.plot_type and "Rose" not in self.plot_type:
            lbl_tile = ctk.CTkLabel(self.control_frame, text="Tile Size Multiplier: 1x (500µm)")
            lbl_tile.pack(pady=(5, 0))
            sl_tile = ctk.CTkSlider(self.control_frame, from_=1, to=5, number_of_steps=4, command=lambda v: [lbl_tile.configure(text=f"Tile Multiplier: {int(v)}x ({int(v)*500}µm)"), self._delayed_update()])
            sl_tile.set(1); sl_tile.pack(pady=5, padx=20, fill="x")
            self.sliders['tile_mult'] = sl_tile
            
            lbl_blur = ctk.CTkLabel(self.control_frame, text="Blur Radius: 3x")
            lbl_blur.pack(pady=(15, 0))
            sl_blur = ctk.CTkSlider(self.control_frame, from_=1, to=9, number_of_steps=8, command=lambda v: self._force_odd_blur(v, lbl_blur))
            sl_blur.set(3); sl_blur.pack(pady=5, padx=20, fill="x")
            self.sliders['blur'] = sl_blur
            
            if self.plot_type == "Relation Heatmap":
                fig_math = Figure(figsize=(3.5, 1.8), dpi=100)
                fig_math.patch.set_facecolor('#2b2b2b')
                ax_math = fig_math.add_subplot(111)
                ax_math.set_facecolor('#2b2b2b')
                ax_math.axis('off')
                
                latex_str = (
                    r"$\mathbf{Relation\ Score\ (R)}$" + "\n\n"
                    r"$R = \frac{A_{local}}{A_{global}} \times \frac{D_{global}}{D_{local}}$" + "\n\n"
                    r"$R > 1: Score = R - 1$" + "\n"
                    r"$R \leq 1: Score = 1 - \frac{1}{R}$"
                )
                
                ax_math.text(0.5, 0.5, latex_str, color='#3498db', fontsize=14, ha='center', va='center')
                canvas_math = FigureCanvasTkAgg(fig_math, master=self.control_frame)
                canvas_math.draw()
                canvas_math.get_tk_widget().pack(pady=(15, 0), padx=10, fill="x")
            
            ctk.CTkLabel(self.control_frame, text="💡 Hover over the map to see\nimage grid. Click a cell\nto open the stack.", text_color="#e74c3c", font=("Arial", 12, "italic")).pack(pady=10)
                         
        elif self.plot_type == "Correlation Scatter":
            ctk.CTkLabel(self.control_frame, text="💡 100 sample points highlighted\nHover to select, click to inspect.", text_color="#00ffff", font=("Arial", 12, "italic")).pack(pady=10)
                         
        elif self.plot_type == "Tile Scatter":
            ctk.CTkLabel(self.control_frame, text="💡 Each point is a tile.\nHover to select, click to inspect.", text_color="#d35400", font=("Arial", 12, "italic")).pack(pady=10)
                         
        elif self.plot_type == "Half-Rose Heatmap":
            lbl_angle = ctk.CTkLabel(self.control_frame, text="Angle Bin (deg): 5")
            lbl_angle.pack(pady=(5, 0))
            sl_angle = ctk.CTkSlider(self.control_frame, from_=1, to=30, number_of_steps=29, command=lambda v: [lbl_angle.configure(text=f"Angle Bin (deg): {int(v)}"), self._delayed_update()])
            sl_angle.set(5); sl_angle.pack(pady=5, padx=20, fill="x")
            self.sliders['angle_bin'] = sl_angle

            lbl_aspect = ctk.CTkLabel(self.control_frame, text="Aspect Bin (x/30): 5")
            lbl_aspect.pack(pady=(15, 0))
            sl_aspect = ctk.CTkSlider(self.control_frame, from_=1, to=30, number_of_steps=29, command=lambda v: [lbl_aspect.configure(text=f"Aspect Bin (x/30): {int(v)}"), self._delayed_update()])
            sl_aspect.set(5); sl_aspect.pack(pady=5, padx=20, fill="x")
            self.sliders['aspect_bin'] = sl_aspect
            
        elif self.plot_type in ["Count X", "Area X", "Aspect X", "Count Y", "Area Y", "Aspect Y"]:
            lbl = ctk.CTkLabel(self.control_frame, text="Bin Width (µm): 500")
            lbl.pack(pady=(5, 0))
            slider = ctk.CTkSlider(self.control_frame, from_=100, to=2000, number_of_steps=19, command=lambda v: [lbl.configure(text=f"Bin Width: {int(v)} µm"), self._delayed_update()])
            slider.set(500); slider.pack(pady=5, padx=20, fill="x")
            self.sliders['bin_width'] = slider

        # ==========================================
        # 2. APPEARANCE
        # ==========================================
        lbl_app = ctk.CTkLabel(self.control_frame, text="Appearance", font=("Arial", 16, "bold"), text_color="#3498db")
        lbl_app.pack(pady=(20, 5), anchor="w", padx=10)
        
        sw_dark = ctk.CTkSwitch(self.control_frame, text="Dark Mode", variable=self.ui_vars['dark_mode'], command=self.update_plot)
        sw_dark.pack(pady=5, padx=20, anchor="w")
        
        sw_title = ctk.CTkSwitch(self.control_frame, text="Show Title", variable=self.ui_vars['show_title'], command=self.update_plot)
        sw_title.pack(pady=5, padx=20, anchor="w")
        
        sw_grid = ctk.CTkSwitch(self.control_frame, text="Show Grid", variable=self.ui_vars['show_grid'], command=self.update_plot)
        sw_grid.pack(pady=5, padx=20, anchor="w")
        
        lbl_ls = ctk.CTkLabel(self.control_frame, text="Label Size: 12")
        lbl_ls.pack(pady=(10, 0))
        sl_ls = ctk.CTkSlider(self.control_frame, from_=6, to=24, number_of_steps=18, 
                              command=lambda v: [self.ui_vars['label_size'].set(int(v)), lbl_ls.configure(text=f"Label Size: {int(v)}"), self._delayed_update()])
        sl_ls.set(12)
        sl_ls.pack(pady=5, padx=20, fill="x")

        # ==========================================
        # 3. COLORS
        # ==========================================
        lbl_col = ctk.CTkLabel(self.control_frame, text="Colors", font=("Arial", 16, "bold"), text_color="#3498db")
        lbl_col.pack(pady=(20, 5), anchor="w", padx=10)
        
        if "Heatmap" in self.plot_type:
            ctk.CTkLabel(self.control_frame, text="Colormap:").pack(anchor="w", padx=20)
            cmap_combo = ctk.CTkComboBox(self.control_frame, values=['lava', 'div', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm'], variable=self.ui_vars['cmap'], command=lambda _: self.update_plot())
            cmap_combo.pack(pady=5, padx=20, fill="x")
        else:
            ctk.CTkLabel(self.control_frame, text="Base Color:").pack(anchor="w", padx=20)
            color_options = [
                'Blue', 'Red', 'Green', 'Purple', 'Orange', 'Cyan', 
                'Teal (#298c8c)', 'Dark Purple (#721085)', 'Warm Orange (#f1a226)'
            ]
            color_combo = ctk.CTkComboBox(self.control_frame, values=color_options, variable=self.ui_vars['base_color'], command=lambda _: self.update_plot())
            color_combo.pack(pady=5, padx=20, fill="x")

        # ==========================================
        # 4. AXIS LIMITS
        # ==========================================
        lbl_lim = ctk.CTkLabel(self.control_frame, text="Axis Limits", font=("Arial", 16, "bold"), text_color="#3498db")
        lbl_lim.pack(pady=(20, 5), anchor="w", padx=10)
        
        lim_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        lim_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(lim_frame, text="X Min:").grid(row=0, column=0, padx=5, pady=5)
        self.entry_x_min = ctk.CTkEntry(lim_frame, width=65)
        self.entry_x_min.grid(row=0, column=1, padx=5, pady=5)
        
        ctk.CTkLabel(lim_frame, text="X Max:").grid(row=0, column=2, padx=5, pady=5)
        self.entry_x_max = ctk.CTkEntry(lim_frame, width=65)
        self.entry_x_max.grid(row=0, column=3, padx=5, pady=5)
        
        ctk.CTkLabel(lim_frame, text="Y Min:").grid(row=1, column=0, padx=5, pady=5)
        self.entry_y_min = ctk.CTkEntry(lim_frame, width=65)
        self.entry_y_min.grid(row=1, column=1, padx=5, pady=5)
        
        ctk.CTkLabel(lim_frame, text="Y Max:").grid(row=1, column=2, padx=5, pady=5)
        self.entry_y_max = ctk.CTkEntry(lim_frame, width=65)
        self.entry_y_max.grid(row=1, column=3, padx=5, pady=5)

        # ==========================================
        # 5. BUTTONS
        # ==========================================
        ctk.CTkButton(self.control_frame, text="Apply Limits & Update", height=40, font=("Arial", 16, "bold"),
                      command=self.update_plot).pack(pady=(30, 20), padx=20, fill="x")
                      
        save_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        save_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(save_frame, text="Save DPI:").pack(side="left", padx=5)
        ctk.CTkComboBox(save_frame, values=['100', '300', '600', '1000'], variable=self.ui_vars['dpi'], width=80).pack(side="left", padx=5)
        
        ctk.CTkButton(self.control_frame, text="Save Plot (Pub. Ready)", height=40, fg_color="#27ae60", hover_color="#2ecc71",
                      command=self.save_plot).pack(pady=10, padx=20, fill="x")

    def _force_odd_blur(self, val, label):
        val = int(val); val += 1 if val % 2 == 0 else 0
        self.sliders['blur'].set(val); label.configure(text=f"Blur Radius: {val}x")
        self._delayed_update()

    def update_plot(self):
        def _get_lim(entry):
            val = entry.get()
            if val.strip() == "": return None
            try: return float(val)
            except ValueError: return None

        plot_kwargs = {
            'dark_mode': self.ui_vars['dark_mode'].get(),
            'show_title': self.ui_vars['show_title'].get(),
            'show_grid': self.ui_vars['show_grid'].get(),
            'label_size': self.ui_vars['label_size'].get(),
            'cmap_name': self.ui_vars['cmap'].get(),
            'base_color': self.ui_vars['base_color'].get(),
            'x_min': _get_lim(self.entry_x_min),
            'x_max': _get_lim(self.entry_x_max),
            'y_min': _get_lim(self.entry_y_min),
            'y_max': _get_lim(self.entry_y_max)
        }

        try:
            if self.current_fig:
                for cid in self.cids: self.current_fig.canvas.mpl_disconnect(cid)
                self.cids.clear()
                plt.close(self.current_fig)
                
            if self.canvas_widget: self.canvas_widget.destroy()
            
            DF = self.data_loader.df
            sh_um = self.data_loader.sample_height_um
            g_dens = self.data_loader.global_density
            g_area = self.data_loader.global_med_area
            figsize = (8, 6)
            
            if "Dist" in self.plot_type:
                b = int(self.sliders['bins'].get())
                self.current_fig = PlotEngine.create_dist(DF, "area_um2" if "Area" in self.plot_type else "aspect_ratio", bins=b, figsize=figsize, is_preview=False, **plot_kwargs)
                self.current_params_str = f"bins_{b}"
                
            elif "Correlation" in self.plot_type:
                self.current_fig = PlotEngine.create_scatter_corr(DF, figsize=figsize, is_preview=False, **plot_kwargs)
            elif "Tile Scatter" in self.plot_type:
                self.current_fig = PlotEngine.create_scatter_tiles(DF, sh_um, g_dens, g_area, figsize=figsize, is_preview=False, **plot_kwargs)
                
            elif self.plot_type == "Tile Heatmap":
                self.current_fig = PlotEngine.create_tile_heatmap(DF, sh_um, figsize=figsize, is_preview=False, **plot_kwargs)
                
            elif self.plot_type == "Half-Rose Heatmap":
                a_bin = int(self.sliders['angle_bin'].get())
                asp_bin = int(self.sliders['aspect_bin'].get())
                self.current_fig = PlotEngine.create_half_rose_heatmap(DF, angle_bin_deg=a_bin, aspect_bin_num=asp_bin, figsize=(6,6), is_preview=False, **plot_kwargs)
                self.current_params_str = f"angle{a_bin}_aspect{asp_bin}of30"
                
            elif "Heatmap" in self.plot_type and "Tile" not in self.plot_type and "Rose" not in self.plot_type:
                tm, bv = int(self.sliders['tile_mult'].get()), int(self.sliders['blur'].get())
                self.current_fig = PlotEngine.create_heatmap(DF, sh_um, g_dens, g_area, self.plot_type, tile_mult=tm, blur_val=bv, figsize=(7, 6), is_preview=False, positions=self.data_loader.positions, **plot_kwargs)
                self.current_params_str = f"tile{tm}x_blur{bv}x"
                
            elif "Rose" in self.plot_type:
                self.current_fig = PlotEngine.create_rose(DF, is_half=("Half" in self.plot_type), figsize=(6,6), is_preview=False, **plot_kwargs)
                
            elif self.plot_type in ["Count X", "Area X", "Aspect X", "Count Y", "Area Y", "Aspect Y"]:
                bw = int(self.sliders['bin_width'].get())
                col = 'x_um' if 'X' in self.plot_type else 'y_um'
                metric = self.plot_type.split()[0].lower()
                self.current_fig = PlotEngine.create_profile(DF, col, metric, bin_width=bw, figsize=figsize, is_preview=False, **plot_kwargs)
                self.current_params_str = f"binwidth_{bw}um"
                
            canvas = FigureCanvasTkAgg(self.current_fig, master=self.canvas_frame)
            canvas.draw()
            
            if ("Heatmap" in self.plot_type and hasattr(self.current_fig, 'hover_rect')) or \
               (self.plot_type == "Correlation Scatter" and hasattr(self.current_fig, 'clickable_points')) or \
               (self.plot_type == "Tile Scatter" and hasattr(self.current_fig, 'tile_points')):
                self.cids.append(self.current_fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move))
                self.cids.append(self.current_fig.canvas.mpl_connect('button_press_event', self.on_click))
                
            if "Heatmap" in self.plot_type and hasattr(self.current_fig, 'hover_rect'):
                self.cids.append(self.current_fig.canvas.mpl_connect('axes_enter_event', self.on_axes_enter))
                self.cids.append(self.current_fig.canvas.mpl_connect('axes_leave_event', self.on_axes_leave))

            self.canvas_widget = canvas.get_tk_widget()
            self.canvas_widget.pack(fill="both", expand=True)
        except Exception as e:
            self.logger.error(f"Error in update_plot: {e}")
            self.logger.error(traceback.format_exc())

    def on_axes_enter(self, event):
        if not hasattr(self.current_fig, 'grid_lines'): return
        for line in self.current_fig.grid_lines: line.set_visible(True)
        self.current_fig.canvas.draw_idle()

    def on_axes_leave(self, event):
        if not hasattr(self.current_fig, 'grid_lines'): return
        for line in self.current_fig.grid_lines: line.set_visible(False)
        self.current_fig.hover_rect.set_visible(False)
        self.current_fig.canvas.draw_idle()

    def on_mouse_move(self, event):
        if not event.inaxes: return
        
        if self.plot_type == "Correlation Scatter" and hasattr(self.current_fig, 'clickable_points'):
            ax = self.current_fig.axes[0]
            pts = self.current_fig.clickable_points
            xy_data = np.c_[pts['area_um2'].values, pts['aspect_ratio'].values]
            xy_display = ax.transData.transform(xy_data)
            dists = np.sqrt((xy_display[:, 0] - event.x)**2 + (xy_display[:, 1] - event.y)**2)
            
            if len(dists) > 0:
                min_idx = np.argmin(dists)
                if dists[min_idx] < 15: 
                    hovered = pts.iloc[min_idx]
                    self.current_fig.hover_highlight.set_offsets([[hovered['area_um2'], hovered['aspect_ratio']]])
                    self.current_fig.hovered_point = hovered
                else:
                    self.current_fig.hover_highlight.set_offsets(np.empty((0, 2)))
                    self.current_fig.hovered_point = None
                self.current_fig.canvas.draw_idle()
            return
            
        if self.plot_type == "Tile Scatter" and hasattr(self.current_fig, 'tile_points'):
            ax = self.current_fig.axes[0]
            pts = self.current_fig.tile_points
            xy_data = np.c_[pts['density'].values, pts['area'].values]
            xy_display = ax.transData.transform(xy_data)
            dists = np.sqrt((xy_display[:, 0] - event.x)**2 + (xy_display[:, 1] - event.y)**2)
            
            if len(dists) > 0:
                min_idx = np.argmin(dists)
                if dists[min_idx] < 15:
                    hovered = pts.iloc[min_idx]
                    self.current_fig.hover_highlight.set_offsets([[hovered['density'], hovered['area']]])
                    self.current_fig.hovered_point = hovered
                else:
                    self.current_fig.hover_highlight.set_offsets(np.empty((0, 2)))
                    self.current_fig.hovered_point = None
                self.current_fig.canvas.draw_idle()
            return

        if hasattr(self.current_fig, 'hover_rect'):
            x, y = event.xdata, event.ydata
            
            x_edges = getattr(self.current_fig, 'image_x_edges', [])
            y_edges = getattr(self.current_fig, 'image_y_edges', [])
            if len(x_edges) == 0 or len(y_edges) == 0: return
            
            x_idx = np.searchsorted(x_edges, x) - 1
            y_idx = np.searchsorted(y_edges, y) - 1
            
            if 0 <= x_idx < len(x_edges)-1 and 0 <= y_idx < len(y_edges)-1:
                x_start, x_end = x_edges[x_idx], x_edges[x_idx+1]
                y_start, y_end = y_edges[y_idx], y_edges[y_idx+1]
                self.current_fig.hover_rect.set_xy((x_start, y_start))
                self.current_fig.hover_rect.set_width(x_end - x_start)
                self.current_fig.hover_rect.set_height(y_end - y_start)
                self.current_fig.hover_rect.set_visible(True)
            else:
                self.current_fig.hover_rect.set_visible(False)
            self.current_fig.canvas.draw_idle()

    def on_click(self, event):
        if not event.inaxes: return
        
        if self.plot_type == "Correlation Scatter" and getattr(self.current_fig, 'hovered_point', None) is not None:
            h = self.current_fig.hovered_point
            stack_name = str(h['source_stack']).replace('.tif', '').replace('.tiff', '')
            parts = stack_name.split('_')
            try:
                g_x, g_y = int(parts[-2]), int(parts[-1])
                best_pos = next((p for p in self.data_loader.positions if p['gridX'] == g_x and p['gridY'] == g_y), None)
                if best_pos:
                    TiffViewerPopup(self, self.data_loader, best_pos, target_hydrate_id=h['global_id'])
            except Exception as e:
                self.logger.error(f"Error parsing stack name {stack_name}: {e}")
            return
            
        if self.plot_type == "Tile Scatter" and getattr(self.current_fig, 'hovered_point', None) is not None:
            h = self.current_fig.hovered_point
            center_x, center_y = h['x_um'], h['y_um']
            
            x_vals = sorted(list(set(round(p['positionX'], 1) for p in self.data_loader.positions)))
            y_vals = sorted(list(set(round(p['positionY'], 1) for p in self.data_loader.positions)))
            dx = [x_vals[i+1] - x_vals[i] for i in range(len(x_vals)-1)]
            dy = [y_vals[i+1] - y_vals[i] for i in range(len(y_vals)-1)]
            step_x = np.median(dx) if dx else 1400.0
            step_y = np.median(dy) if dy else 1400.0
            width_um = step_x / 0.75
            height_um = step_y / 0.75
            
            best_pos = None
            min_dist = float('inf')
            for pos in self.data_loader.positions:
                pos_center_x = pos['positionX'] + width_um / 2
                pos_center_y = pos['positionY'] + height_um / 2
                dist = (pos_center_x - center_x)**2 + (pos_center_y - center_y)**2
                if dist < min_dist:
                    min_dist = dist
                    best_pos = pos
                    
            if best_pos:
                self.logger.info(f"Tile Scatter Click -> Opening Stack [{best_pos['gridX']},{best_pos['gridY']}]")
                TiffViewerPopup(self, self.data_loader, best_pos)
            return
            
        if hasattr(self.current_fig, 'hover_rect') and self.current_fig.hover_rect.get_visible():
            x, y = event.xdata, event.ydata
            
            x_edges = getattr(self.current_fig, 'image_x_edges', [])
            y_edges = getattr(self.current_fig, 'image_y_edges', [])
            if len(x_edges) == 0 or len(y_edges) == 0: return
            
            x_idx = np.searchsorted(x_edges, x) - 1
            y_idx = np.searchsorted(y_edges, y) - 1
            
            if 0 <= x_idx < len(x_edges)-1 and 0 <= y_idx < len(y_edges)-1:
                # Calculate the normalized cell center.
                center_x_norm = (x_edges[x_idx] + x_edges[x_idx+1]) / 2.0
                center_y_norm = (y_edges[y_idx] + y_edges[y_idx+1]) / 2.0
                
                # Restore the offsets to match the original HDF5 and JSON coordinates.
                offset_x = getattr(self.current_fig, 'offset_x', 0.0)
                offset_y = getattr(self.current_fig, 'offset_y', 0.0)
                
                center_x = center_x_norm + offset_x
                center_y = center_y_norm + offset_y
                
                x_vals = sorted(list(set(round(p['positionX'], 1) for p in self.data_loader.positions)))
                y_vals = sorted(list(set(round(p['positionY'], 1) for p in self.data_loader.positions)))
                dx = [x_vals[i+1] - x_vals[i] for i in range(len(x_vals)-1)]
                dy = [y_vals[i+1] - y_vals[i] for i in range(len(y_vals)-1)]
                step_x = np.median(dx) if dx else 1400.0
                step_y = np.median(dy) if dy else 1400.0
                width_um = step_x / 0.75
                height_um = step_y / 0.75
                
                best_pos = None
                min_dist = float('inf')
                for pos in self.data_loader.positions:
                    pos_center_x = pos['positionX'] + width_um / 2
                    pos_center_y = pos['positionY'] + height_um / 2
                    # Compare positions using the restored coordinates.
                    dist = (pos_center_x - center_x)**2 + (pos_center_y - center_y)**2
                    if dist < min_dist:
                        min_dist = dist
                        best_pos = pos
                        
                if best_pos:
                    TiffViewerPopup(self, self.data_loader, best_pos)

    def save_plot(self):
        if not self.current_fig: return
        filename = f"{self.plot_type.replace(' ', '_').lower()}_{self.current_params_str}.png"
        save_path = self.data_loader.out_dir / filename
        if save_path.exists() and not messagebox.askyesno("File Exists", f"'{filename}' already exists.\nOverwrite?"): return
        
        dpi_val = int(self.ui_vars['dpi'].get())
        facecolor = '#2b2b2b' if self.ui_vars['dark_mode'].get() else 'white'
        
        self.current_fig.savefig(save_path, dpi=dpi_val, bbox_inches='tight', facecolor=facecolor)
        messagebox.showinfo("Success", f"Plot saved to:\n{save_path}\n(DPI: {dpi_val})")