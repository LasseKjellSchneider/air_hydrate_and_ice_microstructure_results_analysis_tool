import customtkinter as ctk
import tkinter.messagebox as messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import logging
import traceback

from core.data_loader import DataLoader
from core.plot_engine import PlotEngine
from gui.popup_plot import PlotPopup

class AnalysisFrame(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller
        self.data_loader = None
        self.logger = logging.getLogger("HydrateAnalyzer")
        
        # --- HEADER ---
        header = ctk.CTkFrame(self, height=60, corner_radius=0)
        header.pack(fill="x", side="top")
        
        ctk.CTkButton(header, text="< Back to Start", width=120, height=40,
                      command=lambda: controller.show_frame("StartFrame")).pack(side="left", padx=20, pady=10)
                      
        self.title_label = ctk.CTkLabel(header, text="Single Dataset Analysis", font=("Arial", 24, "bold"))
        self.title_label.pack(side="left", padx=20, pady=10)
        
        self.btn_save_all = ctk.CTkButton(header, text="Save All Plots", width=150, height=40,
                      fg_color="#27ae60", hover_color="#2ecc71", state="disabled",
                      command=self.save_all_plots)
        self.btn_save_all.pack(side="right", padx=20, pady=10)
        
        # --- MAIN CONTAINER ---
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.main_container.grid_rowconfigure((0, 1, 2), weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)
        
        self.boxes = {}
        self.active_canvases = []
        self.setup_ui_sections()

    def on_show(self):
        """Called automatically by HydrateTab when this frame is displayed."""
        if self.data_loader is None:
            self.auto_load_workspace_data()
        
    def setup_ui_sections(self):
        self.boxes['A'] = self.create_category_box("Distribution & Scatter", row=0)
        self.boxes['B'] = self.create_category_box("Spatial Heatmaps", row=1)
        
        frame_c = ctk.CTkFrame(self.main_container, corner_radius=5)
        frame_c.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        ctk.CTkLabel(frame_c, text="Orientation & 1D Spatial Profiles", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=(5, 0))
        
        content_c = ctk.CTkFrame(frame_c, fg_color="transparent")
        content_c.pack(fill="both", expand=True, padx=5, pady=2)
        
        self.box_orient = ctk.CTkFrame(content_c, fg_color="transparent")
        self.box_orient.pack(side="left", fill="y", padx=5)
        
        self.box_profiles = ctk.CTkFrame(content_c, fg_color="transparent")
        self.box_profiles.pack(side="left", fill="both", expand=True, padx=5)

    def create_category_box(self, title_text, row):
        frame = ctk.CTkFrame(self.main_container, corner_radius=5)
        frame.grid(row=row, column=0, sticky="nsew", padx=5, pady=5)
        ctk.CTkLabel(frame, text=title_text, font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=(5, 0))
        plot_container = ctk.CTkFrame(frame, fg_color="transparent")
        plot_container.pack(fill="both", expand=True, padx=5, pady=2)
        return plot_container

    def auto_load_workspace_data(self):
        filepath = self.controller.base_dir / "analysis_results.json"
        
        if not filepath.exists():
            messagebox.showerror(
                "File Not Found", 
                f"Could not locate 'analysis_results.json' in workspace:\n{self.controller.base_dir}"
            )
            return
            
        try:
            self.data_loader = DataLoader(filepath)
            self.title_label.configure(text=f"Dataset: {self.data_loader.sample_name}")
            self.btn_save_all.configure(state="normal")
            
            self.render_previews()
            
        except Exception as e:
            error_msg = f"An error occurred while loading data:\n\n{str(e)}\n\nPlease check 'Analyzer_debug.txt' for details."
            messagebox.showerror("Loading Error", error_msg)
            self.logger.error(traceback.format_exc())

    def render_previews(self):
        self.logger.info("Start rendering Dashboard Previews...")
        try:
            for canvas in self.active_canvases: canvas.get_tk_widget().destroy()
            self.active_canvases.clear()
            
            if self.data_loader.df.empty: 
                self.logger.warning("DataFrame empty, aborting render.")
                return
            
            DPI = 80
            # --- BOX A ---
            for plot_type in ["Area Distribution", "Aspect Ratio Dist", "Correlation Scatter", "Tile Scatter", "Tile Heatmap"]:
                self.logger.debug(f"Rendering: {plot_type}")
                if "Dist" in plot_type:
                    fig = PlotEngine.create_dist(self.data_loader.df, "area_um2" if "Area" in plot_type else "aspect_ratio", figsize=(2.4, 1.8), dpi=DPI, is_preview=True)
                elif "Correlation" in plot_type:
                    fig = PlotEngine.create_scatter_corr(self.data_loader.df, figsize=(2.4, 1.8), dpi=DPI, is_preview=True)
                elif plot_type == "Tile Scatter":
                    fig = PlotEngine.create_scatter_tiles(self.data_loader.df, self.data_loader.sample_height_um, self.data_loader.global_density, self.data_loader.global_med_area, figsize=(2.4, 1.8), dpi=DPI, is_preview=True)
                elif plot_type == "Tile Heatmap":
                    fig = PlotEngine.create_tile_heatmap(self.data_loader.df, self.data_loader.sample_height_um, figsize=(2.4, 1.8), dpi=DPI, is_preview=True)
                self._embed(fig, self.boxes['A'], plot_type, side="left")
                
            # --- BOX B ---
            for pt in ["Density Heatmap", "Area Heatmap", "Aspect Heatmap", "Relation Heatmap"]:
                self.logger.debug(f"Rendering: {pt}")
                fig = PlotEngine.create_heatmap(self.data_loader.df, self.data_loader.sample_height_um, self.data_loader.global_density, self.data_loader.global_med_area, map_type=pt, figsize=(3, 1.8), dpi=DPI, is_preview=True)
                self._embed(fig, self.boxes['B'], pt, side="left")

            # --- BOX C (Orientation) ---
            self.logger.debug("Rendering: Orientations")
            fig_r = PlotEngine.create_rose(self.data_loader.df, is_half=False, figsize=(2, 1.4), dpi=DPI, is_preview=True)
            self._embed(fig_r, self.box_orient, "Rose Diagram", side="top")
            
            fig_hr = PlotEngine.create_rose(self.data_loader.df, is_half=True, figsize=(2, 1.1), dpi=DPI, is_preview=True)
            self._embed(fig_hr, self.box_orient, "Half-Rose Diagram", side="top")
            
            fig_hr_heat = PlotEngine.create_half_rose_heatmap(self.data_loader.df, figsize=(2, 1.1), dpi=DPI, is_preview=True)
            self._embed(fig_hr_heat, self.box_orient, "Half-Rose Heatmap", side="top")

            # --- BOX C (Profiles) ---
            for i in range(3): self.box_profiles.grid_columnconfigure(i, weight=1)
            for i in range(2): self.box_profiles.grid_rowconfigure(i, weight=1)
            
            prof_map = [
                ("Count X", 'x_um', 'count', 0, 0), ("Area X", 'x_um', 'area', 0, 1), ("Aspect X", 'x_um', 'aspect', 0, 2),
                ("Count Y", 'y_um', 'count', 1, 0), ("Area Y", 'y_um', 'area', 1, 1), ("Aspect Y", 'y_um', 'aspect', 1, 2)
            ]
            for pt, col, metric, grid_r, grid_c in prof_map:
                self.logger.debug(f"Rendering: {pt}")
                fig = PlotEngine.create_profile(self.data_loader.df, col, metric, figsize=(2, 1.7), dpi=DPI, is_preview=True)
                self._embed(fig, self.box_profiles, pt, row=grid_r, col=grid_c)
                
            self.logger.info("All previews rendered successfully.")

        except Exception as e:
            self.logger.error(f"CRITICAL ERROR in render_previews: {e}")
            self.logger.error(traceback.format_exc())
            messagebox.showerror("Rendering Error", f"An error occurred during visualization:\n\n{str(e)}\n\nPlease check 'Analyzer_debug.txt' for details.")

    def _embed(self, fig, container, plot_type, side=None, row=None, col=None):
        canvas = FigureCanvasTkAgg(fig, master=container)
        canvas.draw()
        w = canvas.get_tk_widget()
        if side: w.pack(side=side, fill="both", expand=True, padx=2, pady=2)
        else: w.grid(row=row, column=col, sticky="nsew", padx=2, pady=2)
        w.configure(cursor="hand2")
        
        # Connect click event to open the popup plot
        canvas.mpl_connect('button_press_event', lambda e: PlotPopup(self, self.data_loader, plot_type))
        self.active_canvases.append(canvas)

    def save_all_plots(self):
        if not self.data_loader or self.data_loader.df.empty: return
        self.btn_save_all.configure(text="Saving...", state="disabled")
        self.update_idletasks()
        
        try:
            DF = self.data_loader.df
            sh_um = self.data_loader.sample_height_um
            g_dens = self.data_loader.global_density
            g_area = self.data_loader.global_med_area
            out = self.data_loader.out_dir
            figsize, dpi = (8, 6), 300
            
            for pt in ["Area Distribution", "Aspect Ratio Dist"]:
                fig = PlotEngine.create_dist(DF, "area_um2" if "Area" in pt else "aspect_ratio", bins=50, figsize=figsize, is_preview=False)
                fig.savefig(out / f"{pt.replace(' ', '_').lower()}.png", dpi=dpi, bbox_inches='tight')
                plt.close(fig)
                
            fig = PlotEngine.create_scatter_corr(DF, figsize=figsize, is_preview=False)
            fig.savefig(out / "correlation_scatter.png", dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            
            fig = PlotEngine.create_scatter_tiles(DF, sh_um, g_dens, g_area, figsize=figsize, is_preview=False)
            fig.savefig(out / "tile_scatter.png", dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            
            fig = PlotEngine.create_tile_heatmap(DF, sh_um, figsize=figsize, is_preview=False)
            fig.savefig(out / "tile_heatmap.png", dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            
            for pt in ["Density Heatmap", "Area Heatmap", "Aspect Heatmap", "Relation Heatmap"]:
                fig = PlotEngine.create_heatmap(DF, sh_um, g_dens, g_area, map_type=pt, tile_mult=1, blur_val=3, figsize=(7, 6), is_preview=False, positions=self.data_loader.positions)
                fig.savefig(out / f"{pt.replace(' ', '_').lower()}.png", dpi=dpi, bbox_inches='tight')
                plt.close(fig)
                
            fig = PlotEngine.create_rose(DF, is_half=False, figsize=(6,6), is_preview=False)
            fig.savefig(out / "rose_diagram.png", dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            
            fig = PlotEngine.create_rose(DF, is_half=True, figsize=(6,6), is_preview=False)
            fig.savefig(out / "half_rose_diagram.png", dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            
            fig = PlotEngine.create_half_rose_heatmap(DF, figsize=(6,6), is_preview=False)
            fig.savefig(out / "half_rose_heatmap.png", dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            
            for pt in ["Count X", "Area X", "Aspect X", "Count Y", "Area Y", "Aspect Y"]:
                col = 'x_um' if 'X' in pt else 'y_um'
                metric = pt.split()[0].lower()
                fig = PlotEngine.create_profile(DF, col, metric, bin_width=100.0, figsize=figsize, is_preview=False)
                fig.savefig(out / f"profile_{pt.replace(' ', '_').lower()}.png", dpi=dpi, bbox_inches='tight')
                plt.close(fig)
                
            messagebox.showinfo("Success", f"Successfully saved 18 High-Res plots to:\n\n{out}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save plots:\n{e}")
            self.logger.error(f"Error saving plots: {e}")
        finally:
            self.btn_save_all.configure(text="Save All Plots", state="normal")