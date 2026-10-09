import customtkinter as ctk
import tkinter.messagebox as messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from core.compare_engine import ComparePlotEngine

class ComparePopup(ctk.CTkToplevel):
    def __init__(self, parent, data_loaders, plot_type):
        super().__init__(parent)
        self.data_loaders = data_loaders
        self.plot_type = plot_type
        
        self.title(f"Comparison: {plot_type}")
        h = 600 if len(data_loaders) == 2 else 850
        self.geometry(f"1000x{h}")
        self.attributes('-topmost', True) 
        
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.canvas_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.canvas_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        self.current_fig = None
        self.canvas_widget = None
        self.current_params_str = "default"
        
        self.setup_controls()
        self.update_plot()
        
    def setup_controls(self):
        ctk.CTkLabel(self.control_frame, text="Sync Parameters", font=("Arial", 18, "bold")).pack(pady=20)
        self.sliders = {}
        
        if "Dist" in self.plot_type:
            lbl = ctk.CTkLabel(self.control_frame, text="Bins (Resolution): 50")
            lbl.pack(pady=(10, 0))
            slider = ctk.CTkSlider(self.control_frame, from_=10, to=200, number_of_steps=190, command=lambda v: lbl.configure(text=f"Bins: {int(v)}"))
            slider.set(50); slider.pack(pady=10, padx=20, fill="x")
            self.sliders['bins'] = slider
            
        elif "Heatmap" in self.plot_type and "Rose" not in self.plot_type:
            lbl_tile = ctk.CTkLabel(self.control_frame, text="Tile Size Multiplier: 1x (500µm)")
            lbl_tile.pack(pady=(10, 0))
            sl_tile = ctk.CTkSlider(self.control_frame, from_=1, to=5, number_of_steps=4, command=lambda v: lbl_tile.configure(text=f"Tile Multiplier: {int(v)}x ({int(v)*500}µm)"))
            sl_tile.set(1); sl_tile.pack(pady=10, padx=20, fill="x")
            self.sliders['tile_mult'] = sl_tile
            
            lbl_blur = ctk.CTkLabel(self.control_frame, text="Blur Radius: 3x")
            lbl_blur.pack(pady=(20, 0))
            sl_blur = ctk.CTkSlider(self.control_frame, from_=1, to=9, number_of_steps=8, command=lambda v: self._force_odd_blur(v, lbl_blur))
            sl_blur.set(3); sl_blur.pack(pady=10, padx=20, fill="x")
            self.sliders['blur'] = sl_blur
            
            if self.plot_type == "Relation Heatmap":
                # Render the relation-score equation with Matplotlib MathText.
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
                
                ax_math.text(0.5, 0.5, latex_str, color='#3498db', fontsize=14, 
                             ha='center', va='center')
                
                canvas_math = FigureCanvasTkAgg(fig_math, master=self.control_frame)
                canvas_math.draw()
                canvas_math.get_tk_widget().pack(pady=(20, 0), padx=10, fill="x")
                
        elif self.plot_type == "Half-Rose Heatmap":
            lbl_angle = ctk.CTkLabel(self.control_frame, text="Angle Bin (deg): 5")
            lbl_angle.pack(pady=(10, 0))
            sl_angle = ctk.CTkSlider(self.control_frame, from_=1, to=30, number_of_steps=29, command=lambda v: lbl_angle.configure(text=f"Angle Bin (deg): {int(v)}"))
            sl_angle.set(5); sl_angle.pack(pady=10, padx=20, fill="x")
            self.sliders['angle_bin'] = sl_angle

            lbl_aspect = ctk.CTkLabel(self.control_frame, text="Aspect Bin (x/30): 5")
            lbl_aspect.pack(pady=(10, 0))
            sl_aspect = ctk.CTkSlider(self.control_frame, from_=1, to=30, number_of_steps=29, command=lambda v: lbl_aspect.configure(text=f"Aspect Bin (x/30): {int(v)}"))
            sl_aspect.set(5); sl_aspect.pack(pady=10, padx=20, fill="x")
            self.sliders['aspect_bin'] = sl_aspect
            
        elif self.plot_type in ["Count X", "Area X", "Aspect X", "Count Y", "Area Y", "Aspect Y"]:
            lbl = ctk.CTkLabel(self.control_frame, text="Bin Width (µm): 500")
            lbl.pack(pady=(10, 0))
            slider = ctk.CTkSlider(self.control_frame, from_=100, to=2000, number_of_steps=19, command=lambda v: lbl.configure(text=f"Bin Width: {int(v)} µm"))
            slider.set(500); slider.pack(pady=10, padx=20, fill="x")
            self.sliders['bin_width'] = slider

        ctk.CTkButton(self.control_frame, text="Update Plots", height=40, font=("Arial", 16, "bold"),
                      command=self.update_plot).pack(pady=30, padx=20, fill="x")
        ctk.CTkButton(self.control_frame, text="Save Combined Plot", height=40, fg_color="#27ae60", hover_color="#2ecc71",
                      command=self.save_plot).pack(pady=10, padx=20, fill="x")

    def _force_odd_blur(self, val, label):
        val = int(val); val += 1 if val % 2 == 0 else 0
        self.sliders['blur'].set(val); label.configure(text=f"Blur Radius: {val}x")

    def update_plot(self):
        if self.current_fig:
            plt.close(self.current_fig)
        if self.canvas_widget: 
            self.canvas_widget.destroy()
            
        figsize = (8, 3 * len(self.data_loaders))
        
        if "Dist" in self.plot_type:
            b = int(self.sliders['bins'].get())
            self.current_fig = ComparePlotEngine.create_compare_dist(self.data_loaders, "area_um2" if "Area" in self.plot_type else "aspect_ratio", bins=b, figsize=figsize, is_preview=False)
            self.current_params_str = f"bins_{b}"
            
        elif "Correlation" in self.plot_type:
            self.current_fig = ComparePlotEngine.create_compare_scatter_corr(self.data_loaders, figsize=figsize, is_preview=False)
        elif "Tile Scatter" in self.plot_type:
            self.current_fig = ComparePlotEngine.create_compare_scatter_tiles(self.data_loaders, figsize=figsize, is_preview=False)
            
        elif "Heatmap" in self.plot_type and "Rose" not in self.plot_type:
            tm, bv = int(self.sliders.get('tile_mult', ctk.IntVar(value=1)).get()), int(self.sliders.get('blur', ctk.IntVar(value=3)).get())
            self.current_fig = ComparePlotEngine.create_compare_heatmap(self.data_loaders, self.plot_type, tile_mult=tm, blur_val=bv, figsize=figsize, is_preview=False)
            self.current_params_str = f"tile{tm}x_blur{bv}x"
            
        elif self.plot_type == "Half-Rose Heatmap":
            a_bin = int(self.sliders.get('angle_bin', ctk.IntVar(value=5)).get())
            asp_bin = int(self.sliders.get('aspect_bin', ctk.IntVar(value=5)).get())
            self.current_fig = ComparePlotEngine.create_compare_half_rose_heatmap(self.data_loaders, angle_bin_deg=a_bin, aspect_bin_num=asp_bin, figsize=figsize, is_preview=False)
            self.current_params_str = f"angle{a_bin}_aspect{asp_bin}of30"
            
        elif "Rose" in self.plot_type:
            is_half = "Half" in self.plot_type
            self.current_fig = ComparePlotEngine.create_compare_rose(self.data_loaders, is_half=is_half, figsize=figsize, is_preview=False)
            self.current_params_str = "default"
            
        elif self.plot_type in ["Count X", "Area X", "Aspect X", "Count Y", "Area Y", "Aspect Y"]:
            bw = int(self.sliders['bin_width'].get())
            col = 'x_um' if 'X' in self.plot_type else 'y_um'
            metric = self.plot_type.split()[0].lower()
            self.current_fig = ComparePlotEngine.create_compare_profile(self.data_loaders, col, metric, bin_width=bw, figsize=figsize, is_preview=False)
            self.current_params_str = f"binwidth_{bw}um"
            
        canvas = FigureCanvasTkAgg(self.current_fig, master=self.canvas_frame)
        canvas.draw()
        self.canvas_widget = canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

    def save_plot(self):
        if not self.current_fig: return
        filename = f"comparison_{self.plot_type.replace(' ', '_').lower()}_{self.current_params_str}.png"
        
        # Save to the output directory of the first dataset
        save_path = self.data_loaders[0].out_dir / filename
        if save_path.exists() and not messagebox.askyesno("File Exists", f"'{filename}' already exists.\nOverwrite?"): return
        
        self.current_fig.savefig(save_path, dpi=300, bbox_inches='tight')
        messagebox.showinfo("Success", f"Plot saved to:\n{save_path}")