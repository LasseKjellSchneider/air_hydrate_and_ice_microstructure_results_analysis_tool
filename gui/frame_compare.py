import customtkinter as ctk
import tkinter.messagebox as messagebox
from tkinter import filedialog
import pathlib
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from core.data_loader import DataLoader
from core.compare_engine import ComparePlotEngine
from gui.popup_compare import ComparePopup

class CompareFrame(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller
        
        self.selected_folders = []
        self.data_loaders = []
        self.active_canvases = []
        
        # --- HEADER ---
        self.header = ctk.CTkFrame(self, height=60, corner_radius=0)
        self.header.pack(fill="x", side="top")
        
        ctk.CTkButton(self.header, text="< Back to Start", width=120, height=40,
                      command=lambda: controller.show_frame("StartFrame")).pack(side="left", padx=20, pady=10)
                      
        self.title_label = ctk.CTkLabel(self.header, text="Compare Datasets", font=("Arial", 24, "bold"))
        self.title_label.pack(side="left", padx=20, pady=10)
        
        self.btn_back_setup = ctk.CTkButton(self.header, text="New Comparison", width=150, height=40, fg_color="#e67e22", hover_color="#d35400", command=self.show_setup)
        
        # --- SETUP CONTAINER ---
        self.setup_container = ctk.CTkFrame(self, fg_color="transparent")
        self.setup_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(self.setup_container, text="Select 2 or 3 Datasets to compare", font=("Arial", 18, "bold")).pack(pady=10)
        
        self.btn_add = ctk.CTkButton(self.setup_container, text="+ Add Dataset", font=("Arial", 14, "bold"), height=40, command=self.add_folder)
        self.btn_add.pack(pady=10)
        
        self.list_frame = ctk.CTkScrollableFrame(self.setup_container, fg_color="#1e1e1e", width=600, height=200)
        self.list_frame.pack(pady=10)
        
        self.btn_start = ctk.CTkButton(self.setup_container, text="Start Comparison", height=50, font=("Arial", 16, "bold"), 
                                       fg_color="#27ae60", hover_color="#2ecc71", state="disabled", command=self.start_compare)
        self.btn_start.pack(pady=20)
        
        self.lbl_loading = ctk.CTkLabel(self.setup_container, text="", font=("Arial", 16), text_color="#f39c12")
        self.lbl_loading.pack()

        # --- DASHBOARD CONTAINER ---
        self.dashboard_container = ctk.CTkScrollableFrame(self, fg_color="transparent")

    def on_show(self):
        """Called automatically when switching to this tab. Auto-adds the workspace folder as Dataset 1."""
        if not self.selected_folders and self.controller.base_dir:
            json_path = self.controller.base_dir / "analysis_results.json"
            if json_path.exists():
                self.selected_folders.append(json_path)
                self.refresh_list_ui()
                self.check_ready_state()

    # --- SETUP LOGIC ---
    def add_folder(self):
        if len(self.selected_folders) >= 3:
            messagebox.showwarning("Limit Reached", "You can compare a maximum of 3 datasets.")
            return
            
        folder = filedialog.askdirectory(title="Select Preprocessed Data Folder")
        if not folder: return
        p = pathlib.Path(folder)
        
        json_path = p / "analysis_results.json"
        if not json_path.exists():
            messagebox.showwarning("Warning", "No 'analysis_results.json' found in this folder.")
            return
            
        if json_path in self.selected_folders:
            messagebox.showinfo("Duplicate", "This dataset is already in the comparison list.")
            return
            
        self.selected_folders.append(json_path)
        self.refresh_list_ui()
        self.check_ready_state()

    def refresh_list_ui(self):
        for widget in self.list_frame.winfo_children(): widget.destroy()
        for i, json_path in enumerate(self.selected_folders):
            row = ctk.CTkFrame(self.list_frame, fg_color="#2b2b2b", corner_radius=5)
            row.pack(fill="x", pady=5, padx=5)
            
            label_text = f"{i+1}. {json_path.parent.name}"
            if i == 0:
                label_text += " (Current Workspace)"
                
            ctk.CTkLabel(row, text=label_text, font=("Arial", 14, "bold")).pack(side="left", padx=15, pady=10)
            ctk.CTkButton(row, text="X", width=30, fg_color="#e74c3c", hover_color="#c0392b", command=lambda idx=i: self.remove_folder(idx)).pack(side="right", padx=10)

    def start_compare(self):
        self.btn_start.configure(state="disabled")
        self.lbl_loading.configure(text="Loading JSONs and generating plots... Please wait (~5-10s)")
        self.update_idletasks()
        
        def worker():
            try:
                self.data_loaders = [DataLoader(json_path) for json_path in self.selected_folders]
                self.controller.after(0, self.render_dashboard)
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                self.controller.after(0, lambda: messagebox.showerror("Load Error", str(e)))
                self.controller.after(0, lambda: self.btn_start.configure(state="normal"))
                self.controller.after(0, lambda: self.lbl_loading.configure(text=""))
                
        threading.Thread(target=worker, daemon=True).start()

    def remove_folder(self, index):
        self.selected_folders.pop(index)
        self.refresh_list_ui()
        self.check_ready_state()

    def check_ready_state(self):
        if 2 <= len(self.selected_folders) <= 3:
            self.btn_start.configure(state="normal")
        else:
            self.btn_start.configure(state="disabled")

    def show_setup(self):
        self.dashboard_container.pack_forget()
        self.btn_back_setup.pack_forget()
        self.setup_container.pack(fill="both", expand=True, padx=20, pady=20)
        self.title_label.configure(text="Compare Datasets")
        for canvas in self.active_canvases: canvas.get_tk_widget().destroy()
        self.active_canvases.clear()

    # --- DASHBOARD LOGIC ---
    def render_dashboard(self):
        self.setup_container.pack_forget()
        self.btn_back_setup.pack(side="right", padx=20, pady=10)
        
        # Build Legend Header
        legend = " | ".join([f"{i+1}: {l.sample_name}" for i, l in enumerate(self.data_loaders)])
        self.title_label.configure(text=f"Comparison -> {legend}")
        
        self.dashboard_container.pack(fill="both", expand=True, padx=10, pady=5)
        
        for w in self.dashboard_container.winfo_children(): w.destroy()
        self.active_canvases.clear()
        
        # --- STRUCTURED LAYOUT ---
        rows_layout = [
            ("1. Distributions", ["Area Distribution", "Aspect Ratio Dist"]),
            ("2. Scatters & Tile Heatmap", ["Correlation Scatter", "Tile Scatter", "Tile Heatmap"]),
            ("3. Heatmaps (Primary)", ["Density Heatmap", "Area Heatmap", "Aspect Heatmap"]),
            ("4. Heatmaps (Secondary)", ["Relation Heatmap"]),
            ("5. Spatial Profiles (X-Axis)", ["Count X", "Area X", "Aspect X"]),
            ("6. Spatial Profiles (Y-Axis)", ["Count Y", "Area Y", "Aspect Y"]),
            ("7. Orientations & Angles", ["Rose Diagram", "Half-Rose Diagram", "Half-Rose Heatmap"])
        ]
        
        DPI = 80
        figsize = (3.5, 3.5) if len(self.data_loaders) == 3 else (4, 2)
        prof_fig = (3.5, 2.5) if len(self.data_loaders) == 3 else (3, 1.5)
        
        for row_title, plots in rows_layout:
            row_frame = ctk.CTkFrame(self.dashboard_container, corner_radius=5)
            row_frame.pack(fill="x", padx=5, pady=5)
            
            ctk.CTkLabel(row_frame, text=row_title, font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=(5, 0))
            
            plot_container = ctk.CTkFrame(row_frame, fg_color="transparent")
            plot_container.pack(fill="both", expand=True, padx=5, pady=2)
            
            for pt in plots:
                if "Dist" in pt: 
                    fig = ComparePlotEngine.create_compare_dist(self.data_loaders, "area_um2" if "Area" in pt else "aspect_ratio", figsize=figsize, dpi=DPI, is_preview=True)
                elif "Correlation" in pt: 
                    fig = ComparePlotEngine.create_compare_scatter_corr(self.data_loaders, figsize=figsize, dpi=DPI, is_preview=True)
                elif pt == "Tile Scatter": 
                    fig = ComparePlotEngine.create_compare_scatter_tiles(self.data_loaders, figsize=figsize, dpi=DPI, is_preview=True)
                elif "Heatmap" in pt and "Rose" not in pt: 
                    fig = ComparePlotEngine.create_compare_heatmap(self.data_loaders, map_type=pt, figsize=figsize, dpi=DPI, is_preview=True)
                elif pt in ["Count X", "Area X", "Aspect X", "Count Y", "Area Y", "Aspect Y"]:
                    col = 'x_um' if 'X' in pt else 'y_um'
                    metric = pt.split()[0].lower()
                    fig = ComparePlotEngine.create_compare_profile(self.data_loaders, col, metric, bin_width=500.0, figsize=prof_fig, dpi=DPI, is_preview=True)
                elif pt == "Rose Diagram":
                    fig = ComparePlotEngine.create_compare_rose(self.data_loaders, is_half=False, figsize=figsize, dpi=DPI, is_preview=True)
                elif pt == "Half-Rose Diagram":
                    fig = ComparePlotEngine.create_compare_rose(self.data_loaders, is_half=True, figsize=figsize, dpi=DPI, is_preview=True)
                elif pt == "Half-Rose Heatmap":
                    fig = ComparePlotEngine.create_compare_half_rose_heatmap(self.data_loaders, figsize=figsize, dpi=DPI, is_preview=True)
                
                self._embed(fig, plot_container, pt)
            
        self.btn_start.configure(state="normal")
        self.lbl_loading.configure(text="")

    def _embed(self, fig, container, plot_type):
        canvas = FigureCanvasTkAgg(fig, master=container)
        canvas.draw()
        w = canvas.get_tk_widget()
        w.pack(side="left", fill="both", expand=True, padx=2, pady=2)
        w.configure(cursor="hand2")
        canvas.mpl_connect('button_press_event', lambda e: ComparePopup(self, self.data_loaders, plot_type))
        self.active_canvases.append(canvas)