import customtkinter as ctk
import numpy as np
import tifffile
import cv2
from PIL import Image
import logging

class TiffViewerPopup(ctk.CTkToplevel):
    def __init__(self, parent, data_loader, pos_info, target_hydrate_id=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self.pos_info = pos_info
        self.target_hydrate_id = target_hydrate_id
        
        self.logger = logging.getLogger("HydrateAnalyzer")
        
        self.grid_x = pos_info['gridX']
        self.grid_y = pos_info['gridY']
        self.pos_x = pos_info['positionX']
        self.pos_y = pos_info['positionY']
        
        title_addon = f" | Hydrate ID: {target_hydrate_id}" if target_hydrate_id else ""
        self.title(f"TIFF Viewer - Grid: [{self.grid_x}, {self.grid_y}]{title_addon}")
        self.geometry("900x950")
        self.attributes('-topmost', True)
        self.focus_set()
        self.focus_force()
        
        self.stack = None
        self.current_layer = 0
        self.max_layers = 1
        self.source_stack_name = ""
        
        self.controls_frame = ctk.CTkFrame(self)
        self.controls_frame.pack(fill="x", padx=10, pady=10)
        
        self.layer_label = ctk.CTkLabel(self.controls_frame, text="Layer: 0", font=("Arial", 14, "bold"), width=80)
        self.layer_label.pack(side="left", padx=10)
        
        self.slider = ctk.CTkSlider(self.controls_frame, from_=0, to=1, command=self.on_slider_move)
        self.slider.pack(side="left", fill="x", expand=True, padx=20)
        
        self.show_overlay = ctk.BooleanVar(value=True)
        self.cb_overlay = ctk.CTkCheckBox(self.controls_frame, text="Show Detected Hydrates", variable=self.show_overlay, command=self.update_image)
        self.cb_overlay.pack(side="right", padx=10)
        
        self.image_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.image_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        self.image_label = ctk.CTkLabel(self.image_frame, text="Searching and loading image...")
        self.image_label.pack(expand=True)
        
        self.bind("<MouseWheel>", self.on_mouse_wheel)
        self.bind("<Up>", lambda e: self.scroll_layer(1))
        self.bind("<Right>", lambda e: self.scroll_layer(1))
        self.bind("<Down>", lambda e: self.scroll_layer(-1))
        self.bind("<Left>", lambda e: self.scroll_layer(-1))
        self.bind("<Configure>", self.on_resize)

        self.load_tiff()

    def load_tiff(self):
        found_file = None
        for f in self.data_loader.base_dir.glob("*.tif"):
            parts = f.stem.split("_")
            if len(parts) >= 2:
                try:
                    if int(parts[-2]) == self.grid_x and int(parts[-1]) == self.grid_y:
                        found_file = f
                        break
                except ValueError: pass
                    
        if not found_file:
            self.image_label.configure(text=f"No matching TIFF found for [{self.grid_x}, {self.grid_y}]!")
            return
            
        self.source_stack_name = found_file.stem
        
        try:
            with tifffile.TiffFile(found_file) as tif:
                pages = list(tif.pages)
                if len(pages) > 1:
                    self.stack = np.stack([p.asarray() for p in pages], axis=0)
                else:
                    self.stack = pages[0].asarray()
                    if self.stack.ndim == 2:
                        self.stack = np.expand_dims(self.stack, axis=0)
                    elif self.stack.ndim == 3:
                        min_axis = int(np.argmin(self.stack.shape))
                        if min_axis != 0 and self.stack.shape[min_axis] > 4:
                            self.stack = np.moveaxis(self.stack, min_axis, 0)
                
            self.max_layers = self.stack.shape[0]
            
            # Select the target layer when a hydrate ID is provided.
            if self.target_hydrate_id is not None:
                target_row = self.data_loader.df[self.data_loader.df['global_id'] == self.target_hydrate_id]
                if not target_row.empty:
                    self.current_layer = int(target_row.iloc[0]['layer'])
                    self.current_layer = min(self.current_layer, self.max_layers - 1)
            
            if self.max_layers > 1:
                self.slider.configure(from_=0, to=self.max_layers-1, number_of_steps=self.max_layers-1, state="normal")
            else:
                self.slider.configure(state="disabled")
                
            self.slider.set(self.current_layer)
            self.update_image()
        except Exception as e:
            self.image_label.configure(text=f"Error loading image:\n{e}")

    def scroll_layer(self, step):
        if self.stack is None or self.max_layers <= 1: return
        new_layer = max(0, min(self.max_layers - 1, self.current_layer + step))
        if new_layer != self.current_layer:
            self.slider.set(new_layer)
            self.on_slider_move(new_layer)

    def on_slider_move(self, value):
        self.current_layer = int(round(float(value)))
        self.layer_label.configure(text=f"Layer: {self.current_layer}")
        self.update_image()
        
    def on_mouse_wheel(self, event):
        step = -1 if event.delta > 0 else 1
        self.scroll_layer(step)
        
    def on_resize(self, event):
        if event.widget == self.image_frame:
            self.after(50, self.update_image)

    def update_image(self):
        if self.stack is None: return
        
        img_layer = self.stack[self.current_layer].copy()
        
        if img_layer.ndim == 3 and img_layer.shape[-1] in [3, 4]:
            code = cv2.COLOR_RGBA2GRAY if img_layer.shape[-1] == 4 else cv2.COLOR_RGB2GRAY
            img_layer = cv2.cvtColor(img_layer, code)
                
        img_gray = cv2.normalize(img_layer, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        
        if self.show_overlay.get():
            df = self.data_loader.df
            clean_source = df['source_stack'].astype(str).str.replace(r'\.tiff?$', '', case=False, regex=True)
            hyds = df[(clean_source == self.source_stack_name) & (df['layer'] == self.current_layer)]
            
            pixel_scale = getattr(self.data_loader, 'pixel_scale', 1.41)
            
            for _, h in hyds.iterrows():
                if h.get('center_x_px', 0) > 0:
                    cx_px = int(h['center_x_px'])
                    cy_px = int(h['center_y_px'])
                else:
                    cx_px = int((h['x_um'] - self.pos_x) / pixel_scale)
                    cy_px = int((h['y_um'] - self.pos_y) / pixel_scale)
                
                axes_x = max(1, int((h['major_axis_um'] / pixel_scale) / 2))
                axes_y = max(1, int((h['minor_axis_um'] / pixel_scale) / 2))
                angle = h['angle_deg'] + 90
                
                # Highlight the selected hydrate, if applicable.
                thickness = 2
                if self.target_hydrate_id is not None and h.get('global_id') == self.target_hydrate_id:
                    color = (255, 0, 255) # Magenta highlight in BGR format.
                    thickness = 5
                else:
                    path_str = str(h['pipeline_path']).lower()
                    if "good" in path_str: color = (0, 255, 0)
                    elif "rescued" in path_str or "moderate" in path_str: color = (0, 215, 255)
                    else: color = (0, 0, 255)
                    
                cv2.ellipse(img_bgr, (cx_px, cy_px), (axes_x, axes_y), angle, 0, 360, color, thickness)

        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        frame_w = self.image_frame.winfo_width()
        frame_h = self.image_frame.winfo_height()
        
        if frame_w > 10 and frame_h > 10:
            ctk_img = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(frame_w-20, frame_h-20))
            self.image_label.configure(image=ctk_img, text="")