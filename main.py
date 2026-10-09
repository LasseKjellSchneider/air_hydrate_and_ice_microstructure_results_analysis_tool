import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
import pathlib
import traceback

# Import the GUI tab modules.
from gui.tab1_hydrate import HydrateTab
from gui.tab2_planes import PlanesTab
from gui.tab3_mesh import MeshTab
from gui.tab4_caxis import CAxisTab

# Import the summary processing engine.
from core.summary_engine import SummaryEngine

# Configure the application-wide appearance.
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class MasterApp(ctk.CTk):
    def __init__(self, base_dir, master_h5_path, active_summary_path):
        super().__init__()
        self.title("Hydrate Analysis & 3D Reconstruction Dashboard")
        self.geometry("1600x900")
        
        # Store the workspace paths for use by all tabs.
        self.base_dir = base_dir
        self.master_h5_path = master_h5_path
        self.active_summary_path = active_summary_path  # Canonical path for the active summary.
        
        # Derive the sample name from the master file, falling back to the workspace name.
        raw_name = self.master_h5_path.stem.replace("_master", "").replace("master", "").strip("_-")
        self.sample_name = raw_name if raw_name else self.base_dir.name

        # Initialize the tab views.
        self._build_tabs()

    def _build_tabs(self):
        # Initialize the tabview container
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=(10, 10))

        # Add individual tabs without numbering
        self.tab_hydrate = self.tabview.add("Hydrate Analysis")
        self.tab_planes = self.tabview.add("Plane Analysis")
        self.tab_mesh = self.tabview.add("3D Reconstruction")
        self.tab_caxis = self.tabview.add("C-Axis & Hydrates")

        # Initialize GUI logic for each tab
        self.hydrate_gui = HydrateTab(self.tab_hydrate, self.base_dir)
        self.planes_gui = PlanesTab(self.tab_planes, self.master_h5_path)
        self.mesh_gui = MeshTab(self.tab_mesh, self.master_h5_path)
        self.caxis_gui = CAxisTab(self.tab_caxis, self.base_dir)

    def set_active_summary(self, new_summary_path):
        """Update the active summary after a new filter is applied in Tab 3."""
        self.active_summary_path = new_summary_path
        self.reload_all_data()

    def reload_all_data(self):
        """Reload data in dynamic tabs from the active summary."""
        if self.active_summary_path:
            print(f"MasterApp: Reloading data across all tabs from {self.active_summary_path.name}...")
        
        # Reload the C-axis tab when it supports workspace data loading.
        if hasattr(self.caxis_gui, 'load_workspace_data'):
            self.caxis_gui.load_workspace_data()
            
        # Reload the hydrate analysis tab from the active summary.
        if hasattr(self.hydrate_gui, 'load_workspace_data'):
            self.hydrate_gui.load_workspace_data()


def main():
    # Create a hidden root window to host the startup file dialog.
    temp_root = ctk.CTk()
    temp_root.withdraw()

    # Ask the user to select a dataset directory at startup.
    target_dir = filedialog.askdirectory(
        title="Please select the dataset folder containing JSONs and the master.h5 file"
    )
    
    if not target_dir:
        print("No directory selected. Exiting application.")
        temp_root.destroy()
        return

    base_dir = pathlib.Path(target_dir)

    # Find an HDF5 master file in the selected directory.
    h5_files = list(base_dir.glob("*master*.h5"))
    
    if not h5_files:
        messagebox.showerror(
            "Missing Master File",
            f"Could not find any '.h5' file containing 'master' in the name within:\n{base_dir}\n\nPlease check the folder and try again."
        )
        temp_root.destroy()
        return

    master_h5_path = h5_files[0]

    # Generate the initial summary before opening the main application.
    temp_root.deiconify()
    temp_root.title("Loading Workspace")
    temp_root.geometry("450x200")
    temp_root.eval('tk::PlaceWindow . center') 
    
    lbl_splash = ctk.CTkLabel(
        temp_root, 
        text="Processing Data...\nGenerating initiale Summary JSON.", 
        font=("Arial", 18, "bold"), 
        text_color="#38BDF8"
    )
    lbl_splash.pack(expand=True)
    temp_root.update()

    active_summary = None
    try:
        # Generate the initial summary without filtering out any hydrates.
        engine = SummaryEngine(base_dir)
        active_summary = engine.run(min_hydrate_area=0.0)
    except Exception as e:
        traceback.print_exc()
        messagebox.showerror("Summary Error", f"Error while creating Summary JSON:\n{e}")

    # Close the loading window.
    temp_root.destroy()

    print(f"Loading workspace from: {base_dir}")
    print(f"Master H5 found: {master_h5_path.name}")
    if active_summary:
        print(f"Active Summary generated: {active_summary.name}")

    # Launch the main application with the active summary path.
    app = MasterApp(base_dir, master_h5_path, active_summary)
    app.mainloop()

if __name__ == "__main__":
    main()