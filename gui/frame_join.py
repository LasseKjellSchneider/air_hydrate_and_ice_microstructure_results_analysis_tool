import customtkinter as ctk
import tkinter.messagebox as messagebox
from tkinter import filedialog
import pathlib
import threading

# Import the backend module
from core.join_engine import JoinEngine

class JoinFrame(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller
        
        self.selected_folders = []
        self.output_folder = None
        
        # --- HEADER ---
        header = ctk.CTkFrame(self, height=60, corner_radius=0)
        header.pack(fill="x", side="top")
        
        ctk.CTkButton(header, text="< Back to Start", width=120, height=40,
                      command=lambda: controller.show_frame("StartFrame")).pack(side="left", padx=20, pady=10)
                      
        ctk.CTkLabel(header, text="Join Datasets", font=("Arial", 24, "bold")).pack(side="left", padx=20, pady=10)
        
        # --- MAIN CONTENT ---
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Left Panel: Folder list and ordering
        self.left_panel = ctk.CTkFrame(self.main_container)
        self.left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        ctk.CTkLabel(self.left_panel, text="1. Select & Order Datasets", font=("Arial", 18, "bold")).pack(pady=(15, 5))
        ctk.CTkLabel(self.left_panel, text="Top = Leftmost Segment, Bottom = Rightmost Segment.", text_color="gray").pack(pady=(0, 15))
        
        self.btn_add = ctk.CTkButton(self.left_panel, text="+ Add Data Folder", font=("Arial", 14, "bold"), height=40, command=self.add_folder)
        self.btn_add.pack(pady=(0, 10), padx=20, fill="x")
        
        self.list_frame = ctk.CTkScrollableFrame(self.left_panel, fg_color="#1e1e1e")
        self.list_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        # Right Panel: Output configuration & Execution
        self.right_panel = ctk.CTkFrame(self.main_container, width=350)
        self.right_panel.pack(side="right", fill="y", padx=(10, 0))
        self.right_panel.pack_propagate(False) 
        
        ctk.CTkLabel(self.right_panel, text="2. Configure Output", font=("Arial", 18, "bold")).pack(pady=(15, 10))
        
        lbl_name = ctk.CTkLabel(self.right_panel, text="New Dataset Name:", font=("Arial", 14))
        lbl_name.pack(pady=(5, 0), padx=20, anchor="w")
        self.entry_name = ctk.CTkEntry(self.right_panel, placeholder_text="e.g. Joined_Sample_1", height=35)
        self.entry_name.pack(pady=(5, 15), padx=20, fill="x")
        
        lbl_gap = ctk.CTkLabel(self.right_panel, text="Gap between datasets (in mm):", font=("Arial", 14))
        lbl_gap.pack(pady=(5, 0), padx=20, anchor="w")
        self.entry_gap = ctk.CTkEntry(self.right_panel, placeholder_text="e.g. 2.0", height=35)
        self.entry_gap.insert(0, "0.0")
        self.entry_gap.pack(pady=(5, 15), padx=20, fill="x")
        
        self.btn_out = ctk.CTkButton(self.right_panel, text="Select Output Folder", height=40, command=self.select_output)
        self.btn_out.pack(pady=10, padx=20, fill="x")
        
        self.lbl_out = ctk.CTkLabel(self.right_panel, text="No output selected", text_color="gray", wraplength=300)
        self.lbl_out.pack(pady=(0, 20), padx=20)
        
        ctk.CTkLabel(self.right_panel, text="-"*40, text_color="gray").pack(pady=5)
        
        self.btn_start = ctk.CTkButton(self.right_panel, text="Start Joining", height=50, font=("Arial", 16, "bold"), 
                                       fg_color="#27ae60", hover_color="#2ecc71", state="disabled", command=self.start_join)
        self.btn_start.pack(pady=15, padx=20, fill="x")
        
        self.lbl_status = ctk.CTkLabel(self.right_panel, text="", font=("Arial", 14), text_color="#f39c12", wraplength=300)
        self.lbl_status.pack(pady=5)

    def on_show(self):
        """Called automatically when switching to this tab. Auto-adds the workspace folder."""
        if not self.selected_folders and self.controller.base_dir:
            self.selected_folders.append(self.controller.base_dir)
            self.refresh_list_ui()
            self.check_ready_state()

    def add_folder(self):
        folder = filedialog.askdirectory(title="Select Preprocessed Data Folder")
        if not folder: return
        p = pathlib.Path(folder)
        
        if not (p / "analysis_results.json").exists():
            messagebox.showwarning("Warning", "No 'analysis_results.json' found.\nAre you sure this is a processed dataset?")
            return
            
        if p in self.selected_folders:
            messagebox.showinfo("Duplicate", "This dataset is already in the join list.")
            return
            
        self.selected_folders.append(p)
        self.refresh_list_ui()
        self.check_ready_state()

    def move_up(self, index):
        if index > 0:
            self.selected_folders[index - 1], self.selected_folders[index] = self.selected_folders[index], self.selected_folders[index - 1]
            self.refresh_list_ui()

    def move_down(self, index):
        if index < len(self.selected_folders) - 1:
            self.selected_folders[index + 1], self.selected_folders[index] = self.selected_folders[index], self.selected_folders[index + 1]
            self.refresh_list_ui()

    def remove_folder(self, index):
        self.selected_folders.pop(index)
        self.refresh_list_ui()
        self.check_ready_state()

    def refresh_list_ui(self):
        for widget in self.list_frame.winfo_children(): widget.destroy()
        for i, path in enumerate(self.selected_folders):
            row_frame = ctk.CTkFrame(self.list_frame, fg_color="#2b2b2b", corner_radius=5)
            row_frame.pack(fill="x", pady=5, padx=5)
            
            label_text = f"{i+1}. {path.name}"
            if i == 0 and path == self.controller.base_dir:
                label_text += " (Current Workspace)"
                
            lbl = ctk.CTkLabel(row_frame, text=label_text, font=("Arial", 14, "bold"), anchor="w")
            lbl.pack(side="left", padx=15, pady=10, expand=True, fill="x")
            
            btn_up = ctk.CTkButton(row_frame, text="↑", width=30, command=lambda idx=i: self.move_up(idx))
            btn_up.pack(side="left", padx=2)
            if i == 0: btn_up.configure(state="disabled")
            
            btn_down = ctk.CTkButton(row_frame, text="↓", width=30, command=lambda idx=i: self.move_down(idx))
            btn_down.pack(side="left", padx=2)
            if i == len(self.selected_folders) - 1: btn_down.configure(state="disabled")
            
            btn_del = ctk.CTkButton(row_frame, text="X", width=30, fg_color="#e74c3c", hover_color="#c0392b", command=lambda idx=i: self.remove_folder(idx))
            btn_del.pack(side="left", padx=(10, 10))

    def select_output(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder = pathlib.Path(folder)
            self.lbl_out.configure(text=f".../{self.output_folder.name}")
            self.check_ready_state()

    def check_ready_state(self):
        if len(self.selected_folders) >= 2 and self.output_folder is not None:
            self.btn_start.configure(state="normal")
        else:
            self.btn_start.configure(state="disabled")

    def start_join(self):
        try:
            gap_str = self.entry_gap.get().replace(",", ".")
            gap_mm = float(gap_str) if gap_str else 0.0
        except ValueError:
            messagebox.showerror("Error", "Gap must be a valid number (e.g., 2.0).")
            return
            
        dataset_name = self.entry_name.get()
        if not dataset_name:
            dataset_name = "Joined_Dataset"

        self.btn_start.configure(state="disabled")
        self.btn_add.configure(state="disabled")
        self.lbl_status.configure(text="Joining datasets... This involves copying large files, please wait.", text_color="#f39c12")
        
        def worker():
            try:
                engine = JoinEngine(self.selected_folders, self.output_folder, gap_mm, dataset_name)
                engine.run()
                self.controller.after(0, lambda: [
                    self.lbl_status.configure(text="Join completed successfully!", text_color="#2ecc71"),
                    messagebox.showinfo("Success", "Datasets successfully joined!")
                ])
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                self.controller.after(0, lambda: [
                    self.lbl_status.configure(text="Error during join. See terminal.", text_color="#e74c3c"),
                    messagebox.showerror("Error", f"An error occurred:\n{e}")
                ])
            finally:
                self.controller.after(0, lambda: [
                    self.btn_start.configure(state="normal"),
                    self.btn_add.configure(state="normal")
                ])
                
        threading.Thread(target=worker, daemon=True).start()