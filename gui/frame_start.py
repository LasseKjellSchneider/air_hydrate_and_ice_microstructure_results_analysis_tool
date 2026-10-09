import customtkinter as ctk

class StartFrame(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller
        
        # Center everything
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure((0, 1, 2, 3, 4), weight=1)
        
        # Title
        title = ctk.CTkLabel(self, text="Hydrate Analysis Suite", font=("Arial", 40, "bold"))
        title.grid(row=0, column=0, pady=(50, 20))
        
        # Navigation Buttons
        btn_analysis = ctk.CTkButton(
            self, text="1. Single Dataset Analysis", font=("Arial", 22, "bold"),
            height=80, width=400, corner_radius=15,
            command=lambda: controller.show_frame("AnalysisFrame")
        )
        btn_analysis.grid(row=1, column=0, pady=15)
        
        btn_compare = ctk.CTkButton(
            self, text="2. Compare Datasets", font=("Arial", 22, "bold"),
            height=80, width=400, corner_radius=15,
            command=lambda: controller.show_frame("CompareFrame")
        )
        btn_compare.grid(row=2, column=0, pady=15)
        
        btn_join = ctk.CTkButton(
            self, text="3. Join Datasets", font=("Arial", 22, "bold"),
            height=80, width=400, corner_radius=15, fg_color="#27ae60", hover_color="#2ecc71",
            command=lambda: controller.show_frame("JoinFrame")
        )
        btn_join.grid(row=3, column=0, pady=15)