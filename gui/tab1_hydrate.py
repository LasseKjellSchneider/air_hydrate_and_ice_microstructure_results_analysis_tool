import customtkinter as ctk

from gui.frame_start import StartFrame
from gui.frame_analysis import AnalysisFrame
from gui.frame_compare import CompareFrame
from gui.frame_join import JoinFrame

class HydrateTab(ctk.CTkFrame):
    """
    Main controller for Tab 1. Manages the navigation between
    Start, Analysis, Compare, and Join modules.
    """
    def __init__(self, parent, base_dir):
        super().__init__(parent, fg_color="transparent")
        self.pack(fill="both", expand=True)
        
        # Store the workspace directory selected at startup
        self.base_dir = base_dir 
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        self.frames = {}
        
        # Initialize all sub-frames
        for F in (StartFrame, AnalysisFrame, CompareFrame, JoinFrame):
            frame = F(self, self)
            self.frames[F.__name__] = frame
            frame.grid(row=0, column=0, sticky="nsew")
            
        # Display the start menu by default
        self.show_frame("StartFrame")
        
    def show_frame(self, frame_class_name):
        """Brings the requested frame to the front and triggers its on_show method if available."""
        frame = self.frames[frame_class_name]
        frame.tkraise()
        
        # Trigger dynamic loading only when the frame is actually shown
        if hasattr(frame, 'on_show'):
            frame.on_show()

    def load_workspace_data(self):
        """
        Called by MasterApp after a new summary has been generated, for example
        when the volume filter in Tab 3 is applied. Refreshes all child frames.
        """
        for frame in self.frames.values():
            # Prefer the frame's explicit workspace-data loader when available.
            if hasattr(frame, 'load_workspace_data'):
                frame.load_workspace_data()
            # Otherwise, refresh frames that update themselves when displayed.
            elif hasattr(frame, 'on_show'):
                # Allow background refresh failures without interrupting other frames.
                try:
                    frame.on_show()
                except Exception:
                    pass