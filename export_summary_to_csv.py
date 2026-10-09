import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd

def flatten_dict(d, parent_key='', sep='_'):
    """
    Flatten a nested dictionary into a single-level dictionary.

    For example, {'A': {'B': 1}} becomes {'A_B': 1}.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def main():
    # Initialize Tkinter with a hidden root window for file dialogs.
    root = tk.Tk()
    root.withdraw()
    
    file_paths = []
    
    messagebox.showinfo(
        "CSV Export", 
        "Wähle nun nacheinander deine '_summary.json' Dateien aus.\n\n"
        "Wenn du alle gewünschten Proben ausgewählt hast, klicke im Datei-Dialog einfach auf 'Abbrechen'."
    )
    
    # Collect the selected summary files.
    while True:
        file_path = filedialog.askopenfilename(
            title=f"({len(file_paths)} gewählt) _summary.json auswählen (Abbrechen = Fertig)",
            filetypes=[("JSON Summary", "*_summary.json"), ("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if not file_path:  # The user selected Cancel.
            break
            
        file_paths.append(file_path)

    if not file_paths:
        print("Keine Dateien ausgewählt. Abbruch.")
        return

    print(f"{len(file_paths)} Dateien ausgewählt. Verarbeite...")
    
    # Read and format the selected data.
    all_samples_data = {}
    
    for path in file_paths:
        sample_name = os.path.basename(path).replace("_summary.json", "")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Exclude the hydrate list; only global statistics are required.
        if "5_hydrate_list" in data:
            del data["5_hydrate_list"]
            
        # Flatten the nested dictionary.
        flat_data = flatten_dict(data)
        
        # Use the sample name as the corresponding DataFrame column label.
        all_samples_data[sample_name] = flat_data

    # Create a DataFrame with flattened metrics as rows and sample names as columns.
    df = pd.DataFrame(all_samples_data)
    
    # Set the CSV index label.
    df.index.name = "Metric"

    # Select the output file path.
    out_path = filedialog.asksaveasfilename(
        title="Speicherort für die zusammengefasste CSV auswählen",
        defaultextension=".csv",
        initialfile="Proben_Vergleich_Summary.csv",
        filetypes=[("CSV files", "*.csv")]
    )
    
    if out_path:
        # Use semicolon separators and decimal commas for compatibility with localized spreadsheet software.
        # For Python or pandas workflows, use sep=',' and decimal='.'.
        df.to_csv(out_path, sep=';', decimal=',', encoding='utf-8-sig')
        print(f"Erfolgreich gespeichert unter: {out_path}")
        messagebox.showinfo("Erfolg", f"Die Daten wurden erfolgreich als CSV exportiert:\n{out_path}")
    else:
        print("Speichern abgebrochen.")

if __name__ == "__main__":
    main()