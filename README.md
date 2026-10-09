# Air Hydrate and Ice Microstructure Result Analysis

Desktop application for correlation computation and result visualization.

## Requirements

- Windows
- Python
- Packages listed in requirements.txt

## Setup and launch

From this directory, run the following in PowerShell:

`powershell`
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py


After starting the program a folder with the aggregated data needs to be selected.
Subsequently, the summary JSON file is created, and the visualization GUI is opening.


## Project layout

- `main.py`: application entry point
- `gui\`: desktop workflow screens
- `core\`: correlation calculation and plotting engine
- `requirements.txt`: Python dependencies
- `export_summary_to_csv.py`: calculation of the correlation and saving of the pooled results in a JSON file
