# Sci‑Fi Calculator

A lightweight scientific calculator desktop app built with Python and PySide6 (Qt for Python). This repository contains a small GUI calculator (`main.py`) and a Qt Designer .ui file (`sci-calculator.ui`) used for the interface.

## Features

- Standard arithmetic: + − × ÷
- Scientific functions: sin, cos, tan, log, ln, exponent, power, sqrt
- Memory store/recall
- Clear and backspace
- Button-driven UI
- Small, cross-platform desktop app (tested on Windows)

## Quick overview

- Entry point: `main.py`
- UI layout: `sci-calculator.ui` (editable with Qt Designer / PySide6 Designer)
- Dependencies listed in `requirements.txt`

## Requirements

- Python 3.10+ (3.11 recommended)
- PySide6

All Python dependencies are listed in `requirements.txt`.

## Install (Windows PowerShell)

Open PowerShell in the project directory `c:\Users\Administrator\Desktop\scifi-calculator` and run:

```powershell
# create a virtual environment
python -m venv .venv

# activate the virtual environment (PowerShell)
.\.venv\Scripts\Activate.ps1

# upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you run into an execution policy issue when activating the venv (e.g., running scripts is disabled), you can allow the activation script for the current session:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

## Run

With the virtualenv activated, start the calculator:

```powershell
python main.py
```

This should open the GUI window. If the UI file is loaded dynamically, ensure `sci-calculator.ui` is present in the same folder as `main.py`.

## Usage

- Click buttons or type using the keyboard to build expressions.
- Use the scientific function buttons for trig/log operations.
- Use memory buttons to store and recall values.

## Development notes

- The UI was created with Qt Designer (PySide6 Designer). If you want to tweak the layout open `sci-calculator.ui` in the designer and re-save it.
- If `main.py` imports the UI file using `uic` or loads it at runtime, keep the relative path consistent.

### Rebuilding the Python UI (optional)

If you prefer to compile the `.ui` file into a Python module, you can use PySide6's `pyside6-uic` tool:

```powershell
pyside6-uic sci-calculator.ui -o ui_sci_calculator.py
```

Then update `main.py` to import the generated `ui_sci_calculator` module instead of loading the `.ui` file.

## Troubleshooting

- Blank window / UI not appearing: check that `sci-calculator.ui` is in the same directory and not corrupted.
- Missing dependency errors: ensure the virtual environment is activated and `pip install -r requirements.txt` completed successfully.
- Permission errors activating venv in PowerShell: see the `Set-ExecutionPolicy` step above.

## Tests

There are no automated tests included by default. For small GUI projects, consider adding a few unit tests for any pure computation logic (e.g., expression evaluation) and keep GUI tests manual or via an automation tool (pytest-qt) if needed.

## Contributing

Contributions are welcome. For small fixes or improvements:

1. Fork the repo
2. Create a feature branch
3. Open a pull request with a clear description of your changes

Keep changes small and focused. If you change any UI layout, include a screenshot and a short note about why the change helps.
