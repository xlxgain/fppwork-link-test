# Sinusoidal fringe test

Generate a vertical 8-bit grayscale sinusoidal fringe image, 1024 × 768 pixels, with 16 horizontal periods. Parameters and the formula are recorded in `parameters.json`; results are recorded in `report.md`.

From this directory, using the parent workspace virtual environment:

```powershell
..\.venv\Scripts\python.exe generate_stripe.py
```

The script requires Python, NumPy and Pillow, and writes `stripe.png` beside the script.
