# Experiment report

- Result: successful; the script ran and generated `stripe.png`.
- Environment: Windows, Python 3.13.14 (workspace `.venv`), NumPy 2.5.4, Pillow 13.0.0.
- Command (from repository root): `..\.venv\Scripts\python.exe generate_stripe.py`
- Image: PNG, 8-bit grayscale (Pillow mode L), 1024 × 768 pixels.
- Measured grayscale range: 0–255; measured mean: 127.4990234375 (after 8-bit rounding).
- Verification: image dimensions and mode checked; every pixel compared against the specified cosine formula rounded to 8-bit values. Parameters are recorded in `parameters.json`.
