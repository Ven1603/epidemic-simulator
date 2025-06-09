# 🧬 Epidemic Simulator and Forecasting Dashboard

An interactive web-based tool for simulating the spread of infectious diseases using mathematical compartmental models (SIR, SEIR, SIRD, SEIRD, etc.).

This app was developed as part of a BSc project and features real-time parameter adjustment, animated infection curves, disease forecasting, and vaccination dynamics.

## Features

Supports multiple epidemic models: **SIS, SIR, SIRV, SEIR, SEIRV, SIRD, SEIRD**
Interactive parameter tuning (infection, recovery, vaccination, etc.)
30-day forecast of infections using ARIMA
Vaccination dynamics included
Download results as CSV and Excel

## Technologies Used

- Python
- Streamlit
- NumPy
- Pandas
- SciPy (ODE solving)
- Matplotlib (visualization & animation)
- Statsmodels (forecasting)
- XlsxWriter
