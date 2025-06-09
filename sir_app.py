import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from io import BytesIO
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Epidemic Simulator with Forecast and Vaccination", layout="wide")
st.title("Epidemic Simulator")

# ---------------- Model Selection ----------------
model_type = st.sidebar.selectbox("Choose a Model", ["SIS", "SIR", "SIRV", "SEIR", "SEIRV", "SIRD", "SEIRD"])
manual_input = st.sidebar.checkbox("Manually Input Parameters", False)

# ---------------- Parameter Input ----------------
if manual_input:
    beta = st.sidebar.number_input("Infection rate (β)", value=0.3)
    gamma = st.sidebar.number_input("Recovery rate (γ)", value=0.1)
    sigma = st.sidebar.number_input("Incubation rate (σ)", value=0.2)
    mu = st.sidebar.number_input("Death rate (μ)", value=0.01)
    nu = st.sidebar.number_input("Vaccination rate (ν)", value=0.01)
    I0 = st.sidebar.number_input("Initial Infected", value=1)
    E0 = st.sidebar.number_input("Initial Exposed", value=0)
    R0 = st.sidebar.number_input("Initial Recovered", value=0)
    V0 = st.sidebar.number_input("Initial Vaccinated", value=0)
    D0 = st.sidebar.number_input("Initial Deceased", value=0)
    N = st.sidebar.number_input("Total Population", value=1000)
    days = st.sidebar.slider("Simulation Days", 30, 365, 160)
else:
    beta = st.sidebar.slider("Infection rate (β)", 0.0, 1.0, 0.3, 0.01)
    gamma = st.sidebar.slider("Recovery rate (γ)", 0.0, 1.0, 0.1, 0.01)
    sigma = st.sidebar.slider("Incubation rate (σ)", 0.0, 1.0, 0.2, 0.01)
    mu = st.sidebar.slider("Death rate (μ)", 0.0, 0.5, 0.01, 0.01)
    nu = st.sidebar.slider("Vaccination rate (ν)", 0.0, 0.5, 0.01, 0.01)
    I0 = st.sidebar.slider("Initial Infected", 1, 100, 1)
    E0 = st.sidebar.slider("Initial Exposed", 0, 100, 0)
    R0 = st.sidebar.slider("Initial Recovered", 0, 100, 0)
    V0 = st.sidebar.slider("Initial Vaccinated", 0, 100, 0)
    D0 = st.sidebar.slider("Initial Deceased", 0, 100, 0)
    N = st.sidebar.slider("Total Population", 100, 10000, 1000)
    days = st.sidebar.slider("Simulation Days", 30, 365, 160)

# ---------------- Initial Values ----------------
S0 = N - I0 - R0 - D0 - V0 - (E0 if model_type in ["SEIR", "SEIRV"] else 0)
t = np.linspace(0, days, days)

# ---------------- Model Definitions ----------------
def sis(y, t, beta, gamma):
    S, I = y
    return [-beta * S * I / N + gamma * I, beta * S * I / N - gamma * I]

def sir(y, t, beta, gamma):
    S, I, R = y
    return [-beta * S * I / N, beta * S * I / N - gamma * I, gamma * I]

def sirv(y, t, beta, gamma, nu):
    S, I, R, V = y
    dS = -beta * S * I / N - nu * S
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    dV = nu * S
    return [dS, dI, dR, dV]

def seir(y, t, beta, sigma, gamma):
    S, E, I, R = y
    return [-beta * S * I / N, beta * S * I / N - sigma * E, sigma * E - gamma * I, gamma * I]

def seirv(y, t, beta, sigma, gamma, nu):
    S, E, I, R, V = y
    dS = -beta * S * I / N - nu * S
    dE = beta * S * I / N - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I
    dV = nu * S
    return [dS, dE, dI, dR, dV]

def sird(y, t, beta, gamma, mu):
    S, I, R, D = y
    return [-beta * S * I / N, beta * S * I / N - gamma * I - mu * I, gamma * I, mu * I]

def seird(y, t, beta, sigma, gamma, mu):
    S, E, I, R, D = y
    return [-beta * S * I / N, beta * S * I / N - sigma * E, sigma * E - gamma * I - mu * I, gamma * I, mu * I]

# ---------------- Solve ODE ----------------
if model_type == "SIS":
    y0 = [S0, I0]
    result = odeint(sis, y0, t, args=(beta, gamma))
    df = pd.DataFrame(result, columns=["Susceptible", "Infected"])
elif model_type == "SIR":
    y0 = [S0, I0, R0]
    result = odeint(sir, y0, t, args=(beta, gamma))
    df = pd.DataFrame(result, columns=["Susceptible", "Infected", "Recovered"])
elif model_type == "SIRV":
    y0 = [S0, I0, R0, V0]
    result = odeint(sirv, y0, t, args=(beta, gamma, nu))
    df = pd.DataFrame(result, columns=["Susceptible", "Infected", "Recovered", "Vaccinated"])
elif model_type == "SEIR":
    y0 = [S0, E0, I0, R0]
    result = odeint(seir, y0, t, args=(beta, sigma, gamma))
    df = pd.DataFrame(result, columns=["Susceptible", "Exposed", "Infected", "Recovered"])
elif model_type == "SEIRV":
    y0 = [S0, E0, I0, R0, V0]
    result = odeint(seirv, y0, t, args=(beta, sigma, gamma, nu))
    df = pd.DataFrame(result, columns=["Susceptible", "Exposed", "Infected", "Recovered", "Vaccinated"])
elif model_type == "SIRD":
    y0 = [S0, I0, R0, D0]
    result = odeint(sird, y0, t, args=(beta, gamma, mu))
    df = pd.DataFrame(result, columns=["Susceptible", "Infected", "Recovered", "Deceased"])
elif model_type == "SEIRD":
    y0 = [S0, E0, I0, R0, D0]
    result = odeint(seird, y0, t, args=(beta, sigma, gamma, mu))
    df = pd.DataFrame(result, columns=["Susceptible", "Exposed", "Infected", "Recovered", "Deceased"])

df["Day"] = t

# ---------------- Plot Results ----------------
st.subheader(f" {model_type} Model Simulation")
fig, ax = plt.subplots(figsize=(10, 6))
for col in df.columns:
    if col != "Day":
        ax.plot(df["Day"], df[col], label=col)
ax.set_xlabel("Days")
ax.set_ylabel("Population")
ax.set_title(f"{model_type} Model Dynamics with Vaccination")
ax.legend()
st.pyplot(fig)

# ---------------- Forecasting ----------------
st.subheader("Infected Case Forecast (ARIMA)")
forecast_days = st.slider("Forecast Days", 7, 60, 30)
try:
    model = ARIMA(df["Infected"], order=(2,1,2)).fit()
    forecast = model.forecast(steps=forecast_days)
    forecast_df = pd.DataFrame({
        "Day": np.arange(days, days + forecast_days),
        "Forecasted Infected": forecast
    })
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(df["Day"], df["Infected"], label="Actual Infected", color="blue")
    ax2.plot(forecast_df["Day"], forecast_df["Forecasted Infected"], label="Forecast", color="orange")
    ax2.set_title("Forecast of Future Infections")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Infected")
    ax2.legend()
    st.pyplot(fig2)
except Exception as e:
    st.warning("Forecasting failed. Try adjusting parameters or duration.")

# ---------------- Download Buttons ----------------
st.subheader("Download Simulation Data")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download as CSV", csv, "simulation_data.csv", "text/csv")

excel_data = BytesIO()
with pd.ExcelWriter(excel_data, engine="xlsxwriter") as writer:
    df.to_excel(writer, index=False, sheet_name="Simulation")
st.download_button("Download as Excel", excel_data.getvalue(), "simulation_data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


