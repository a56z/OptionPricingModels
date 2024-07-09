import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from OptionPricingModels import BlackScholes, MertonJumpDiffusion, HestonStochasticVolatility
from matplotlib.colors import LinearSegmentedColormap

# Page configuration
st.set_page_config(
    page_title="Option Pricing Models",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to inject into Streamlit
st.markdown("""
<style>
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px;
    width: auto;
    margin: 0 auto;
}
.metric-call {
    background-color: #90ee90;
    color: black;
    margin-right: 10px;
    border-radius: 10px;
}
.metric-put {
    background-color: #ffcccb;
    color: black;
    border-radius: 10px;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: bold;
    margin: 0;
}
.metric-label {
    font-size: 1rem;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar for User Inputs
with st.sidebar:
    st.title("📊 Black-Scholes Model")
    st.write("`Created by:`")
    linkedin_url = "https://www.linkedin.com/in/enriquefolte/"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Enrique Folte, Buenos Aires`</a>', unsafe_allow_html=True)
    
    model_choice = st.selectbox("Select Option Pricing Model", [
        "Black-Scholes",
        "Merton Jump Diffusion",
        "Heston Stochastic Volatility"
    ])

    current_price = st.number_input("Current Asset Price", value=100.0)
    strike = st.number_input("Strike Price", value=100.0)
    time_to_maturity = st.number_input("Time to Maturity (Years)", value=1.0)
    volatility = st.number_input("Volatility (σ)", value=0.2)
    interest_rate = st.number_input("Risk-Free Interest Rate", value=0.05)

    extra_params = {}
    if model_choice == "Merton Jump Diffusion":
        extra_params['lambda_j'] = st.number_input("Jump Intensity (λ)", value=0.1)
        extra_params['mu_j'] = st.number_input("Average Jump Size (μ)", value=-0.1)
        extra_params['sigma_j'] = st.number_input("Jump Size Volatility (σ_j)", value=0.3)
    elif model_choice == "Heston Stochastic Volatility":
        extra_params['kappa'] = st.number_input("Rate of mean reversion (κ)", value=2.0)
        extra_params['theta'] = st.number_input("Long run average variance (θ)", value=0.04)
        extra_params['sigma'] = st.number_input("Volatility of variance (σ)", value=0.5)
        extra_params['rho'] = st.number_input("Correlation (ρ)", value=-0.5)
        extra_params['v0'] = st.number_input("Initial variance (v0)", value=0.04)

    st.markdown("---")
    calculate_btn = st.button('Heatmap Parameters')
    spot_min = st.number_input('Min Spot Price', min_value=0.01, value=current_price*0.8, step=0.01)
    spot_max = st.number_input('Max Spot Price', min_value=0.01, value=current_price*1.2, step=0.01)
    vol_min = st.slider('Min Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*0.5, step=0.01)
    vol_max = st.slider('Max Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*1.5, step=0.01)
    
    spot_range = np.linspace(spot_min, spot_max, 10)
    vol_range = np.linspace(vol_min, vol_max, 10)

def create_custom_colormap():
    # Create a custom red to green colormap
    colors = sns.color_palette("RdYlGn", as_cmap=True)
    return colors

def plot_heatmap(model_class, spot_range, vol_range, strike, time_to_maturity, interest_rate, **kwargs):
    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))
    call_moneyness = np.zeros((len(vol_range), len(spot_range)))
    put_moneyness = np.zeros((len(vol_range), len(spot_range)))
    
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            model = model_class(time_to_maturity, strike, spot, vol, interest_rate, **kwargs)
            call_price, put_price = model.calculate_prices()
            call_prices[i, j] = call_price
            put_prices[i, j] = put_price
            call_moneyness[i, j] = spot - strike
            put_moneyness[i, j] = strike - spot
    
    custom_cmap = create_custom_colormap()
    
    # Plotting Call Price Heatmap
    fig_call, ax_call = plt.subplots(figsize=(10, 8))
    sns.heatmap(call_moneyness, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=call_prices, fmt=".2f", cmap=custom_cmap, ax=ax_call, center=0)
    ax_call.set_title('CALL')
    ax_call.set_xlabel('Spot Price')
    ax_call.set_ylabel('Volatility')
    
    # Plotting Put Price Heatmap
    fig_put, ax_put = plt.subplots(figsize=(10, 8))
    sns.heatmap(put_moneyness, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=put_prices, fmt=".2f", cmap=custom_cmap, ax=ax_put, center=0)
    ax_put.set_title('PUT')
    ax_put.set_xlabel('Spot Price')
    ax_put.set_ylabel('Volatility')
    
    return fig_call, fig_put

# Main Page for Output Display
st.title("Option Pricing Models")

# Table of Inputs
input_data = {
    "Current Asset Price": [current_price],
    "Strike Price": [strike],
    "Time to Maturity (Years)": [time_to_maturity],
    "Volatility (σ)": [volatility],
    "Risk-Free Interest Rate": [interest_rate],
}
if model_choice == "Merton Jump Diffusion":
    input_data.update({
        "Jump Intensity (λ)": [extra_params['lambda_j']],
        "Average Jump Size (μ)": [extra_params['mu_j']],
        "Jump Size Volatility (σ_j)": [extra_params['sigma_j']]
    })
elif model_choice == "Heston Stochastic Volatility":
    input_data.update({
        "Rate of mean reversion (κ)": [extra_params['kappa']],
        "Long run average variance (θ)": [extra_params['theta']],
        "Volatility of variance (σ)": [extra_params['sigma']],
        "Correlation (ρ)": [extra_params['rho']],
        "Initial variance (v0)": [extra_params['v0']]
    })

input_df = pd.DataFrame(input_data)
st.table(input_df)

# Initialize the selected model
model_mapping = {
    "Black-Scholes": BlackScholes,
    "Merton Jump Diffusion": MertonJumpDiffusion,
    "Heston Stochastic Volatility": HestonStochasticVolatility
}

model_instance = model_mapping[model_choice](time_to_maturity, strike, current_price, volatility, interest_rate, **extra_params)

# Calculate Call and Put values
call_price, put_price = model_instance.calculate_prices()

# Display Call and Put Values in colored tables
col1, col2 = st.columns([1, 1], gap="small")

with col1:
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">CALL Value</div>
                <div class="metric-value">${call_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">PUT Value</div>
                <div class="metric-value">${put_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("")
st.title("Options Price - Interactive Heatmap")
st.info("Explore how option prices fluctuate with varying 'Spot Prices and Volatility' levels using interactive heatmap parameters, all while maintaining a constant 'Strike Price'.")

# Interactive Sliders and Heatmaps for Call and Put Options
col1, col2 = st.columns([1, 1], gap="small")

with col1:
    st.subheader("Call Price Heatmap")
    heatmap_fig_call, _ = plot_heatmap(model_mapping[model_choice], spot_range, vol_range, strike, time_to_maturity, interest_rate, **extra_params)
    st.pyplot(heatmap_fig_call)
    plt.close(heatmap_fig_call)

with col2:
    st.subheader("Put Price Heatmap")
    _, heatmap_fig_put = plot_heatmap(model_mapping[model_choice], spot_range, vol_range, strike, time_to_maturity, interest_rate, **extra_params)
    st.pyplot(heatmap_fig_put)
    plt.close(heatmap_fig_put)
