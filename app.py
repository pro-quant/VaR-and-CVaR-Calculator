import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Function to calculate daily returns from asset prices


def calculate_returns(data):
    returns = data.pct_change().dropna()
    return returns

# Function to calculate Value at Risk (VaR) and Conditional VaR (CVaR)


def calculate_var_cvar(returns, confidence_level=0.95):
    var = np.percentile(returns, (1 - confidence_level) * 100)
    cvar = returns[returns <= var].mean()
    return var, cvar


# Function to plot the distribution of portfolio returns
def plot_returns_distribution(returns, var, cvar, confidence_level):
    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(returns, kde=True, color="blue", stat="density", ax=ax)
    ax.axvline(var, color='r', linestyle='--',
               label=f"VaR ({100*(1-confidence_level):.1f}%)")
    ax.axvline(cvar, color='g', linestyle='--', label="CVaR")
    ax.legend()
    ax.set_title("Distribution of Portfolio Returns with VaR and CVaR")
    st.pyplot(fig)


# Function to simulate portfolio returns using Monte Carlo simulation
def monte_carlo_simulation(returns, num_simulations=10000):
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    simulations = np.random.normal(mean_return, std_return, num_simulations)
    return simulations


# Streamlit UI
st.title("Risk Management Tool: VaR and CVaR Calculator")


# Add a link to the data download page
st.markdown(
    """
    **To get started, you can download data using the** [Proquant Stock Data Downloader](https://proquant.se/apps/DownloadData/)
    """,
    unsafe_allow_html=True,
)

# Step 1: Upload Portfolio Data
uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file with historical asset prices", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Detect file format and load the data
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
    elif uploaded_file.name.endswith('.xlsx'):
        data = pd.read_excel(uploaded_file, index_col=0, parse_dates=True)
    else:
        st.error("Unsupported file format!")
        st.stop()

    st.write("Portfolio Data (First 5 Rows):")
    st.dataframe(data.head())

    # Step 2: Calculate Returns
    st.write("Calculating daily returns for each ticker...")
    returns = calculate_returns(data)
    st.write("Daily Returns (First 5 Rows):")
    st.dataframe(returns.head())

    # Step 3: Select Methodology
    methodology = st.selectbox("Select Methodology", [
                               "Historical Simulation", "Monte Carlo", "Parametric"])

    # Step 4: Set Parameters
    confidence_level = st.slider(
        "Select Confidence Level", 0.90, 0.99, 0.95, 0.01)
    time_horizon = st.number_input("Time Horizon (days)", 1, 30, 1)

    # Step 5: Calculate Portfolio Returns (Equal Weighting Assumed)
    portfolio_returns = returns.mean(axis=1)

    # Step 6: Calculate VaR and CVaR based on selected methodology
    if methodology == "Historical Simulation":
        var, cvar = calculate_var_cvar(portfolio_returns, confidence_level)
    elif methodology == "Monte Carlo":
        simulations = monte_carlo_simulation(portfolio_returns)
        var, cvar = calculate_var_cvar(simulations, confidence_level)
    else:
        mean_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)
        var = mean_return - std_return * norm.ppf(1 - confidence_level)
        cvar = mean_return - std_return * \
            (norm.pdf(norm.ppf(1 - confidence_level)) / (1 - confidence_level))

    st.write(f"VaR ({100*(1-confidence_level):.1f}%): {var:.2f}")
    st.write(f"CVaR: {cvar:.2f}")

    # Step 7: Visualize Results
    plot_returns_distribution(portfolio_returns, var, cvar, confidence_level)

    # Step 8: Generate Reports
    if st.button("Generate Risk Report"):
        # Generate the detailed risk report
        report_details = []

        # Portfolio Overview
        portfolio_stats = portfolio_returns.describe()
        skewness = portfolio_returns.skew()
        kurtosis = portfolio_returns.kurtosis()

        report_details.append("### Portfolio Overview\n")
        report_details.append(portfolio_stats.to_string())
        report_details.append(f"\nSkewness: {skewness:.2f}")
        report_details.append(f"\nKurtosis: {kurtosis:.2f}\n")

        # Individual Asset Performance
        report_details.append("\n### Individual Asset Performance\n")
        asset_stats = returns.describe()
        report_details.append(asset_stats.to_string())

        # Correlation Matrix
        report_details.append("\n### Correlation Matrix\n")
        correlation_matrix = returns.corr()
        report_details.append(correlation_matrix.to_string())

        # VaR and CVaR Summary
        report_details.append("\n### VaR and CVaR Summary\n")
        report_details.append(
            f"VaR ({100*(1-confidence_level):.1f}%): {var:.2f}")
        report_details.append(f"\nCVaR: {cvar:.2f}")

        # Generate a string for the report
        detailed_report = "\n".join(report_details)

        # Instant download button
        st.download_button("Download Risk Report", detailed_report,
                           file_name="detailed_risk_report.txt")
