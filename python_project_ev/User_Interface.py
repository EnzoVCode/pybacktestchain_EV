import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from src.pybacktestchain_EV.data_module import DataModule, EnhancedInformation, preprocess_data, get_stocks_data
from src.pybacktestchain_EV.strategies import (
    equal_weight_strategy,
    min_variance_strategy,
    max_sharpe_ratio_strategy,
    risk_parity_strategy,
)
from src.pybacktestchain_EV.broker import Broker
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.pybacktestchain_EV.risk_metrics import calculate_var, calculate_expected_shortfall

# Initialize Streamlit app
st.title("Portfolio Backtesting Interface")
st.write("An interactive interface for backtesting trading strategies.")

# Sidebar configuration
st.sidebar.header("Configuration")

# Input tickers
tickers = st.sidebar.text_area(
    "Enter Stock Tickers (comma-separated)",
    "NVDA, AAPL, ,GC=F, CL=F, TLT, LQD, EURUSD=X, JPY=X"
)
tickers_list = [ticker.strip() for ticker in tickers.split(",") if ticker.strip()]

# Define backtesting parameters
initial_cash = st.sidebar.number_input("Initial Cash", min_value=1000, value=100000)
start_date = st.sidebar.date_input("Start Date", value=datetime.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", value=datetime.today())

# Select Strategy
strategy = st.sidebar.selectbox(
    "Select Strategy",
    [
        "Equal Weight",
        "Minimum Variance",
        "Maximum Sharpe Ratio",
        "Risk-Parity"
    ]
)

confidence_level = st.sidebar.slider("Risk Metrics Confidence Level (%)", 90, 99, 95) / 100

# Add stop-loss and take-profit inputs in the interface
st.sidebar.header("Risk Management")
stop_loss = st.sidebar.number_input(
    "Stop Loss (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1
) / 100  # Convert percentage to decimal
take_profit = st.sidebar.number_input(
    "Take Profit (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1
) / 100  # Convert percentage to decimal

st.write(f"Selected Stop Loss: {stop_loss * 100:.2f}%")
st.write(f"Selected Take Profit: {take_profit * 100:.2f}%")

# Add transaction cost input in the interface
st.sidebar.header("Additional Parameters")
transaction_cost = st.sidebar.number_input(
    "Transaction Cost (%)", min_value=0.0, max_value=5.0, value=0.1, step=0.01
) / 100  # Convert percentage to decimal

st.write(f"Selected Transaction Cost: {transaction_cost * 100:.2f}%")

# Run Backtest
if st.sidebar.button("Run Backtest"):
    try:
        st.write("Fetching data for tickers...")
        data = get_stocks_data(tickers_list, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        if data.empty:
            st.error("No data retrieved for the provided tickers.")
        else:
            st.write("### Fetched Data Preview")
            st.write(data.head())

            # Preprocess the data
            try:
                data = data.dropna()
                processed_data = preprocess_data(data)
                st.write("### Processed Data for Strategies")
                st.write(processed_data.head())
            except Exception as e:
                st.error(f"Error in preprocessing data: {e}")

            # Initialize DataModule and Enhanced Information
            data_module = DataModule(data)
            enhanced_info = EnhancedInformation(data_module=data_module)

            # Initialize Broker
            broker = Broker(cash=initial_cash, verbose=True)
            
            # Simulate backtest
            portfolio_value = []
            transaction_log = []

            for date in processed_data.index[1:]:
                prices = processed_data.loc[date]

                # Skip if prices have NaN
                if prices.isnull().any():
                    st.warning(f"Skipping date {date} due to missing data.")
                    continue
                
                broker.apply_risk_management(prices, stop_loss, take_profit, date)

                # Compute information set
                try:
                    information_set = enhanced_info.compute_information(date)
                    covariance_matrix = information_set['covariance_matrix']
                    expected_return = information_set['expected_return']
                except ValueError as e:
                    st.warning(f"Data issue on {date}: {e}")
                    continue

                # Strategy selection
                try:
                    if strategy == "Equal Weight":
                        portfolio = equal_weight_strategy(prices)
                    elif strategy == "Minimum Variance":
                        portfolio = min_variance_strategy(prices, covariance_matrix)
                    elif strategy == "Maximum Sharpe Ratio":
                        portfolio = max_sharpe_ratio_strategy(expected_return, covariance_matrix)
                    elif strategy == "Risk-Parity":
                        portfolio = risk_parity_strategy(prices, covariance_matrix)
                except ValueError as e:
                    st.error(f"Strategy error on {date}: {e}")
                    continue

                # Ensure no NaN weights
                if any(pd.isnull(list(portfolio.values()))):
                    raise ValueError("Portfolio contains NaN values.")


                # Execute portfolio and handle errors
                try:
                    broker.execute_portfolio(portfolio, prices, date)
                    portfolio_value.append({"Date": date, "Portfolio Value": broker.get_portfolio_value(prices)})
                    transaction_log.append(broker.get_transaction_log())
                except ValueError as e:
                    st.error(f"Execution error on {date}: {e}")
                    continue

            # Convert portfolio value list to a DataFrame for analysis
            portfolio_value_df = pd.DataFrame(portfolio_value).set_index("Date")
            portfolio_value_df["Returns"] = portfolio_value_df["Portfolio Value"].pct_change()

            # Calculate VaR and Expected Shortfall
            current_portfolio_value = portfolio_value_df["Portfolio Value"].iloc[-1]  # Get the latest portfolio value

            var = calculate_var(portfolio_value_df["Returns"].dropna(), current_portfolio_value, confidence_level=confidence_level)
            es = calculate_expected_shortfall(portfolio_value_df["Returns"].dropna(), current_portfolio_value, confidence_level=confidence_level)

            # Display Risk Metrics
            st.write("### Risk Metrics")
            st.write(f"Value-at-Risk (VaR) at {confidence_level*100:.0f}% confidence level: {var:.2f} USD")
            st.write(f"Expected Shortfall (ES) at {confidence_level*100:.0f}% confidence level: {es:.2f} USD")


            # Display transaction log
            st.write("### Transaction Log")
            if transaction_log:
                flat_transaction_log = pd.concat(transaction_log, ignore_index=True)
                st.write(flat_transaction_log)

            # Plot portfolio value evolution
            st.write("### Portfolio Value Over Time")
            st.line_chart(portfolio_value_df["Portfolio Value"])


            # Plot portfolio returns evolution over time
            st.write("### Portfolio Returns Over Time")
            st.line_chart(portfolio_value_df["Returns"].dropna())

            # Plot histogram of portfolio returns distribution
            st.write("### Portfolio Returns Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(portfolio_value_df["Returns"].dropna(), bins=20, edgecolor='k', alpha=0.7)
            ax.set_title("Portfolio Returns Distribution", fontsize=14)
            ax.set_xlabel("Returns", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.grid(True)
            st.pyplot(fig)

            # Correlation heatmap of asset returns
            st.write("### Correlation Heatmap of Asset Returns")
            returns = processed_data.pct_change().dropna()
            correlation_matrix = returns.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap of Asset Returns", fontsize=16)
            st.pyplot(fig)


            # Calculate portfolio returns and display return statistics
            return_stats = portfolio_value_df["Returns"].describe()
            st.write("### Portfolio Return Summary Statistics")
            st.write(return_stats)

    except Exception as e:
        st.error(f"Error running backtest: {e}")
