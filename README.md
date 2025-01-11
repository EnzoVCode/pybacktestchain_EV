# pybacktestchain-ev

Package for the Python Course based on pybacktestchain library by J. Imbet.

## Overview

`pybacktestchain-ev` is a Python package designed to facilitate portfolio backtesting using various trading strategies. The package incorporates risk management tools, blockchain-based backtest storage, and a user-friendly interface built with Streamlit.

The package includes:
- A fully interactive Streamlit-based user interface (UI).
- Support for multiple asset classes (stocks, commodities, FX, fixed income, etc.).
- Customizable backtesting parameters and risk management tools.
- Pre-processing capabilities for data compatibility across all time zones.

## Installation

You can install the package via pip:

```bash
$ pip install pybacktestchain-ev
```

## Features

- **Interactive UI**: A Streamlit-based interface for configuring backtests, visualizing results, and analyzing performance and risk metrics.
- **Portfolio Backtesting**: Supports backtesting of standard trading strategies:
  1. Equal Weight
  2. Minimum Variance
  3. Maximum Sharpe Ratio
  4. Risk Parity
- **Customizable Transaction Costs**: Allows precise configuration of trading fees for realistic simulations.
- **Risk Management**:
  - Includes stop-loss and take-profit mechanisms with customizable thresholds.
  - Computes Value-at-Risk (VaR) and Expected Shortfall (ES) with user-defined confidence levels.
- **Pre-Processing**:
  - Handles data across various asset classes and ensures compatibility across time zones.
- **Blockchain Integration**: Stores backtest results in a tamper-proof blockchain for future reference.

## Usage

### Running the Streamlit Interface

To start the Streamlit interface:

```bash
$ streamlit run <your_app_file>.py
```

### Configuring a Backtest

1. **Select Tickers**: Enter assets' tickers (comma-separated) to define the universe (supports all asset classes).
2. **Set Parameters**:
   - Initial cash amount.
   - Start and end dates for the backtest.
   - Choose a strategy:
     - Equal Weight
     - Minimum Variance
     - Maximum Sharpe Ratio
     - Risk Parity
   - Configure risk management:
     - Stop-loss and take-profit thresholds.
     - Transaction costs.
     - Confidence levels for VaR and Expected Shortfall.
3. **Run the Backtest**: Click the "Run Backtest" button.

### Viewing Results

- **Summary Statistics**: Includes key metrics for the backtested portfolio.
- **Visualizations**:
  - Portfolio value and returns over time.
  - Distribution of returns.
  - Correlation heatmap of asset returns.
- **Risk Metrics**: Displays VaR and Expected Shortfall with selected confidence levels.
- **Transaction Log**: Provides detailed insights into executed trades.
- **Blockchain Storage**: Saves each backtest under a randomly generated name for future reference.

### Blockchain Storage

Each backtest is saved in a blockchain file under a randomly generated name. This ensures results are tamper-proof and can be referenced later. Blockchain files are stored in the `blockchain/` directory within the package folder.

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`pybacktestchain-ev` was created by Enzo Volpato. It is licensed under the terms of the MIT license.

## Credits

`pybacktestchain-ev` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
