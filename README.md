# python_project_ev

Package for the Master 203 project at Paris-Dauphine University.

## Overview

`python_project_ev` is a Python package designed to facilitate portfolio backtesting using various trading strategies. The package incorporates risk management tools, blockchain-based backtest storage, and a user-friendly interface built with Streamlit.

## Installation

You can install the package via pip:

```bash
$ pip install python_project_ev
```

## Features

- **Interactive UI**: A Streamlit-based interactive interface for configuring backtests, visualizing results, analyzing performance and risk metrics.
- **Portfolio Backtesting**: Supports backtesting of standard trading strategies such as Equal Weight, Minimum Variance, Maximum Sharpe Ratio, and Risk Parity.
- **Customizable Transaction Costs**: Includes a feature to account for transaction costs in backtesting.
- **Risk Management**: Includes stop-loss and take-profit mechanisms with customizable thresholds, as well as Value-at-Risk (VaR) and Expected Shortfall for the backtested portfolio.
- **Blockchain Integration**: Stores backtest results in a blockchain.

## Usage

### Running the Streamlit Interface

To start the Streamlit interface:

```bash
$ streamlit run app.py
```

### Configuring a Backtest

1. **Select Tickers**: Enter stock tickers (comma-separated) to define the universe (works across all asset classes).
2. **Set Parameters**:
   - Initial cash amount.
   - Start and end dates for the backtest.
   - Choose a strategy: Equal Weight, Minimum Variance, Maximum Sharpe Ratio, or Risk Parity.
   - Configure risk management: Stop-loss, take-profit percentages, and transaction costs.
3. **Run the Backtest**: Click the "Run Backtest" button.

### Viewing Results

- Portfolio value and returns are displayed over time, as well as the distribution of returns.
- Risk metrics, such as Value-at-Risk (VaR) and Expected Shortfall (ES), are calculated.
- A transaction log is provided for detailed insights and store to the blockchain.
- Additional visualizations include correlation heatmaps and return distributions.

### Blockchain Storage

Each backtest is saved in a blockchain under a randomly generated name. This ensures results are tamper-proof and can be referenced later. Blockchain files are stored in the `blockchain/` directory.

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`python_project_ev` was created by Enzo Volpato. It is licensed under the terms of the MIT license.

## Credits

`python_project_ev` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
