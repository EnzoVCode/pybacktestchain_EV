import yfinance as yf
import pandas as pd
from sec_cik_mapper import StockMapper
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy.optimize import minimize
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)

#---------------------------------------------------------
# Constants
#---------------------------------------------------------

UNIVERSE_SEC = list(StockMapper().ticker_to_cik.keys())

#---------------------------------------------------------
# Functions
#---------------------------------------------------------

def get_stock_data(ticker, start_date, end_date):
    """Retrieve historical data for a single stock."""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date, auto_adjust=False, actions=False)
        df = data.reset_index()
        df['ticker'] = ticker
        return df
    except Exception as e:
        logging.warning(f"Failed to retrieve data for {ticker}: {e}")
        return pd.DataFrame()

def get_stocks_data(tickers, start_date, end_date):
    """Retrieve historical data for a list of stocks."""
    dfs = []
    for ticker in tickers:
        df = get_stock_data(ticker, start_date, end_date)
        if not df.empty:
            dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        logging.warning("No data retrieved for the given tickers.")
        return pd.DataFrame()

#---------------------------------------------------------
# Classes
#---------------------------------------------------------
@dataclass
class DataModule:
    data: pd.DataFrame

@dataclass
class Information:
    s: timedelta = timedelta(days=360)
    data_module: DataModule = None
    time_column: str = 'Date'
    company_column: str = 'ticker'
    adj_close_column: str = 'Close'

    def slice_data(self, t: datetime):
        data = self.data_module.data
        s = self.s
        
        data[self.time_column] = pd.to_datetime(data[self.time_column])
        sliced_data = data[(data[self.time_column] >= t - s) & (data[self.time_column] < t)]
        return sliced_data

    def get_prices(self, t: datetime):
        data = self.slice_data(t)
        prices = data.groupby(self.company_column)[self.adj_close_column].last().to_dict()
        return prices

def compute_statistics(data, company_column, adj_close_column):
    """Compute additional statistics for data analysis."""
    stats = {}
    data['return'] = data.groupby(company_column)[adj_close_column].pct_change()
    stats['expected_return'] = data.groupby(company_column)['return'].mean().to_numpy()
    stats['volatility'] = data.groupby(company_column)['return'].std().to_numpy()
    stats['skewness'] = data.groupby(company_column)['return'].skew().to_numpy()
    stats['kurtosis'] = data.groupby(company_column)['return'].apply(pd.Series.kurt).to_numpy()
    return stats

@dataclass
class EnhancedInformation(Information):
    def compute_information(self, t: datetime):
        data = self.slice_data(t)
        information_set = {}

        data['return'] = data.groupby(self.company_column)[self.adj_close_column].pct_change()

        information_set['expected_return'] = data.groupby(self.company_column)['return'].mean().to_numpy()
        information_set['volatility'] = data.groupby(self.company_column)['return'].std().to_numpy()

        pivot_data = data.pivot(index=self.time_column, columns=self.company_column, values=self.adj_close_column)
        pivot_data = pivot_data.dropna()

        information_set['covariance_matrix'] = pivot_data.cov().to_numpy()
        information_set['companies'] = pivot_data.columns.to_numpy()
        return information_set

    def compute_additional_statistics(self, t: datetime):
        data = self.slice_data(t)
        additional_stats = {}

        additional_stats.update(compute_statistics(data, self.company_column, self.adj_close_column))
        return additional_stats

@dataclass
class OptimizedPortfolio(EnhancedInformation):
    def compute_portfolio(self, t: datetime, information_set):
        try:
            mu = information_set['expected_return']
            Sigma = information_set['covariance_matrix']
            gamma = 1
            n = len(mu)

            obj = lambda x: -x.dot(mu) + gamma / 2 * x.dot(Sigma).dot(x)
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = [(0.0, 1.0)] * n
            x0 = np.ones(n) / n

            res = minimize(obj, x0, constraints=cons, bounds=bounds)

            portfolio = {k: None for k in information_set['companies']}
            if res.success:
                for i, company in enumerate(information_set['companies']):
                    portfolio[company] = res.x[i]
            else:
                raise Exception("Optimization did not converge")

            return portfolio
        except Exception as e:
            logging.warning(f"Optimization error: {e}")
            return {k: 1 / len(information_set['companies']) for k in information_set['companies']}


def preprocess_data(data):
    """Prepare data for strategies."""
    data["Date"] = pd.to_datetime(data["Date"]).dt.tz_localize(None)
    wide_data = data.pivot(index="Date", columns="ticker", values="Adj Close")
    wide_data = wide_data.dropna().sort_index()  # Clean and sort data
    return wide_data


