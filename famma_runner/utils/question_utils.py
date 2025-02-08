import pandas as pd
import numpy as np
from scipy.stats import norm

class QuestionValidator:
    """
    Utils for validating questions
    """
    
    def __init__(self):
        pass
    
    @staticmethod
    def validate_strategy_question(csv_dir: str) -> bool:
        """
        Validate a strategy question
        """
        data_df = pd.read_csv(csv_dir, header=0)

        def sharpe(account_values: np.array, risk_free_rate, annualize_coefficient):
            # log-return
            diff = np.log(account_values[1:]) - np.log(account_values[:-1])
            # we'll get the mean and calculate everything
            # we multply the mean of returns by the annualized coefficient and divide by annualized std
            annualized_std = diff.std() * np.sqrt(annualize_coefficient)
            print(diff.std())
            return (diff.mean() * annualize_coefficient - risk_free_rate) / annualized_std
        
        def sortino(account_values: np.array, risk_free_rate, annualize_coefficient):
            diff = np.log(account_values[1:]) - np.log(account_values[:-1])
            # we'll get the mean and calculate everything
            # remember, we're only using the negative returns
            neg_returns = diff[diff < 0]
            annualized_std = neg_returns.std() * np.sqrt(annualize_coefficient)
            return (diff.mean() * annualize_coefficient - risk_free_rate) / annualized_std
        
        def value_at_risk(account_values: np.array, confidence_level: float):
            # ref: https://medium.com/@serdarilarslan/value-at-risk-var-and-its-implementation-in-python-5c9150f73b0e
            # ref: https://github.com/ibaris/VaR/blob/main/src/var/methods.py
            returns = np.log(account_values[1:]) - np.log(account_values[:-1])
            returns_sorted = np.sort(returns) # sort our returns (should be estimated normal)
            index = int(confidence_level * len(returns_sorted))
            print(np.percentile(returns, 100 - (confidence_level * 100), method="lower"))
            print(abs(returns_sorted[index]))
            return np.percentile(returns, 100 - (confidence_level * 100), method="lower")


        risk_free_rate = 0.0005
        annualize_coefficient = 250
        sortino_ratio = sortino(data_df['fund_1'].values, risk_free_rate, annualize_coefficient)
        print(f'sortino ratio with risk-free rate of {risk_free_rate} and annualization coefficient of {annualize_coefficient}: {sortino_ratio}')

        risk_free_rate = 0.0003
        annualize_coefficient = 1
        sortino_ratio = sortino(data_df['fund_1'].values, risk_free_rate/250, annualize_coefficient)
        print(f'sortino ratio with risk-free rate of {risk_free_rate} and annualization coefficient of {annualize_coefficient}: {sortino_ratio}')

        risk_free_rate = 0.0001
        annualize_coefficient = 250
        sharpe_ratio = sharpe(data_df['fund_2'].values, risk_free_rate, annualize_coefficient)
        print(f'sharpe ratio with risk-free rate of {risk_free_rate} and annualization coefficient of {annualize_coefficient}: {sharpe_ratio}')

        var = value_at_risk(data_df['fund_2'].values, 0.99)
        print(f'value at risk with confidence level of 0.99: {var}')

    @staticmethod
    def validate_option_question(s_0: float, k: float, r: float, t: float, sigma: float, c: float = 0) -> bool:
        """
        Validate an option pricing question
        """
        # black scholes
        d1 = (np.log(s_0 / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        term_1 = s_0 * norm.cdf(d1)
        term_2 = k * np.exp(-r * t) * norm.cdf(d2)
        term_3 = -0.5 * c * s_0 * norm.cdf(-d1)
        call_price = term_1 - term_2 + term_3
        print(f'call price: {call_price}')


if __name__ == '__main__':
    question_validator = QuestionValidator()
    question_validator.validate_strategy_question(csv_dir='fund_performance.csv')

    question_validator.validate_option_question(s_0=48, k=45, r=0.02, t=1.25, sigma=0.1, c=0.0001)