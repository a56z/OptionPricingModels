from numpy import exp, sqrt, log
from scipy.stats import norm
import numpy as np
import scipy.integrate as integrate

class BlackScholes:
    def __init__(self, time_to_maturity, strike, current_price, volatility, interest_rate):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate

    def calculate_prices(self):
        time_to_maturity = self.time_to_maturity
        strike = self.strike
        current_price = self.current_price
        volatility = self.volatility
        interest_rate = self.interest_rate

        d1 = (log(current_price / strike) + (interest_rate + 0.5 * volatility ** 2) * time_to_maturity) / (volatility * sqrt(time_to_maturity))
        d2 = d1 - volatility * sqrt(time_to_maturity)

        call_price = current_price * norm.cdf(d1) - (strike * exp(-(interest_rate * time_to_maturity)) * norm.cdf(d2))
        put_price = (strike * exp(-(interest_rate * time_to_maturity)) * norm.cdf(-d2)) - current_price * norm.cdf(-d1)

        self.call_price = call_price
        self.put_price = put_price

        # GREEKS
        # Delta
        self.call_delta = norm.cdf(d1)
        self.put_delta = 1 - norm.cdf(d1)

        # Gamma
        self.call_gamma = norm.pdf(d1) / (strike * volatility * sqrt(time_to_maturity))
        self.put_gamma = self.call_gamma

        return call_price, put_price


class MertonJumpDiffusion:
    def __init__(self, time_to_maturity, strike, current_price, volatility, interest_rate, lambda_j, mu_j, sigma_j):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate
        self.lambda_j = lambda_j
        self.mu_j = mu_j
        self.sigma_j = sigma_j

    def calculate_prices(self):
        time_to_maturity = self.time_to_maturity
        strike = self.strike
        current_price = self.current_price
        volatility = self.volatility
        interest_rate = self.interest_rate
        lambda_j = self.lambda_j
        mu_j = self.mu_j
        sigma_j = self.sigma_j

        call_price = 0.0
        put_price = 0.0
        for k in range(50):  # Sum up to 50 terms for numerical approximation
            r_k = interest_rate - lambda_j * (exp(mu_j + 0.5 * sigma_j**2) - 1) + (k * log(1 + mu_j)) / time_to_maturity
            sigma_k = sqrt(volatility**2 + (k * sigma_j**2) / time_to_maturity)
            
            d1_k = (log(current_price / strike) + (r_k + 0.5 * sigma_k**2) * time_to_maturity) / (sigma_k * sqrt(time_to_maturity))
            d2_k = d1_k - sigma_k * sqrt(time_to_maturity)
            
            call_price_k = (exp(-lambda_j * time_to_maturity) * (lambda_j * time_to_maturity)**k / np.math.factorial(k)) * (
                current_price * norm.cdf(d1_k) - strike * exp(-r_k * time_to_maturity) * norm.cdf(d2_k))
            put_price_k = (exp(-lambda_j * time_to_maturity) * (lambda_j * time_to_maturity)**k / np.math.factorial(k)) * (
                strike * exp(-r_k * time_to_maturity) * norm.cdf(-d2_k) - current_price * norm.cdf(-d1_k))
            
            call_price += call_price_k
            put_price += put_price_k

        self.call_price = call_price
        self.put_price = put_price

        return self.call_price, self.put_price


class HestonStochasticVolatility:
    def __init__(self, time_to_maturity, strike, current_price, volatility, interest_rate, kappa, theta, sigma, rho, v0):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate
        self.kappa = kappa  # Rate at which variance reverts to theta
        self.theta = theta  # Long-run variance
        self.sigma = sigma  # Volatility of variance
        self.rho = rho  # Correlation between the asset price and its variance
        self.v0 = v0  # Initial variance

    def characteristic_function(self, u):
        time_to_maturity = self.time_to_maturity
        current_price = self.current_price
        interest_rate = self.interest_rate
        kappa = self.kappa
        theta = self.theta
        sigma = self.sigma
        rho = self.rho
        v0 = self.v0

        xi = kappa - sigma * rho * 1j * u
        d = np.sqrt(xi**2 + sigma**2 * (1j * u + u**2))
        g = (xi - d) / (xi + d)

        cf = np.exp(1j * u * (np.log(current_price) + (interest_rate - 0.5 * v0) * time_to_maturity))
        cf *= np.exp((v0 / sigma**2) * ((1 - np.exp(-d * time_to_maturity)) / (1 - g * np.exp(-d * time_to_maturity))) * (xi - d))
        cf *= np.exp(kappa * theta * ((time_to_maturity * (xi - d)) - 2 * np.log((1 - g * np.exp(-d * time_to_maturity)) / (1 - g))))

        return cf

    def integrand(self, u, sign):
        return (np.exp(-1j * u * np.log(self.strike)) * self.characteristic_function(u - 1j * sign)) / (1j * u * self.strike**sign)

    def calculate_prices(self):
        P1 = 0.5 + (1 / np.pi) * integrate.quad(lambda u: np.real(self.integrand(u, 1)), 0, 100)[0]
        P2 = 0.5 + (1 / np.pi) * integrate.quad(lambda u: np.real(self.integrand(u, -1)), 0, 100)[0]

        call_price = (self.current_price * P1) - (self.strike * np.exp(-self.interest_rate * self.time_to_maturity) * P2)
        put_price = call_price - self.current_price + (self.strike * np.exp(-self.interest_rate * self.time_to_maturity))

        self.call_price = call_price
        self.put_price = put_price

        return self.call_price, self.put_price


class BatesStochasticVolatilityJumpDiffusion:
    def __init__(self, *args, **kwargs):
        pass
    def calculate_prices(self):
        return 0.0, 0.0

class SABRStochasticVolatility:
    def __init__(self, *args, **kwargs):
        pass
    def calculate_prices(self):
        return 0.0, 0.0

class VasicekMeanRevertingDiffusion:
    def __init__(self, *args, **kwargs):
        pass
    def calculate_prices(self):
        return 0.0, 0.0

class CoxIngersollRossSquareRootDiffusion:
    def __init__(self, *args, **kwargs):
        pass
    def calculate_prices(self):
        return 0.0, 0.0

class SquareRootJumpDiffusion:
    def __init__(self, *args, **kwargs):
        pass
    def calculate_prices(self):
        return 0.0, 0.0

if __name__ == "__main__":
    time_to_maturity = 2
    strike = 90
    current_price = 100
    volatility = 0.2
    interest_rate = 0.05
    lambda_j = 0.1  # Jump intensity
    mu_j = -0.1    # Average jump size
    sigma_j = 0.3  # Jump size volatility
    kappa = 2.0    # Rate of mean reversion of variance
    theta = 0.04   # Long run average variance
    sigma = 0.5    # Volatility of the variance
    rho = -0.5     # Correlation between the asset and variance
    v0 = 0.04      # Initial variance

    # Example using Merton Jump Diffusion
    MJD = MertonJumpDiffusion(time_to_maturity, strike, current_price, volatility, interest_rate, lambda_j, mu_j, sigma_j)
    MJD.calculate_prices()

    # Example using Heston Stochastic Volatility
    HSV = HestonStochasticVolatility(time_to_maturity, strike, current_price, volatility, interest_rate, kappa, theta, sigma, rho, v0)
    HSV.calculate_prices()
