#Monte Carlo Method to simulate stock returns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr #yahoo specific data

#Import data
def get_data(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start, end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

stockList = ["RIG","SU", "CCJ", "VRTX", "PYPL", "APPS", "BNTX", "OSTK",
            "PHGP.L", "PHSP.L", "FRES.L", "RICA.L", "BMY", "LMT"]
#stocks = [stock + '.L' for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days = 300)

meanReturns, covMatrix = get_data(stockList, startDate, endDate)

weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)

print(weights)

##MonteCarlo Method
mc_sims = 100
T = 100 #tf in days

#creating matrices
meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
meanM = meanM.T #Transforming mean

portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0) #empty array with number of sims and tf

initialPortfolio = 10000

for m in range(0, mc_sims):
    #MC Loops
    #cholesky decomposition
    Z = np.random.normal(size = (T, len(weights)))
    L = np.linalg.cholesky(covMatrix) #lower triangle
    dailyReturns = meanM + np.inner(L,Z)
    portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initialPortfolio #cumulative returns over the 100 days of simulation

plt.plot(portfolio_sims)
plt.ylabel("Portfolio Value")
plt.xlabel("Days")
plt.title("MC Simulation of The Spark Portfolio")
plt.show()