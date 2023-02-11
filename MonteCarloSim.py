#Monte Carlo Method to simulate stock returns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr 

#Import data
def get_data(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start, end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

stockList = ["RIG","SU", "CCJ", "VRTX", "PYPL", "OSTK",
             "APPS","FRES.L", "BMY", "LMT", "PHGP.L", "PHSP.L"] 

endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days = 300)

meanReturns, covMatrix = get_data(stockList, startDate, endDate)

weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)

print(weights)

##MonteCarlo Method
mc_sims = 1000
T = 10 #tf in days

#creating matrices
meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
meanM = meanM.T #Transforming mean

portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0) #empty array with number of sims and tf

initialPortfolio = 10000

for m in range(0, mc_sims):
    #MC Loops
    Z = np.random.normal(size = (T, len(weights)))
    L = np.linalg.cholesky(covMatrix) #lower triangle,cholesky decomposition
    dailyReturns = meanM + np.inner(L,Z)
    if m == 0:
        portfolio_sims[:,m] = initialPortfolio
    else:
        portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initialPortfolio #cumulative returns over the 100 days of simulation

plt.plot(portfolio_sims)
plt.legend(stockList)
plt.ylabel("Portfolio Value")
plt.xlabel("Days")
plt.title("MC Simulation of The Spark Portfolio")
plt.show()
