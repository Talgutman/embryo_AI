from pandas import DataFrame
import statsmodels.api as sm
import pandas as pd
import numpy as np
import scipy.stats as st


def pre_graph_calcs():
    # obtaining regression model from the original healthy data
    healthyData = pd.read_csv(r'database.csv')
    df = DataFrame(healthyData, columns=['MA squared', 'MA', 'ln of brain volume'])
    X = df[['MA', 'MA squared']]
    Y = df['ln of brain volume']
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    predictions = model.predict(X)
    regressionParameters = model.params

    # use obtained parameters to predict volume for each age
    gestation = np.arange(18.0, 40.5, 0.5)
    lnBrainVolume = regressionParameters[0] + regressionParameters[1] * gestation + regressionParameters[2] * (
            gestation ** 2)

    # calculate precentiles
    lnPrecentiles = np.zeros((len(lnBrainVolume), 9))
    sdlog = 0.18078
    Zp = [st.norm.ppf(0.03), st.norm.ppf(0.05), st.norm.ppf(0.1), st.norm.ppf(0.25), st.norm.ppf(0.5), st.norm.ppf(0.75), st.norm.ppf(0.9), st.norm.ppf(0.95), st.norm.ppf(0.97)]


    for i in range(len(lnBrainVolume)):
        for j in range(9):
            lnPrecentiles[i][j] = lnBrainVolume[i] + Zp[j] * sdlog
    temp = np.zeros((len(lnBrainVolume), 10))
    temp[:, 0] = gestation
    temp[:, 1:10] = lnPrecentiles
    lnPrecentiles = temp
    precentiles = np.zeros(np.shape(lnPrecentiles))
    precentiles[:, 0] = lnPrecentiles[:, 0]
    precentiles[:, 1:10] = np.exp(lnPrecentiles[:, 1:10])

    return gestation, precentiles, lnPrecentiles, sdlog

