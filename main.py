from pandas import DataFrame
import statsmodels.api as sm
import csv
import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import os

#TODO: append date that the data was added and the name of the file it was added from
def applyA():
    added_file = input("name of file containing added data: ")
    append_data = np.genfromtxt(added_file, delimiter=',')  # augmenting the new data to fit the regression
    dim = np.shape(append_data)
    addSqauredVol = np.zeros((dim[0], 3))
    addSqauredVol[:, 1:3] = append_data[:, 0:2]
    addSqauredVol[:, 0] = (append_data[:, 0]) ** 2
    with open('database.csv', 'a', newline='') as my_csv:  # adding the new data to the data of the curves
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(addSqauredVol)


    gestation, precentiles, lnPrecentiles, sdlog = pre_graph_calcs()  # creating the new curves
    # creating precentiles graph
    dims = np.shape(precentiles)
    nametags = ['3rd', '5th', '10th', '25th', '50th', '75th', '90th', '95th', '97th']
    colors = ['b', 'g', 'y', 'm', 'c', 'm', 'y', 'g', 'b']
    for i in range(1, dims[1]):
        specificPrecentage = precentiles[:, i]
        plt.plot(gestation, specificPrecentage, '%s' % colors[i - 1], label='%s precentile' % nametags[i - 1])
    plt.legend()
    plt.xlabel('gestation [weeks]')
    plt.ylabel('brain volume [cubed centimeters]')
    plt.title('brain volume vs. gestation')
    plt.show()


def applyB():
    gestation, percentiles, lnPercentiles, sdlog = pre_graph_calcs()  # get relevant data to create graph and specific p
    # calculate exact percentile for data point
    added_file = input("name of file containing added data: ")
    data_points = np.genfromtxt(added_file, delimiter=',')
    dim = np.shape(data_points)
    ln50p_for_point = np.zeros((dim[0], 1))  # get the 50th precentile from chart, needed for the calc of exact precentile
    for i in range(dim[0]):
        current_point = data_points[i]
        row = np.where(gestation == current_point[0])
        ln50p_for_point[i] = lnPercentiles[row, 5]
    actual_lnvolume = data_points[:, 1]
    fixed_ln50p_for_point = np.array([x[0] for x in ln50p_for_point])
    sigma = (np.subtract(actual_lnvolume, fixed_ln50p_for_point)) / sdlog
    exact_p = np.round(st.norm.cdf(sigma), 3)


    # plot graph with data points
    data_points[:, 1] = np.exp(data_points[:, 1])
    fig, ax = plt.subplots()
    ax.scatter(data_points[:, 0], data_points[:, 1])

    for i, txt in enumerate(exact_p):
        ax.annotate(txt, (data_points[:, 0][i], data_points[:, 1][i]))

    dims = np.shape(percentiles)
    nametags = ['3rd', '5th', '10th', '25th', '50th', '75th', '90th', '95th', '97th']
    colors = ['b', 'g', 'y', 'm', 'c', 'm', 'y', 'g', 'b']
    for i in range(1, dims[1]):
        specificPrecentage = percentiles[:, i]
        plt.plot(gestation, specificPrecentage, '%s' % colors[i - 1], label='%s precentile' % nametags[i - 1])
    plt.legend()
    plt.xlabel('gestation [weeks]')
    plt.ylabel('brain volume [cubed centimeters]')
    plt.title('brain volume vs. gestation')

    plt.show()


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
    SDdistance = [st.norm.ppf(0.03), st.norm.ppf(0.05), st.norm.ppf(0.1), st.norm.ppf(0.25), st.norm.ppf(0.5), st.norm.ppf(0.75), st.norm.ppf(0.9), st.norm.ppf(0.95), st.norm.ppf(0.97)]

    #sdlog=np.zeros((len(gestation), 1))
    #indx_of_age = np.where(df["MA"] == 20.5)
    #temp = df['ln of brain volume']
    #print(np.std(Y[indx_of_age[0]]))
    # counter = 0
    # for i in gestation:
    #     indx_of_age = np.where(df["MA"] == i)
    #     sdlog[counter] = np.std(Y[indx_of_age[0]])
    #     counter += 1
    # print(sdlog)


    for i in range(len(lnBrainVolume)):
        for j in range(9):
            lnPrecentiles[i][j] = lnBrainVolume[i] + SDdistance[j] * sdlog
    temp = np.zeros((len(lnBrainVolume), 10))
    temp[:, 0] = gestation
    temp[:, 1:10] = lnPrecentiles
    lnPrecentiles = temp
    precentiles = np.zeros(np.shape(lnPrecentiles))
    precentiles[:, 0] = lnPrecentiles[:, 0]
    precentiles[:, 1:10] = np.exp(lnPrecentiles[:, 1:10])

    return gestation, precentiles, lnPrecentiles, sdlog


def main():
    print("Choose option:\n1 - update percentile curves\n2 - calculate the data's percentile")
    options = input('your choice is: ')
    if options == '1':
        applyA()
    elif options == '2':
        applyB()
    else:
        print("option doesn't exist")
        main()

main()
