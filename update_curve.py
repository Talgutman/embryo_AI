import csv
import numpy as np
import matplotlib.pyplot as plt
from pre_graph_calcs import pre_graph_calcs

#adding healthy data to change the curves

def update_curve():
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


