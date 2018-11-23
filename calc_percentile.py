import numpy as np
import matplotlib.pyplot as plt
from pre_graph_calcs import pre_graph_calcs
import scipy.stats as st

#calculating percentile for the input volume and displaying it on the curves

def calc_percentile(MA, volume):
    gestation, percentiles, lnPercentiles, sdlog = pre_graph_calcs()  # get relevant data to create graph and specific p
    # calculate exact percentile for data point

    # fixed_ln50p_for_point = np.array([x[0] for x in ln50p_for_point]) ???? cant remember what this is

    row = np.where(gestation == MA)
    ln50p_of_current_volume = lnPercentiles[row, 5]
    fixed_ln50p_current_volume=ln50p_of_current_volume[0][0]
    ln_current_volume = np.log(volume)
    sigma = (np.subtract(ln_current_volume, fixed_ln50p_current_volume)) / sdlog
    exact_p = np.round(st.norm.cdf(sigma), 3)

    fig, ax = plt.subplots()
    ax.scatter(MA,volume)

    dims = np.shape(percentiles)
    nametags = ['3rd', '5th', '10th', '25th', '50th', '75th', '90th', '95th', '97th']
    colors = ['b', 'g', 'y', 'm', 'c', 'm', 'y', 'g', 'b']
    for i in range(1, dims[1]):
        specificPrecentage = percentiles[:, i]
        plt.plot(gestation, specificPrecentage, '%s' % colors[i - 1], label='%s precentile' % nametags[i - 1])
    plt.text(30,40,'percentile: {}'.format(exact_p))
    plt.legend()
    plt.xlabel('gestation [weeks]')
    plt.ylabel('brain volume [cubed centimeters]')
    plt.title('brain volume vs. gestation')
    plt.show()

    # TODO: add exact p curve