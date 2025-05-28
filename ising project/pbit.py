import numpy
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import scienceplots
from matplotlib import ticker

p_reco = list(numpy.loadtxt('p_num3.txt'))
step_reco = list(numpy.loadtxt('v_num3.txt'))

p_reco.append(1)
step_reco.append(4.3)

p_reco.append(0)
step_reco.append(3.2)

p_reco.append(1)
step_reco.append(4.4)

p_reco.append(0)
step_reco.append(3.1)

p_reco.append(0)
step_reco.append(3.0)

p_reco.append(1)
step_reco.append(4.5)
an = interp1d(step_reco, p_reco, kind=3)
def func(p_reco=p_reco, step_reco=step_reco, x=0):
    # an = numpy.polyfit(step_reco, p_reco, 2)
    if x < 3.2:
        return 0
    elif x > 4.2:
        return 1
    else:
        yvals = an(x)
        if yvals > 1:
            return 1
        elif yvals < 0:
            return 0
        else:
            return yvals

if __name__ == '__main__':
    x = numpy.arange(3, 4.5, 0.02)
    y = []
    for i in range(0,len(x)):
        y.append(func(x=x[i]))

    y = np.array(y)
    plt.style.use(['science', 'nature'])
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    #
    font1 = {'family': 'Arial',
                     'weight': 'normal',
                     'size': 14,
            }
    font2 = {'family': 'Arial',
                     'weight': 'normal',
                     'size': 10,
            }
    visible_ticks = {
       "top": False,
       "right": False,
    }
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tick_params(axis="x", which="both", **visible_ticks)
    plt.tick_params(axis="y", which="both", **visible_ticks)
    plt.xlabel("Vin (V)", font1)
    plt.ylabel(u'SET Probability', font1)
    plt.yticks(fontproperties='Arial', size=9, weight='semibold')
    plt.xticks(fontproperties='Arial', size=9, weight='semibold')

    plt.plot(x, y, linewidth = 2.0, zorder=1, label="Interpolate Curve")
    plt.scatter(step_reco, p_reco, c="r", marker='o',zorder=2,alpha=0.5, label="Data point")
    plt.legend(loc="upper left",prop=font2)
    ax = plt.gca()
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    plt.show()