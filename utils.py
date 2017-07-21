import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from skopt.acquisition import gaussian_ei
from skopt.acquisition import gaussian_lcb


def plot_optimizer(opt, x, acq_name='EI'):
    model = opt.models[-1]
    x_model = opt.space.transform(x.tolist())

    # Plot Model(x) + contours
    y_pred, sigma = model.predict(x_model, return_std=True)
    plt.plot(x, y_pred, "g--", label=r"$\mu(x)$")
    plt.fill(np.concatenate([x, x[::-1]]),
             #np.concatenate([y_pred - 1.9600 * sigma,
             #                 (y_pred + 1.9600 * sigma)[::-1]]),
             np.concatenate([y_pred - sigma,
                             (y_pred + sigma)[::-1]]),
             alpha=.2, fc="g", ec="None")

    # Plot sampled points
    plt.plot(opt.Xi, opt.yi,
             "ro", label="Observations")

    if acq_name == 'EI':
        acq = gaussian_ei(x_model, model, y_opt=np.min(opt.yi),
                          **opt.acq_func_kwargs)
        acq /= acq.max()
    elif acq_name == 'LCB':
        acq = gaussian_lcb(x_model, model, **opt.acq_func_kwargs)
        acq /= acq.min()

    # shift down to make a better plot
    acq = acq - 2
    plt.plot(x, acq, "b", label="%s(x)" % acq_name)
    plt.fill_between(x.ravel(), -2.0, acq.ravel(), alpha=0.3, color='blue')

    # Adjust plot layout
    plt.grid()
    plt.legend(loc='best')


def midpoint(x):
    return x[0] + (x[1] - x[0])/2


def spread(x):
    return np.power((x[1] - x[0]) / 2., 2)


def beer_gauss(ABV, IBU):
    means = midpoint(ABV), midpoint(IBU)
    cov = ((spread(ABV), 0.), (0, spread(IBU)))
    return stats.multivariate_normal(mean=means, cov=cov)


def beer_score(points):
    # alcohol and IBU dimensions
    # session beers
    session = beer_gauss((3.5, 5.), (10, 35))
    # american IPA
    american_ipa = beer_gauss((6.3, 7.5), (50, 70))
    # american imperial red ale
    american_imperial = beer_gauss((8, 10.6), (55, 85))
    american_wheat_ale = beer_gauss((8.5, 12.2), (45, 85))

    # english bitter
    bitter = beer_gauss((3, 4.2), (20, 35))

    # belgian pale ale
    belgian_pale_ale = beer_gauss((4, 6), (20, 30))
    # belgian dubble
    belgian_dubble = beer_gauss((6.3, 7.6), (20, 35))
    belgian_triple = beer_gauss((7.1, 10.1), (20, 45))
    # belgian quadruple
    belgian_quad = beer_gauss((7.2, 11.2), (25, 50))
    belgian_golden = beer_gauss((7, 11), (20, 50))

    # vienna lager
    vienna_lager = beer_gauss((4.5, 5.5), (22, 28))

    # weizen bock
    weizen_bock = beer_gauss((7, 9.5), (15, 35))
    maibock = beer_gauss((6, 8), (20, 38))
    weizen_dunkel = beer_gauss((4.8, 5.4), (10, 15))

    background = beer_gauss((2, 14), (0, 80))

    Z = (10 * background.pdf(points) +
         3 * session.pdf(points) +
         14.2 * american_ipa.pdf(points) +
         2 * american_imperial.pdf(points) +
         15 * american_wheat_ale.pdf(points) +
         5 * bitter.pdf(points) +
         5 * belgian_pale_ale.pdf(points) +
         1 * belgian_dubble.pdf(points) +
         7 * belgian_triple.pdf(points) +
         5 * belgian_quad.pdf(points) +
         6 * belgian_golden.pdf(points) +
         1 * vienna_lager.pdf(points) +
         3.2 * weizen_bock.pdf(points) +
         2.9 * weizen_dunkel.pdf(points) +
         7 * maibock.pdf(points))
    return Z / 0.66
