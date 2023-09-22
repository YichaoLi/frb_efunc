import numpy as np
import h5py as h5
import pandas as pd
from tqdm.autonotebook import tqdm

from scipy.ndimage import gaussian_filter

import emcee
import corner
from IPython.display import display, Math

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm


phi_func = lambda E, E_star, phi_star, alpha: phi_star * (E/E_star)**(alpha + 1) * np.exp(-E/E_star)

def plot_steps(samples, labels):
    ndim = samples.shape[1]
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, i], alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    
def log_prior(theta, theta_min=None, theta_max=None):
    #E_star, phi_star, alpha = theta
    #if (38 < E_star < 42) and (1 < phi_star < 5) and (-6 < alpha < 4):
    #    return 0.0
    if theta_min is None:
        theta_min = theta
    if theta_max is None:
        theta_max = theta

    for i in range(len(theta)):
        if theta[i] > theta_max[i] or theta[i] < theta_min[i]:
            return -np.inf
    return 0.

def log_likelihood(theta, x, y, err, theta_min=None, theta_max=None):
    
    lp = log_prior(theta, theta_min=theta_min, theta_max=theta_max)
    if not np.isfinite(lp):
        return -np.inf
    
    E_star, phi_star, alpha = theta
    phi_th = phi_func(x, 10**E_star, 10**phi_star, alpha)
    
    return lp + ( -0.5 * np.sum( (y - phi_th)**2/err**2 ) )
    #return lp + ( -0.5 * np.sum( (np.log(y) - np.log(phi_th) )**2/(err/y)**2 ) )

def peak_upper_lower_band(samples, threshold=0.64, debug=False):
    
    bins_e = np.linspace(samples.min(), samples.max(), 500)
    bins_c = bins_e[:-1] + 0.5 * (bins_e[1:]-bins_e[:-1])
    d      = bins_e[1] - bins_e[0]
    N      = bins_c.shape[0]

    prob = np.histogram(samples, bins=bins_e)[0]
    prob = gaussian_filter(prob, sigma=10)

    _peak_arg = np.argmax(prob)
    v_peak = bins_c[_peak_arg]
    
    if debug:
        fig = plt.figure(figsize=(4,4))
        ax  = fig.subplots()
        ax.plot(bins_c, prob)
        ax.axvline(v_peak, 0, 1)
    
    v_upper = np.percentile(samples[samples>v_peak], threshold*100)
    v_lower = np.percentile(samples[samples<v_peak], (1-threshold)*100)
    
    return v_lower, v_peak, v_upper
    

def fit_model(result, theta_init, theta_min, theta_max, nwalkers=100, nsteps=5000, 
              labels=None, show_steps=False, show_corner=False):

    y    = result[0, :]
    yerr = result[1, :]
    x    = result[2, :]
    
    good = y > 0
    y    = y[good]
    yerr = yerr[good]
    x    = x[good]
    args = (x, y, yerr)

    E_star, phi_star, alpha = theta_init
    if theta_min[2] == theta_max[2]:
        print('Fix alpha to %4.2f'%theta_init[2])
        _L = log_likelihood_fixalpha
        ndim  = len(theta_init) - 1
        pos = np.array([E_star, phi_star])[None, :]
        pos = pos + np.random.randn(nwalkers, ndim) * 1.e-2
        kwargs={'alpha': alpha, 'theta_min': theta_min, 'theta_max':theta_max}
        labels = labels[:-1]
    else:
        _L = log_likelihood
        ndim = len(theta_init)
        pos = np.array([E_star, phi_star, alpha])[None, :]
        pos = pos + np.random.randn(nwalkers, ndim) * 1.e-2
        kwargs={'theta_min': theta_min, 'theta_max':theta_max}

    if labels is None:
        labels = ['Param %02d'%i for i in range(ndim)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, _L, args=args, kwargs=kwargs)
    state = sampler.run_mcmc(pos, nsteps, progress=True)
    
    tau = sampler.get_autocorr_time()
    thin = int(np.max(tau) // 2)
    
    samples = sampler.get_chain(discard=thin, thin=thin, flat=True)
    # add frb rate parameter log Phi
    phi_sample = est_frb_rate(samples, alpha=alpha, logE_min=39, logE_max=41.5)
    samples = np.concatenate([samples, phi_sample[:, None]], axis=1)
    labels.append(r'$\lg \Phi$')

    log_prob = sampler.get_log_prob(discard=thin, thin=thin, flat=True)
    log_prob = np.ma.masked_invalid(log_prob)
    log_prob_argmin = np.ma.argmin(log_prob)
    #print(log_prob.shape, samples.shape)
    #print(log_prob.min(), log_prob.max())
    best_fit_param = samples[np.argmax(log_prob)]
    
    if show_steps:
        plot_steps(samples, labels)
        
    truths = []
    for i in range(ndim + 1):
        #mcmc = np.percentile(samples[:, i], [16, 50, 84])
        mcmc = peak_upper_lower_band(samples[:, i], threshold=0.64)
        q = np.diff(mcmc)
        txt = "{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i][1:-1])
        truths.append([mcmc[1], q[0], q[1], best_fit_param[i]])
        display(Math(txt))

    truths = np.array(truths)
    
    if show_corner:
        corner.corner(samples, labels=labels, truths=truths[:, -1])
        
    return truths, samples

def log_likelihood_fixalpha(theta, x, y, err, alpha=-1.5, theta_min=None, theta_max=None):
    
    lp = log_prior(theta, theta_min=theta_min, theta_max=theta_max)
    if not np.isfinite(lp):
        return -np.inf
    
    E_star, phi_star= theta
    phi_th = phi_func(x, 10**E_star, 10**phi_star, alpha)
    
    return lp + ( -0.5 * np.sum( (y - phi_th)**2/err**2 ) )
    #return lp + ( -0.5 * np.sum( (np.log(y) - np.log(phi_th) )**2/(err/y)**2 ) )


def est_frb_rate(sample, alpha=-1.5, logE_min=39, logE_max=41.5):

    e_bins_e = np.logspace(logE_min, logE_max, 201)
    e_bins_c = e_bins_e[:-1] * (e_bins_e[1:]/e_bins_e[:-1]) ** 0.5
    e_bins_d = np.log10(e_bins_e[1:]) - np.log10(e_bins_e[:-1])

    E_star   = 10.**sample[:, 0][:, None]
    phi_star = 10.**sample[:, 1][:, None]
    if sample.shape[1] == 3:
        alpha    = sample[:, 2][:, None]

    e_bins_c = e_bins_c[None, :]
    e_bins_d = e_bins_d[None, :]

    phi = phi_func(e_bins_c, E_star, phi_star, alpha) * np.log(10.) * e_bins_d
    phi = np.log10(np.sum(phi, axis=1))

    return phi



def plot_results(result_path, param_label, alpha=-1.53):
    
    #samples = []
    with h5.File(result_path, 'r') as fp:
        result   = fp['result'][:] 
        best_fit = fp['best_fit'][:]
        z_bin    = fp['z_bin'][:]
        #for ii in range(z_bin.shape[0]-1):
        #    samples.append(fp['samples_%02d'%ii][:])

    fig_e = plt.figure(figsize=(6, 4))
    #ax_e  = fig_e.subplots()
    ax_e = fig_e.add_axes([0.15, 0.15, 0.8, 0.8])

    index = []
    data = []
    for ii in range(result.shape[2]):

        z_min = z_bin[ii]
        z_max = z_bin[ii+1]
        label = r'$%3.2f < z < %3.2f$'%(z_min, z_max)
        
        ef     = result[0, :, ii]
        ef_err = result[1, :, ii]
        bc     = result[2, :, ii]
        db = ( bc[1]/bc[0] )**0.5
        bc_err = [bc - bc/db, bc * db - bc]
        l = ax_e.errorbar(bc, ef, ef_err, bc_err, fmt='o', ms=8, 
                      elinewidth=2.5, label=label)[0]
        
        E_star   = 10.**best_fit[ii][0, -1]
        phi_star = 10.**best_fit[ii][1, -1]
        if best_fit[ii].shape[0] == 4:
            alpha    = best_fit[ii][2, -1]
        xx = np.logspace(np.log10(1.e38), np.log10(1.e42), 100)
        ax_e.plot(xx, phi_func(xx, E_star, phi_star, alpha), '-', color=l.get_color())

        #sample = samples[ii]
        #phi_sample = est_frb_rate(sample, logE_min=39, logE_max=41.5)
        #phi_result = peak_upper_lower_band(phi_sample, threshold=0.64)
        #phi_best   = est_frb_rate(np.array([[np.log10(E_star), np.log10(phi_star), alpha],]))[0]
        #phi_lower, phi_upper = np.diff(phi_result)

        #display(Math(label))
        #for jj in range(best_fit[ii].shape[0]):
        #    txt = "{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{+{2:.3f}}}"
        #    txt = txt.format(best_fit[ii][jj, 0], best_fit[ii][jj, 1], 
        #                     best_fit[ii][jj, 2], param_label[jj][1:-1])
        #    display(Math(txt))
        #print()
        _data = []
        for jj in range(best_fit[ii].shape[0]):
            txt = "${0:.3f}_{{-{1:.3f}}}^{{+{2:.3f}}}$"
            txt = txt.format(best_fit[ii][jj, -1], best_fit[ii][jj, 1], best_fit[ii][jj, 2])
            _data.append(txt)

        #txt = "${0:.3f}_{{-{1:.3f}}}^{{+{2:.3f}}}$"
        #txt = txt.format(phi_best, phi_lower, phi_upper)
        #_data.append(txt)

        data.append(_data)

        index.append(label)

    df = pd.DataFrame(data, index=pd.Index(index), columns=pd.MultiIndex.from_product(
                    [ param_label ]))


    ax_e.set_xlabel(r'$E\,[{\rm erg}]$')
    ax_e.set_ylabel(r'$E\,{\rm d}n/{\rm d}E\,[{\rm Gpc}^{-3}{\rm yr}^{-1}]$')
    #ax_e.set_ylabel(r'$E{\phi}(E)\,[{\rm Gpc}^{-3}{\rm yr}^{-1}]$')
    ax_e.set_ylim(1.e-1,   1.e5)
    ax_e.set_xlim(1.e38, 2.e42)
    ax_e.loglog()
    ax_e.legend()
    
    return fig_e, ax_e, df

def plot_phi(result_path_list, label_list, shift=None, alpha=-1.5, fmt='o', **kwargs):

    fig = plt.figure(figsize=(6, 4))
    ax  = fig.subplots()

    if shift is None:
        shift = [0, ] * len(result_path_list)

    for rr in range(len(result_path_list)):
        result_path = result_path_list[rr]
        label = label_list[rr]

        with h5.File(result_path, 'r') as fp:
            best_fit = fp['best_fit'][:]
            z_bin    = fp['z_bin'][:]

        phi = []
        phi_err = []
        z = []
        z_err = []
        for ii in range(z_bin.shape[0] - 1):
            z_min = z_bin[ii]
            z_max = z_bin[ii+1]
            z_c = 0.5 * (z_min + z_max) 
            z.append( z_c )
            z_err.append([z_c - z_min, z_max-z_c])
            phi.append(best_fit[ii][-1, -1])
            phi_err.append([best_fit[ii][-1,  1], best_fit[ii][-1,  2]])

        phi     = np.array(phi)
        phi_err = np.array(phi_err)
        z       = np.array(z)
        z_err   = np.array(z_err)
        ax.errorbar(z+shift[rr], phi, yerr=phi_err.T, xerr=z_err.T, fmt=fmt, 
                label=label_list[rr], **kwargs)

    leg = ax.legend(fontsize='large')

    return fig, ax, leg
