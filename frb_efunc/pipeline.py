import numpy as np
import h5py as h5

from frb_efunc import utils
from frb_efunc import est_functions as efunc
from frb_efunc import energy_func as ef
from frb_efunc import fitting_functions as ff


def match_galaxy_est_energy_func(cat_path, cat_name, galaxy_path, z_bin=None, dm_mw='ne2001', 
                                 none_repeat=False, gal_cat=True, use_nbar=False, suffix='',
                                 match_gal=True, do_matching=True, fix_dm_host=False, size=2000,
                                 dm_host_model='Zhang', threshold_dm_max=1.e3):
    
    if gal_cat:
        suffix += '_galcat'
    else:
        suffix += '_grpcat'
    
    if none_repeat:
        suffix += '_onceoff'
    else:
        suffix += '_repeat'

    name, data = utils.read_chime_frb_cat(cat_path, cat_name, dm_mw=dm_mw, 
            none_repeat=none_repeat, threshold_dm_max=threshold_dm_max)
    
    #galaxy_path = '/home/ycli/data/group/groups_DESI/'
    with h5.File(galaxy_path, 'r') as f:
        galaxy_cat = f['cat'][:]
    
    #print(data.shape)
    n_frb = data.shape[0]
    suffix += '%d'%n_frb
    
    suffix += '_%s'%dm_mw
    
    if use_nbar:
        suffix += '_nbar'
        
    if fix_dm_host:
        suffix += '_fixDMhost'
    else:
        suffix += '_dm%s'%dm_host_model
    
    output_name ='match_result%s'%suffix
    output_file = './output/%s.h5'%output_name
    if match_gal and do_matching:
        efunc.match_galaxy_cat(data, name, galaxy_cat, output_file, dm_halo = 30., 
                plot=True, use_nbar=use_nbar, timing=False, fix_dm_host=fix_dm_host,
                dm_host_model=dm_host_model)
    if not match_gal:
        output_name = output_name.replace('match', 'nomatch')

    print(output_name)
    
    frb_cat_file = output_file
    if z_bin is None:
        z_bin = np.linspace(0.05, 1.0, 3)
    results = ef.est_energy_function_zbin(output_file, z_bin, nbin=10, size=size,
            use_nbar=use_nbar, E_min=1.e30, E_bin_min = 1.e38, E_bin_max = 1.e42, alpha=-1.53,
            match_gal=match_gal, fix_dm_host=fix_dm_host)

    output_file_ef = './output/%s_EF.h5'%output_name

    with h5.File(output_file_ef, 'w') as fp:
        fp['z_bin']  = z_bin
        fp['result'] = results


def fit_EF_model(file_ef, theta_init, theta_min, theta_max, labels = None,
                 nwalkers=100, nsteps=10000, show_steps=True, show_corner=True, 
                 output_path=None):
    
    with h5.File(file_ef, 'r') as fp:
        z_bin  = fp['z_bin'][:]
        result = fp['result'][:]
    
    #param_init = [E_star, phi_star, alpha]
    
    best_fit = []
    samples = []
    for i in range(z_bin.shape[0]-1):
        _r = ff.fit_model(result[:, :, i], theta_init, theta_min, theta_max, 
                labels=labels, nwalkers=nwalkers, nsteps=nsteps, 
                show_steps=show_steps, show_corner=show_corner)
        best_fit.append(_r[0])
        samples.append(_r[1])
    
    #samples = np.array(samples)
    if output_path is not None:
        with h5.File(output_path, 'w') as fp:
            fp['result']   = result
            fp['best_fit'] = np.array(best_fit)
            fp['z_bin'] = z_bin
            for i in range(z_bin.shape[0]-1):
                fp['samples_%02d'%i] = samples[i]
            
    return result, best_fit, z_bin, samples

