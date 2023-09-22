import numpy as np
import healpy as hp
import h5py as h5

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from astropy import constants as const
from frb_efunc import likelihood_functions as lfunc
from frb_efunc import utils

import warnings
warnings.filterwarnings('ignore')

from tqdm.autonotebook import tqdm



# ------------------------------------------------------------------------------------
def _L(i, args):
    dm_ext, sigma_dm, z = args
    return Likelihood_DM_dblquad(dm_ext, sigma_dm, z[i])[0]

def match_galaxy_cat(data, name, group_cat, output_file, dm_halo = 30., plot=False, use_nbar=True, 
                     timing=False, fix_dm_host=False, dm_host_model='Zhang'):
    
    frb_ra   = data[:, 0]
    frb_dra  = data[:, 1]
    frb_dec  = data[:, 2]
    frb_ddec = data[:, 3]
    dm_ext   = data[:, 4] - dm_halo # DM_measured - DM_MW - DM_halo
    dm_ext_err = dm_ext * 0.0001
    fluence  = data[:, 5]
    fluence_err = data[:, 6]
    high_freq = data[:, 12]
    low_freq  = data[:, 13]
    weights  = data[:, -1]
    
    g_N   = group_cat.shape[0]
    g_ra  = group_cat[:, 2]
    g_dec = group_cat[:, 3]
    g_z   = group_cat[:, 4]
    g_mass= group_cat[:, 5]
    
    if fix_dm_host:
        _Like_func = lfunc.Likelihood_DM_fixDMhost
    else:
        _Like_func = lfunc.Likelihood_DM
    
    #if use_nbar:
    nbar_func = est_nbar(group_cat)
    
    if plot:
        fig = plt.figure(figsize=[10, 4])
        ax  = fig.subplots(1)
    
    results = []
    
    with h5.File(output_file, 'w') as fp:
        
        for ii in tqdm(range(frb_ra.shape[0]), colour='green'):
            ra_l = (frb_ra[ii]  - frb_dra[ii])
            ra_r = (frb_ra[ii]  + frb_dra[ii])
            
            if ra_l < 0:
                sel = ((g_ra > 0) * (g_ra <= ra_r)) + (g_ra > ra_l + 360.)
            elif ra_r > 360:
                sel = ((g_ra > ra_l) * (g_ra <= 360)) + g_ra < ra_r - 360
            else:
                sel = (g_ra > ra_l) * (g_ra <= ra_r)
        
            dec_l = (frb_dec[ii] - frb_ddec[ii])
            dec_r = (frb_dec[ii] + frb_ddec[ii])
            sel *= ( g_dec > dec_l ) * ( g_dec <= dec_r )
            
            if np.any(sel):
                #print(ra_l, ra_r, dec_l, dec_r)
                #print(np.sum(sel.astype('int')))
                zz = g_z[sel]
                mass = g_mass[sel]
    
                if timing:
                    t0 = time.time()
                    print('FRB %3d: %5d galaxies. '%(ii, zz.shape[0]), end='')
                
                #with mp.Pool(32) as p:
                #    l = p.map(partial(_L, args=args), range(zz.shape[0]))
                l = _Like_func(dm_ext[ii], dm_ext_err[ii], zz, dm_host_model=dm_host_model)
                
                if timing:
                    t1 = time.time()
                    print('\t Use %6.3f min '%((t1 - t0)/60.))
    
                l = np.ma.masked_invalid(l)
                
                if plot: ax.plot(zz, l, '.')
                
                l = np.ma.filled(l, 0)
                if not np.all(np.isfinite(l)):
                    print(l.max(),l.min())
                if np.all(l==0):
                    print('all 0 in likeli, no matched gal')
                w = l * mass
                if use_nbar:
                    w /= nbar_func(zz)
                
                z_est = np.sum(w * zz) / np.sum(w)
                
                _r = [frb_ra[ii], frb_dra[ii], frb_dec[ii], frb_ddec[ii], dm_ext[ii], 
                      z_est, fluence[ii], fluence_err[ii], high_freq[ii]-low_freq[ii], 
                      weights[ii]]
                
                results.append(_r)
                
                #fp['%s/result'%name[ii]] = [frb_ra[ii], frb_dra[ii], frb_dec[ii], frb_ddec[ii], dm_ext[ii], z_est, ]
                fp['%s/result'%name[ii]] = _r
                fp['%s/likeli'%name[ii]] = l
                fp['%s/mass_g'%name[ii]] = mass
                fp['%s/nbar_g'%name[ii]] = nbar_func(zz)
                fp['%s/z_g'%name[ii]   ] = zz
                
            #pbar.update(n=1)
    
    #return results


def est_nbar(group_cat, plot=False):

    nside = 256
    des_pix_bin = np.arange(hp.nside2npix(nside) + 1) - 0.5
    group_pix = hp.ang2pix(nside, group_cat[:, 2], group_cat[:, 3], lonlat=True)
    des_footprint = np.histogram(group_pix, bins=des_pix_bin)[0]
    des_footprint[des_footprint>0] = 1
    
    n_pix = np.sum(des_footprint)
    s_pix = hp.nside2pixarea(nside, degrees=False)
    f_sky = n_pix * s_pix / (4. * np.pi) 
    print("Survey Area %6.2f degree square (%3.2f )"%(n_pix * s_pix / (np.pi/180.)**2, f_sky))
    
    
    z_bins = np.linspace(0.1, 1., 30)
    z_cent = 0.5 * (z_bins[1:] + z_bins[:-1])
    hist, bins = np.histogram(group_cat[:, 4], bins=z_bins)
    
    r = cosmo.comoving_distance(z_bins)
    v = f_sky * 4 * np.pi * ( 0.5 * ( r[1:] + r[:-1] ) ) **2 * (r[1:] - r[:-1])
    
    n_bar = hist/v
    n_fit_func = np.poly1d( np.polyfit(z_cent, n_bar, 6) )
    
    if plot:
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        ax.plot(z_cent, 1./n_bar, 'r.-', lw=2, drawstyle='steps-mid')
        zzz = np.linspace(0, z_cent.max(), 200)
        ax.plot(zzz, 1./n_fit_func(zzz), 'k-', lw=1.5)
        
    return n_fit_func


def est_z_with_error(dm_ext, dm_ext_err, z_best, threshold=0.90, N = 100):
    z_max = max(1, 2 * z_best)
    zz = np.linspace(0, z_max, N)
    dz = zz[1] - zz[0]
    likeli_th = lfunc.Likelihood_DM(dm_ext, dm_ext_err, zz)
    likeli_th = np.ma.masked_invalid(likeli_th)
    likeli_th = np.ma.filled(likeli_th, 0)
    
    z_peak_arg = np.argmax(likeli_th)
    z_peak = zz[z_peak_arg]
    
    # find upper lim
    z_upper = z_peak
    _upper = 0.5 * likeli_th[z_peak_arg] * dz
    norm_upper = np.sum(likeli_th[z_peak_arg + 1:] * dz) + _upper
    #print(norm_upper)
    for ii in range(z_peak_arg + 1, N):
        if _upper >= norm_upper * threshold:
            break
        _upper += likeli_th[ii] * dz
        z_upper = zz[ii]
    #print(ii)
        
    # find lower lim
    z_lower = z_peak
    _lower = 0.5 * likeli_th[z_peak_arg] * dz
    norm_lower = np.sum(likeli_th[0:z_peak_arg] * dz) + _lower
    for ii in range(z_peak_arg -1, 0, -1):    
        if _lower >= norm_lower * threshold:
            break
        _lower += likeli_th[ii] * dz
        z_lower = zz[ii]
    
    return likeli_th, zz, z_upper, z_lower
