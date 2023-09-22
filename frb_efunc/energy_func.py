import numpy as np
import h5py as h5

from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from astropy import constants as const
from astropy.io import fits
from scipy.integrate import quad, dblquad

from tqdm.autonotebook import tqdm

from frb_efunc import likelihood_functions as lfunc


def bandwidth_corr(bandwidth, z):
    
    '''
    correct the bandwith to the rest frame according to Eq.10 and 11 in 
    Hashimoto T., et al., 2022, MNRAS, 511, 1961. doi:10.1093/mnras/stac065
    
    bandwidth in unit of MHz
    '''
    
    dv_itg = 400./ (1 + z)
    dv_frb = bandwidth

    #dv_itg < dv_frb:
    corr = dv_itg / dv_frb
    corr[corr>1] = 1
    return corr
    
    

def frb_random_z(fp, size=100, use_nbar=False, E_min=1.e30, alpha=-1.53, 
        match_gal=True, fix_dm_host=False):
    '''
    generate random z catalogue according to the likelihood function
    '''
    
    rng = np.random.default_rng()
    
    n_frb = len(fp.keys())
    z_sample = np.zeros((n_frb, size))
    energy = np.zeros((n_frb, size))
    weight = np.zeros(n_frb)
    bandwidth = np.zeros((n_frb, size))
    ii = 0
    for key in fp.keys():
        
        #print(key)
        result = fp[key]['result'][:]
        z_g    = fp[key]['z_g'][:]

        if match_gal :
            likeli = fp[key]['likeli'][:]
            if use_nbar:
                likeli /= fp[key]['nbar_g'][:]
        else:
            z_g = np.linspace(z_g.min(), z_g.max(), 1000)
            if fix_dm_host:
                _Like_func = lfunc.Likelihood_DM_fixDMhost
            else:
                _Like_func = lfunc.Likelihood_DM
            dm_ext = result[4]
            likeli = _Like_func(dm_ext, dm_ext * 0.0001, z_g)
        
        likeli /= np.sum(likeli)
        
        _size = size
        _l = 0
        for loop in range(100):
            _z = rng.choice(z_g, _size, p=likeli)
            #_e = fluence_to_energy(result[6], _z, alpha, bandwidth=result[8]*1.e-3)
            bw_corr = bandwidth_corr(result[8], _z)
            _e = fluence_to_energy(result[6], _z, alpha, bandwidth=0.4 * bw_corr)
            
            good = _e > E_min
            _n = np.sum(good.astype('int'))
            z_sample[ii][_l:_l+_n] = _z[good]
            energy[ii][_l:_l+_n] = _e[good]
            bandwidth[ii][_l:_l+_n] = 0.4 * bw_corr[good]
            _l += _n
            _size = size - _l
            if _size == 0: break
            #print(_l, end=' ')
        if loop == 99:
            print("Warning: %s has only %d good z sample"%(key, _l))
        
        weight[ii] = result[-1]
        ii += 1
    
    return z_sample, energy, weight, bandwidth
        
def fluence_to_energy(fluence, z, alpha=-1.53, bandwidth=0.4):
    
    '''
    fluence: Jy ms = 1.e-23 erg / s / cm^2 / Hz * ms = 1.e-26 erg / cm^2 / Hz
    bandwidth: GHz = 1.e9 Hz
    
    '''

    factor = fluence * bandwidth * 1.e-26 * 1.e9 # erg / cm^2
    
    dc = cosmo.comoving_distance(z).to(u.cm).value
    dl = (1 + z) * dc

    return 4 * np.pi * dl**2 / ( (1 + z) ** (2 + alpha) ) * factor

def energy_to_fluence(e, z, alpha=-1.53, bandwidth=0.4):
    
    #print(bandwidth, e)
    factor = e / (bandwidth * 1.e-26 * 1.e9)
    
    dc = cosmo.comoving_distance(z).to(u.cm).value
    dl = (1 + z) * dc
    
    return ( (1 + z) ** (2 + alpha) ) * factor / ( 4 * np.pi * dl**2 )

def V_max_func(e, b, z_min=0.01, z_max=1.0, f_min=10.**(0.5), alpha=-1.53):
    
    '''
    Estimate the maximum Volumn within which each FRB could 
    still be detected (ln(F/[Jy ms])>0.5).
    
    according to https://ui.adsabs.harvard.edu/abs/1968ApJ...151..393S/abstract
    
    
    '''

    #z_min = 0.05
    #z_max = 1.0
    
    d_min = cosmo.comoving_distance(z_min).to(u.Gpc).value
    #d_max = cosmo.comoving_distance(z_max).to(u.Gpc).value
    zz = np.linspace(z_min, z_max, 500)
    
    #f_min = np.exp(0.5)
    Vmax = np.zeros(e.shape)
    for ii, _e in enumerate(e):
        #print(energy_to_fluence(_e, zz, alpha, bandwidth=b[ii]), f_min)
        _f = energy_to_fluence(_e, zz, alpha, bandwidth=b[ii])
        good = _f > f_min
        z_max_idx = np.where(good)[0]
        if len(z_max_idx) != 0:
            z_max_ii = zz[z_max_idx[-1]]
        else:
            z_max_ii = z_min
        d_max = cosmo.comoving_distance(z_max_ii).to(u.Gpc).value
        Vmax[ii] = 4 * np.pi / 3. * (d_max**3 - d_min**3)
        if Vmax[ii] == 0:
            print(z_max_ii, z_max, z_min, _e, _f.min(), f_min)
    
    return Vmax
    

def est_energy_function(frb_cat_file, size=2000, nbin=20, use_nbar=True, 
                        E_min=1.e30, E_bin_min = 1.e37, E_bin_max = 1.e43,
                        z_min=0.1, z_max = 1.0, use_selection=True, 
                        f_sel=0.44, f_min=10**(0.5),alpha=-1.53,
                        match_gal=True, fix_dm_host=False):
    
    '''
    
    number density of FRB per unit time:
    rho = 1 + z / factor
    
    factor: is used for correcting the effect of missing FRBs
    see https://ui.adsabs.harvard.edu/abs/2022MNRAS.511.1961H/abstract
    
    factor = V_max * Omega_sky * t_obs * f_sel
    
    f_sel is the fraction of the sky overlaped with optical survey
    V_max see V_max_func
    '''
    Omega_sky = 0.003 # fractional coverage of the CHIME FoV
    t_obs = 214.8 / 365. # = 0.59[year] survey time.
    
    
    e_bin_e = np.logspace(np.log10(E_bin_min), np.log10(E_bin_max), nbin + 1)
    e_bin_c = e_bin_e[:-1] * (e_bin_e[1:] / e_bin_e[:-1]) ** 0.5
    
    with h5.File(frb_cat_file, 'r') as fp:
        
        z_sample, energy, weight, bandwidth = frb_random_z(fp, size=size, 
                                                           use_nbar=use_nbar, 
                                                           E_min=E_min,
                                                           alpha=alpha,
                                                           match_gal=match_gal, 
                                                           fix_dm_host=fix_dm_host,
                                                           )
    
    energy_list = np.zeros((size, nbin))
    #for ii in range(size):
    for ii in tqdm(range(size), colour='green'):
        _z = z_sample[:, ii]
        _e = energy[:, ii]
        sel = ( _z > z_min ) * (_z < z_max)
        _z = _z[sel]
        _e = _e[sel]
        _b = bandwidth[:, ii][sel]
        Vmax = V_max_func(_e, _b, z_min=z_min, z_max = z_max, f_min=f_min, alpha=alpha)
        factor = Omega_sky * t_obs * Vmax * f_sel
        factor[factor==0] = np.inf
        rho = ( 1 + _z ) / factor
        if use_selection:
            rho *= weight[sel]
        energy_list[ii] = np.histogram(_e, bins=e_bin_e, weights=rho)[0]
        
        energy_list[ii] /= (np.log10(e_bin_e[1:]) - np.log10(e_bin_e[:-1]))
        #energy_list[ii] /= (np.log(e_bin_e[1:]) - np.log(e_bin_e[:-1]))
        
    energy_list = np.array(energy_list)
    energy_mean = np.mean(energy_list, axis=0)
    energy_erro = np.std(energy_list, axis=0)
    
    return energy_mean, energy_erro, e_bin_c, e_bin_e

def est_energy_function_zbin(frb_cat_file, z_bin, nbin=10, size=2000, **keyargs):

    #nbin = 10
    results = np.zeros((3, nbin, z_bin.shape[0]-1))
    for ii in range(z_bin.shape[0]-1):
        z_min = z_bin[ii]
        z_max = z_bin[ii+1]
        ef, ef_err, bc, be = est_energy_function(frb_cat_file, size=size, nbin=nbin, 
                                                 z_min=z_min, z_max = z_max,
                                                 **keyargs)
        
        #bc_err = [bc - be[:-1], be[1:] - bc]
        print(ef)
        
        results[0, :, ii] = ef
        results[1, :, ii] = ef_err
        results[2, :, ii] = bc
    return results

