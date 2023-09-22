import numpy as np
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from astropy import constants as const
from astropy.io import fits
from scipy.integrate import quad, dblquad

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

import warnings
warnings.filterwarnings('ignore')

from tqdm.autonotebook import tqdm

import h5py as h5
import healpy as hp


# ------------------------------------------------------------------------------------------
def Likelihood_DM_host_Zhang(dm_host, z):
    """
    G. Q. Zhang et al 2020 ApJ 900 170
    https://iopscience.iop.org/article/10.3847/1538-4357/abaa4a
    """
    
    #mu_host       = lambda z: 0.4042 * z + 3.606
    mu_host   = lambda z: np.log( 32.97 * (1 + z) ** 0.84 )
    sig_host  = 1.248 
    sig2      = lambda z: sig_host ** 2
    
    L = lambda dm_host, z: np.exp( -0.5 * ( np.log(dm_host) - mu_host(z) ) ** 2 / sig2(z)) \
                          / ( dm_host *  ( 2 * np.pi * sig2(z) )**0.5 )

    return L(dm_host, z) #/ norm[:, None]

def Likelihood_DM_host_Mo(dm_host, z):
    """
    Jian-Feng Mo, Weishan Zhu, Yang Wang, Lin Tang, Long-Long Feng
    MNRAS, Volume 518, Issue 1, January 2023, Pages 539–561, 
    https://doi.org/10.1093/mnras/stac3104

    Table 2 TNG100-1 PStar-VNum DM_host
    """
    
    #mu_host       = lambda z: 0.4042 * z + 3.606
    mu_host   = lambda z: np.log( 63.55 )
    sig_host  = 1.25
    sig2      = lambda z: sig_host ** 2
    
    L = lambda dm_host, z: np.exp( -0.5 * ( np.log(dm_host) - mu_host(z) ) ** 2 / sig2(z)) \
                          / ( dm_host *  ( 2 * np.pi * sig2(z) )**0.5 )

    return L(dm_host, z) #/ norm[:, None]

# ------------------------------------------------------------------------------------------
def DM_IGM_th(z):
    
    # estimate DM IGM according to Deng, Wei and Zhang, Bing ApJ 783:L35 2014
    
    y1   = 1
    y2   = 4 - 3 * y1
    Y_H  = 3./4. * y1
    Y_He = 1./4. * y2
    
    # Assume H is fully ionized at z < 6 
    # Fan, X., Carilli, C. L., & Keating, B. 2006, ARA&A, 44, 415
    # Assume He is fully ionized at z < 2 
    # McQuinn, M., Lidz, A., Zaldarriaga, M., et al. 2009, ApJ, 694, 842
    chi_e_H  = lambda z: 1
    chi_e_He = lambda z: 1
    
    
    Ob0 = cosmo.Ob0
    f_IGM = 0.83 
    H0 = cosmo.H0.value # km / s / Mpc
    rho_c = cosmo.critical_density0.value # g / cm3
    m_p = const.m_p.value * 1.e3 # g
    c = const.c.value * 1.e-3 # km / s
    
    n_e = lambda z: rho_c / m_p * Ob0 * f_IGM * (1 + z)**3 \
                    * (Y_H * chi_e_H(z) + 2 * Y_He * chi_e_He(z) / 4) 
    
    dl =  lambda z: c /(1 + z) / ( cosmo.H(z).value / 1.e6)
    #print(n_e(1))
    
    func_int = lambda z: n_e(z) / (1 + z) * dl(z)
    dm_IGM = lambda z: quad(func_int, 0, z)[0]
    
    #return func_int(z)
    return np.vectorize(dm_IGM)(z)

def Likelihood_DM_IGM(dm_igm, z):
    
    sig_IGM = lambda z: 173.8 * z ** 0.4
    sig2    = lambda z: sig_IGM(z)**2

    L = lambda dm_igm, z: np.exp( -0.5*(dm_igm-DM_IGM_th(z))**2/sig2(z) ) / ( (2 * np.pi * sig2(z))**0.5 )
    
    #zz = np.linspace(0, 1, 2000)
    #norm = np.ma.masked_invalid( L(zz, dm_ext) )
    #norm = np.ma.sum(norm, axis=1) #* ( zz[1] - zz[0] )
    
    return L(dm_igm, z) #/ norm[:, None]

# ------------------------------------------------------------------------------------------
def Likelihood_DM_dblquad(dm_ext, sigma_dm, z):
    
    # dm_ext = DM - DM_mw - DM_halo = DM_host + DM_IGM
    
    L_DM = lambda dm_host, dm_igm, dm_ext, sigma_dm, z: \
        np.exp( -0.5*(dm_ext-(dm_host+dm_igm))**2/(sigma_dm**2) )
    L_DM_host = Likelihood_DM_host
    L_DM_igm  = Likelihood_DM_IGM
    
    func_int = lambda dm_host, dm_igm, dm_ext, sigma_dm, z: \
    L_DM(dm_host, dm_igm, dm_ext, sigma_dm, z) * L_DM_host(dm_host, z) * L_DM_igm(dm_igm, z)
    
    dm_host_min = 0
    
    L = lambda dm_ext, sigma_dm, z: dblquad(func_int, 0, dm_ext, 0, lambda x: dm_ext - x, 
                                            args=(dm_ext, sigma_dm, z))
    
    return np.vectorize(L)(dm_ext, sigma_dm, z)

@np.vectorize
def Likelihood_DM(dm_ext, sigma_dm, z, N=1000, dm_host_model = 'Zhang'):
    
    # dm_ext = DM - DM_mw - DM_halo = DM_host + DM_IGM
    
    L_DM = lambda dm_host, dm_igm, dm_ext, sigma_dm, z: \
        np.exp( -0.5 * ( dm_ext-(dm_host+dm_igm))**2/(sigma_dm**2) )  \
        / ( np.sqrt(2. * np.pi) * sigma_dm )

    L_DM_host = globals()['Likelihood_DM_host_%s'%dm_host_model]
    L_DM_igm  = Likelihood_DM_IGM
    
    func_int = lambda dm_host, dm_igm, dm_ext, sigma_dm, z: \
    L_DM(dm_host, dm_igm, dm_ext, sigma_dm, z) * L_DM_host(dm_host*(1+z), z) * L_DM_igm(dm_igm, z)
    
    dm_host_min = 0
    
    #L = lambda dm_ext, sigma_dm, z: dblquad(func_int, 0, dm_ext, 0, lambda x: dm_ext - x, 
    #                                        args=(dm_ext, sigma_dm, z))
    
    dm_igm_e  =   np.linspace(10, dm_ext, N + 1, dtype='float32')
    dm_igm_d  =   dm_igm_e[1:] - dm_igm_e[:-1]
    dm_igm_c  =  (dm_igm_e[1:] + dm_igm_e[:-1]) * 0.5
    dm_igm_c  =   dm_igm_c[:, None]
    dm_igm_d  =   dm_igm_d[:, None]
    
    dm_host_e =   np.linspace(0, dm_ext, N + 1, dtype='float32')
    dm_host_d =   dm_host_e[1:] - dm_host_e[:-1]
    dm_host_c =  (dm_host_e[1:] + dm_host_e[:-1]) * 0.5
    dm_host_c =   dm_host_c[None, :]
    dm_host_d =   dm_host_d[None, :]
    
    mask = ( dm_igm_c + dm_host_c ) >= dm_ext
    
    L = func_int(dm_host_c, dm_igm_c, dm_ext, sigma_dm, z)
    L[mask] = 0
    L *= dm_igm_d * dm_host_d
    
    L = np.sum(L, axis=(0, 1))
    
    return L

# ------------------------------------------------------------------------------------------
@np.vectorize
def Likelihood_DM_fixDMhost(dm_ext, sigma_dm, z, N=1000, dm_host_model=''):
    
    #mu_host = lambda z: np.log( 32.97 * (1 + z) ** 0.84 )
    mu_host = lambda z: np.log( 50./(1+z) ) 
    
    # dm_ext = DM - DM_mw - DM_halo = DM_host + DM_IGM
    
    L_DM = lambda dm_host, dm_igm, dm_ext, sigma_dm, z: \
        np.exp( -0.5 * ( dm_ext-(dm_host+dm_igm))**2/(sigma_dm**2) )  / ( np.sqrt(2. * np.pi) * sigma_dm )
    #L_DM_host = Likelihood_DM_host
    L_DM_igm  = Likelihood_DM_IGM
    
    func_int = lambda dm_host, dm_igm, dm_ext, sigma_dm, z: \
    L_DM(dm_host, dm_igm, dm_ext, sigma_dm, z) * L_DM_igm(dm_igm, z)
    
    dm_host_min = 0
    
    #L = lambda dm_ext, sigma_dm, z: dblquad(func_int, 0, dm_ext, 0, lambda x: dm_ext - x, 
    #                                        args=(dm_ext, sigma_dm, z))
    
    dm_igm_e  =   np.linspace(10, dm_ext, N + 1, dtype='float32')
    dm_igm_d  =   dm_igm_e[1:] - dm_igm_e[:-1]
    dm_igm_c  =  (dm_igm_e[1:] + dm_igm_e[:-1]) * 0.5
    dm_igm_c  =   dm_igm_c[:, None]
    dm_igm_d  =   dm_igm_d[:, None]
    
    #dm_host_e =   np.linspace(0, dm_ext, N + 1, dtype='float32')
    #dm_host_d =   dm_host_e[1:] - dm_host_e[:-1]
    #dm_host_c =  (dm_host_e[1:] + dm_host_e[:-1]) * 0.5
    #dm_host_c =   dm_host_c[None, :]
    #dm_host_d =   dm_host_d[None, :]
    dm_host = np.exp(mu_host(z))
    
    mask = ( dm_igm_c + dm_host ) >= dm_ext
    
    L = func_int(dm_host, dm_igm_c, dm_ext, sigma_dm, z)
    L[mask] = 0
    #print(L.min(),L.max())
    if not np.all(np.isfinite(L)):
        print(L.max(), L.min())
    L *= dm_igm_d
    
    L = np.sum(L, axis=(0))
    
    return L
