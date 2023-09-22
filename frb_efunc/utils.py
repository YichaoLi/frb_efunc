import openpyxl
import numpy as np

from astropy.io import fits


def read_chime_frb_cat_txt(cat_path, cat_name, dm_min=100, dm_max=1200):
    
    wb = openpyxl.load_workbook(cat_path + cat_name).active
    
    name         = np.array([x.value for x in  wb['A'][1:]])
    ra           = np.array([x.value for x in  wb['B'][1:]])[:, None]
    ra_err       = np.array([x.value for x in  wb['C'][1:]])[:, None]
    dec          = np.array([x.value for x in  wb['D'][1:]])[:, None]
    dec_err      = np.array([x.value for x in  wb['E'][1:]])[:, None]
    dm_exc_ymw16 = np.array([x.value for x in  wb['F'][1:]])[:, None]
    fluence      = np.array([x.value for x in  wb['J'][1:]])[:, None]
    fluence_err  = np.array([x.value for x in  wb['K'][1:]])[:, None]
    
    sel = ( dm_exc_ymw16[:, 0] > dm_min ) * ( dm_exc_ymw16[:, 0] < dm_max )
    
    results = [ra, ra_err, dec, dec_err, dm_exc_ymw16, fluence, fluence_err]
    return name[sel], np.concatenate(results, axis=1)[sel]

# --------------------------------------------------------------------------------

# selection function from https://arxiv.org/abs/2201.03574

s_DM  = lambda DM : -0.7707 * np.log10(DM )**2 + 4.5601 * np.log10(DM ) - 5.6291
s_tau = lambda tau: -0.2922 * np.log10(tau)**2 - 1.0196 * np.log10(tau) + 1.4592
s_w   = lambda w  : -0.0785 * np.log10(w  )**2 - 0.5435 * np.log10(w  ) + 0.9574
s_F   = lambda F  : 10.**(1.7173 * (1.0 - np.exp(-2.0348 * np.log10(F))) - 1.7173)

#s_DM  = lambda DM : -0.7707 * np.log(DM )**2 + 4.5601 * np.log(DM ) - 5.6291
#s_tau = lambda tau: -0.2922 * np.log(tau)**2 - 1.0196 * np.log(tau) + 1.4592
#s_w   = lambda w  : -0.0785 * np.log(w  )**2 - 0.5435 * np.log(w  ) + 0.9574
#s_F   = lambda F  : np.exp(1.7173 * (1.0 - np.exp(-2.0348 * np.log(F))) - 1.7173)


def read_chime_frb_cat(cat_path, cat_name, threshold_snr=10, threshold_dm_min=100,
                       threshold_dm_max = 1000, threshold_tau = 0.8, none_repeat=True, 
                       full_sample=False, dm_mw='ymw16'):
    
    hdul = fits.open(cat_path + cat_name)
    data = hdul[1].data
    
    # selection 
    bonsai_snr    = data.field('bonsai_snr')
    dm_fitb       = data.field('dm_fitb')
    dm_obs        = data.field('bonsai_dm')
    dm_exc_ymw16  = data.field('dm_exc_ymw16')
    ymw16         = dm_fitb - dm_exc_ymw16
    dm_exc_ne2001 = data.field('dm_exc_ne2001')
    ne2001        = dm_fitb - dm_exc_ne2001
    dm_mk         = np.max(np.concatenate([ymw16[:, None], ne2001[:, None]], axis=1), axis=1)
    excluded_flag = data.field('excluded_flag') == 0
    scat_time     = data.field('scat_time')*1.e3
    fluence       = data.field('fluence')
    sub_num       = data.field('sub_num')
    flux_notes    = data.field('flux_notes')
    
    repeater_name = data.field('repeater_name')
    
    sel = (bonsai_snr>threshold_snr) * (dm_obs>1.5*dm_mk)\
        * (dm_obs<threshold_dm_max) \
        * excluded_flag * (np.log10(scat_time)<threshold_tau) * (np.log10(fluence) > 0.5)\
        * (sub_num == 0) * ( flux_notes == '-9999')
    if none_repeat:
        sel *= repeater_name == '-9999'
    else:
        sel *= repeater_name != '-9999'
        
    N_frb = np.sum(sel.astype('int'))
    
    field_list = ['ra', 'ra_err', 'dec', 'dec_err', 'dm_exc_%s'%dm_mw, 
                  'fluence', 'fluence_err', 'width_fitb', 'width_fitb_err', 
                  'scat_time', 'scat_time_err', 'bonsai_dm', 
                  'high_freq', 'low_freq']
    
    tns_name = data.field('tns_name')[sel]
    results = []
    for _f in field_list:
        results.append(data.field(_f)[sel][:, None])
    
    hdul.close()

    w_DM  = 1./s_DM(results[11])
    w_tau = 1./s_tau(results[9]*1.e3)
    w_w   = 1./s_tau(results[7]*1.e3)
    w_F   = 1./s_F(results[5])
    
    weights = w_DM * w_tau * w_w * w_F
    weights[~np.isfinite(weights)] = 0
    
    weights = weights/np.sum(weights) * 84697./39638. * N_frb
    
    results.append(weights)
    
    results = np.concatenate(results, axis=1)
    
    return tns_name, results
