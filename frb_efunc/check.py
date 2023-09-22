import h5py as h5
import numpy as np
import healpy as hp


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

from astropy.io import fits

from reproject import reproject_from_healpix, reproject_to_healpix
from astropy.wcs import WCS
from astropy.visualization.wcsaxes.frame import RectangularFrame, EllipticalFrame


def plot_likeli(match_cat_file, output='./plot.png', use_nbar=False, 
                DM_min=100, DM_max=1000):
    
    norm = mpl.colors.Normalize(vmin=DM_min, vmax=DM_max)
    cmap = cm.jet
    
    fig_e = plt.figure(figsize=(7, 4))
    ax_e  = fig_e.add_axes([0.09, 0.12, 0.79, 0.85])
    ax_c  = fig_e.add_axes([0.89, 0.12, 0.02, 0.85])
    
    with h5.File(match_cat_file, 'r') as fp:

        n_frb = len(fp.keys())
        ii = 0
        for key in fp.keys():
            
            #print(key)
            result = fp[key]['result'][:]
            likeli = fp[key]['likeli'][:]
            z_g    = fp[key]['z_g'][:]
            
            dm_ext = result[4]
            
            if use_nbar:
                likeli /= fp[key]['nbar_g'][:]
            
            #likeli /= np.sum(likeli)
            #likeli /= np.max(likeli)
            
            ax_e.plot(z_g, likeli*1.e3, '.', ms=1, c=cmap(norm(dm_ext)))
    
    ax_e.set_ylabel(r'$P(z|{\rm DM})\times 10^{-3}$')
    ax_e.set_xlabel(r'z')
    
    
    fig_e.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax_e, cax=ax_c)
    ax_c.set_ylabel(r'$({\rm DM}_{\rm IGM} + {\rm DM}_{\rm host})\,[{\rm pc}\,{\rm cm}^{-3}]$')
    ax_c.set_yticklabels(ax_c.get_yticks(), rotation = 90)
    
    #fig_e.savefig(output, dpi=300)
    fig_e.savefig(output)


# plot catalogue

def get_projection(proj, imap_shp, field_center, pix):

    target_header = "NAXIS   = 2\n"\
                  + "NAXIS1  = %d\n"%imap_shp[0]\
                  + "NAXIS2  = %d\n"%imap_shp[1]\
                  + "CTYPE1  = \'RA---%s\'\n"%proj\
                  + "CRPIX1  = %d\n"%(imap_shp[0]/2)\
                  + "CRVAL1  = %f\n"%field_center[0]\
                  + "CDELT1  = -%f\n"%pix\
                  + "CUNIT1  = \'deg\'\n"\
                  + "CTYPE2  = \'DEC--%s\'\n"%proj\
                  + "CRPIX2  = %d\n"%(imap_shp[1]/2)\
                  + "CRVAL2  = %f\n"%field_center[1]\
                  + "CDELT2  = %f\n"%pix\
                  + "CUNIT2  = \'deg\'\n"\
                  + "COORDSYS= \'icrs\'"

    target_header = fits.Header.fromstring(target_header, sep = '\n')

    return target_header, WCS(target_header)

def plot_des_footprint(des_footprint, figsize = (8, 6), imap_shp = (1300, 1100), 
                       field_center = (15, -23), pix = 0.1,
                       proj='ZEA', cmap='Greens', plot_cat=False, frb_cat=None, 
                       output=None, title='', axes=None):

    if axes is None:
        fig = plt.figure(figsize=figsize)
        target_header, projection = get_projection(proj, imap_shp, field_center, pix)
        ax = fig.add_axes([0.06, 0.1, 0.92, 0.8], projection=projection, 
                          frame_class=RectangularFrame)
    else:
        fig, ax, target_header, projection = axes
    
    #print(des_footprint.min(), des_footprint.max())
    array, footprint = reproject_from_healpix((des_footprint, 'icrs'), target_header,
                                              nested=False, order='nearest-neighbor')

    if plot_cat:
        array = np.ma.masked_invalid(array)
        array[array==0] = np.ma.masked
        vmin = np.ma.min(array)
        vmax = np.ma.max(array)
        #print(vmin, vmax)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    
        im = ax.pcolormesh(array, cmap=cmap, norm=norm)
    else:
        array[array>0] = 1.
        ax.contour(array, levels=[0.1, 1,], colors='g', linewidths=1.0)
        ax.contourf(array, levels=[0.1, 1,], colors='g', linewidths=2.5, alpha=0.3)
        
    if frb_cat is not None:
        frb_ra   = frb_cat[:, 0]
        frb_dec  = frb_cat[:, 2]

        ax.plot(frb_ra[:], frb_dec[:], 'o', mec='k', mfc='none', ms=5, mew=1,
                transform=ax.get_transform('icrs'))
    
    ax.minorticks_on()
    ax.set_aspect('equal')
    #ax.coords[0].set_ticks(spacing=300 * u.arcmin,)
    ax.coords[0].set_major_formatter('hh:mm')
    ax.coords[0].set_separator((r'$^{\rm h}$', r'$^{\rm m}$', r'$^{\rm s}$'))
    ax.coords[1].set_major_formatter('dd:mm')
    ax.coords[1].set_axislabel('Dec. (J2000)', minpad=0.5)
    ax.coords.grid(color='0.5', linestyle='--', lw=0.5)
    ax.coords[0].set_axislabel('R.A. (J2000)', minpad=0.5)
    
    ax.set_title(title)
    
    if output is not None:
        
        fig.savefig(output + 'frb_footprint.png', format='png', dpi=300)
        


def plot_catalogue(group_cat, data, des_footprint=None, figsize=(10, 5), output=None, 
        plot_cat=False):

    if des_footprint is None:
        nside = 256
        des_pix_bin = np.arange(hp.nside2npix(nside) + 1) - 0.5
        group_pix = hp.ang2pix(nside, group_cat[:, 2], group_cat[:, 3], lonlat=True)
        des_footprint = np.histogram(group_pix, bins=des_pix_bin)[0]
    else:
        nside = hp.npix2nside(des_footprint.shape[0])
    
    # SGCP
    sel  = ( data[:, 4] > 0 ) * ( data[:, 4] < 1000 )
    sel *= ( data[:, 2] < 35 ) * (data[:, 2] > -60)
    #sel *= ( data[:, 0] < 100 ) * ( data[:, 0] > -50)
    sel *= ( data[:, 0] < 100 ) + ( data[:, 0] > 360-50)
    frb_pix = hp.ang2pix(nside, data[:, 0], data[:, 2], lonlat=True)
    sel *= des_footprint[frb_pix] > 0
    
    
    fig = plt.figure(figsize=figsize)
    target_header, projection = get_projection(proj='ZEA', imap_shp = (1300, 1100), 
                                               field_center = (15, -23), pix = 0.1)
    ax = fig.add_axes([0.07, 0.1, 0.43, 0.8], projection=projection, 
                      frame_class=RectangularFrame)
    axes = [fig, ax, target_header, projection]
    print(data[sel].shape)
    plot_des_footprint(des_footprint, frb_cat=data[sel], title='SGC', axes=axes, plot_cat=plot_cat)
    
    # NGCP
    sel  = ( data[:, 4] > 0 ) * ( data[:, 4] < 1000 )
    sel *= ( data[:, 2] > -10 ) 
    #sel *= ( data[:, 0] > 90 ) + ( data[:, 0] < -50)
    sel *= ( data[:, 0] > 100 ) * ( data[:, 0] < 360-50)
    frb_pix = hp.ang2pix(nside, data[:, 0], data[:, 2], lonlat=True)
    sel *= des_footprint[frb_pix] > 0
    
    
    target_header, projection = get_projection(proj='ZEA', imap_shp = (1300, 1100), 
                                               field_center = (195, 40), pix = 0.1)
    ax = fig.add_axes([0.50, 0.1, 0.43, 0.8], projection=projection, 
                      frame_class=RectangularFrame)
    axes = [fig, ax, target_header, projection]
    print(data[sel].shape)
    plot_des_footprint(des_footprint, frb_cat=data[sel], title='NGC', axes=axes, plot_cat=plot_cat)
    ax.coords[1].set_axislabel_position('r')
    ax.coords[1].set_ticklabel_position('r')
    
    if output is not None:
        #fig.savefig(output + 'frb_footprint.png', format='png', dpi=300)
        fig.savefig(output + 'frb_footprint.pdf', format='pdf')
    
