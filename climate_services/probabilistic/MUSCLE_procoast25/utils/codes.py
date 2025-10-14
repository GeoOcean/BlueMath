import os
import os.path as op
import sys
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize,ListedColormap, LinearSegmentedColormap
import matplotlib as mpl
from matplotlib import cm
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle as pk
from scipy.stats import entropy

import cartopy
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

#########################################################################################################
##                                             CALCULATIONS                                            ##
#########################################################################################################

def fix_dir(base_dirs):
    'fix csiro direction for wavespectra (from -> to)'
    new_dirs = base_dirs + 180
    new_dirs[np.where(new_dirs>=360)] = new_dirs[np.where(new_dirs>=360)] - 360
    
    return new_dirs

def Set_t_h_sp(sp):
    
    sp=sp.rename({'frequency':'freq', 'direction':'dir'})
    sp['efth']=sp['efth']*(np.pi/180)
        
    sp['dir']=fix_dir(sp['dir'])
    
    sp['t']=(['freq'],1/sp.freq.values)    
    h=np.full(np.shape(sp.efth.values),0.0)
    freq_d=np.diff(sp.freq.values)
    freq_d=np.append(0.00001,freq_d)
    for i in range(len(sp.freq.values)):
        h[:,i,:]=((sp.efth.values[:,i,:])*(sp.dir.values[0]-sp.dir.values[1])*freq_d[i])
    sp['h']=(['time','freq','dir'],h)
    
    return sp

def sp_group_bins(sp, dir_bins, t_bins):
    
    print('Direction bins:' + str(dir_bins))
    print('---------------------------------')
    print('Period bins:' + str(t_bins))
    
    sp_mod = xr.Dataset(
        {
            'h': (['time','t','dir'], sp.h.values),
        },
        coords={
            't': sp.t.values,
            'dir': sp.dir.values,
            'time': sp.time.values,
        },
    )
    sp_mod=sp_mod.groupby_bins('dir',dir_bins).sum()
    sp_mod=sp_mod.groupby_bins('t',t_bins).sum()
    sp_mod['dir']=('dir_bins',(dir_bins)[:-1])#+0.5*(dir_bins[0]-dir_bins[1]))
    sp_mod['t']=('t_bins',t_bins[1:])
    # sp_mod['h']=4*np.sqrt(sp_mod['h'])
    sp_mod['h_t']=4*np.sqrt(sp_mod['h'])
    
    return sp_mod


def n34_format(n34, rolling_mean=[]):

    n34 = xr.Dataset(
        {
            'index': (['time'], n34[:,1:].reshape(-1)),
        },
        coords={
            'time': pd.date_range(datetime(n34[0,0].astype('int'),1,1), datetime(n34[-1,0].astype('int')+1,1,1) , freq='1M') ,
        },
    )
    n34.index[np.where(n34.index==-99.99)]=np.nan

    #Means are calculated from the period 1991-2020, same as the forecast
    n34['index']=n34.index-np.nanmean(n34.sel(time=slice('1991-01-01','2020-12-31')).index)
    
    
    if rolling_mean:
        n34_rol=n34.rolling(time=3, center=True).mean()
#         n34_rol=n34_rol.sel(time=slice('1979-01-01','2021-01-01'))
#         n34=n34.sel(time=slice('1979-01-01','2021-01-01'))
        n34=n34_rol
    
    return n34


def pdo_format(pdo, rolling_mean=[], rol_months=3):
    
    years=pdo[:,0]

    pdo = xr.Dataset(
        {
            'index': (['time'], pdo[:,1:].reshape(-1)),
        },
        coords={
            'time': np.arange(str(years[0].astype('int')) + '-01', str(years[-1].astype('int')+1) + '-01', dtype='datetime64[M]') ,
        },
    )
    pdo.index[np.where(pdo.index==-99.99)]=np.nan

    if rolling_mean:
        
        pdo_rol=pdo.rolling(time=rol_months, center=True).mean()
        
        ## 3-month rolling mean

#         pdo_rol=pdo_rol.sel(time=slice('1979-01-01','2021-01-01'))
        pdo=pdo_rol
    
    return pdo

def nao_format(nao, rolling_mean=[], rol_months=3):
    
    years=nao[:,0]

    nao = xr.Dataset(
        {
            'index': (['time'], nao[:,1:].reshape(-1)),
        },
        coords={
            'time': np.arange(str(years[0].astype('int')) + '-01', str(years[-1].astype('int')+1) + '-01', dtype='datetime64[M]') ,
        },
    )
    nao.index[np.where(nao.index==-99.99)]=np.nan

    if rolling_mean:
        
        nao_rol=nao.rolling(time=rol_months, center=True).mean()
        
        ## 3-month rolling mean

#         nao_rol=nao_rol.sel(time=slice('1979-01-01','2021-01-01'))
        nao=nao_rol
    
    return nao

def pna_format(pna, rolling_mean=[], rol_months=3):
    
    years=pna[:,0]

    pna = xr.Dataset(
        {
            'index': (['time'], pna[:,1:].reshape(-1)),
        },
        coords={
            'time': np.arange(str(years[0].astype('int')) + '-01', str(years[-1].astype('int')+1) + '-01', dtype='datetime64[M]') ,
        },
    )
    pna.index[np.where(pna.index==-99.99)]=np.nan

    if rolling_mean:
        
        pna_rol=pna.rolling(time=rol_months, center=True).mean()
        
        ## 3-month rolling mean

        pna=pna_rol
    
    return pna

def sam_format(sam, rolling_mean=[], rol_months=3):
    
    years=sam[:,0]

    sam = xr.Dataset(
        {
            'index': (['time'], sam[:,1:].reshape(-1)),
        },
        coords={
            'time': np.arange(str(years[0].astype('int')) + '-01', str(years[-1].astype('int')+1) + '-01', dtype='datetime64[M]') ,
        },
    )
    sam.index[np.where(sam.index==-99.99)]=np.nan

    if rolling_mean:
        
        sam_rol=sam.rolling(time=rol_months, center=True).mean()
        
        ## 3-month rolling mean

#         sam_rol=sam_rol.sel(time=slice('1979-01-01','2021-01-01'))
        sam=sam_rol
    
    return sam

def ao_format(ao, rolling_mean=[], rol_months=3):
    
    years=ao[:,0]

    ao = xr.Dataset(
        {
            'index': (['time'], ao[:,1:].reshape(-1)),
        },
        coords={
            'time': np.arange(str(years[0].astype('int')) + '-01', str(years[-1].astype('int')+1) + '-01', dtype='datetime64[M]') ,
        },
    )
    ao.index[np.where(ao.index==-99.99)]=np.nan

    if rolling_mean:
        
        ao_rol=ao.rolling(time=rol_months, center=True).mean()
        
        ## 3-month rolling mean

#         sam_rol=sam_rol.sel(time=slice('1979-01-01','2021-01-01'))
        ao=ao_rol
    
    return ao


def olr_format(olr, rolling_mean=[], rol_months=3):
    
    years=olr[:,0]

    olr = xr.Dataset(
        {
            'index': (['time'], olr[:,1:].reshape(-1)),
        },
        coords={
            'time': np.arange(str(years[0].astype('int')) + '-01', str(years[-1].astype('int')+1) + '-01', dtype='datetime64[M]') ,
        },
    )
    olr.index[np.where(olr.index==-999.9)]=np.nan

    if rolling_mean:
        
        olr_rol=olr.rolling(time=rol_months, center=True).mean()
        
        ## 3-month rolling mean

#         sam_rol=sam_rol.sel(time=slice('1979-01-01','2021-01-01'))
        olr=olr_rol
    
    return olr

def soi_format(soi, rolling_mean=[], rol_months=3):
    
    years=soi[:,0]

    soi = xr.Dataset(
        {
            'index': (['time'], soi[:,1:].reshape(-1)),
        },
        coords={
            'time': np.arange(str(years[0].astype('int')) + '-01', str(years[-1].astype('int')+1) + '-01', dtype='datetime64[M]') ,
        },
    )
    soi.index[np.where(soi.index==-999.9)]=np.nan

    if rolling_mean:
        
        soi_rol=soi.rolling(time=rol_months, center=True).mean()
        
        ## 3-month rolling mean

#         sam_rol=sam_rol.sel(time=slice('1979-01-01','2021-01-01'))
        soi=soi_rol
    
    return soi

def amo_format(amo, rolling_mean=[], rol_months=3):
    
    years=amo[:,0]

    amo = xr.Dataset(
        {
            'index': (['time'], amo[:,1:].reshape(-1)),
        },
        coords={
            'time': np.arange(str(years[0].astype('int')) + '-01', str(years[-1].astype('int')+1) + '-01', dtype='datetime64[M]') ,
        },
    )
    amo.index[np.where(amo.index==-99.99)]=np.nan

    if rolling_mean:
        
        amo_rol=amo.rolling(time=rol_months, center=True).mean()
        
        ## 3-month rolling mean

#         sam_rol=sam_rol.sel(time=slice('1979-01-01','2021-01-01'))
        amo=amo_rol
    
    return amo

def scand_format(scand, rolling_mean=[], rol_months=3):
    
    years=scand[:,0]

    scand = xr.Dataset(
        {
            'index': (['time'], scand[:,2].reshape(-1)),
        },
        coords={
            'time': np.arange(str(years[0].astype('int')) + '-01', str(years[-1].astype('int')+1) + '-01', dtype='datetime64[M]') ,
        },
    )
    scand.index[np.where(scand.index==-99.99)]=np.nan

    if rolling_mean:
        
        scand_rol=scand.rolling(time=rol_months, center=True).mean()
        
        ## 3-month rolling mean

        scand=scand_rol
    
    return scand

def obtain_combination_positions(sp_mod_daily, comb):
    
    for c in range(len(comb)):
        
        if comb[c][0]=='month':
            a=[]
            for m in range(len(comb[c][1])):
                a=np.concatenate([a,np.where(sp_mod_daily.time.dt.month.values==comb[c][1][m])[0]])
        elif comb[c][0]=='year':
            a=[]
            for m in range(len(comb[c][1])):
                a=np.concatenate([a,np.where(sp_mod_daily.time.dt.year.values==comb[c][1][m])[0]])    
        else:
            a=np.where(sp_mod_daily[comb[c][0]].values==comb[c][1])[0]
            
        if c==0: comb_pos=a
        else: comb_pos=np.intersect1d(comb_pos,a)

    print('Number of times in intersection : ' + str(len(comb_pos)))
    
    return comb_pos.astype('int')

#########################################################################################################
##                                               PLOTTING                                              ##
#########################################################################################################


def Plot_Map_Locations(locations, coordinates, figsize=[25,11], iloc=-1):
    fig = plt.figure(figsize=figsize)

    ax = plt.axes(projection = ccrs.PlateCarree(central_longitude=190))
    ax.stock_img()

    # cartopy land feature
    land_10m = cartopy.feature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='darkgrey', facecolor='gainsboro',  zorder=1)
    ax.add_feature(land_10m)
    ax.gridlines()

    # scatter data points
    vmin, vmax =96000, 103000

#     p1=plt.pcolor(est.longitude.values, est.latitude.values, est.F.values, transform=ccrs.PlateCarree(),zorder=2, cmap='plasma')
    for i in range(len(locations)):
        plt.plot(coordinates[i][0], coordinates[i][1], '.', color='navy', markersize=20, transform=ccrs.PlateCarree(),zorder=2)
        if iloc>=0:
            if i != iloc:
                plt.text(coordinates[i][0]-6, coordinates[i][1]+2.5, locations[i], color='navy', transform=ccrs.PlateCarree(),zorder=2, fontsize=15)
        else:
            plt.text(coordinates[i][0]-6.5, coordinates[i][1]+2.5, locations[i], color='navy', transform=ccrs.PlateCarree(),zorder=2, fontsize=15)
        
    if iloc>=0:
        plt.plot(coordinates[iloc][0], coordinates[iloc][1], '.', color='magenta', markersize=25, transform=ccrs.PlateCarree(),zorder=2)
        plt.text(coordinates[iloc][0]-6, coordinates[iloc][1]+2.5, locations[iloc], color='magenta', transform=ccrs.PlateCarree(),zorder=2, fontsize=15, fontweight='bold')
        

def plot_spectrum(ax,x,y,z, vmax=0.3, ylim=0.49):
    x1=np.append(x,x[0])
    y1=np.append(0,y)
    z1=np.column_stack((z[:,:],z[:,-1]))
    p1=ax.pcolormesh(x1,y1,np.sqrt(z1), vmin=0, vmax=vmax)   
    p1.set_cmap('inferno')
    ax.set_theta_zero_location('N', offset=0)
    ax.set_theta_direction(-1)
    ax.set_ylim(0,ylim)
    ax.tick_params(axis='y', colors='plum',labelsize=14,grid_linestyle=':',grid_alpha=0.75,grid_color='plum')
    ax.tick_params(axis='x', colors='purple',labelsize=14,pad=5,grid_linestyle=':',grid_alpha=0.75,grid_color='plum')
    ax.grid(color='plum', linestyle='--', linewidth=0.7,alpha=0.2)
    return p1        
        

def plot_spectrum_hs(ax,x,y,z,z1=[], vmin=0, vmax=0.6,  vmin_z1=0, vmax_z1=0.3, ylim=0.49, size_point=5,point_edge_color=None, alpha_bk=1, cmap='inferno', cmap_z1= 'inferno',remove_axis=0, prob=None, prob_max=0.06, lw=8):
    
    xx,yy=np.meshgrid(x,y)
    
    x=np.append(x,x[0]); 
    y=np.append(0,y); 
    #if vmax<np.nanmax(z):
    #    vmax=1.1*np.nanmax(z)
        
    if cmap=='RdBu_r':
        norm = mcolors.TwoSlopeNorm(0,vmin, vmax)
        p1=ax.pcolormesh(x,y,z, vmin=vmin,vmax=vmax, cmap=plt.cm.RdBu_r, norm=norm, alpha=alpha_bk, linewidths=0.0000001)
    else:
        p1=ax.pcolormesh(x,y,z, vmin=vmin, vmax=vmax,shading='flat', cmap=cmap, alpha=alpha_bk, linewidths=0.00000001)   
    
    if len(z1):
        dx=(x[1]-x[0])/2
        dy=(y[1]-y[0])/2
        p_z1=ax.scatter(xx-dx,yy-dy,size_point, z1, edgecolors=point_edge_color,vmin=vmin_z1,vmax=vmax_z1, cmap=cmap_z1)   
        
    ax.set_theta_zero_location('N', offset=0)
    ax.set_theta_direction(-1)
    ax.set_ylim(0,ylim)
    
    if prob:
        norm = Normalize(vmin=0, vmax=prob_max)        
        cmap = cm.get_cmap('Blues')
        ax.spines['polar'].set_color(cmap(norm(prob)))
        ax.spines['polar'].set_linewidth(lw)
        
    if remove_axis:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.tick_params(axis='y', colors='plum',labelsize=14,grid_linestyle=':',grid_alpha=0.75,grid_color='plum')
        ax.tick_params(axis='x', colors='purple',labelsize=14,pad=5,grid_linestyle=':',grid_alpha=0.75,grid_color='plum')
    ax.grid(color='plum', linestyle='--', linewidth=0.7,alpha=0.2)
    
    if len(z1):
        return p1, p_z1
    else:
        return p1
    
def Plot_Sp_Transformation(sp, sp_mod, time=[],  vmax=0.2, vmax2=1.5):
    
    fig = plt.figure(figsize=[20,8])

    gs1=gridspec.GridSpec(1,1)
    ax1=fig.add_subplot(gs1[0],projection='polar')
    if time:
        z=sp.isel(time=time).efth.values
        lb=str(sp.isel(time=time).time.values)
    else:
        z=np.nanmean(sp.efth.values, axis=0)
        lb='Mean'
        
    p1=plot_spectrum(ax1,np.deg2rad(sp.dir.values), sp.freq.values,z, vmax=vmax)
    ax1.set_title(lb, fontsize=16)
    gs1.tight_layout(fig, rect=[[], [], 0.4, []])

    gs2=gridspec.GridSpec(1,1)
    ax0=fig.add_subplot(gs2[0])
    plt.colorbar(p1,cax=ax0)
    ax0.set_ylabel('Sqrt(Efth)')
    gs2.tight_layout(fig, rect=[0.4, 0.1, 0.46, 0.9])

    gs3=gridspec.GridSpec(1,1)
    ax2=fig.add_subplot(gs3[0],projection='polar')
    if time:
        z=sp_mod.isel(time=time).h.values
        z=4*np.sqrt(z)
    else:
        z=np.mean(sp_mod.h.values,axis=2)
        z=4*np.sqrt(z)

    p2=plot_spectrum_hs(ax2,np.deg2rad(sp_mod.dir.values), sp_mod.t.values,z,vmax=vmax2, ylim=0.99*np.max(sp_mod.t.values))
    ax2.set_title(lb, fontsize=16)
    gs3.tight_layout(fig, rect=[0.5, [], 0.9, []])

    gs4=gridspec.GridSpec(1,1)
    ax0=fig.add_subplot(gs4[0])
    plt.colorbar(p2,cax=ax0)
    ax0.set_ylabel('Hs (m)')
    gs4.tight_layout(fig, rect=[0.92, 0.1, 0.98, 0.9])
    
    
def plot_mjo(mjo, figsize=[7.5, 6]):
    
    fig, ax = plt.subplots(figsize=figsize)
    plt.plot([-4.5, 4.5],[0,0], color='grey')
    plt.plot([0,0],[-4.5, 4.5], color='grey')
    plt.plot([-4.5,4.5],[-4.5, 4.5], color='grey')
    plt.plot([-4.5, 4.5],[4.5,-4.5], color='grey')
    plt.xlim([-4.5,4.5])
    plt.ylim([-4.5,4.5])
    cax=plt.scatter(mjo.rmm1, mjo.rmm2,3, mjo.phase, cmap='Set2')
    plt.xlabel('RMM1', fontsize=16)
    plt.ylabel('RMM2', fontsize=16)
    tk=np.linspace(0.5,7.5,9)
    cbar=fig.colorbar(cax, ticks=tk)
    cbar.ax.set_yticklabels(['0','1','2','3','4','5','6','7','8']) 
    cbar.set_label('MJO phase', fontsize=15)
    
def plot_index(index, name=' ', l1=0.5, l2=-0.5, figsize=[22,6]):
    
    plt.figure(figsize=figsize)
    plt.plot(index.time,index.index,'.-',color='grey', linewidth=2, markersize=8, alpha=0.5, label='Rolling mean')
    plt.grid(color='plum')
    plt.xlabel('Time',fontsize=16)
    plt.ylabel(name,fontsize=16)

    s1=np.where(index.index>l1)[0]
    s2=np.where(index.index<l2)[0]

    #Decision on wether to use the rolling mean or the original monthly values

    clasif=np.full([len(index.index)],1.0)
    clasif[s1]=0
    clasif[s2]=2
    index['classification']=('time',clasif)
    
    if name=='El Niño':
        up_n='El Niño'
        low_n='La Niña'
    else:
        up_n = 'Upper limit' 
        low_n = 'Lower limit' 
        
    plt.plot(index.time[s1],index.index[s1],'.',color='indianred', markersize=12,label=up_n)
    plt.plot(index.time[s2],index.index[s2],'.',color='cornflowerblue', markersize=12,label=low_n)


    plt.plot([index.time.values[0],index.time.values[-1]], [l1,l1],'-',color='indianred', linewidth=2, label= up_n + ' threshold')
    plt.plot([index.time.values[0],index.time.values[-1]], [0,0],'--',color='grey', linewidth=2)
    plt.plot([index.time.values[0],index.time.values[-1]], [l2,l2],'-',color='cornflowerblue', linewidth=2, label=low_n + ' threshold')
    plt.xlim([index.time.values[0],index.time.values[-1]])

    plt.legend(ncol=5, fontsize=13)
    
    return index

def plot_n34(n34, l1=0.5, l2=-0.5, figsize=[22,6]):
    plt.figure(figsize=figsize)
    plt.plot(n34.time,n34.index,'.-',color='grey', linewidth=2, markersize=8, alpha=0.5, label='Rolling mean')
    plt.grid(color='plum')
    plt.xlabel('Time',fontsize=16)
    plt.ylabel('Niño 3.4',fontsize=16)

    s1=np.where(n34.index>l1)[0]
    s2=np.where(n34.index<l2)[0]

    #Decision on wether to use the rolling mean or the original monthly values

    clasif=np.full([len(n34.index)],1.0)
    clasif[s1]=0
    clasif[s2]=2
    n34['classification']=('time',clasif) #0:Niño, 1:Neutral, 2:Niña

    plt.plot(n34.time[s1],n34.index[s1],'.',color='indianred', markersize=12,label='El Niño')
    plt.plot(n34.time[s2],n34.index[s2],'.',color='cornflowerblue', markersize=12,label='La Niña')


    plt.plot([n34.time.values[0],n34.time.values[-1]], [l1,l1],'-',color='indianred', linewidth=2, label='El Niño threshold')
    plt.plot([n34.time.values[0],n34.time.values[-1]], [0,0],'--',color='grey', linewidth=2)
    plt.plot([n34.time.values[0],n34.time.values[-1]], [l2,l2],'-',color='cornflowerblue', linewidth=2, label='La Niña threshold')
    plt.xlim([n34.time.values[0],n34.time.values[-1]])

    plt.legend(ncol=5, fontsize=13)

def plot_spec_n34(sp_mod_daily, annomaly=[], vmax_z1=0.7):
    
    fig = plt.figure(figsize=[18,4])

    gr=3
    name=['El Niño', 'Neutral', 'La Niña']
    gs3=gridspec.GridSpec(1,3,hspace=0.01, wspace=0.01)
    for b in range(gr):

        s=np.where(sp_mod_daily.n34==b)[0]
        prob=len(s)/len(sp_mod_daily.time)
        ax2=fig.add_subplot(gs3[b],projection='polar')
        ax2.set_title(name[b], fontsize=16)
        if annomaly:
            mean_spec=4*np.sqrt(np.mean(sp_mod_daily.h.values,axis=0)) #z1 plots points
            mean_prob=np.sum(sp_mod_daily.is_h.values,axis=0) / len(sp_mod_daily.time) # z is the background
            z1=4*np.sqrt(np.mean(sp_mod_daily.h.isel(time=s).values,axis=0))
            z1=z1 - mean_spec #z1 plots points
            z=np.sum(sp_mod_daily.is_h.isel(time=s).values,axis=0) / len(s) - mean_prob # z is the background
            prob_max=0.3
            [p2,p_z1]=plot_spectrum_hs(ax2,np.deg2rad(sp_mod_daily.dir_bins.values), sp_mod_daily.t_bins.values,z,point_edge_color='Grey', cmap='RdBu_r', vmin=-0.04, vmax=0.04,alpha_bk=0.4,
                                    z1=z1, vmin_z1=-vmax_z1, vmax_z1=vmax_z1, cmap_z1='RdBu_r', size_point=50, lw=5,
                                    remove_axis=1, prob=prob, prob_max=prob_max, ylim=np.nanmax(sp_mod_daily.t_bins))  
        else:
            z1=4*np.sqrt(np.mean(sp_mod_daily.h.isel(time=s).values,axis=0)) #z1 plots points
            z=np.sum(sp_mod_daily.is_h.isel(time=s).values,axis=0) / len(s) # z is the background
            prob_max=0.2
            [p2,p_z1]=plot_spectrum_hs(ax2,np.deg2rad(sp_mod_daily.dir_bins.values), sp_mod_daily.t_bins.values,z, cmap='Greys', vmin=0, vmax=1,alpha_bk=0.4,
                                    z1=z1, vmin_z1=0, vmax_z1=vmax_z1, cmap_z1='CMRmap_r', size_point=40,lw=5,
                                    remove_axis=1, prob=prob, prob_max=prob_max, ylim=np.nanmax(sp_mod_daily.t_bins))  
    gs3.tight_layout(fig, rect=[[], [], 0.78, []])

    gs4=gridspec.GridSpec(1,1)
    ax0=fig.add_subplot(gs4[0])
    plt.colorbar(p2,cax=ax0, extend='both')
    ax0.set_ylabel('Cell Probability', fontsize=14)
    gs4.tight_layout(fig, rect=[0.77, 0.1, 0.85, 0.9])

    gs5=gridspec.GridSpec(1,1)
    ax0=fig.add_subplot(gs5[0])
    plt.colorbar(p_z1,cax=ax0, extend='both')
    ax0.set_ylabel('Hs (m)', fontsize=14)
    gs5.tight_layout(fig, rect=[0.85, 0.1, 0.93, 0.9])

    gs6=gridspec.GridSpec(1,1)
    ax0=fig.add_subplot(gs6[0])
    norm = Normalize(vmin=0, vmax=prob_max)        
    cmap = cm.get_cmap('Blues')
    cb1 = mpl.colorbar.ColorbarBase(ax0, cmap=cmap, norm=norm, orientation='vertical', extend='both')
    cb1.set_label('Cluster Probability', fontsize=14)
    gs6.tight_layout(fig, rect=[0.93, 0.1, 1.02, 0.9])

    
def plot_spec_index(sp_mod_daily, var, name, annomaly=[], vmax_z1=0.7):
    
    fig = plt.figure(figsize=[18,4])

    gr=3
    gs3=gridspec.GridSpec(1,3,hspace=0.01, wspace=0.01)
    for b in range(gr):

        s=np.where(sp_mod_daily[var]==b)[0]
        prob=len(s)/len(sp_mod_daily.time)
        ax2=fig.add_subplot(gs3[b],projection='polar')
        ax2.set_title(name[b], fontsize=16)
        if annomaly:
            mean_spec=4*np.sqrt(np.mean(sp_mod_daily.h.values,axis=0)) #z1 plots points
            mean_prob=np.sum(sp_mod_daily.is_h.values,axis=0) / len(sp_mod_daily.time) # z is the background
            z1=4*np.sqrt(np.mean(sp_mod_daily.h.isel(time=s).values,axis=0))
            z1=z1 - mean_spec #z1 plots points
            z=np.sum(sp_mod_daily.is_h.isel(time=s).values,axis=0) / len(s) - mean_prob # z is the background
            prob_max=0.3
            [p2,p_z1]=plot_spectrum_hs(ax2,np.deg2rad(sp_mod_daily.dir_bins.values), sp_mod_daily.t_bins.values,z,point_edge_color='Grey', cmap='RdBu_r', vmin=-0.04, vmax=0.04,alpha_bk=0.4,
                                    z1=z1, vmin_z1=-vmax_z1, vmax_z1=vmax_z1, cmap_z1='RdBu_r', size_point=50, lw=5,
                                    remove_axis=1, prob=prob, prob_max=prob_max, ylim=np.nanmax(sp_mod_daily.t_bins))  
        else:
            z1=4*np.sqrt(np.mean(sp_mod_daily.h.isel(time=s).values,axis=0)) #z1 plots points
            z=np.sum(sp_mod_daily.is_h.isel(time=s).values,axis=0) / len(s) # z is the background
            prob_max=0.2
            [p2,p_z1]=plot_spectrum_hs(ax2,np.deg2rad(sp_mod_daily.dir_bins.values), sp_mod_daily.t_bins.values,z, cmap='Greys', vmin=0, vmax=1,alpha_bk=0.4,
                                    z1=z1, vmin_z1=0, vmax_z1=vmax_z1, cmap_z1='CMRmap_r', size_point=40,lw=5,
                                    remove_axis=1, prob=prob, prob_max=prob_max, ylim=np.nanmax(sp_mod_daily.t_bins))  
    gs3.tight_layout(fig, rect=[[], [], 0.78, []])

    gs4=gridspec.GridSpec(1,1)
    ax0=fig.add_subplot(gs4[0])
    plt.colorbar(p2,cax=ax0, extend='both')
    ax0.set_ylabel('Cell Probability', fontsize=14)
    gs4.tight_layout(fig, rect=[0.77, 0.1, 0.85, 0.9])

    gs5=gridspec.GridSpec(1,1)
    ax0=fig.add_subplot(gs5[0])
    plt.colorbar(p_z1,cax=ax0, extend='both')
    ax0.set_ylabel('Hs (m)', fontsize=14)
    gs5.tight_layout(fig, rect=[0.85, 0.1, 0.93, 0.9])

    gs6=gridspec.GridSpec(1,1)
    ax0=fig.add_subplot(gs6[0])
    norm = Normalize(vmin=0, vmax=prob_max)        
    cmap = cm.get_cmap('Blues')
    cb1 = mpl.colorbar.ColorbarBase(ax0, cmap=cmap, norm=norm, orientation='vertical', extend='both')
    cb1.set_label('Cluster Probability', fontsize=14)
    gs6.tight_layout(fig, rect=[0.93, 0.1, 1.02, 0.9])
    
    
def plot_spec_n34_mjo_month(sp_mod_daily, n34_sel=[], mjo_sel=[], month=[], an_range=0.1, mean_month=[], figsize=[22,7], vmax_z1=0.8, vmax_an=0.07):

    fig = plt.figure(figsize=figsize)
    
    gr=6
    gs3=gridspec.GridSpec(1,2,hspace=0.1, wspace=0.1)
    
    if n34_sel and mjo_sel and month:
        label= 'n34= ' + str(n34_sel)+ ' |  MJO= ' + str(mjo_sel) + ' |  Month= ' + str(month)
        s=np.where((sp_mod_daily.n34==n34_sel-1) & (sp_mod_daily.mjo==mjo_sel) & (sp_mod_daily.time.dt.month.values==month))[0]
    elif n34_sel and mjo_sel:
        label= 'n34= ' + str(n34_sel)+ ' |  MJO= ' + str(mjo_sel) 
        s=np.where((sp_mod_daily.n34==n34_sel-1) & (sp_mod_daily.mjo==mjo_sel))[0]
    elif n34_sel and month:
        label= 'n34= ' + str(n34_sel)+ ' |  Month= ' + str(month)
        s=np.where((sp_mod_daily.n34==n34_sel-1) & (sp_mod_daily.time.dt.month.values==month))[0]
    elif mjo_sel and month:
        label= ' |  MJO= ' + str(mjo_sel) + ' |  Month= ' + str(month)
        s=np.where((sp_mod_daily.mjo==mjo_sel) & (sp_mod_daily.time.dt.month.values==month))[0]
    elif mjo_sel:
        label=  ' MJO= ' + str(mjo_sel) 
        s=np.where(sp_mod_daily.mjo==mjo_sel)[0]
    elif n34_sel:
        label= 'n34= ' + str(n34_sel)
        s=np.where((sp_mod_daily.n34==n34_sel-1))[0]
    elif month:
        label=' Month= ' + str(month)
        s=np.where((sp_mod_daily.time.dt.month.values==month))[0]
    else:
        label='ALL DATA'
        s=np.arange(len(sp_mod_daily.time))
    
    print('Number of data: ' + str(len(s)))
    prob=len(s)/len(sp_mod_daily.time)
    ax2=fig.add_subplot(gs3[0],projection='polar')
    ax2.set_ylabel(label, labelpad=40, fontsize=20)

    z1=np.mean(sp_mod_daily.h.isel(time=s).values,axis=0) #z1 plots points
    z1=4*np.sqrt(z1)
    z=np.sum(sp_mod_daily.is_h.isel(time=s).values,axis=0) / len(s) # z is the background

    [p2,p_z1]=plot_spectrum_hs(ax2,np.deg2rad(sp_mod_daily.dir_bins.values), sp_mod_daily.t_bins.values,z, cmap='Greys', vmin=0, vmax=1, alpha_bk=0.4,
                            z1=z1, vmin_z1=0, vmax_z1=vmax_z1, cmap_z1='CMRmap_r', size_point=100,
                            remove_axis=1, prob=prob, prob_max=0.2, ylim=np.nanmax(sp_mod_daily.t_bins))  

    ax2=fig.add_subplot(gs3[1],projection='polar')   
    
    if mean_month:
        s1=np.where((sp_mod_daily.time.dt.month.values==month))[0]
        mean_spec=4*np.sqrt(np.mean(sp_mod_daily.isel(time=s1).h.values,axis=0))
        mean_prob=np.sum(sp_mod_daily.isel(time=s1).is_h.values,axis=0) / len(sp_mod_daily.isel(time=s1).time)
    else:
        mean_spec=4*np.sqrt(np.mean(sp_mod_daily.h.values,axis=0))
        mean_prob=np.sum(sp_mod_daily.is_h.values,axis=0) / len(sp_mod_daily.time)
    z1 = z1 - mean_spec #z1 plots points 
    z = z - mean_prob
    
    [p3,p_z1_an]=plot_spectrum_hs(ax2,np.deg2rad(sp_mod_daily.dir_bins.values), sp_mod_daily.t_bins.values,z,point_edge_color='Grey', cmap='RdBu_r', vmin=-vmax_an, vmax=vmax_an ,alpha_bk=0.3,
                            z1=z1, vmin_z1=-an_range, vmax_z1=an_range, cmap_z1='RdBu_r', size_point=90,
                            remove_axis=1, prob=prob, prob_max=0.2, ylim=np.nanmax(sp_mod_daily.t_bins))  

    gs3.tight_layout(fig, rect=[0, [], 0.8, []])

    gs4=gridspec.GridSpec(1,1)
    ax0=fig.add_subplot(gs4[0])
    plt.colorbar(p2,cax=ax0, extend='both')
    ax0.set_ylabel('Cell Probability', fontsize=14)
    gs4.tight_layout(fig, rect=[0.755, 0.1, 0.82, 0.9])

    gs4=gridspec.GridSpec(1,1)
    ax0=fig.add_subplot(gs4[0])
    plt.colorbar(p3,cax=ax0, extend='both')
    ax0.set_ylabel('Cell Probability - Anomaly', fontsize=14)
    gs4.tight_layout(fig, rect=[0.82, 0.1, 0.89, 0.9])

    gs5=gridspec.GridSpec(1,1)
    ax0=fig.add_subplot(gs5[0])
    plt.colorbar(p_z1,cax=ax0, extend='both')
    ax0.set_ylabel('Hs (m)', fontsize=14)
    gs5.tight_layout(fig, rect=[0.89, 0.1, 0.95, 0.9])

    gs6=gridspec.GridSpec(1,1)
    ax0=fig.add_subplot(gs6[0])
    plt.colorbar(p_z1_an,cax=ax0, extend='both')
    ax0.set_ylabel('Hs (m) - Anomaly', fontsize=14)
    gs6.tight_layout(fig, rect=[0.95, 0.1, 1.02, 0.9])
    
    
def plot_spec_mjo(sp_mod_daily, annomaly=[], vmax_z1=0.7, figsize=[19,7], grd=[2,4]):
    
    fig = plt.figure(figsize=figsize)

    gr=8
    gs3=gridspec.GridSpec(grd[0],grd[1],hspace=0.01, wspace=0.01)
    hs_m=[]

    for b in range(gr):

        s=np.where(sp_mod_daily.mjo==b+1)[0]
        prob=len(s)/len(sp_mod_daily.time)
        ax2=fig.add_subplot(gs3[b],projection='polar')
        ax2.set_title('MJO' + str(b+1), fontsize=16)
        if annomaly:
            
            mean_spec=4*np.sqrt(np.mean(sp_mod_daily.h.values,axis=0)) #z1 plots points
            mean_prob=np.sum(sp_mod_daily.is_h.values,axis=0) / len(sp_mod_daily.time) # z is the background
            
            z1=4*np.sqrt(np.mean(sp_mod_daily.h.isel(time=s).values,axis=0)) - mean_spec #z1 plots points
            z=np.sum(sp_mod_daily.is_h.isel(time=s).values,axis=0) / len(s)  - mean_prob# z is the background
            
            prob_max=0.15
    
            [p2,p_z1]=plot_spectrum_hs(ax2,np.deg2rad(sp_mod_daily.dir_bins.values), sp_mod_daily.t_bins.values,z, point_edge_color='Grey', cmap='RdBu_r', vmin=-0.04, vmax=0.04,
                            alpha_bk=0.4, z1=z1, vmin_z1=-vmax_z1, vmax_z1=vmax_z1, cmap_z1='RdBu_r', size_point=45, lw=5,
                            remove_axis=1, prob=prob, prob_max=prob_max, ylim=np.nanmax(sp_mod_daily.t_bins))
        else:
            z1=4*np.sqrt(np.mean(sp_mod_daily.h.isel(time=s).values,axis=0)) #z1 plots points
            z=np.sum(sp_mod_daily.is_h.isel(time=s).values,axis=0) / len(s) # z is the background
            prob_max=0.2
            [p2,p_z1]=plot_spectrum_hs(ax2,np.deg2rad(sp_mod_daily.dir_bins.values), sp_mod_daily.t_bins.values,z, cmap='Greys', vmin=0, vmax=1,alpha_bk=0.4,
                                z1=z1, vmin_z1=0, vmax_z1=vmax_z1, cmap_z1='CMRmap_r', size_point=45, lw=5,
                                remove_axis=1, prob=prob, prob_max=prob_max, ylim=np.nanmax(sp_mod_daily.t_bins))  

    gs3.tight_layout(fig, rect=[[], [], 0.78, []])

    gs4=gridspec.GridSpec(1,1)
    ax0=fig.add_subplot(gs4[0])
    plt.colorbar(p2,cax=ax0, extend='both')
    ax0.set_ylabel('Cell Probability', fontsize=14)
    gs4.tight_layout(fig, rect=[0.77, 0.1, 0.85, 0.9])

    gs5=gridspec.GridSpec(1,1)
    ax0=fig.add_subplot(gs5[0])
    plt.colorbar(p_z1,cax=ax0, extend='both')
    ax0.set_ylabel('Hs (m)', fontsize=14)
    gs5.tight_layout(fig, rect=[0.85, 0.1, 0.93, 0.9])

    gs6=gridspec.GridSpec(1,1)
    ax0=fig.add_subplot(gs6[0])
    norm = Normalize(vmin=0, vmax=prob_max)        
    cmap = cm.get_cmap('Blues')
    cb1 = mpl.colorbar.ColorbarBase(ax0, cmap=cmap, norm=norm, orientation='vertical', extend='both')
    cb1.set_label('Cluster Probability', fontsize=14)
    gs6.tight_layout(fig, rect=[0.93, 0.1, 1.02, 0.9])
    

    
def spec_n34_mjo_month_ax(sp_mod_daily, n34_sel=[], mjo_sel=[], month=[], an_range=0.1, ax2=[], anomaly=[], mean_month=[], size_point=18, lw=3, vmax_z1=0.7):

   
    if n34_sel and mjo_sel and month:
        label= 'n34= ' + str(n34_sel)+ ' |  MJO= ' + str(mjo_sel) + ' |  Month= ' + str(month)
        s=np.where((sp_mod_daily.n34==n34_sel-1) & (sp_mod_daily.mjo==mjo_sel) & (sp_mod_daily.time.dt.month.values==month))[0]
    elif n34_sel and mjo_sel:
        label= 'n34= ' + str(n34_sel)+ ' |  MJO= ' + str(mjo_sel) 
        s=np.where((sp_mod_daily.n34==n34_sel-1) & (sp_mod_daily.mjo==mjo_sel))[0]
    elif n34_sel and month:
        label= 'n34= ' + str(n34_sel)+ ' |  Month= ' + str(month)
        s=np.where((sp_mod_daily.n34==n34_sel-1) & (sp_mod_daily.time.dt.month.values==month))[0]
    elif mjo_sel and month:
        label= ' |  MJO= ' + str(mjo_sel) + ' |  Month= ' + str(month)
        s=np.where((sp_mod_daily.mjo==mjo_sel) & (sp_mod_daily.time.dt.month.values==month))[0]
    elif mjo_sel:
        label=  ' MJO= ' + str(mjo_sel) 
        s=np.where(sp_mod_daily.mjo==mjo_sel)[0]
    elif n34_sel:
        label= 'n34= ' + str(n34_sel)
        s=np.where((sp_mod_daily.n34==n34_sel-1))[0]
    elif month:
        label=' Month= ' + str(month)
        s=np.where((sp_mod_daily.time.dt.month.values==month))[0]
    else:
        label='ALL DATA'
        s=np.arange(len(sp_mod_daily.time))
    
    prob=len(s)/len(sp_mod_daily.time)
    
    z1=4*np.sqrt(np.mean(sp_mod_daily.h.isel(time=s).values,axis=0)) #z1 plots points
    z=np.sum(sp_mod_daily.is_h.isel(time=s).values,axis=0) / len(s) # z is the background

    
    if anomaly:
        if mean_month:
            s1=np.where((sp_mod_daily.time.dt.month.values==month))[0]
            mean_spec=4*np.sqrt(np.mean(sp_mod_daily.isel(time=s1).h.values,axis=0))
            mean_prob=np.sum(sp_mod_daily.isel(time=s1).is_h.values,axis=0) / len(sp_mod_daily.isel(time=s1).time)
        else:
            mean_spec=4*np.sqrt(np.mean(sp_mod_daily.h.values,axis=0))
            mean_prob=np.sum(sp_mod_daily.is_h.values,axis=0) / len(sp_mod_daily.time)
        z1 = z1 - mean_spec #z1 plots points
        z = z - mean_prob
        [p2,p_z1]=plot_spectrum_hs(ax2,np.deg2rad(sp_mod_daily.dir_bins.values), sp_mod_daily.t_bins.values,z,point_edge_color='Grey', cmap='RdBu_r', 
                            vmin=-0.04, vmax=0.04 ,alpha_bk=0.3, lw=lw,
                            z1=z1, vmin_z1=-an_range, vmax_z1=an_range, cmap_z1='RdBu_r', size_point=size_point,
                            remove_axis=1, prob=prob, prob_max=0.02, ylim=np.nanmax(sp_mod_daily.t_bins))
    else:
        [p2,p_z1]=plot_spectrum_hs(ax2,np.deg2rad(sp_mod_daily.dir_bins.values), sp_mod_daily.t_bins.values,z, cmap='Greys', vmin=0, vmax=1, alpha_bk=0.4,
                            z1=z1, vmin_z1=0, vmax_z1=vmax_z1, cmap_z1='CMRmap_r', size_point=size_point, lw=lw,
                            remove_axis=1, prob=prob, prob_max=0.02, ylim=np.nanmax(sp_mod_daily.t_bins))
        
    
    
    return p2, p_z1


def Plot_Map_Locations_mean(locations,  path):
    
    
    fig = plt.figure(figsize=[28,14])

    ax = plt.axes(projection = ccrs.PlateCarree(central_longitude=180))
    ax.stock_img()

    # cartopy land feature
    land_10m = cartopy.feature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='darkgrey', facecolor='gainsboro',  zorder=1)
    ax.add_feature(land_10m)
    ax.gridlines()
    
    for iloc in range(len(locations)):
        
        sp_mod_daily=xr.open_dataset(os.path.join(path, 'Spec_Mod_Daily_'+locations[iloc]+'.nc'))
        
        ll=sp_mod_daily.coordinates.values[0]
        if ll<0: 
            ll=ll+360
            
        ax.plot(ll, sp_mod_daily.coordinates.values[1], '.', color='darkmagenta', markersize=20, transform=ccrs.PlateCarree(),zorder=2)
        
        axin = inset_axes(ax, bbox_to_anchor=[ll-180, sp_mod_daily.coordinates.values[1]],
                          height=1.2,
                          width=1.2,
                          loc='center', 
                          bbox_transform=ax.transData,
                          axes_class = matplotlib.projections.get_projection_class('polar'))
        axin.set_title(locations[iloc], color='darkmagenta', fontweight='bold')
        spec_n34_mjo_month_ax(sp_mod_daily, n34_sel=[], mjo_sel=[], month=[], an_range=0.1, ax2=axin, anomaly=0, mean_month=[], vmax_z1=0.8, size_point=10, lw=1)

        
# def Plot_Map_Locations_n34(locations, coordinates, path):
    
    
#     fig = plt.figure(figsize=[28,14])

#     ax = plt.axes(projection = ccrs.PlateCarree(central_longitude=180))
#     ax.stock_img()

#     # cartopy land feature
#     land_10m = cartopy.feature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='darkgrey', facecolor='gainsboro',  zorder=1)
#     ax.add_feature(land_10m)
#     ax.gridlines()
    
#     for iloc in range(len(locations)):
        
#         ll=coordinates[iloc][0]
#         if ll<0: 
#             ll=ll+360
            
#         ax.plot(ll, coordinates[iloc][1], '.', color='darkmagenta', markersize=20, transform=ccrs.PlateCarree(),zorder=2)
#         ax.text(coordinates[iloc][0]-2, coordinates[iloc][1]+5, locations[iloc], color='darkmagenta', transform=ccrs.PlateCarree(),zorder=2, rotation=80, alpha=0.5, fontsize=15)
        
#         sp_mod_daily=xr.open_dataset(os.path.join(path, 'Spec_Mod_Daily_' + locations[iloc] + '.nc'))
        
#         n34_sel=1 #Niño
#         axin = inset_axes(ax, bbox_to_anchor=[ll-180-12, coordinates[iloc][1]],
#                           height=1.2,
#                           width=1.2,
#                           loc='center', 
#                           bbox_transform=ax.transData,
#                           axes_class = matplotlib.projections.get_projection_class('polar'))
#         axin.set_title('El Niño', color='firebrick', fontweight='bold')
#         spec_n34_mjo_month_ax(sp_mod_daily, n34_sel=n34_sel, mjo_sel=[], month=[], an_range=0.1, ax2=axin, anomaly=1, mean_month=[], vmax_z1=0.07,size_point=10, lw=1)


#         n34_sel=3 #Niña
#         axin2 = inset_axes(ax, 
#                           bbox_to_anchor=[ll-180+12, coordinates[iloc][1]],
#                           height=1.2,
#                           width=1.2,
#                           loc='center', 
#                           bbox_transform=ax.transData,
#                           axes_class = matplotlib.projections.get_projection_class('polar'))
#         axin2.set_title('La Niña', color='navy', fontweight='bold')
#         spec_n34_mjo_month_ax(sp_mod_daily, n34_sel=n34_sel, mjo_sel=[], month=[], an_range=0.1, ax2=axin2, anomaly=1, mean_month=[], vmax_z1=0.07,size_point=10, lw=1)


        
def spec_index_ax(sp_mod_daily, index, index_sel, label, ax2=[], anomaly=[], an_range=0.1,  size_point=18, lw=3, vmax_z1=0.8):
    
    '''
    
    sp_mod_daily : spectra with climate indexes associated
    index        : cliamte index to select (pdo, n34, mjo, nao)
    index_sel    : value of the index to select (Generally 1-3)
    label        : label for plotting
    
    '''
    
    s=np.where(sp_mod_daily[index]==index_sel)[0]
    prob=len(s)/len(sp_mod_daily.time)

    z1=4*np.sqrt(np.mean(sp_mod_daily.h.isel(time=s).values,axis=0)) #z1 plots points
    z=np.sum(sp_mod_daily.is_h.isel(time=s).values,axis=0) / len(s) # z is the background

    if anomaly:
        
        mean_spec=4*np.sqrt(np.mean(sp_mod_daily.h.values,axis=0))
        mean_prob=np.sum(sp_mod_daily.is_h.values,axis=0) / len(sp_mod_daily.time)
            
        z1 = z1 - mean_spec #z1 plots points
        z = z - mean_prob
        
        [p2,p_z1]=plot_spectrum_hs(ax2,np.deg2rad(sp_mod_daily.dir_bins.values), sp_mod_daily.t_bins.values,z,point_edge_color='Grey', cmap='RdBu_r', 
                            vmin=-0.04, vmax=0.04 ,alpha_bk=0.4, lw=lw,
                            z1=z1, vmin_z1=-an_range, vmax_z1=an_range, cmap_z1='RdBu_r', size_point=size_point,
                            remove_axis=1, prob=prob, prob_max=0.02, ylim=np.nanmax(sp_mod_daily.t_bins))
    else:
        
        [p2,p_z1]=plot_spectrum_hs(ax2,np.deg2rad(sp_mod_daily.dir_bins.values), sp_mod_daily.t_bins.values,z, cmap='Greys', vmin=0, vmax=1, alpha_bk=0.4,
                            z1=z1, vmin_z1=0, vmax_z1=vmax_z1, cmap_z1='CMRmap_r', size_point=size_point, lw=lw,
                            remove_axis=1, prob=prob, prob_max=0.02, ylim=np.nanmax(sp_mod_daily.t_bins))
        
    
    return p2, p_z1


def Plot_Map_Locations_index(path, locations, index, labels = ['Positive', 'Negative'], index_sel = [0,2], anomaly=1, vmax_z1=0.04, an_range=0.1):
    
    '''
    
    labels: Labels for plotting
    index_sel : Only 2 can be selected
    
    '''
    
    fig = plt.figure(figsize=[27,14])
    
    gs3=gridspec.GridSpec(1,1,hspace=0.01, wspace=0.01)
    ax=fig.add_subplot(gs3[0],projection= ccrs.PlateCarree(central_longitude=180))

#     ax = plt.axes(projection = ccrs.PlateCarree(central_longitude=180))
    ax.stock_img()

    # cartopy land feature
    land_10m = cartopy.feature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='darkgrey', facecolor='gainsboro',  zorder=1)
    ax.add_feature(land_10m)
    ax.gridlines()
    
    if len(index)==1:
        index = np.full(len(locations), index[0])
    
    for iloc in range(len(locations)):
        
        sp_mod_daily=xr.open_dataset(os.path.join(path, 'Spec_Mod_Daily_' + locations[iloc] + '.nc'))
        
        ll=sp_mod_daily.coordinates.values[0]
        if ll<0: 
            ll=ll+360
            
        ax.plot(ll, sp_mod_daily.coordinates.values[1], '.', color='darkmagenta', markersize=20, transform=ccrs.PlateCarree(),zorder=2)
        ax.text(ll-2, sp_mod_daily.coordinates.values[1]+5, locations[iloc], color='darkmagenta', transform=ccrs.PlateCarree(),zorder=2, rotation=80, alpha=0.7, fontsize=15)

        for isl in range(len(index_sel)):
            
            if isl==0:
                posit = ll-180-12
            else: 
                posit = ll-180+12
            
            axin = inset_axes(ax, bbox_to_anchor=[posit, sp_mod_daily.coordinates.values[1]],
                          height=1.3,
                          width=1.3,
                          loc='center', 
                          bbox_transform=ax.transData,
                          axes_class = matplotlib.projections.get_projection_class('polar'))
            
            axin.set_title(labels[isl] + ' ' +  index[iloc], color='navy', fontweight='bold')
            
            [p2, p_z1] = spec_index_ax(sp_mod_daily, index[iloc], index_sel[isl], labels[isl] + ' ' +  index[iloc],
                          ax2=axin, anomaly=anomaly, an_range=an_range, 
                          size_point=18, lw=3, vmax_z1=vmax_z1)
            
    gs3.tight_layout(fig, rect=[0, 0.1, 1, 1])

    gs4=gridspec.GridSpec(1,1)
    ax0=fig.add_subplot(gs4[0])
    plt.colorbar(p2,cax=ax0, extend='both', orientation='horizontal')
    ax0.set_xlabel('Cell Probability', fontsize=16)
    gs4.tight_layout(fig, rect=[0.1, 0, 0.5, 0.1])

    gs5=gridspec.GridSpec(1,1)
    ax0=fig.add_subplot(gs5[0])
    plt.colorbar(p_z1,cax=ax0, extend='both', orientation='horizontal')
    ax0.set_xlabel('Hs (m)', fontsize=16)
    gs5.tight_layout(fig, rect=[0.5, 0, 0.9, 0.1])
    
    
def plot_spec_index_combined(sp_mod_daily, comb_pos, annomaly=[], vmax_z1=0.7, mk_size=80, lw=10, colorbars=1):
    
    fig = plt.figure(figsize=[12,7])

    gr=3
    gs3=gridspec.GridSpec(1,1,hspace=0.01, wspace=0.01)

    s=comb_pos
    prob=len(s)/len(sp_mod_daily.time)
    ax2=fig.add_subplot(gs3[0],projection='polar')
#     ax2.set_title(name, fontsize=16)
    if annomaly:
        mean_spec=4*np.sqrt(np.mean(sp_mod_daily.h.values,axis=0)) #z1 plots points
        mean_prob=np.sum(sp_mod_daily.is_h.values,axis=0) / len(sp_mod_daily.time) # z is the background
        z1=4*np.sqrt(np.mean(sp_mod_daily.h.isel(time=s).values,axis=0))
        z1=z1 - mean_spec #z1 plots points
        z=np.sum(sp_mod_daily.is_h.isel(time=s).values,axis=0) / len(s) - mean_prob # z is the background
        prob_max=0.3
        [p2,p_z1]=plot_spectrum_hs(ax2,np.deg2rad(sp_mod_daily.dir_bins.values), sp_mod_daily.t_bins.values,z,point_edge_color='Grey', cmap='RdBu_r', vmin=-0.04, vmax=0.04,alpha_bk=0.4,
                                z1=z1, vmin_z1=-vmax_z1, vmax_z1=vmax_z1, cmap_z1='RdBu_r', size_point=mk_size, lw=lw,
                                remove_axis=1, prob=prob, prob_max=prob_max, ylim=np.nanmax(sp_mod_daily.t_bins))  
    else:
        z1=4*np.sqrt(np.mean(sp_mod_daily.h.isel(time=s).values,axis=0)) #z1 plots points
        z=np.sum(sp_mod_daily.is_h.isel(time=s).values,axis=0) / len(s) # z is the background
        prob_max=0.2
        [p2,p_z1]=plot_spectrum_hs(ax2,np.deg2rad(sp_mod_daily.dir_bins.values), sp_mod_daily.t_bins.values,z, cmap='Greys', vmin=0, vmax=1,alpha_bk=0.4,
                                z1=z1, vmin_z1=0, vmax_z1=vmax_z1, cmap_z1='CMRmap_r', size_point=mk_size,lw=lw,
                                remove_axis=1, prob=prob, prob_max=prob_max, ylim=np.nanmax(sp_mod_daily.t_bins))  
    if colorbars:
        
        gs3.tight_layout(fig, rect=[[], [], 0.85, []])

        gs4=gridspec.GridSpec(1,1)
        ax0=fig.add_subplot(gs4[0])
        plt.colorbar(p2,cax=ax0, extend='both')
        ax0.set_ylabel('Cell Probability', fontsize=14)
        gs4.tight_layout(fig, rect=[0.79, 0.1, 0.9, 0.9])

        gs5=gridspec.GridSpec(1,1)
        ax0=fig.add_subplot(gs5[0])
        plt.colorbar(p_z1,cax=ax0, extend='both')
        ax0.set_ylabel('Hs (m)', fontsize=14)
        gs5.tight_layout(fig, rect=[0.9, 0.1, 1.01, 0.9])

#     gs6=gridspec.GridSpec(1,1)
#     ax0=fig.add_subplot(gs6[0])
#     norm = Normalize(vmin=0, vmax=prob_max)        
#     cmap = cm.get_cmap('Blues')
#     cb1 = mpl.colorbar.ColorbarBase(ax0, cmap=cmap, norm=norm, orientation='vertical', extend='both')
#     cb1.set_label('Cluster Probability', fontsize=14)
#     gs6.tight_layout(fig, rect=[0.93, 0.1, 1.03, 0.9])