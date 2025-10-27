"""
Interpolate MALI model basal sliding speed and melt rate to ISSM mesh
"""

import os
import pathlib
import sys
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.patches import Rectangle
from scipy import interpolate as interp
import cmocean

yts = 365*86400

def main():
    datadir = os.path.abspath('../../../../data/lanl-mali')
    ## BASAL VELOCITY
    ais_outputs = os.path.join(datadir, 'AIS_4kmto20km_hist04.nc')
    with xr.open_dataset(ais_outputs, engine='scipy') as ais:
        ub = ais['uReconstructX'][0, :, -1]
        vb = ais['uReconstructY'][0, :, -1]
        temp = np.mean(ais['temperature'][0, :, :], axis=-1)

        xMali = ais['xCell']
        yMali = ais['yCell']

    vvMali = np.sqrt(ub**2 + vb**2)*yts

    mesh = np.load('../geom/mesh.npy', allow_pickle=True)


    xyMali = (xMali, yMali)
    xyMesh = np.array([mesh['x'], mesh['y']]).T
    vvMesh = interp.griddata(xyMali, vvMali, xyMesh, method='nearest')
    
    tempMesh = interp.griddata(xyMali, temp, xyMesh, method='nearest')
    tempMesh[tempMesh==0] = np.median(tempMesh)
    np.save('temperature_mali.npy', tempMesh)

    mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)

    np.save('basal_velocity_mali.npy', vvMesh)
    fig,ax = plt.subplots()
    tpc = ax.tripcolor(mtri, np.log10(vvMesh), vmin=0, vmax=3, cmap=cmocean.cm.speed)
    fig.colorbar(tpc, label='log10 Basal velocity (m a$^{-1}$)')
    fig.savefig('mali_vvMesh.png', dpi=400)

    fig,ax = plt.subplots()
    tpc = ax.tripcolor(mtri, tempMesh, vmax=273.15, cmap=cmocean.cm.thermal)
    fig.colorbar(tpc, label='Temperature (K)')
    fig.savefig('mali_tempMesh.png', dpi=400)

    ## BASAL MELT RATE
    output_state = os.path.abspath(os.path.join(datadir, 'output_state_2060.nc'))
    with xr.open_dataset(output_state, engine='scipy') as output:
        basalmeltMali = output['basalMeltInput'][0]

    kgm2s_to_mwea = 365*86400/910
    basalmeltMali *= kgm2s_to_mwea

    basalmeltMesh = interp.griddata(xyMali, basalmeltMali, xyMesh, method='nearest')

    np.save('basal_melt_mali.npy', basalmeltMesh)
    fig,ax = plt.subplots()
    tpc = ax.tripcolor(mtri, np.log10(basalmeltMesh), cmap=cmocean.cm.thermal, vmin=-3, vmax=0)
    fig.colorbar(tpc, label='log10 Basal melt rate (m w.e. a$^{-1}$)')
    fig.savefig('mali_basalmeltMesh.png', dpi=400)
    return vvMesh, basalmeltMali

if __name__=='__main__':
    main()