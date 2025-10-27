import os
import numpy as np
import xarray as xr

from InterpFromGridToMesh import InterpFromGridToMesh
from cuffey import cuffey
from m1qn3inversion import m1qn3inversion
from SetMarineIceSheetBC import SetMarineIceSheetBC
from hydrologyglads import hydrologyglads
from setflowequation import setflowequation

from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation

# Parameters to change/Try
friction_coefficient = 10 # default [10]
Temp_change          =  0  # default [0 K]

#Name and Coordinate system
md.mesh.epsg=3031


#Geometry
md.mask.ocean_levelset = np.load('../data/geom/ocean_levelset.npy')
md.mask.ice_levelset = np.load('../data/geom/ice_levelset.npy')
md.geometry.base      = np.load('../data/geom/base.npy')
md.geometry.surface   = np.load('../data/geom/surface.npy')
md.geometry.thickness = np.load('../data/geom/thick.npy')
md.geometry.bed       = np.load('../data/geom/bed.npy')

#Initialization parameters
print('   Interpolating temperatures')
md.initialization.temperature = np.load('../data/lanl-mali/temperature_mali.npy')

print('   Set observed velocities')
vx = np.load('../data/geom/vx.npy')
vy = np.load('../data/geom/vy.npy')
md.initialization.vx = vx
md.initialization.vy = vy
md.initialization.vz = np.zeros((md.mesh.numberofvertices,1))
md.initialization.vel = np.sqrt(vx**2 + vy**2)

print('   Set Pressure')
md.initialization.pressure=md.materials.rho_ice*md.constants.g*md.geometry.thickness

print('   Construct ice rheological properties')
md.materials.rheology_n=3*np.ones((md.mesh.numberofelements,1))
md.materials.rheology_B=cuffey(md.initialization.temperature)


#Friction and inversion set up
print('   Construct basal friction parameters')
md.friction.coefficient=friction_coefficient*np.ones((md.mesh.numberofvertices,1))
md.friction.p=np.ones((md.mesh.numberofelements,1))
md.friction.q=np.ones((md.mesh.numberofelements,1))

#no friction applied on floating ice
md.friction.coefficient[md.mask.ocean_levelset<0]=0
# md.groundingline.migration='SubelementMigration'

## GLADS SETTINGS

print('Setting hydrology defaults')
# HYDROLOGY
# parameters
onevec = np.ones((md.mesh.numberofvertices, 1))
md.hydrology = hydrologyglads()
md.hydrology.sheet_conductivity = 1e-2*onevec
md.hydrology.sheet_alpha = 3./2.
md.hydrology.sheet_beta = 2.
md.hydrology.cavity_spacing = 2
md.hydrology.bump_height = 0.25*onevec
md.hydrology.channel_sheet_width = 2
md.hydrology.englacial_void_ratio = 1e-5
md.hydrology.rheology_B_base = (2.4e-24)**(-1./3.)*onevec
md.hydrology.istransition = 1
md.hydrology.ischannels = 1
md.hydrology.channel_conductivity = 0.05*onevec
md.hydrology.channel_alpha = 5./4.
md.hydrology.channel_beta = 3./2.
md.hydrology.creep_open_flag = 0
md.hydrology.isincludesheetthickness = 1
md.hydrology.requested_outputs = [
        'HydraulicPotential',
        'EffectivePressure',
        'HydrologySheetThickness',
        'ChannelDischarge',
        'ChannelArea',
        'HydrologyWaterVx',
        'HydrologyWaterVy',
]

# INITIAL CONDITIONS
md.initialization.watercolumn = 0.25*md.hydrology.bump_height*onevec
md.initialization.channelarea = 0*np.zeros((md.mesh.numberofedges, 1))

phi_bed = md.constants.g*md.materials.rho_freshwater*md.geometry.bed
p_ice = md.constants.g*md.materials.rho_ice*md.geometry.thickness
md.initialization.hydraulic_potential = phi_bed + 0.95*p_ice

# BOUNDARY CONDITIONS
md.hydrology.spcphi = np.nan*onevec
md.hydrology.neumannflux = np.zeros((md.mesh.numberofelements, 1))
md.hydrology.spcphi[md.mask.ocean_levelset.squeeze()<0] = 0

# md = setflowequation(md, 'SSA', 'all')

# FORCING
md.hydrology.melt_flag = 1
md.basalforcings.geothermalflux = 0
melt = np.load('../data/lanl-mali/basal_melt_mali.npy')
md.basalforcings.groundedice_melting_rate = melt
md.hydrology.moulin_input = np.zeros((md.mesh.numberofvertices, 1))

# Execution path
SLURM_TMPDIR = os.getenv('SLURM_TMPDIR')
if SLURM_TMPDIR:
    md.cluster.executionpath = SLURM_TMPDIR
else:
    cwd = os.getcwd()
    expath = os.path.join(cwd, 'TMP/')
    if not os.path.exists(expath):
        os.makedirs(expath)
    md.cluster.executionpath = expath
