import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import cmocean

nu = 1.79e-6

# N = np.load('../../issm/thwaites/train/train_N.npy')
# h = np.load('../../issm/thwaites/train/train_hs.npy')
# v = np.load('../../issm/thwaites/train/train_vv.npy')
N = np.load('../../issm/thwaites-consistent/glads/RUN/output_001/steady/N.npy')[:,-1]
h = np.load('../../issm/thwaites-consistent/glads/RUN/output_001/steady/h_s.npy')[:,-1]
vx = np.load('../../issm/thwaites-consistent/glads/RUN/output_001/steady/vx.npy')[:,-1]
vy = np.load('../../issm/thwaites-consistent/glads/RUN/output_001/steady/vy.npy')[:,-1]
v = np.sqrt(vx**2 + vy**2) / 86400/365

ks = 2.137962e-02

print('v:', np.mean(v))

mesh = np.load('../../issm/thwaites-consistent/data/geom/mesh.npy', allow_pickle=True)
mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)

# vel = np.load('../../issm/thwaites/data/lanl-mali/')

fig,ax = plt.subplots()
mappable = ax.tripcolor(mtri, h, vmin=0, vmax=0.5, cmap=cmocean.cm.amp)
ax.set_aspect('equal')
ax.set_title('Sheet thickness (m)')
fig.colorbar(mappable, ax=ax)
fig.savefig('figures/thwaites_001_hs.png', dpi=400)

Re = h*v/nu
fig,ax = plt.subplots()
mappable = ax.tripcolor(mtri, Re, vmin=0, vmax=4000, cmap=cmocean.cm.balance)
ax.set_aspect('equal')
ax.set_title('Reynolds number')
fig.colorbar(mappable, ax=ax)
fig.savefig('figures/thwaites_001_Re.png', dpi=400)

T = ks*h**3/(1 + Re/2000)
# T = ks*h**3
fig,ax = plt.subplots()
mappable = ax.tripcolor(mtri, T, vmin=0, cmap=cmocean.cm.ice)
ax.set_aspect('equal')
ax.set_title('Transmissivity ()')
fig.colorbar(mappable, ax=ax)
fig.savefig('figures/thwaites_001_T.png', dpi=400)

q= h*v
gradphi = h*v/T
k_turb = q/(h**1.25)/(gradphi**0.5)
k_turb[np.isnan(k_turb)] = 0
fig,ax = plt.subplots()
mappable = ax.tripcolor(mtri, k_turb, vmin=0, cmap=cmocean.cm.haline)
ax.set_aspect('equal')
ax.set_title('Equivalent turbulent conductivitiy (m$^{7/4}$ kg$^{-1/2}$)')
fig.colorbar(mappable, ax=ax)
fig.savefig('figures/thwaites_001_kturb.png', dpi=400)

fig,ax = plt.subplots()
mappable = ax.tripcolor(mtri, gradphi, vmin=0, vmax=200, cmap=cmocean.cm.turbid)
ax.set_aspect('equal')
ax.set_title('Hydraulic potential gradient (Pa/m)')
fig.colorbar(mappable, ax=ax)
fig.savefig('figures/thwaites_001_gradphi.png', dpi=400)

print('median gradient:', np.nanmedian(gradphi))


# plt.show()
