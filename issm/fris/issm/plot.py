import numpy as np

from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation


mesh = np.load('../data/geom/mesh.npy', allow_pickle=True)
mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)

coef = np.load('friction_coefficient.npy').squeeze()
fig,ax = plt.subplots()
pc = ax.tripcolor(mtri, coef, vmin=0)
ax.set_title('friction coef')
fig.colorbar(pc)
fig.savefig('friction_coefficient.png', dpi=400)

model_vel = np.load('model_vel.npy').squeeze()

# obs_vel = np.load('obs_vel.npy').squeeze()
vx = np.load('../data/geom/vx.npy')
vy = np.load('../data/geom/vy.npy')
obs_vel = np.sqrt(vx**2 + vy**2)

fig, axs = plt.subplots(figsize=(8, 8), ncols=2, nrows=2)
axs = axs.flatten()

abs_pc = axs[0].tripcolor(mtri, np.log10(obs_vel), vmin=0, vmax=3)
axs[1].tripcolor(mtri, np.log10(model_vel), vmin=0, vmax=3)
diff_pc = axs[2].tripcolor(mtri, model_vel-obs_vel,
    vmin=-500, vmax=500, cmap='RdBu_r')


for ax in axs:
    ax.set_aspect('equal')
    ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

fig.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.9, wspace=0.05, hspace=0.1)


cb1 = fig.colorbar(abs_pc, ax=(axs[:2]), label='log$_10$ speed (m/a)')

cb2 = fig.colorbar(diff_pc, ax=(axs[2:]), label='Speed error (m/a)')

axs[0].set_title('Observed')
axs[1].set_title('Modelled')
axs[2].set_title('Obs - model')

fig.savefig('velocitySolution.png', dpi=400)

# plt.show()


