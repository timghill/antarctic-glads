import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import cmocean
from scipy import interpolate


mesh = np.load('mesh.npy', allow_pickle=True)
levelset = np.load('ocean_levelset.npy')

vx = np.load('vx.npy')
vy = np.load('vy.npy')
vv = np.sqrt(vx**2 + vy**2)

vv[levelset<=0] = np.nan

points = [
    np.array([
        [-5.344e5, -8.656e5],
        [-7.009e5, -6.579e5],
    ]),
    np.array([
        [-5.286e5, -9.137e5],
        [-7.754e5, -8.238e5]
    ])
]

npoints = len(points)
print('npoints:', npoints)

colors = [
    'magenta',
    'cyan',
]

nsteps = 101
maxlen = 200e3

mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
fig,ax = plt.subplots()
ax.tripcolor(mtri, np.log10(vv), vmin=0, vmax=3, cmap=cmocean.cm.balance)
ax.set_aspect('equal')

profig, proax = plt.subplots()

for i in range(len(points)):
    line = points[i]
    nsegments = len(line)-1

    segnorms = []
    for seg in range(nsegments):
        segnorm = np.sqrt(np.sum((line[seg+1] - line[seg])**2))
        segnorms.append(segnorm)
    
    print('segnorms:', segnorms)
    print('nsegments:', nsegments)
    # Remove segments that are too far from the GL
    for seg in range(nsegments):
        if np.sum(segnorms[:seg+1])>maxlen:
            segnorms[seg] = maxlen - np.sum(segnorms[:seg])
            nnz = seg
            segnorms = segnorms[:seg+1]
            line = line[:seg+2]
    
    print('segnorms:', segnorms)
    
    print('line:', line)
    xxline = []
    yyline = []
    ssline = []
    for seg in range(nsegments):
        xynorm = np.sqrt(np.sum((line[seg+1] - line[seg])**2))
        xdir = (line[seg+1][0] - line[seg][0])/xynorm
        ydir = (line[seg+1][1] - line[seg][1])/xynorm
        # xdir = (p1[i][0]-p0[i][0])/pnorm
        # ydir = (p1[i][1]-p0[i][1])/pnorm

        num_this_segment = int(np.ceil((nsteps-1)*segnorms[seg]/np.sum(segnorms)))
        print('num_this_segment:', num_this_segment)
        s = np.linspace(0, segnorms[seg], num_this_segment)
        print('s:', s)

        print('xdir:', xdir)
        print('ydir:', ydir)
        print('ss dir:', xdir**2 + ydir**2)
        # xline = p0[i][0] + s*xdir
        # yline = p0[i][1] + s*ydir
        xline = line[seg][0] + s*xdir
        yline = line[seg][1] + s*ydir
        
        if seg>0:
            xline = xline[1:]
            yline = yline[1:]
            s = s[1:] + np.sum(segnorms[:seg])
        xxline.extend(list(xline))
        yyline.extend(list(yline))
        ssline.extend(list(s))
    
    xxline = np.array(xxline)
    yyline = np.array(yyline)
    ssline = np.array(ssline)

        # ax.plot(xline, yline, color=colors[i])

        # Interpolate onto the line
    vvline = interpolate.griddata((mesh['x'], mesh['y']), vv, (xxline, yyline), method='linear')

    ax.plot(xxline, yyline, color=colors[i])
    proax.plot(ssline/1e3, vvline, color=colors[i])
    proax.set_xlabel('Distance from the grounding line (km)')
    proax.set_ylabel('Velocity (m a$^{-1}$)')

plt.show()