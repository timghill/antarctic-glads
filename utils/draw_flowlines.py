import argparse
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import cmocean
from scipy import interpolate


def draw_flowlines(files):
        

    mesh = np.load('mesh.npy', allow_pickle=True)
    levelset = np.load('ocean_levelset.npy')

    #vx = np.load('vx.npy')
    #vy = np.load('vy.npy')
    #vv = np.sqrt(vx**2 + vy**2)
    vv = np.load('../lanl-mali/basal_velocity_mali.npy')

    vv[levelset<=0] = np.nan

    # points = [
    #     np.array([
    #         [-1.5271e6, -4.751e5],
    #         [-1.3428e6, -4.435e5],
    #     ]),
    #     np.array([
    #         [-1.5889e6, -2.528e5],
    #         [-1.5849e6, -7.17e4],
    #     ]),
    # ]

    npoints = len(files)
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

    for i in range(npoints):
        # line = points[i]
        line = np.loadtxt(files[i], delimiter=',')
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
        vvline[vvline<50] = np.nan

        ax.plot(xxline, yyline, color=colors[i])
        proax.plot(ssline/1e3, vvline, color=colors[i])
        proax.set_xlabel('Distance from the grounding line (km)')
        proax.set_ylabel('Velocity (m a$^{-1}$)')

        np.save('flowline_{:02d}.npy'.format(i), np.array([ssline, xxline, yyline]))

    profig.savefig('flowlines.png', dpi=400)
    fig.savefig('flowlines_map.png', dpi=400)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()
    draw_flowlines(args.files)
