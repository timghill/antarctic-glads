import shapefile
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import scipy.spatial
import contourpy
from pyproj import CRS, Transformer

region1 = 'Jpp-K'
region2 = 'J-Jpp'
# region2 = 'Jpp-K'

# Read bedmachine mask
bm = xr.open_dataset('../bedmachine/BedMachineAntarctica-v3.nc')
print('bm:', bm)

dd = 10
x = bm['x'][::dd].values
y = bm['y'][::dd].values
mask = bm['mask'][::dd, ::dd].values

fig, ax = plt.subplots()
ax.pcolormesh(x, y, mask)
ax.set_aspect('equal')

# Read basins *.shp
sf = shapefile.Reader('ANT_Basins_IMBIE2_v1.6.shp')
records = sf.records()
shapes = sf.shapes()
region_names = np.array([rec[1] for rec in records])
print('Regions:', region_names)
print(region_names==region1)
region1 = np.where(region_names==region1)[0][0]
region2= np.where(region_names==region2)[0][0]
print('Rignot region:', region1, region2)
outlines = [np.array(shape.points) for shape in shapes]
for i in range(1, len(region_names)):
    ox = outlines[i][:, 0]
    oy = outlines[i][:, 1]
    xmid = 0.5*(np.min(ox) + np.max(ox))
    ymid = 0.5*(np.min(oy) + np.max(oy))
    ax.plot(ox, oy, color='k')
    ax.text(xmid, ymid, region_names[i], color='k')

# Use interior Rignot basins and the BedMachine ice front
cntr = contourpy.contour_generator(x=x, y=y, z=mask)
contour_lines = cntr.lines(0.5)
contour_lens = [len(cl) for cl in contour_lines]
imax = np.argmax(contour_lens)
contour = contour_lines[imax]

outline1 = outlines[region1]
outline2 = outlines[region2]

outlines = [outline1, outline2]

# Manually adjust ice shelf fronts

# outline = np.delete(outline, np.arange(415, 583), axis=0)
# outline = np.insert(outline, 415, contour[4155:4205], axis=0)

# outline = np.delete(outline, np.arange(502, 767), axis=0)
# outline = np.insert(outline, 502, contour[4218:4250], axis=0)

# outline = np.delete(outline, np.arange(662, 1007), axis=0)
# outline = np.insert(outline, 662, contour[4294:4300], axis=0)

# Merge the outlines
# colors= ['m', 'r']
# for j,outline in enumerate(outlines):
#     npoint = len(outline)
#     ax.plot(outline[:, 0], outline[:, 1], color=colors[j], zorder=5)
#     ax.plot(contour[:, 0], contour[:, 1], color='b', linewidth=1.5)
#     xmin = np.min(outline[:,0])
#     xmax = np.max(outline[:,0])
#     ymin = np.min(outline[:,1])
#     ymax = np.max(outline[:,1])
#     for i in range(npoint):
#         xi = outline[i,0]
#         yi = outline[i,1]
#         if xi>=xmin and xi<=xmax and yi>=ymin and yi<=ymax:
#             ax.text(xi, yi, i, fontsize=10, color=colors[j])
# for i in range(len(contour)):
#     if i%5==0:
#         ax.text(contour[i,0], contour[i,1], i, fontsize=10, color='r')

outline = outline1.copy()
print(outline.shape)
outline = np.delete(outline, np.arange(914, 967), axis=0)

outline_insert = np.vstack((outline2[1465:], outline2[:1415]))
outline = np.insert(outline, 914, outline_insert, axis=0)
print(outline.shape)

# Now merge the inland outline with the ice front

outline = outline[:1117]
outline = outline[783:]

outline = np.insert(outline, 0, contour[6400:6600], axis=0)
outline = np.vstack((outline, outline[0][None,:]))

ax.plot(outline[:, 0], outline[:, 1], color='m', zorder=5)
ax.plot(contour[:, 0], contour[:, 1], color='b', linewidth=1.5)

print('shape:', outline.shape)
dmin = 0
while dmin<50 and len(outline)>3:
    print(outline.shape)
    for i in range(1, len(outline)-1):
        xi = outline[i,0]
        yi = outline[i,1]
        dd = np.sqrt((xi-outline[:,0])**2 + (yi-outline[:,1])**2)
        dd[i] = np.nan
        # dd[0] = np.nan
        # dd[-1] = np.nan
        dmin = np.nanmin(dd)
        if dmin<50:
            print('dmin:', dmin)
            outline = np.delete(outline, i, axis=0)
            print('Removed index ', i)
            break

print('shape:', outline.shape)

# for i in range(len(contour)):
#     if i%5==0:
#         ax.text(contour[i,0], contour[i,1], i, fontsize=10, color='b')

u,c = np.unique(outline, axis=0, return_counts=True)
for i in range(len(c)):
    if c[i]>1:
        ax.plot(u[i][0], u[i][1], 'go')
print('shape:', outline.shape)
print('unique:', np.unique(outline, axis=0).shape)
np.save('basin_fris.npy', outline)

plt.show()
