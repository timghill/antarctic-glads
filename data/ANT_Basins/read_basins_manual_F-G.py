import shapefile
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import scipy.spatial
import contourpy
from pyproj import CRS, Transformer

region = 'F-G'

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
print(region_names==region)
region_num = np.where(region_names==region)[0][0]
print('Rignot region:', region_num)
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

outline = outlines[region_num]

# Manually adjust ice shelf fronts
# outline = np.delete(outline, np.arange(95, 872), axis=0)
outline = outline[:1613]
outline = outline[66:]
outline = np.insert(outline, 0, contour[3717:3772], axis=0)

outline = np.delete(outline, np.arange(57, 120), axis=0)
outline = np.insert(outline, 57, contour[3775:3780], axis=0)

outline = np.delete(outline, np.arange(210, 262), axis=0)
outline = np.insert(outline, 210, contour[3811:3815], axis=0)

outline = np.delete(outline, np.arange(325, 1330), axis=0)
outline = np.insert(outline, 325, contour[3845:4058], axis=0)

print(outline.shape)
outline = np.vstack((outline, outline[0][None,:]))
print(outline.shape)

npoint = len(outline)
ax.plot(outline[:, 0], outline[:, 1], color='m', zorder=5)
ax.plot(contour[:, 0], contour[:, 1], color='b', linewidth=1.5)


# for i in range(npoint):
#     ax.text(outline[i,0], outline[i,1], i, fontsize=10, color='m')

# for i in range(len(contour)):
#     if i%5==0:
#         ax.text(contour[i,0], contour[i,1], i, fontsize=10, color='b')

ax.set_xlim((np.min(outline[:,0]), np.max(outline[:,0])))
ax.set_ylim((np.min(outline[:,1]), np.max(outline[:,1])))

np.save('basin_F-G.npy', outline)


plt.show()
