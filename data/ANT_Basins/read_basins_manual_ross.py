import shapefile
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import scipy.spatial
import contourpy
from pyproj import CRS, Transformer

region1 = 'Ep-F'
region2 = 'E-Ep'
# region2 = 'Jpp-K'

# Read bedmachine mask
bm = xr.open_dataset('../bedmachine/BedMachineAntarctica-v3.nc')
print('bm:', bm)

dd = 8
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

# Merge the outlines
outline1_clip = outline1[132:1812]
outline2_clip = outline2[49:169]

outline = np.vstack((outline2_clip, outline1_clip))
outline = np.vstack((outline, outline[0]))

outline = np.insert(outline, 120, contour[4265:4620], axis=0)

outline = np.delete(outline, np.arange(822, 924), axis=0)
outline = np.insert(outline, 822, contour[4710:4712], axis=0)

outline = np.delete(outline, np.arange(846, 975), axis=0)
outline = np.insert(outline, 846, contour[4722:4727], axis=0)

outline = np.delete(outline, np.arange(875, 1701), axis=0)
outline = np.insert(outline, 875, contour[4740:4805], axis=0)

# Now merge the inland outline with the ice front
ax.plot(outline[:, 0], outline[:, 1], color='m', zorder=5)
# for i in range(len(outline)):
#     ax.text(outline[i,0], outline[i,1], i, fontsize=10, color='m')

ax.plot(contour[:,0], contour[:,1], color='b', zorder=4)
xmin = np.min(outline[:,0])
xmax = np.max(outline[:,0])
ymin = np.min(outline[:,1])
ymax = np.max(outline[:,1])

ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))
# for i in range(len(contour)):
#     xi,yi = contour[i]
#     if i%5==0 and xi<=xmax and xi>=xmin and yi<=ymax and yi>=ymin:
#         ax.text(contour[i,0], contour[i,1], i, fontsize=10, color='b')




np.save('basin_ross.npy', outline)

plt.show()
