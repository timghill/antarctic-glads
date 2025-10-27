import shapefile
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import scipy.spatial
import contourpy
from pyproj import CRS, Transformer

region = 'Cp-D'

dist_thresh = 2.5e3

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

# # Manually adjust ice shelf fronts

outline = np.delete(outline, np.arange(192, 220), axis=0)
outline = np.insert(outline, 192, contour[1964:1969], axis=0)

outline = np.delete(outline, np.arange(250, 440), axis=0)
outline = np.insert(outline, 250, contour[1984:2010], axis=0)

outline = np.delete(outline, np.arange(590, 1100), axis=0)
outline = np.insert(outline, 590, contour[2101:2170], axis=0)

outline = np.delete(outline, np.arange(676, 788), axis=0)
outline = np.insert(outline, 676, contour[2178:2196], axis=0)

outline = np.delete(outline, np.arange(760, 888), axis=0)
outline = np.insert(outline, 760, contour[2215:2225], axis=0)

outline = np.delete(outline, np.arange(853, 894), axis=0)
outline = np.insert(outline, 853, contour[2243:2249], axis=0)

outline = np.delete(outline, np.arange(904, 1032), axis=0)
outline = np.insert(outline, 904, contour[2255:2277], axis=0)

outline = np.delete(outline, np.arange(926, 990), axis=0)
outline = np.insert(outline, 926, contour[2277:2286], axis=0)

outline = np.delete(outline, np.arange(1054, 1095), axis=0)
outline = np.insert(outline, 1054, contour[2319:2327], axis=0)

# outline = np.delete(outline, np.arange(1786, 1971), axis=0)
# outline = np.insert(outline, 1786, contour[2254:2287], axis=0)

# outline = np.delete(outline, np.arange(1485, 1610), axis=0)
# outline = np.insert(outline, 1485, contour[2215:2225], axis=0)

# outline = np.delete(outline, np.arange(502, 767), axis=0)
# outline = np.insert(outline, 502, contour[4218:4250], axis=0)

# outline = np.delete(outline, np.arange(662, 1007), axis=0)
# outline = np.insert(outline, 662, contour[4294:4300], axis=0)


npoint = len(outline)
ax.plot(outline[:, 0], outline[:, 1], color='m', zorder=5, marker='.')
ax.plot(contour[:, 0], contour[:, 1], color='b', linewidth=1.5, marker='.')

# for i in range(len(outline)):
#     if i%5==0:
#         ax.text(outline[i,0], outline[i,1], i, color='b', fontsize=10)

# for i in range(len(contour)):
#     if i%5==0:
#         ax.text(contour[i,0], contour[i,1], i, color='r', fontsize=10)


ax.set_xlim([np.min(outline[:,0]), np.max(outline[:, 0])])
ax.set_ylim([np.min(outline[:,1]), np.max(outline[:, 1])])


np.save('basin_Cp-D.npy', outline)


plt.show()
