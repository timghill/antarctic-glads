import shapefile
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import scipy.spatial
import shapely
import contourpy
from pyproj import CRS, Transformer

region = 'G-H'

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
npoint = len(outline)
intersect = []
ix2 = []
for i in range(npoint):
    d = np.sqrt((outline[i,0]-contour[:,0])**2 + (outline[i,1]-contour[:,1])**2)
    dclose = np.where(d<dist_thresh)[0]
    intersect.extend(dclose)
    ix2.extend(0*dclose + i)


print('intersection:', intersect)

glmin = np.min(intersect)
glmax = np.max(intersect)

amin = np.argmin(intersect)
amax = np.argmax(intersect)

i1 = ix2[amin]
i2 = ix2[amax]

if i1>i2:
    # tmp = i1.copy()
    # i1 = i2.copy()
    # i2 = tmp

    glmin = np.max(intersect)
    glmax = np.min(intersect)

interior_outline = np.delete(outline, np.arange(i1, i2), axis=0)
print('interior_outline:', interior_outline.shape)
print('i1:', i1)
print('i2:', i2)
interior_outline = np.insert(interior_outline, i1, contour[glmin:glmax], axis=0)
ax.plot(contour[:,0], contour[:, 1], color='blue', marker='.')
ax.plot(contour[glmin:glmax, 0], contour[glmin:glmax, 1], color='magenta', marker='.')

ax.plot(outline[:, 0], outline[:, 1], color='red', marker='.')

ax.plot(interior_outline[:, 0], interior_outline[:, 1], color='y')
# print(contour_lines[imax].shape)
# cx = contour_lines[imax][:, 0]
# cy = contour_lines[imax][:, 1]
# ax.plot(cx, cy, color='blue')

# gl = shapely.LinearRing(contour_lines[imax])
# basin = shapely.LinearRing(outlines[region_num])

# un = shapely.intersection_all((gl, basin))
# print('un:', un)
# print(dir(un))

# xy = shapely.get_coordinates(un)
# print('xy:', xy)

# fig,ax = plt.subplots()
# # ax.plot(un.boundary.xy)
# # for line in un.geoms:
# ax.scatter(xy[:,0], xy[:,1])
# ax.set_aspect('equal')

plt.show()
