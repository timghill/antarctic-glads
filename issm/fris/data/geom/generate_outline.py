"""
Read CSV file of boundary nodes and reformat to ISSM format
"""

import numpy as np

_header_template = """## Name:{fname}
## Icon:0
# Points Count  Value
{points} 1.000000
# X pos Y pos"""

fname_out = 'outline.exp'


# outline = np.loadtxt(fname_in, skiprows=1, quotechar='"', delimiter=',', usecols=(1,2)).astype(float)
outline = np.load('../../../../data/ANT_Basins/basin_fris.npy')

# print(outline[:5])

# For the true boundary
# outline = outline[::5]
# vertexid = outline[:, 0].astype(int)
n_vertices = outline.shape[0]

# reformat = np.zeros((n_vertices+1, 2))
# reformat[:-1, :] = outline[:]
# reformat[-1, :] = reformat[0, :]

# print(reformat[:5])

header = _header_template.format(fname=fname_out, points=n_vertices)
np.savetxt(fname_out, outline, header=header, delimiter=' ', comments='', fmt='%.2f')
