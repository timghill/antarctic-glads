"""
Read CSV file of boundary nodes and reformat to ISSM format
"""

import argparse
import numpy as np

def main(basin):
    _header_template = """## Name:{fname}
    ## Icon:0
    # Points Count  Value
    {points} 1.000000
    # X pos Y pos"""

    fname_out = 'outline.exp'

    outline = np.load(f'../../../../data/ANT_Basins/basin_{basin}.npy')
    n_vertices = outline.shape[0]
    header = _header_template.format(fname=fname_out, points=n_vertices)
    np.savetxt(fname_out, outline, header=header, delimiter=' ', comments='', fmt='%.2f')
    return fname_out

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('basin')
    args = parser.parse_args()
    main(args.basin)
