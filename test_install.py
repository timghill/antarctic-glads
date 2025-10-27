import sys
import os

issmdir = os.getenv('ISSM_DIR')

sys.path.append(os.path.join(issmdir, 'bin'))
sys.path.append(os.path.join(issmdir, 'lib'))

from issmversion import issmversion
