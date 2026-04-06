# create_aimtool_file
An unofficial tool for creating aimooe tool files.

## Installation

```bash
pip install create_aimtool_file
```

## Usage

```python
import numpy as np
from create_aimtool_file import create_aimtool_file

P = np.array([
    [-154.349,  70.446, 835.555], # 3rd closest to centroid
    [-114.986,  70.007, 830.095], # Farthest from center
    [-150.089,  99.024, 813.499], # Closest to centroid
    [-129.166, 102.473, 807.977], # 2nd closest to centroid
])

# Create tool file
create_aimtool_file(".", "BONE-1", P)
```
