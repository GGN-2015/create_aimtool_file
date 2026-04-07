from solve_rigid_point_set_rt import compute_best_rigid_transform
from solve_rigid_point_set_rt import apply_transform
from fit_3d_plane             import fit_3d_plane
import numpy as np
import os

def cross_3d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the cross product of two 3D vectors (supports list / np.array input)
    :param a: vector a, shape (3,)
    :param b: vector b, shape (3,)
    :return: cross product result vector, shape (3,)
    """
    # Convert to numpy array (automatically compatible with list input)
    a = np.asarray(a)
    b = np.asarray(b)
    
    # Safety check
    if a.shape != (3,) or b.shape != (3,):
        raise ValueError("Both vectors must be 1D arrays of length 3!")
    
    # 3D cross product formula
    cx = a[1] * b[2] - a[2] * b[1]
    cy = a[2] * b[0] - a[0] * b[2]
    cz = a[0] * b[1] - a[1] * b[0]
    
    return np.array([cx, cy, cz])

# Compute unit vector
def unit(vec:np.ndarray):
    if np.linalg.norm(vec) < 1e-6:
        return np.array([0, 0, 0]).astype(vec.dtype)
    return vec / np.linalg.norm(vec)

# Compute tool file data
def calculate_tool_file_data(node_set:np.ndarray, dtype=np.float32, according_to_manual:bool=False):
    if len(node_set.shape) != 2: # Must be a 2D array
        raise ValueError()
    if node_set.shape[1] != 3: # Must be 3D points
        raise ValueError()
    if node_set.shape[0] < 3: # At least three points required
        raise ValueError()
    
    # Use float32 for all calculations
    node_set = node_set.astype(dtype)
    
    # Sort descending by distance from the origin
    if not according_to_manual:
        arr = [
            (-np.linalg.norm(node_set[i]), node_set[i])
            for i in range(node_set.shape[0])
        ]
        arr.sort()

    # Sort descending by abs(Y) from less to more
    else:
        arr = [
            ((
                float(node_set[i][1]), # Y
                float(node_set[i][0])  # X
            ), node_set[i])
            for i in range(node_set.shape[0])
        ]
        arr.sort()

    # Get sorted numpy array
    np_arr = np.array([
        val for _, val in arr
    ]).astype(dtype)

    # Translate centroid to origin
    raw_centroid = np_arr.mean(axis=0)
    np_move = (np_arr - raw_centroid).astype(dtype)

    # Compute current X, Y, Z axes
    x_axis = unit(np_move[0])
    y_axis = unit(np_move[1])

    # Make z_axis
    z_axis = fit_3d_plane(np_move)[0]
    if np.dot(raw_centroid, z_axis) > 0: # Make z_axis inner
        z_axis = -z_axis
    z_axis = unit(z_axis)

    # Caculate real_y, recalculate X
    y_axis = unit(cross_3d(z_axis, x_axis).astype(dtype))
    x_axis = unit(cross_3d(y_axis, z_axis).astype(dtype))

    # Construct a transformation
    P = np.array([
        x_axis,
        y_axis,
        z_axis,
        np_move.mean(axis=0)
    ]).astype(dtype)

    # Target coordinate system
    Q = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ]).astype(dtype)

    # Compute rotation matrix
    rotmat, travec = compute_best_rigid_transform(P, Q)

    # Get coordinate representation in the tool file
    aim_exp = apply_transform(np_move, rotmat, travec).astype(dtype)
    return aim_exp

# Create tool file
def create_aimtool_file(aim_dir:str, tool_name:str, node_set:np.ndarray, according_to_manual:bool=False):
    aim_dir = os.path.abspath(aim_dir)
    if not os.path.isdir(aim_dir):
        raise FileNotFoundError(f"{aim_dir} is not a directory.")
    aim_path = os.path.join(aim_dir, f"{tool_name}.aimtool")

    # Compute tool file content
    np_coord = calculate_tool_file_data(
        node_set, 
        according_to_manual=according_to_manual)

    # Create tool file
    with open(aim_path, "w") as fp:
        fp.write(f"{tool_name}\n")
        fp.write(f"T\n")
        fp.write(f"{np_coord.shape[0]}\n")
        for i in range(np_coord.shape[0]):
            fp.write(" ".join([
                "%.6f" % float(item) for item in np_coord[i]
            ]) + " 15\n")
        fp.write(f"2\n")
        fp.write(f"0.000000 0.000000 0.000000\n")
        fp.write(f"0.000000 0.000000 0.000000\n")

if __name__ == "__main__":
    P = np.array([
        [-154.349,  70.446, 835.555],
        [-114.986,  70.007, 830.095],
        [-150.089,  99.024, 813.499],
        [-129.166, 102.473, 807.977],
    ])

    # Create tool file
    create_aimtool_file(".", "BONE-1", P, False)
