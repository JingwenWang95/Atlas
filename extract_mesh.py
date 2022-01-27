import os
import numpy as np
import torch
from skimage import measure
import trimesh

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
exp_dir = "results/release/semseg/test_final"
data = np.load(os.path.join(exp_dir, "living_room_traj2_frei.npz"))
tsdf, voxel_size, origin = data["tsdf"], data["voxel_size"], data["origin"]


def get_scene_mesh(tsdf, voxel_size):
    vertices, faces, normals, _ = measure.marching_cubes(tsdf,
                                                         0.,
                                                         spacing=(voxel_size,
                                                                  voxel_size,
                                                                  voxel_size),
                                                         allow_degenerate=False)
    vertices = np.array(vertices)
    normals = np.array(normals)

    return trimesh.Trimesh(vertices, faces, vertex_normals=normals)


mesh = get_scene_mesh(tsdf, voxel_size)
mesh.export(os.path.join(exp_dir, 'mesh-ICL.ply'))
mesh.show()