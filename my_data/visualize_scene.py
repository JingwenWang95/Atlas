import os
import numpy as np
import open3d as o3d
# from TUM_RGBD import load_K_Rt_from_P
from vis_cameras import visualize


def get_box(vol_bnds):
    points = [
        [vol_bnds[0, 0], vol_bnds[1, 0], vol_bnds[2, 0]],
        [vol_bnds[0, 1], vol_bnds[1, 0], vol_bnds[2, 0]],
        [vol_bnds[0, 0], vol_bnds[1, 1], vol_bnds[2, 0]],
        [vol_bnds[0, 1], vol_bnds[1, 1], vol_bnds[2, 0]],
        [vol_bnds[0, 0], vol_bnds[1, 0], vol_bnds[2, 1]],
        [vol_bnds[0, 1], vol_bnds[1, 0], vol_bnds[2, 1]],
        [vol_bnds[0, 0], vol_bnds[1, 1], vol_bnds[2, 1]],
        [vol_bnds[0, 1], vol_bnds[1, 1], vol_bnds[2, 1]],
    ]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


if __name__ == "__main__":
    base_dir = "../DATAROOT/ICL/living_room_traj2_frei"
    # cam_dict = np.load(os.path.join(base_dir, "processed", "cameras.npz"))
    # world_mats = cam_dict["world_mats"]
    # poses = []
    #
    # for i, P in enumerate(world_mats):
    #     if i % 30 != 0:
    #         continue
    #     P = P[:3, :4]
    #     K, pose = load_K_Rt_from_P(P)
    #     poses += [pose[None, ...]]
    # extrinsics = np.concatenate(poses, axis=0)
    mesh = o3d.io.read_triangle_mesh(os.path.join(base_dir, "mesh-aligned.ply"))
    # mesh = o3d.io.read_triangle_mesh("/home/jingwen/vision/dev/Atlas/results/release/semseg/test_final/mesh.ply")
    xmin, xmax = -2., 3.5
    ymin, ymax = -1.5, 5.
    zmin, zmax = -1.3, 1.5
    vol_bnds = np.array([[xmin, xmax],
                         [ymin, ymax],
                         [zmin, zmax]])
    inner_sphere = get_box(vol_bnds)

    things_to_draw = [inner_sphere, mesh]
    # mesh = o3d.io.read_triangle_mesh("/home/jingwen/Vision/dev/sdf-nerf/logs/fr3_long_office_depth/3/testset_050000/mesh.ply")
    # T = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    # cube = draw_cuboid(T=T)
    # cube.scale(2., [0., 0., 0.])
    visualize(things_to_draw=things_to_draw)