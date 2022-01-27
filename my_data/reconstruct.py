import os
import tqdm
import open3d
import numpy as np
import configargparse
from preprocess import tum2matrix, get_association


def integrate_rgbd_frames(basedir, camera_intrinsics):
    transform = np.array([[1, 0, 0, 0],
                          [0, 0, -1, 0],
                          [0, 1, 0, 0],
                          [0, 0, 0, 1]])
    print('Reading association file')
    rgb_files = []
    depth_files = []
    poses = []  # c2w
    with open(os.path.join(basedir, "dep_rgb_traj.txt"), "r") as f:
        for line in f.readlines():
            items = line.strip().split(" ")
            rgb_files += [items[3]]
            depth_files += [items[1]]
            tum_poses = [float(x) for x in items[-7:]]
            poses += [transform @ tum2matrix(tum_poses)]

    print("Integrating depth maps...")
    voxel_length = 10. / 256.0
    volume = open3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel_length,
                                                   sdf_trunc=0.04,
                                                   color_type=open3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    assert(len(rgb_files) == len(depth_files) == len(poses))
    for i in tqdm.tqdm(range(len(rgb_files))):

        path_rgb = os.path.join(basedir, rgb_files[i])
        path_depth = os.path.join(basedir, depth_files[i])
        color = open3d.io.read_image(str(path_rgb))
        depth = open3d.io.read_image(str(path_depth))
        rgbd = open3d.geometry.RGBDImage.create_from_tum_format(color,
                                                                depth,
                                                                convert_rgb_to_intensity=False)
        # requires w2c
        volume.integrate(rgbd, camera_intrinsics, np.linalg.inv(poses[i]))

    print("Extract a triangle mesh from the volume and visualize it.")
    cloud = volume.extract_point_cloud()
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    print("Writing file:", os.path.join(basedir, "mesh.ply"))
    open3d.io.write_triangle_mesh(os.path.join(basedir, "mesh-aligned.ply"), mesh)

    print("Writing file:", os.path.join(basedir, "pcd.ply"))
    open3d.io.write_point_cloud(os.path.join(basedir, "pcd-aligned.ply"), cloud)

    # np.savetxt(os.path.join(basedir, "pcd.txt"), cloud)

    mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, origin=[0, 0, 0])

    open3d.visualization.draw_geometries([cloud, mesh_frame])
    open3d.visualization.draw_geometries([mesh, mesh_frame])


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')

    basedir = "../DATAROOT/ICL/living_room_traj2_frei"
    if not os.path.isfile(os.path.join(basedir, "dep_rgb_traj.txt")):
        get_association(os.path.join(basedir, "associations.txt"), os.path.join(basedir, "groundtruth.txt"), os.path.join(basedir, "dep_rgb_traj.txt"))

    # making z pointing upward
    transform = np.array([[1, 0, 0, 0],
                          [0, 0, -1, 0],
                          [0, 1, 0, 0],
                          [0, 0, 0, 1]])
    poses = []
    poses_aligned = []
    with open(os.path.join(basedir, "dep_rgb_traj.txt")) as f:
        for i, line in enumerate(f.readlines()):
            line_list = line.strip().split(" ")
            c2w = tum2matrix([float(x) for x in line_list[-7:]])
            poses += [c2w]
            poses_aligned += [transform @ c2w]

    np.savez(os.path.join(basedir, "raw_poses.npz"), c2w_mats=poses)
    np.savez(os.path.join(basedir, "aligned_poses.npz"), c2w_mats=poses_aligned)
    camera_intrinsics = open3d.camera.PinholeCameraIntrinsic(640, 480, 481.2, -480., 319.5, 239.5)
    integrate_rgbd_frames(basedir, camera_intrinsics)