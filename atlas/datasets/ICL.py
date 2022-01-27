import json
import os

import numpy as np


def prepare_ICL_scene(scene, path, path_meta, verbose=2):
    """Generates a json file for a sample scene in our common format

    This wraps all the data about a scene into a single common format
    that is readable by our dataset class. This includes paths to all
    the images, depths, etc, as well as other metadata like camera
    intrinsics and pose. It also includes scene level information like
    a path to the mesh.

    Args:
        scene: name of the scene.
            examples: 'scans/scene0000_00'
                      'scans_test/scene0708_00'
        path: path to the original data
        path_meta: path to where the generated data is saved.
            This can be the same as path, but it is recommended to
            keep them seperate so the original data is not accidentally
            modified. The generated data is saved into a mirror directory
            structure.

    Output:
        Creates the file path_meta/scene/info.json

    JSON format:
        {'dataset': 'sample',
         'path': path,
         'scene': scene,
         'frames': [{'file_name_image': '',
                     'intrinsics': intrinsics,
                     'pose': pose,
                     }
                   ]
         }

    """

    if verbose>0:
        print('preparing %s' % scene)

    data = {'dataset': 'ICL',
            'path': path,
            'scene': scene,
            'frames': []
            }

    intrinsics = np.loadtxt(os.path.join(path, scene, 'intrinsics.txt'))
    poses = np.load(os.path.join(path, scene, "aligned_poses.npz"))["c2w_mats"]
    frame_ids = os.listdir(os.path.join(path, scene, 'rgb'))
    frame_ids = [int(os.path.splitext(frame)[0]) for frame in frame_ids]
    frame_ids = sorted(frame_ids)

    for i, frame_id in enumerate(frame_ids[1:]):
        if verbose>1 and i % 25 == 0:
            print('preparing %s frame %d/%d' % (scene, i, len(frame_ids)))

        pose = poses[i]

        # skip frames with no valid pose
        if not np.all(np.isfinite(pose)):
            continue

        frame = {'file_name_image': os.path.join(path, scene, "rgb/{:d}.png".format(frame_id)),
                 'intrinsics': intrinsics.tolist(),
                 'pose': pose.tolist(),
                 }
        data['frames'].append(frame)

    os.makedirs(os.path.join(path_meta, scene), exist_ok=True)
    json.dump(data, open(os.path.join(path_meta, scene, 'info.json'), 'w'))
