import open3d as o3d
import os.path as osp
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
hfov = float(90) * np.pi / 180.

def depth2points(depth_im, extrinsic_info, color_im=None):
    H, W = depth_im.shape
    vfov = 2 * np.arctan(np.tan(hfov/2)*H/W)
    fl_x = W / (2 * np.tan(hfov / 2.)) # 320
    fl_y = H / (2 * np.tan(vfov / 2.)) # 320

    cx=320
    cy=240

    # xs, ys = np.meshgrid(np.arange(W), np.arange(H-1,-1,-1))
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    depth = depth_im.reshape(1,H,W)/1000
    xs = xs.reshape(1,H,W)
    ys = ys.reshape(1,H,W)
    
    xs = (xs - cx) / fl_x
    ys = (ys - cy) / fl_y

    xys = np.vstack((xs * depth , -ys * depth, -depth, np.ones(depth.shape)))

    # msk = depth > 0
    # msk = msk.flatten()
    xys = xys.reshape(4, -1)
    # xy_c0 = xys[:, msk]
    xy_c0 = xys

    colors = None
    if color_im is not None:
        colors = color_im[:,:,:3]
        colors = colors.reshape(-1, 3)
        colors = colors[msk, :]

    T_world_camera0 = np.array(extrinsic_info)

    # Finally transform actual points
    pcd = np.matmul(T_world_camera0, xy_c0)
    return np.transpose(pcd)[:,:3], colors

def pc2normals(pcd, cam_pose):
    cam_pose = np.array(cam_pose)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
    pcd.orient_normals_towards_camera_location(cam_pose[:3, 3])
    normal_vecs = np.asarray(pcd.normals)

    # NOTE: My implementation of orient_normals_towards_camera_location below is not correct
    # cam_dir = np.matmul(cam_pose, np.array([[0],[0],[0],[1]]))
    # cam_dir = np.transpose(cam_dir)[:,:3]
    # cam_dir /= np.linalg.norm(cam_dir)

    # inner_product = np.inner(normal_vecs, cam_dir) # if inner > 
    # print(np.amin(inner_product), np.amax(inner_product))
    # error_dirs_msk = (inner_product<0)
    # normal_vecs[error_dirs_msk[:,0], :] = -normal_vecs[error_dirs_msk[:,0], :]

    return normal_vecs

def run(data_root):
    file_paths = []
    rgb_paths = []
    cam_poses = []
    cam_poses = []
    meta = json.load(open(osp.join(data_root, 'transforms.json')))
    scale_factor = meta['scale_factor']

    for line in meta['frames']:
        i = int(line['file_path'].split('/')[-1].split('.')[0])
        file_path = f"depth/{i}.npy"
        file_paths.append(file_path)
        rgb_paths.append(f"color/{i}.png")
        cam_poses.append(line["transform_matrix"])
    
    # file_paths = file_paths[::10]
    # cam_poses = cam_poses[::10]
    # rgb_paths = rgb_paths[::10]
    # file_paths = [file_paths[0], file_paths[1]]
    # cam_poses = [cam_poses[0], cam_poses[1]]
    # file_paths = file_paths[:1]
    # cam_poses = cam_poses[:1]
    # rgb_paths = rgb_paths[:1]

    # file_paths = [file_paths[0], file_paths[1]]
    # cam_poses = [cam_poses[0], cam_poses[1]]
    # rgb_paths = [rgb_paths[0], rgb_paths[1]]
    

    pcd_nps = []
    color_list = []
    for i, (file_name, rgb_path, cam_pose) in enumerate(zip(file_paths, rgb_paths, cam_poses)):
        depth_im = np.load(osp.join(data_root, file_name)) * scale_factor

        pcd_np, colors = depth2points(depth_im, extrinsic_info=cam_pose)
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np)
        
        normal_map = pc2normals(pcd_o3d, cam_pose)
        normal_map = normal_map.reshape(depth_im.shape[0], depth_im.shape[1], 3)

        msk = depth_im==0 
        normal_map[msk, :] = 0

        img_idx = file_name.split('/')[1].split('.')[0]
        np.save(os.path.join(data_root, 'normal', f"{img_idx}.npy"), {"normal": normal_map, 'msk':msk})

        Image.fromarray(((normal_map+1)/2 * 255).astype(np.uint8)).save(os.path.join(data_root, 'normal', f"{img_idx}.png"))
        # pcd_o3d.colors = o3d.utility.Vector3dVector((np.asarray(pcd_o3d.normals)+1)/2)
        # o3d.visualization.draw_geometries([pcd_o3d])

if __name__ == '__main__':
    run('../data/replica_dinning_room')