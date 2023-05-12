"""
ShapeNerf dataset.
"""

from pathlib import Path
from typing import List, Tuple, Union, Dict
import torch
import numpy as np

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
import cv2

from PIL import Image
import os

def get_normal_map_from_path(
    filepath: Path,
    height: int,
    width: int,
    interpolation: int = cv2.INTER_NEAREST,
) -> torch.Tensor:
    """Loads, rescales and resizes depth images.
    Filepath points to a 16-bit or 32-bit depth image, or a numpy array `*.npy`.

    Args:
        filepath: Path to depth image.
        height: Target depth image height.
        width: Target depth image width.
        scale_factor: Factor by which to scale depth image.
        interpolation: Depth value interpolation for resizing.

    Returns:
        Depth image torch tensor with shape [width, height, 1].
    """
    if filepath.suffix == ".npy":
        normal_data = np.load(filepath, allow_pickle=True)
        normal = normal_data.item().get('normal')
        msk = ~normal_data.item().get('msk')

        normal = cv2.resize(normal, (width, height), interpolation=interpolation)
        msk = cv2.resize(msk.astype(int), (width, height), interpolation=interpolation)
    else:
        raise NotImplementedError()

    return torch.from_numpy(normal).float(), torch.from_numpy(msk).bool()

def get_surface_normal_by_depth(depth, fx, fy):
    """
    depth: (h, w) of float, the unit of depth is meter
    K: (3, 3) of float, the depth camere's intrinsic
    """

    msk = depth == 0
    dz_dv, dz_du = np.gradient(depth)  # u, v mean the pixel coordinate in the image
    # u*depth = fx*x + cx --> du/dx = fx / depth
    du_dx = fx / depth  # x is xyz of camera coordinate
    dv_dy = fy / depth

    dz_dx = dz_du * du_dx
    dz_dy = dz_dv * dv_dy
    # cross-product (1,0,dz_dx)X(0,1,dz_dy) = (-dz_dx, -dz_dy, 1)
    normal_cross = np.dstack((-dz_dx, -dz_dy, np.ones_like(depth)))
    # normalize to unit vector
    normal_unit = normal_cross / np.linalg.norm(normal_cross, axis=2, keepdims=True)
    # set default normal to [0, 0, 1]
    normal_unit[~np.isfinite(normal_unit).all(2)] = [0, 0, 1]
    return normal_unit, msk

def get_depth_from_path(
    filepath: Path,
    height: int,
    width: int,
    interpolation: int = cv2.INTER_NEAREST,
) -> torch.Tensor:
    """Loads, rescales and resizes depth images.
    Filepath points to a 16-bit or 32-bit depth image, or a numpy array `*.npy`.

    Args:
        filepath: Path to depth image.
        height: Target depth image height.
        width: Target depth image width.
        scale_factor: Factor by which to scale depth image.
        interpolation: Depth value interpolation for resizing.

    Returns:
        Depth image torch tensor with shape [width, height, 1].
    """
    if filepath.suffix == ".npy":
        depth_data = np.load(filepath, allow_pickle=True)
        msk = depth_data==0
        depth_data = depth_data/1000

        fx, fy, cx, cy= 400, 400, 400, 300
        # K = np.array([
        #     [fx, 0, cx],
        #     [0, fy, cy],
        #     [0,0,1]
        # ])

        # plane to radial depth
        for i in range(depth_data.shape[1]):
            for j in range(depth_data.shape[0]):
                depth_data[j, i] = np.sqrt(fx**2 + (i-cx)**2 + (j-cy)**2) * depth_data[j,i]/fx


        normal, msk = get_surface_normal_by_depth(depth_data, fx, fy)

        msk = ~msk

        depth_data = cv2.resize(depth_data, (width, height), interpolation=interpolation)   
        normal = cv2.resize(normal, (width, height), interpolation=interpolation)   
        msk = cv2.resize(msk.astype(int), (width, height), interpolation=interpolation)
    else:
        raise NotImplementedError()

    return torch.from_numpy(depth_data).float(), torch.from_numpy(normal).float(), torch.from_numpy(msk).bool()


class ShapeNerfDataset(InputDataset):
    """Dataset that returns images and depths.
    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        assert (
            "normal_filenames" in dataparser_outputs.metadata.keys()
            and dataparser_outputs.metadata["normal_filenames"] is not None
        )
        self.normal_filenames = self.metadata["normal_filenames"]
        self.depth_filenames = self.metadata["depth_filenames"]

    def plane_depth2radial(depth, f, cx, cy):
        f = transform['fl_x']
        for i in range(self.W):
            for j in range(self.H):
                depth_data[j, i] = np.sqrt(f**2 + (i-transform['cx'])**2 + (j-transform['cy'])**2) * depth_data[j,i]/f

    def get_metadata(self, data: Dict) -> Dict:
        filepath = self.normal_filenames[data["image_idx"]]
        height = int(self._dataparser_outputs.cameras.height[data["image_idx"]])
        width = int(self._dataparser_outputs.cameras.width[data["image_idx"]])

        # NOTE: load plane depth created normal
        # normal_map, normal_msk = get_normal_map_from_path(filepath=filepath, height=height, width=width)
        # NOTE: Use radial depth create normal
        depth, normal_map, normal_msk = get_depth_from_path(filepath=self.depth_filenames[data["image_idx"]], height=height, width=width)

        # transfer the normal to world coordinate
        rotation = self.cameras[data["image_idx"]].camera_to_worlds[..., :3, :3]
        # NOTE: output
        # if os.environ.get('DEBUG', False):
        #     n_ori = normal_map.numpy() / np.linalg.norm(normal_map.numpy(), axis=2, keepdims=True)
        #     n_ori = ((n_ori + 1)/2 * 255)
        #     Image.fromarray(n_ori.astype(np.uint8), mode="RGB").save(f"outputs/tmp_debug/{data['image_idx']}_normal.png")

        # # normal_map[:, :, 0] *= -1
        # perm= [ 0, 1, 2]
        # normal_map = normal_map[:, :, perm]
        # # normal_map[:,:, 0]*=-1 
        # normal_map[:,:, 1]*=-1 
        # normal_map[:,:, 2]*=-1 

        # normal_map = normal_map.view(-1, 3) @ rotation.T
        # normal_map = torch.nn.functional.normalize(-normal_map)
        # normal_map = normal_map.view(normal_msk.shape[0], normal_msk.shape[1], 3)
        
        # if os.environ.get('DEBUG', False):
        #     tmp_msk = normal_msk[..., None].repeat(1,1,3)
        #     normal_map[~tmp_msk] = 0
        #     n_transfer = normal_map.numpy() #/ np.linalg.norm(normal_map.numpy(), axis=2, keepdims=True)
        #     n_transfer = ((n_transfer + 1)/2 * 255)
        #     # n_transfer = n_transfer / np.linalg.norm(n_transfer, axis=2, keepdims=True)
        #     Image.fromarray(n_transfer.astype(np.uint8), mode="RGB").save(f"outputs/tmp_debug/{data['image_idx']}_normal_t.png")

        # NOTE: if use global normal sup
        # perm= [ 0, 1, 2]
        # normal_map = normal_map[:, :, perm]
        # normal_map[:,:, 1]*=-1 
        # normal_map[:,:, 2]*=-1 

        # normal_map = normal_map.view(-1, 3) @ rotation.T
        # normal_map = torch.nn.functional.normalize(-normal_map)
        # normal_map = normal_map.view(normal_msk.shape[0], normal_msk.shape[1], 3)

        return {"normal_map": normal_map, 'normal_msk':normal_msk}