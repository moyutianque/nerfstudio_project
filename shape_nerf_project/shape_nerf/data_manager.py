from dataclasses import dataclass, field
from typing import Type
from nerfstudio.data.datamanagers import base_datamanager

from shape_nerf.dataset import ShapeNerfDataset


@dataclass
class ShapeNerfDataManagerConfig(base_datamanager.VanillaDataManagerConfig):
    """A datamanager - required to use with .setup()"""

    _target: Type = field(default_factory=lambda: ShapeNerfDataManager)


class ShapeNerfDataManager(base_datamanager.VanillaDataManager):  # pylint: disable=abstract-method
    """Data manager implementation for data that also requires processing data.
    Args:
        config: the DataManagerConfig used to instantiate class
    """
    def next_train(self, step: int):
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)

        camera_index = ray_indices[:, 0]
        cameras = self.train_dataset.cameras[camera_index]
        batch['cameras'] = cameras #camera_to_worlds
        self.train_dataset.metadata['patch_size'] = self.config.patch_size

        return ray_bundle, batch

    def create_train_dataset(self) -> ShapeNerfDataset:
        self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(split="train")
        return ShapeNerfDataset(
            dataparser_outputs=self.train_dataparser_outputs,
        )

    def create_eval_dataset(self) -> ShapeNerfDataset:
        return ShapeNerfDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
        )