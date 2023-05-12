from typing import Dict

import tyro
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from shape_nerf.data_parser import ShapeNerfDataParserConfig
from shape_nerf.model import ShapeNerfModelConfig
from shape_nerf.data_manager import ShapeNerfDataManagerConfig

shape_nerf_base=MethodSpecification(
    config=TrainerConfig(
        method_name="shape_nerf",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=ShapeNerfDataManagerConfig(
                dataparser=ShapeNerfDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                # camera_optimizer=CameraOptimizerConfig(
                #     mode="SO3xR3",
                #     optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                #     scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
                # ),
                camera_optimizer=CameraOptimizerConfig(
                    mode="off",
                ),
            ),
            model=ShapeNerfModelConfig(eval_num_rays_per_chunk=1 << 15),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Base config for shape nerf",
)