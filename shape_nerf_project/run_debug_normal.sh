data_path=../data/replica_dinning_room 
# data_path=../data/replica_multiroom2 

# rm -rf outputs/tmp_debug/*

# DEBUG=1 ns-train shape_nerf --pipeline.model.predict-normals True \
DEBUG=1 ns-train shape_nerf \
--pipeline.datamanager.patch_size 32 \
--pipeline.model.depth_normal_sup True \
--pipeline.datamanager.train_num_rays_per_batch 10240 --vis viewer+wandb \
--data $data_path

config_path=outputs/replica_dinning_room/shape_nerf/2023-05-11_121619/config.yml
config_path=outputs/replica_dinning_room/shape_nerf/2023-05-11_181612/config.yml

# ns-viewer --load-config $config_path 

## Output to mesh
# input_path=outputs/replica_dinning_room/shape_nerf/density_normal_vs_gt_normal-cosloss-camoff
# ns-export poisson --load-config $input_path/config.yml --normal_output_name "normals" --output-dir $input_path

# input_path=outputs/replica_dinning_room/shape_nerf/rendered_normal_vs_gt_normal-cosloss-camon
# ns-export poisson --load-config $input_path/config.yml --normal_output_name "pred_normals" --output-dir $input_path


# config_path=outputs/replica_multiroom2/shape_nerf/rendered_normal_vs_gt_normal-cosloss-camoff-largeset/config.yml

# ns-render --rendered-output-names rgb depth normals pred_normals depth_pred_normal \
# ns-render --rendered-output-names rgb depth depth_pred_normal \
# --load-config $config_path --traj filename \
# --camera-path-filename ../camera_path_dining.json \
# --output-path renders/replica_multiroom2/rendered_normal2.mp4 