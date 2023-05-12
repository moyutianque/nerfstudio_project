data_path=../data/replica_dinning_room 
# data_path=../data/replica_multiroom2 

# rm -rf outputs/tmp_debug/*

DEBUG=1 ns-train shape_nerf --pipeline.model.predict-normals True \
--pipeline.model.sup_gt_normals True --vis viewer+wandb \
--data $data_path

# ns-viewer --load-config outputs/replica_multiroom2/shape_nerf/rendered_normal_vs_gt_normal-cosloss-camoff-largeset/config.yml

## Output to mesh
# input_path=outputs/replica_dinning_room/shape_nerf/density_normal_vs_gt_normal-cosloss-camoff
# ns-export poisson --load-config $input_path/config.yml --normal_output_name "normals" --output-dir $input_path

# input_path=outputs/replica_dinning_room/shape_nerf/rendered_normal_vs_gt_normal-cosloss-camon
# ns-export poisson --load-config $input_path/config.yml --normal_output_name "pred_normals" --output-dir $input_path


# config_path=outputs/replica_multiroom2/shape_nerf/rendered_normal_vs_gt_normal-cosloss-camoff-largeset/config.yml

# ns-render --rendered-output-names rgb depth normals pred_normals \
# --load-config $config_path --traj filename \
# --camera-path-filename ../camera_path.json \
# --output-path renders/replica_multiroom2/rendered_normal.mp4 