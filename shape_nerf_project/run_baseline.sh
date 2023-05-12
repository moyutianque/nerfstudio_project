data_path=../data/replica_dinning_room 
data_path=../data/replica_multiroom2 

# rm -rf outputs/tmp_debug/*

# ns-train shape_nerf --pipeline.model.predict-normals True \
# --vis viewer+wandb \
# --data $data_path


config_path=outputs/replica_multiroom2/shape_nerf/baseline-refnerf-ingp-largeset/config.yml

# ns-viewer --load-config $config_path


# ns-render --rendered-output-names rgb depth normals pred_normals \
# --load-config $config_path --traj filename \
# --camera-path-filename ../camera_path.json \
# --output-path renders/replica_multiroom2/baseline.mp4 