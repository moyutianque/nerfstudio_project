# data_path=../data/replica_dinning_room 
data_path=../data/replica_multiroom2 

ns-train shape_nerf --pipeline.model.predict-normals True \
--pipeline.model.sup_gt_normals True --vis viewer+wandb \
--data $data_path


# ns-viewer --load-config outputs/replica_multiroom2/shape_nerf/project_gt_normal_vs_pred_normal/config.yml