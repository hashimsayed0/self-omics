python train.py
    --use_sample_list True --gpus -1 --split_B True 
    --data_format tsv
    --split_A True --num_workers 4 --exp_name train 
    --deterministic True --mask_A True --num_mask_A 19 
    --change_ch_to_mask_every_epoch True 
    --choose_masking_method_every_epoch True
    --add_distance_loss_to_latent True 
    --recon_all_thrice True --ds_freeze_ae True 