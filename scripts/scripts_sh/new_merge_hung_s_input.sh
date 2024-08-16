CUDA_VISIBLE_DEVICES=1 python training.py \
    --prefix_name new_merge_hung_s_input_p256_180_180_161_v256_256_32 \
    --pi 256 180 180 161 \
    --vf 256 256 32 \
    --obs-order none_tile color_1 color_2 color_3 color_4 color_5 pu blocker monster monster_match_dmg_mask monster_inside_dmg_mask self_dmg_mask legal_action \
    --mid_channels 32 \
    --num_first_cnn_layer 10 \
    --n_steps 32768 \
    --lr 0.00002 \
    --gamma 0.95 \
    --wandb