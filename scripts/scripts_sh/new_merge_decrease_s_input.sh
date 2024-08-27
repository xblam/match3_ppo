CUDA_VISIBLE_DEVICES=1 python training.py \
    --prefix_name new_merge_decrease_s_input_p256_180_180_161_v256_256_32 \
    --pi 256 180 180 161 \ # linear layers actor
    --vf 256 256 32 \ # critic
    --mid_channels 32 \ # cnn
    --num_first_cnn_layer 10 \
    --n_steps 32768 \
    --lr 0.00002 \
    --gamma 0.95 \
    --num_envs 4 \
    --wandb