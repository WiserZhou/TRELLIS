# Example command:
export http_proxy=http://192.168.48.17:18000; export https_proxy=http://192.168.48.17:18000
nohup python train.py \
    --config configs/generation/ss_flow_img_dit_L_16l8_fp16.json \
    --output_dir outputs/ss_flow_img_dit_L_16l8_fp16_1node \
    --data_dir /mnt/pfs/users/yangyunhan/yufan/TRELLIS/datasets/Parts > train_finetune_ss.log 2>&1 &

python train.py \
    --config configs/generation/ss_flow_img_dit_L_16l8_fp16.json \
    --output_dir outputs/ss_flow_img_dit_L_16l8_fp16_1node \
    --data_dir /mnt/pfs/users/yangyunhan/yufan/TRELLIS/datasets/Parts

# finetune
export http_proxy=http://192.168.48.17:18000; export https_proxy=http://192.168.48.17:18000
nohup python train.py \
    --config configs/generation/ss_flow_img_dit_L_16l8_fp16_finetune.json \
    --output_dir outputs/ss_flow_img_dit_L_16l8_fp16_1node_finetune \
    --data_dir /mnt/pfs/users/yangyunhan/yufan/TRELLIS/datasets/Parts > train_finetune_ss.log 2>&1 &

python train.py \
    --config configs/generation/ss_flow_img_dit_L_16l8_fp16_finetune.json \
    --output_dir outputs/ss_flow_img_dit_L_16l8_fp16_1node_finetune \
    --data_dir /mnt/pfs/users/yangyunhan/yufan/TRELLIS/datasets/Parts