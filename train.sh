# Example command:
export http_proxy=http://192.168.48.17:18000; export https_proxy=http://192.168.48.17:18000
nohup python train.py \
    --config configs/generation/ss_flow_img_dit_L_16l8_fp16.json \
    --load_dir outputs/ss_flow_img_dit_L_16l8_fp16_1node \
    --output_dir outputs/ss_flow_img_dit_L_16l8_fp16_1node \
    --data_dir /mnt/pfs/users/yangyunhan/yufan/TRELLIS/datasets/Parts > train_finetune_ss.log 2>&1 &

python train.py \
    --config configs/generation/ss_flow_img_dit_L_16l8_fp16.json \
    --load_dir outputs/ss_flow_img_dit_L_16l8_fp16_1node \
    --output_dir outputs/ss_flow_img_dit_L_16l8_fp16_1node \
    --data_dir /mnt/pfs/users/yangyunhan/yufan/TRELLIS/datasets/Parts

# finetune the first stage
export http_proxy=http://192.168.48.17:18000; export https_proxy=http://192.168.48.17:18000
nohup python train.py \
    --config configs/generation/ss_flow_img_dit_L_16l8_fp16_finetune.json \
    --load_dir outputs/ss_flow_img_dit_L_16l8_fp16_1node_finetune \
    --output_dir outputs/ss_flow_img_dit_L_16l8_fp16_1node_finetune \
    --data_dir /mnt/pfs/users/yangyunhan/yufan/TRELLIS/datasets/Parts > finetune_ss.log 2>&1 &

python train.py \
    --config configs/generation/ss_flow_img_dit_L_16l8_fp16_finetune.json \
    --load_dir outputs/ss_flow_img_dit_L_16l8_fp16_1node_finetune \
    --output_dir outputs/ss_flow_img_dit_L_16l8_fp16_1node_finetune \
    --data_dir /mnt/pfs/users/yangyunhan/yufan/TRELLIS/datasets/Parts

# finetune the second stage
export http_proxy=http://192.168.48.17:18000; export https_proxy=http://192.168.48.17:18000
nohup python train.py \
    --config configs/generation/slat_flow_img_dit_L_64l8p2_fp16_finetune.json \
    --load_dir outputs/slat_flow_img_dit_L_64l8p2_fp16_1node_finetune \
    --output_dir outputs/slat_flow_img_dit_L_64l8p2_fp16_1node_finetune \
    --data_dir /mnt/pfs/users/yangyunhan/yufan/TRELLIS/datasets/Parts > finetune_slat.log 2>&1 &

python train.py \
    --config configs/generation/slat_flow_img_dit_L_64l8p2_fp16_finetune.json \
    --load_dir outputs/slat_flow_img_dit_L_64l8p2_fp16_1node_finetune \
    --output_dir outputs/slat_flow_img_dit_L_64l8p2_fp16_1node_finetune \
    --data_dir /mnt/pfs/users/yangyunhan/yufan/TRELLIS/datasets/Parts

# resume
python train.py \
  --config configs/generation/ss_flow_img_dit_L_16l8_fp16_finetune.json \
  --output_dir outputs/slat_flow_img_dit_L_64l8p2_fp16_resume \
  --data_dir /mnt/pfs/users/yangyunhan/yufan/TRELLIS/datasets/Parts \
  --load_dir /mnt/pfs/users/yangyunhan/yufan/TRELLIS/outputs2/ss_flow_img_dit_L_16l8_fp16_1node/ckpts \
  --ckpt 10000

# visual ss_latent
python sparse_vae.py --input_path /mnt/pfs/users/yangyunhan/yufan/TRELLIS/datasets/Parts/voxels/a36a48897cc62006f0b49bc12d30cd38f28c83704feb203815365405247295ee.ply