python for_Parts_dataset_toolkits/build_metadata.py Parts --output_dir datasets/Parts

python for_Parts_dataset_toolkits/download.py Parts --output_dir datasets/Parts
python for_Parts_dataset_toolkits/build_metadata.py Parts --output_dir datasets/Parts

# render
python3 for_Parts_dataset_toolkits/render.py Parts --output_dir datasets/Parts
python3 for_Parts_dataset_toolkits/transfer_ply.py Parts --output_dir datasets/Parts

python for_Parts_dataset_toolkits/build_metadata.py Parts --output_dir datasets/Parts

python for_Parts_dataset_toolkits/voxelize.py Parts --output_dir datasets/Parts
python for_Parts_dataset_toolkits/build_metadata.py Parts --output_dir datasets/Parts

export http_proxy=http://192.168.48.17:18000; export https_proxy=http://192.168.48.17:18000

python for_Parts_dataset_toolkits/extract_feature.py --output_dir datasets/Parts
python for_Parts_dataset_toolkits/build_metadata.py Parts --output_dir datasets/Parts

python for_Parts_dataset_toolkits/encode_ss_latent.py --output_dir datasets/Parts
python for_Parts_dataset_toolkits/build_metadata.py Parts --output_dir datasets/Parts

python for_Parts_dataset_toolkits/encode_latent.py --output_dir datasets/Parts
python for_Parts_dataset_toolkits/build_metadata.py Parts --output_dir datasets/Parts

# render
# python3 for_Parts_dataset_toolkits/render_cond.py Parts --output_dir datasets/Parts
python3 for_Parts_dataset_toolkits/render_parts_cond.py Parts --output_dir datasets/Parts

python for_Parts_dataset_toolkits/build_metadata.py Parts --output_dir datasets/Parts


# render pip
pip install pandas easydict bpy tqdm trimesh

/mnt/pfs/users/yangyunhan/blender-4.0.0-linux-x64/blender -b -P for_Parts_dataset_toolkits/render_parts_cond.py Parts --output_dir datasets/Parts