python dataset_toolkits/build_metadata.py Parts --output_dir datasets/Parts

python dataset_toolkits/download.py Parts --output_dir datasets/Parts
python dataset_toolkits/build_metadata.py Parts --output_dir datasets/Parts

# render
python3 dataset_toolkits/render.py Parts --output_dir datasets/Parts
python3 dataset_toolkits/transfer_ply.py Parts --output_dir datasets/Parts

python dataset_toolkits/build_metadata.py Parts --output_dir datasets/Parts

python dataset_toolkits/voxelize.py Parts --output_dir datasets/Parts
python dataset_toolkits/build_metadata.py Parts --output_dir datasets/Parts

python dataset_toolkits/extract_feature.py --output_dir datasets/Parts
python dataset_toolkits/build_metadata.py Parts --output_dir datasets/Parts

python dataset_toolkits/encode_ss_latent.py --output_dir datasets/Parts
python dataset_toolkits/build_metadata.py Parts --output_dir datasets/Parts

python dataset_toolkits/encode_latent.py --output_dir datasets/Parts
python dataset_toolkits/build_metadata.py Parts --output_dir datasets/Parts

# render
python3 dataset_toolkits/render_cond.py Parts --output_dir datasets/Parts

python dataset_toolkits/build_metadata.py Parts --output_dir datasets/Parts


# render pip
pip install pillow bpy pandas numpy trimesh easydict tqdm