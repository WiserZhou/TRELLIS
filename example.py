import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from process_utils import save_outputs, load_render_info

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

# parts_info, while_model_paths = load_render_info("/mnt/pfs/users/yangyunhan/yufan/VRenderer/render_info.json")

# image_part_list = []

# for part_name, image_paths in parts_info.items():
#     image_part_list.append(Image.open(image_paths[0])) # front part image

# num_images = len(image_part_list)

# for i in range(num_images):
#     # Run the pipeline
#     outputs = pipeline.run(
#         image_part_list[:i+1],
#         seed=1,
#         # Optional parameters
#         # sparse_structure_sampler_params={
#         #     "steps": 12,
#         #     "cfg_strength": 7.5,
#         # },
#         # slat_sampler_params={
#         #     "steps": 12,
#         #     "cfg_strength": 3,
#         # },
#     )

#     save_outputs(outputs, filename_prefix=f"sample_image_{i+1}", save_video=True, save_glb=False)


image = Image.open("/mnt/pfs/users/yangyunhan/yufan/TRELLIS/datasets/Parts/renders/a36a48897cc62006f0b49bc12d30cd38f28c83704feb203815365405247295ee/0027.webp") # image to be rendered

# Run the pipeline
outputs = pipeline.run(
    image,
    seed=1,
    # Optional parameters
    # sparse_structure_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 7.5,
    # },
    # slat_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 3,
    # },
)

save_outputs(outputs, filename_prefix=f"sample_image_{i+1}", save_video=True, save_glb=False)