import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from process_utils import save_outputs, load_render_cond_info

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

parts_info_list = load_render_cond_info(sh256="a36a48897cc62006f0b49bc12d30cd38f28c83704feb203815365405247295ee", 
        json_dir="/mnt/pfs/users/yangyunhan/yufan/TRELLIS/datasets/Parts/renders_cond")

num_images = len(parts_info_list)

# for i in range(num_images):
    # Run the pipeline
print(parts_info_list[:3][0])
outputs = pipeline.run(
    parts_info_list[:3][0],
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

save_outputs(outputs, filename_prefix=f"part_{3}", save_video=True, save_glb=True)

# image = Image.open("/mnt/pfs/users/yangyunhan/yufan/TRELLIS/datasets/Parts/renders/a36a48897cc62006f0b49bc12d30cd38f28c83704feb203815365405247295ee/0027.webp") # image to be rendered

# # Run the pipeline
# outputs = pipeline.run(
#     image,
#     seed=1,
#     # Optional parameters
#     # sparse_structure_sampler_params={
#     #     "steps": 12,
#     #     "cfg_strength": 7.5,
#     # },
#     # slat_sampler_params={
#     #     "steps": 12,
#     #     "cfg_strength": 3,
#     # },
# )

# save_outputs(outputs, filename_prefix=f"sample_image_model", save_video=True, save_glb=True)