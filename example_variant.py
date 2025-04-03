import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import open3d as o3d
from trellis.pipelines import TrellisTextTo3DPipeline
import os
from process_utils import save_outputs

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisTextTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-text-xlarge")
pipeline.cuda()

# Load mesh to make variants
base_mesh = o3d.io.read_triangle_mesh("assets/T.ply")

# Run the pipeline
outputs = pipeline.run_variant(
    base_mesh,
    # "Rugged, metallic texture with orange and white paint finish, suggesting a durable, industrial feel.",
    # "Full of vitality, with many strawberries hanging on it and some small bees",
    # "Change the shape into Z-shaped",
    "Turn the leaves into blue",
    seed=1,
    # Optional parameters
    # slat_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 7.5,
    # },
)

save_outputs(outputs, filename_prefix="sample_variant", save_video=False, save_glb=True)