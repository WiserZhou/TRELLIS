import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

from trellis.pipelines import TrellisTextTo3DPipeline
from process_utils import save_outputs

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisTextTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-text-xlarge")
pipeline.cuda()

# Run the pipeline
outputs = pipeline.run(
    # "A chair looking like a avocado.",
    "a woman in a red dress sitting on a chair",
    seed=1,
    # Optional parameters
    # sparse_structure_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 7.5,
    # },
    # slat_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 7.5,
    # },
)

save_outputs(outputs, output_dir="./output", filename_prefix="sample_text", save_video=False, save_glb=True)