import os
import argparse
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
# os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from process_utils import save_outputs, load_render_cond_info

def parse_args():
    parser = argparse.ArgumentParser(description="TRELLIS Image-to-3D Pipeline")
    
    # Environment settings
    parser.add_argument("--attn_backend", type=str, default="flash-attn", 
                       help="Attention backend: 'flash-attn' or 'xformers'")
    parser.add_argument("--spconv_algo", type=str, default="native", 
                       help="SPCONV algorithm: 'native' or 'auto'")
    
    # Model settings
    parser.add_argument("--model_path", type=str, default="JeffreyXiang/TRELLIS-image-large",
                       help="Path to pretrained model")
    parser.add_argument("--finetune_component", type=str, default="sparse_structure_flow_model",
                       help="Component to finetune")
    parser.add_argument("--finetune_path", type=str, 
                       default="/mnt/pfs/users/yangyunhan/yufan/TRELLIS/outputs/ss_flow_img_dit_L_16l8_fp16_1node_finetune2/ckpts/denoiser_step0020000.pt",
                       help="Path to finetuned model")
    
    # Data settings
    parser.add_argument("--sh256", type=str, 
                       default="a36a48897cc62006f0b49bc12d30cd38f28c83704feb203815365405247295ee",
                       help="SHA256 hash for render condition info")
    parser.add_argument("--json_dir", type=str,
                       default="/mnt/pfs/users/yangyunhan/yufan/TRELLIS/datasets/Parts/renders_cond",
                       help="Directory for JSON render condition files")
    parser.add_argument("--num_samples", type=int, default=3,
                       help="Number of samples to process")
    parser.add_argument("--seed", type=int, default=1,
                       help="Random seed")
    
    # Output settings
    parser.add_argument("--output_prefix", type=str, default="finetune2_3_parts",
                       help="Prefix for output filenames")
    parser.add_argument("--save_video", action="store_true", default=False,
                       help="Save video output")
    parser.add_argument("--save_glb", action="store_true", default=True,
                       help="Save GLB output")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set environment variables
    os.environ['ATTN_BACKEND'] = args.attn_backend  # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
    os.environ['SPCONV_ALGO'] = args.spconv_algo    # Can be 'native' or 'auto', default is 'auto'.
                                                   # 'auto' is faster but will do benchmarking at the beginning.
                                                   # Recommended to set to 'native' if run only once.

    # Load a pipeline from a model folder or a Hugging Face model hub.
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.model_path)

    # "sparse_structure_decoder": "ckpts/ss_dec_conv3d_16l8_fp16",
    # "sparse_structure_flow_model": "ckpts/ss_flow_img_dit_L_16l8_fp16",
    # "slat_decoder_gs": "ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16",
    # "slat_decoder_rf": "ckpts/slat_dec_rf_swin8_B_64l8r16_fp16",
    # "slat_decoder_mesh": "ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16",
    # "slat_flow_model": "ckpts/slat_flow_img_dit_L_64l8p2_fp16"

    pipeline.finetune_from_pretrained(model_name=args.finetune_component, path=args.finetune_path)

    pipeline.cuda()

    parts_info_list = load_render_cond_info(sh256=args.sh256, json_dir=args.json_dir)

    num_images = len(parts_info_list)

    # for i in range(num_images):
        # Run the pipeline
    print(parts_info_list[:args.num_samples][0])
    outputs = pipeline.run(
        parts_info_list[:args.num_samples][0],
        seed=args.seed,
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

    save_outputs(outputs, filename_prefix=args.output_prefix, save_video=args.save_video, save_glb=args.save_glb)

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

if __name__ == "__main__":
    main()