"""
Classifier-Free Guidance Mixin for Flow Matching Models

This file implements a mixin class that adds classifier-free guidance capabilities to flow matching models.
Classifier-free guidance is a technique that combines conditional and unconditional generation to improve
sample quality and control. This mixin provides functionality for training with randomly dropped conditions
and for sampling with guidance during inference.
"""

import torch
import numpy as np
from ....utils.general_utils import dict_foreach
from ....pipelines import samplers


class ClassifierFreeGuidanceMixin:
    def __init__(self, *args, p_uncond: float = 0.1, **kwargs):
        """
        Initialize the ClassifierFreeGuidanceMixin.
        
        Args:
            p_uncond: Probability of using the unconditional branch during training.
                      Higher values mean more unconditional training.
            *args, **kwargs: Arguments to pass to the parent class.
        """
        super().__init__(*args, **kwargs)
        self.p_uncond = p_uncond

    def get_cond(self, cond, neg_cond=None, **kwargs):
        """
        Get the conditioning data for training with classifier-free guidance.
        Randomly replaces conditioning with negative conditioning based on p_uncond.
        
        Args:
            cond: The positive conditioning data.
            neg_cond: The negative conditioning data (usually null/empty conditioning).
            **kwargs: Additional arguments.
            
        Returns:
            Modified conditioning data with some entries replaced by negative conditioning.
        """
        assert neg_cond is not None, "neg_cond must be provided for classifier-free guidance" 

        if self.p_uncond > 0:
            # Helper function to determine batch size from conditioning data
            def get_batch_size(cond):
                if isinstance(cond, torch.Tensor):
                    return cond.shape[0]
                elif isinstance(cond, list):
                    return len(cond)
                else:
                    raise ValueError(f"Unsupported type of cond: {type(cond)}")
                
            # Get the batch size from the first available conditioning element
            ref_cond = cond if not isinstance(cond, dict) else cond[list(cond.keys())[0]]
            B = get_batch_size(ref_cond)
            
            # Helper function to selectively replace conditioning with negative conditioning
            def select(cond, neg_cond, mask):
                if isinstance(cond, torch.Tensor):
                    # For tensor data, use torch.where for efficient conditional replacement
                    mask = torch.tensor(mask, device=cond.device).reshape(-1, *[1] * (cond.ndim - 1))
                    return torch.where(mask, neg_cond, cond)
                elif isinstance(cond, list):
                    # For list data, use list comprehension to selectively replace elements
                    return [nc if m else c for c, nc, m in zip(cond, neg_cond, mask)]
                else:
                    raise ValueError(f"Unsupported type of cond: {type(cond)}")
            
            # Generate random mask for which samples to replace with negative conditioning
            mask = list(np.random.rand(B) < self.p_uncond)
            
            # Apply the conditioning replacement based on data type
            if not isinstance(cond, dict):
                cond = select(cond, neg_cond, mask)
            else:
                # For dictionary conditioning, apply the operation to each key-value pair
                cond = dict_foreach([cond, neg_cond], lambda x: select(x[0], x[1], mask))
    
        return cond

    def get_inference_cond(self, cond, neg_cond=None, **kwargs):
        """
        Get the conditioning data prepared for inference with classifier-free guidance.
        During inference, we need both conditional and unconditional branches.
        
        Args:
            cond: The positive conditioning data.
            neg_cond: The negative/null conditioning data.
            **kwargs: Additional arguments.
            
        Returns:
            Dictionary containing both positive and negative conditioning for the sampler.
        """
        assert neg_cond is not None, "neg_cond must be provided for classifier-free guidance"
        return {'cond': cond, 'neg_cond': neg_cond, **kwargs}
    
    def get_sampler(self, **kwargs) -> samplers.FlowEulerCfgSampler:
        """
        Get the specialized sampler for classifier-free guidance flow matching.
        
        Args:
            **kwargs: Additional arguments to pass to the sampler.
            
        Returns:
            An instance of FlowEulerCfgSampler configured with the model's sigma_min.
        """
        return samplers.FlowEulerCfgSampler(self.sigma_min)
