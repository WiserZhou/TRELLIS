from typing import *
import torch
import torch.nn as nn
from .. import models
from ..utils.transfer_st_pt import convert_pt_to_safetensors, save_json

class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: Dict[str, nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()
    
    def finetune_from_pretrained(self, model_name:str, path: str):
        """
        Load a finetuned part model.
        """
        st_path_name = convert_pt_to_safetensors(path)
        save_json(model_name, st_path_name)
        try:
            self.models[model_name] = models.from_pretrained(f"{st_path_name}")
        except:
            raise RuntimeError(f"Model {model_name} not found in {st_path_name}")
    
    def finetune_from_pretrained_list(self, model_name_paths: List[Tuple[str, str]]):
        """
        Load a finetuned part model.
        """
        for model_name, path in model_name_paths:
            self.finetune_from_pretrained(model_name, path)

    @staticmethod
    def from_pretrained(path: str) -> "Pipeline":
        """
        Load a pretrained model.
        """
        import os
        import json
        is_local = os.path.exists(f"{path}/pipeline.json")

        if is_local:
            config_file = f"{path}/pipeline.json"
        else:
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(path, "pipeline.json")

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        _models = {}
        for k, v in args['models'].items():
            try:
                _models[k] = models.from_pretrained(f"{path}/{v}")
            except:
                _models[k] = models.from_pretrained(v)

        new_pipeline = Pipeline(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    @property
    def device(self) -> torch.device:
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))
