"""Load the pinned official CPU vision implementation for development-time numerical qualification."""
import hashlib
import importlib.util
from types import SimpleNamespace

import numpy as np

from check_checkpoint_projection import read


SOURCE_SHA256 = "a4f089069310398d42ca17fd4496cec82da64cbbfde9b0230679ce1537cc0bb1"


class Official:
    def __init__(self, path, source, tensors, config):
        import torch
        if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256:
            raise ValueError("vision.py does not match pinned model revision 6821d6ad3681a4b137b066b76094fa82ebd0a380")
        spec = importlib.util.spec_from_file_location("deepseek_vision_reference", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        old_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            args = SimpleNamespace(**config, dim=config["hidden_size"])
            self.vit, self.aligner = module.ViT(args).eval(), module.Aligner(args).eval()
        finally:
            torch.set_default_dtype(old_dtype)

        def weights(prefix):
            state = {}
            for key in tensors:
                if key.startswith(prefix):
                    value = torch.from_numpy(read(source, tensors, key).copy())
                    if tensors[key]["dtype"] == "BF16":
                        value = value.view(torch.bfloat16)
                    state[key[len(prefix):]] = value
            return state

        self.vit.load_state_dict(weights("vision."))
        self.aligner.load_state_dict(weights("aligner."))
        self.torch = torch

    def run(self, patches, height, width):
        torch = self.torch
        bits = (patches.view(np.uint32) >> 16).astype(np.uint16)
        with torch.no_grad():
            x = torch.from_numpy(bits).view(torch.bfloat16)
            return self.aligner(self.vit(x, height, width), height, width).float().numpy()
