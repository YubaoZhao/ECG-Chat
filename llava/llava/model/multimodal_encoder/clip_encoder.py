import torch
import torch.nn as nn
import sys

from training import get_ecg_encoder


class CLIPECGTower(nn.Module):
    def __init__(self, ecg_tower, args, delay_load=False):
        super().__init__()

        self.model_config = None
        self.ecg_processor = None
        self.ecg_tower = None
        self.is_loaded = False

        self.ecg_tower_name = ecg_tower
        self.model_name = getattr(args, 'open_clip_config', None)
        if self.model_name is None:
            raise ValueError('No open_clip config for building ECG encoder!')

        self.load_model(self.model_name)

        ecg_config = self.model_config.get('ecg_cfg', {})

        self.hidden_size = ecg_config.get('width', 768)
        self.seq_length = ecg_config.get('seq_length', 5000)
        self.patch_size = ecg_config.get('patch_size', 50)
        self.device = self.ecg_tower.state_dict()['class_embedding'].device
        self.dtype = self.ecg_tower.state_dict()['class_embedding'].dtype

        self.num_patches_per_side = self.seq_length // self.patch_size
        self.num_patches = self.seq_length // self.patch_size


    def load_model(self, model_name, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.ecg_tower, self.ecg_processor, self.model_config = get_ecg_encoder(model_name, checkpoint_path=self.ecg_tower_name, device='cpu')
        self.ecg_tower.requires_grad_(False)

        self.is_loaded = True
        print("Loaded {} model".format(self.ecg_tower_name))

    @torch.no_grad()
    def forward(self, ecgs):
        self.device = self.ecg_tower.state_dict()['class_embedding'].device
        self.dtype = self.ecg_tower.state_dict()['class_embedding'].dtype
        if type(ecgs) is list:
            ecg_features = []
            for ecg in ecgs:
                ecg_feature = self.ecg_tower(ecg.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_last_transformer_layer=True)
                ecg_feature = ecg_feature.to(ecg.dtype)
                ecg_features.append(ecg_feature)
        else:
            ecg_features = self.ecg_tower(ecgs.to(device=self.device, dtype=self.dtype), output_last_transformer_layer=True)
            ecg_features = ecg_features.to(ecgs.dtype)

        return ecg_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

