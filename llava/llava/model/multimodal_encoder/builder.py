import os
from .clip_encoder import CLIPECGTower


def build_ecg_tower(ecg_tower_cfg, **kwargs):
    model_name = getattr(ecg_tower_cfg, 'mm_ecg_tower', getattr(ecg_tower_cfg, 'ecg_tower', None))
    checkpoint_path = getattr(ecg_tower_cfg, 'mm_ecg_tower', getattr(ecg_tower_cfg, 'ecg_tower', None))
    is_absolute_path_exists = os.path.exists(checkpoint_path)
    if is_absolute_path_exists:
        return CLIPECGTower(checkpoint_path, args=ecg_tower_cfg, **kwargs)

    raise ValueError(f'Unknown ecg tower: {checkpoint_path}')
