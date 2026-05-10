from functools import partial

from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate, t2m_prefix_collate

def get_dataset_class(name):
    if name == "amass":
        from .amass import AMASS
        return AMASS
    elif name == "uestc":
        from .a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    elif name == "lafan_g1":
        from data_loaders.g1 import LAFANG1
        return LAFANG1
    elif name == "latent_tennis_g1":
        from data_loaders.g1 import LatentTennisG1
        return LatentTennisG1
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train', pred_len=0, batch_size=1):
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        if pred_len > 0:
            return partial(t2m_prefix_collate, pred_len=pred_len)
        return partial(t2m_collate, batch_size=batch_size)
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='train', abs_path='.', fixed_len=0,
                device=None, autoregressive=False, cache_path=None, pred_len=0, motion_filter='*.npz',
                prefix_motion_filter='', prefix_file='', prefix_start=-1):
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode, abs_path=abs_path, fixed_len=fixed_len, 
                       device=device, autoregressive=autoregressive)
    elif name in ["lafan_g1", "latent_tennis_g1"]:
        dataset = DATA(split=split, num_frames=num_frames, data_dir=abs_path, fixed_len=fixed_len,
                       pred_len=pred_len, motion_filter=motion_filter,
                       prefix_motion_filter=prefix_motion_filter, prefix_file=prefix_file,
                       prefix_start=prefix_start)
    else:
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train', fixed_len=0, pred_len=0,
                       device=None, autoregressive=False, num_workers=8, data_dir='', motion_filter='*.npz',
                       prefix_motion_filter='', prefix_file='', prefix_start=-1):
    dataset = get_dataset(name, num_frames, split=split, hml_mode=hml_mode, fixed_len=fixed_len,
                device=device, autoregressive=autoregressive, abs_path=data_dir, pred_len=pred_len,
                motion_filter=motion_filter, prefix_motion_filter=prefix_motion_filter,
                prefix_file=prefix_file, prefix_start=prefix_start)
    
    collate = get_collate_fn(name, hml_mode, pred_len, batch_size)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True, collate_fn=collate,
        persistent_workers=num_workers > 0,
        pin_memory=device is not None and device.type == 'cuda',
    )

    return loader
