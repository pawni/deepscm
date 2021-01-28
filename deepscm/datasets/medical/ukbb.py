import os
import pandas as pd
import SimpleITK as sitk
import warnings

import torch

from torch.utils.data.dataset import Dataset


def center_crop(img: torch.Tensor, crop_size):
    img_size = img.shape

    assert len(img_size) == 3
    assert len(img_size) >= len(crop_size)

    crop_size = tuple([1 for _ in range(len(img_size) - len(crop_size))] + list(crop_size))

    offset = tuple((i - c) // 2 for i, c in zip(img_size, crop_size))

    img = img[
        offset[0]:offset[0] + crop_size[0],
        offset[1]:offset[1] + crop_size[1],
        offset[2]:offset[2] + crop_size[2],
    ]

    return img.squeeze()


class UKBBDataset(Dataset):
    def __init__(self, csv_path, base_path='/vol/biobank/12579/brain/rigid_to_mni/images/', crop_type=None, crop_size=(192, 192), downsample: int = None):
        super().__init__()
        self.csv_path = csv_path
        df = pd.read_csv(csv_path)
        self.num_items = len(df)
        self.metrics = {col: torch.as_tensor(df[col]).float() for col in df.columns}
        self.base_path = base_path

        assert len(crop_size) in [2, 3]
        self.crop_type = crop_type
        self.crop_size = crop_size
        self.n_dim = len(self.crop_size)

        self.downsample = downsample

    def __len__(self):
        return self.num_items

    def __getitem__(self, index):
        item = {col: values[index] for col, values in self.metrics.items()}

        img_path = os.path.join(self.base_path, '{}/T1_unbiased_brain_rigid_to_mni.nii.gz'.format(int(item['eid'])))
        img = torch.as_tensor(sitk.GetArrayFromImage(sitk.ReadImage(str(img_path))), dtype=torch.float32)
        img -= img.min()
        img /= img.max()

        if self.crop_type is not None:
            if self.crop_type == 'center':
                img = center_crop(img, self.crop_size)
            elif self.crop_type == 'random':
                raise NotImplementedError('random cropping not implemented.')
            else:
                raise ValueError(f'unknown crop type: {self.crop_type}')

        # add channel dim
        img = img.unsqueeze(0)

        if self.downsample is not None and self.downsample > 1:
            img = img.unsqueeze(0)
            mode = 'bilinear' if self.n_dim == 2 else 'trilinear'
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                img = torch.nn.functional.interpolate(img, scale_factor=1. / self.downsample, align_corners=True, mode=mode)[0]

        item['image'] = img

        return item
