import torch
from torch.utils.data import Dataset
from Register import Registers
import os
import numpy as np
import h5py
import torch.nn.functional as F


@Registers.datasets.register_with_name('custom_aligned')
class CustomAlignedDataset(Dataset):
    def __init__(self, dataset_config, stage='test'):
        self.h5_path = os.path.join(dataset_config.dataset_path, f'{dataset_config.dataset_name}.h5')
        print(f"🔍 [Dataset Info] 尝试加载 H5 文件: {os.path.abspath(self.h5_path)}")
        self.target_size = (dataset_config.image_size, dataset_config.image_size)
        self.transform = None

        with h5py.File(self.h5_path, 'r') as f:
            self.CT_dataset = [np.array(f['CT_dataset'][i], dtype=np.float32) for i in range(len(f['CT_dataset']))]
            self.MR_dataset = [np.array(f['MR_dataset'][i], dtype=np.float32) for i in range(len(f['MR_dataset']))]
            self.CT_shape = np.array(f['CT_shape'], dtype=int)
            self.MR_shape = np.array(f['MR_shape'], dtype=int)
            self.pid = np.array(f['Patient_index']).astype(str)
            self.sid = np.array(f['Slice_index']).astype(str)
            raw_max_slices = np.array(f['Max_slice_index'], dtype=int)
            self.patient_max_slices = dict(zip(self.pid, raw_max_slices))
        self.num_samples = len(self.pid)
        self.patient_max_slices = {}
        for p, s in zip(self.pid, self.sid):
            try:
                s_num = int(s)
            except ValueError:
                s_num = 0

            if p not in self.patient_max_slices:
                self.patient_max_slices[p] = s_num
            else:
                if s_num > self.patient_max_slices[p]:
                    self.patient_max_slices[p] = s_num
        print(f"✅ Max slices calculated. Total patients: {len(self.patient_max_slices)}")

    def _restore(self, data_list, shape_ds, idx):
        raw = np.array(data_list[idx])
        h, w = map(int, shape_ds[idx])
        return raw.reshape((h, w))

    def resize_to_target(self, img_tensor):
        th, tw = self.target_size
        img_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]
        img_resized = F.interpolate(img_tensor, size=(th, tw), mode='bilinear', align_corners=False)
        return img_resized.squeeze(0)  # [C, th, tw]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        ct = self._restore(self.CT_dataset, self.CT_shape, idx)
        mr = self._restore(self.MR_dataset, self.MR_shape, idx)

        ct_tensor = torch.tensor(ct, dtype=torch.float32).unsqueeze(0) * 2 - 1
        mr_tensor = torch.tensor(mr, dtype=torch.float32).unsqueeze(0) * 2 - 1

        p_id = self.pid[idx]
        try:
            current_s = int(self.sid[idx])
            max_s = self.patient_max_slices.get(p_id, current_s)
            pos_val = (current_s / max_s) * 2 - 1 if max_s > 0 else 0.0
        except:
            pos_val = 0.0
        pos_map = torch.full((1, self.target_size[0], self.target_size[1]), pos_val, dtype=torch.float32)

        ct_final = self.resize_to_target(ct_tensor)
        mr_final = self.resize_to_target(mr_tensor)
        cond_final = torch.cat([ct_final, pos_map], dim=0)

        return {
            'CT': cond_final,
            'MR': mr_final,
            'pid': self.pid[idx],
            'sid': self.sid[idx],
            'subject': f"{self.pid[idx]}_slice_{self.sid[idx]}"
        }
