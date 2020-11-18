import os
from base_dataset import BaseDataset, get_transform
from PIL import Image
import random


class Dataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.A_paths = opt.path_A # path content
        self.B_paths = opt.path_B # path style
        self.transform_A = get_transform(opt) # Edit content images
        self.transform_B = get_transform(opt) # Edit style images

    def __getitem__(self, index):
        path_A = self.A_paths[index]
        im_A = Image.open(path_A).convert('RGB')

        path_B = self.B_paths
        im_B = Image.open(path_B).convert('RGB')

        data_A = self.transform_A(im_A)
        data_B = self.transform_B(im_B)

        return {'A': data_A, 'B': data_B, 'A_paths': path_A, 'B_paths': path_B}

    def __len__(self):
        return len(self.A_paths)

def create_dataset(opt):
    data = Dataset(opt)
    dataset = torch.utils.data.DataLoader(data)
    return dataset
