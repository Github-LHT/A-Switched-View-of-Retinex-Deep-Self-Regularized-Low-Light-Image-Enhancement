import os
import torch
from torch.utils import data
from PIL import Image
from torchvision import transforms
import kornia

transform = transforms.Compose([
    transforms.ToTensor()
])


class DataSet(data.Dataset):

    def __init__(self, root):
        imgs = os.listdir(root)
        self.lens = len(root) + 1
        self.imgs = [os.path.join(root, k) for k in imgs]
        self.transforms = transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img_index = img_path[self.lens:-4]
        img = Image.open(img_path)
        img = self.transforms(img)
        img_hsv = kornia.color.rgb_to_hsv(img)
        sample = {"index": img_index, "image": img_hsv}

        return sample

    def __len__(self):
        return len(self.imgs)


def load_images(path, batchsize):
    dataset = DataSet(path)
    data = torch.utils.data.DataLoader(dataset, batchsize, shuffle=True, num_workers=4, drop_last=True)
    print("Loading images is over.")

    return data
