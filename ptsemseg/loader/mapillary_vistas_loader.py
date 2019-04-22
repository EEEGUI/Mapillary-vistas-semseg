import os
import json
import torch
import numpy as np

from torch.utils import data
from PIL import Image

from ptsemseg.utils import recursive_glob
from ptsemseg.augmentations import Compose, RandomHorizontallyFlip, RandomRotate

class mapillaryVistasLoader(data.Dataset):
    def __init__(
        self,
        root,
        split="training",
        img_size=(1025, 2049),
        is_transform=True,
        augmentations=None,
        test_mode=False,
    ):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 9

        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([80.5423, 91.3162, 81.4312])
        self.files = {}

        if not test_mode:
            self.images_base = os.path.join(self.root, self.split, "images")
            self.annotations_base = os.path.join(self.root, self.split, "labels")
            self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".jpg")
            if not self.files[split]:
                raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

            print("Found %d %s images" % (len(self.files[split]), split))
        self.class_names, self.class_ids, self.class_colors, self.class_major_ids = self.parse_config()

        self.ignore_id = 250



    def parse_config(self):
        with open(os.path.join(self.root, "config.json")) as config_file:
            config = json.load(config_file)

        labels = config["labels"]

        class_names = []
        class_ids = []
        class_colors = []
        class_major_ids = []

        for label_id, label in enumerate(labels):
            class_names.append(label["readable"])
            class_ids.append(label_id)
            class_colors.append(label["color"])
            class_major_ids.append(label['majorclass'])
        print("There are {} labels in the config file".format(len(set(class_major_ids))))
        return class_names, class_ids, class_colors, class_major_ids

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base, os.path.basename(img_path).replace(".jpg", ".png")
        )

        img = Image.open(img_path)
        lbl = Image.open(lbl_path)
        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)
        if self.is_transform:
            img, lbl = self.transform(img, lbl)
        return img, lbl

    def transform(self, img, lbl):
        if self.img_size == ("same", "same"):
            pass
        else:
            img = img.resize(
                (self.img_size[1], self.img_size[0]), resample=Image.LANCZOS
            )  # uint8 with RGB mode
            lbl = lbl.resize((self.img_size[1], self.img_size[0]))
        img = np.array(img).astype(np.float64) / 255.0
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()  # From HWC to CHW
        #
        # lbl = torch.from_numpy(np.array(lbl)).long()
        # lbl[lbl == 65] = self.ignore_id
        #
        lbl = torch.from_numpy(np.array(lbl)).long()
        lbl[lbl == self.ignore_id] = 65
        lbl = self.encode_segmap(lbl)
        lbl[lbl == 0] = self.ignore_id
        return img, lbl

    def decode_segmap(self, temp):
        class_major_colors = [[0, 0, 0],
                              [70, 70, 70],
                              [180, 165, 180],
                              [128, 64, 64],
                              [220, 20, 60],
                              [255, 255, 255],
                              [70, 130, 180],
                              [250, 170, 30],
                              [0, 0, 142]]
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, len(class_major_colors)):
            r[temp == l] = class_major_colors[l][0]
            g[temp == l] = class_major_colors[l][1]
            b[temp == l] = class_major_colors[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        # rgb[:, :, 0] = r / 255.0
        # rgb[:, :, 1] = g / 255.0
        # rgb[:, :, 2] = b / 255.0
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for id in self.class_ids:
            mask[mask == id] = self.class_major_ids[id]+100

        mask = mask - 100
        return mask


if __name__ == "__main__":
    augment = Compose([RandomHorizontallyFlip(0.5), RandomRotate(6)])

    local_path = "/home/lin/Documents/dataset/mapillary"
    dst = mapillaryVistasLoader(
        local_path, split='validation', img_size=(512, 1024), is_transform=True, augmentations=None
    )
    bs = 1
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=4, shuffle=True)
    for i, data_samples in enumerate(trainloader):
        x = dst.decode_segmap(data_samples[1][0].numpy())
        x = Image.fromarray(np.uint8(x))
        x.show()

