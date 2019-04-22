import os
import torch
import yaml
import numpy as np
import scipy.misc as misc
from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.utils import convert_state_dict


def test(cfg):
    device = torch.device(cfg['device'])
    data_loader = get_loader('vistas')
    loader = data_loader(root=cfg['testing']['config_path'], is_transform=True, test_mode=True)
    n_classes = loader.n_classes
    # Setup Model
    model_dict = {"arch": 'icnetBN'}
    model = get_model(model_dict, n_classes)
    state = convert_state_dict(torch.load(cfg['testing']['model_path'])["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    for img_name in os.listdir(cfg['testing']['img_fold']):
        img_path = os.path.join(cfg['testing']['img_fold'], img_name)
        img = misc.imread(img_path)
        orig_size = img.shape[:-1]

        # uint8 with RGB mode, resize width and height which are odd numbers
        # img = misc.imresize(img, (orig_size[0] // 2 * 2 + 1, orig_size[1] // 2 * 2 + 1))
        img = misc.imresize(img, (cfg['testing']['img_rows'], cfg['testing']['img_cols']))
        img = img.astype(np.float64)
        img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img).float()

        img = img.to(device)
        outputs = model(img)

        outputs = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
        outputs = outputs.astype(np.float32)
        # float32 with F mode, resize back to orig_size
        outputs = misc.imresize(outputs, orig_size, "nearest", mode="F")

        decoded = loader.decode_segmap(outputs)
        output_path = os.path.join(cfg['testing']['output_fold'], 'mask_%s.png' % img_name.split('.')[0])
        misc.imsave(output_path, decoded)


if __name__ == "__main__":
    with open('configs/icnet_mapillary.yml') as fp:
        cfg = yaml.load(fp)
    test(cfg)
