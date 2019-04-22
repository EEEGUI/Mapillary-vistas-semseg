import cv2
import yaml
import torch
from scipy import misc
import numpy as np
from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.utils import convert_state_dict


class VideoSegmentation:
    def __init__(self, cfg, video_dir):
        filename = video_dir.split('/')[-1].split('.')[0]
        self.cfg = cfg
        self.cap = cv2.VideoCapture(video_dir)
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.out = cv2.VideoWriter('outputs/seg_%s.avi' % filename, fourcc, 15.0, (int(self.w), int(self.h)))
        self.model = self._load_model(cfg)

    def _load_model(self, cfg):
        self.device = torch.device(cfg['device'])
        data_loader = get_loader('vistas')
        self.loader = data_loader(root=cfg['testing']['config_path'], is_transform=True, test_mode=True)
        n_classes = self.loader.n_classes
        # Setup Model
        model_dict = {"arch": 'icnetBN'}
        model = get_model(model_dict, n_classes)
        state = convert_state_dict(torch.load(cfg['testing']['model_path'])["model_state"])
        model.load_state_dict(state)
        model.eval()
        model.to(self.device)
        return model

    def predict(self, imgs):
        orig_size = imgs[0].shape[:-1]
        batchimgs = []
        for img in imgs:
            # uint8 with RGB mode, resize width and height which are odd numbers
            # img = misc.imresize(img, (orig_size[0] // 2 * 2 + 1, orig_size[1] // 2 * 2 + 1))
            img = misc.imresize(img, (self.cfg['testing']['img_rows'], self.cfg['testing']['img_cols']))
            img = img.astype(np.float64)
            img = img.astype(float) / 255.0
            # NHWC -> NCHW
            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, 0)
            # img = torch.from_numpy(img).float()
            batchimgs.append(torch.from_numpy(img))
        batchimgs = torch.cat(batchimgs).float()
        batchimgs = batchimgs.to(self.device)
        batchimgs = self.model(batchimgs)

        imgs = []
        for img in batchimgs:
            c = img.data.max(0)
            img = img.data.max(0)[1].cpu().numpy()
            img = img.astype(np.float32)
            # float32 with F mode, resize back to orig_size
            img = misc.imresize(img, orig_size, "nearest", mode="F")

            imgs.append(self.loader.decode_segmap(img))
        return imgs

    def process(self):
        i = 1
        imgs = []
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret is True:
                print(i)
                imgs.append(frame[:, :, ::-1])
                if i % self.cfg['testing']['bs'] == 0:
                    imgs = torch.from_numpy(np.array(imgs))
                    imgs = self.predict(imgs)
                    for img in imgs:
                        img = np.uint8(np.array(img))
                        self.out.write(img[:, :, ::-1])
                    imgs = []
                i += 1
            else:
                break
            # Release everything if job is finished
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()


def test_video_generate():
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out1 = cv2.VideoWriter('videos/video2.avi', fourcc, 15.0, (1280, 720))
    img = cv2.imread("videos/VDO_0008.avi")

    out1.write(np.uint8(img))
    out1.release()


if __name__ == '__main__':


    # test_video_generate()
    #
    with open('configs/icnet_mapillary.yml') as fp:
        cfg = yaml.load(fp)
    vs = VideoSegmentation(cfg, "videos/video1.avi")
    vs.process()