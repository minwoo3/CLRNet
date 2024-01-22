import os
import os.path as osp
import numpy as np
from .base_dataset import BaseDataset
from .registry import DATASETS
# import clrnet.utils.culane_metric as culane_metric
import clrnet.utils.nia_metric as nia_metric
import cv2
from tqdm import tqdm
import logging
import pickle as pkl

LIST_FILE = {
    'train': 'train.txt',
    'val': 'val.txt',
    'test': 'test.txt',
}

@DATASETS.register_module
class NIA(BaseDataset):
    def __init__(self, data_root, split, processes=None, cfg=None):
        super().__init__(data_root, split, processes=processes, cfg=cfg)
        self.list_path = osp.join(data_root, LIST_FILE[split])
        self.split = split
        self.load_annotations()

    def load_annotations(self):
        self.logger.info('Loading NIA annotations...')
        self.data_infos = []

        with open(self.list_path) as list_file:
            for line in list_file:
                infos = self.load_annotation(line)
                self.data_infos.append(infos)

    def load_annotation(self, line):
        infos = {}

        line = line[: - 1]
        img_path = os.path.join(self.data_root, line.replace(".txt", ".jpg"))
        infos['img_name'] = img_path.replace(".jpg", "")
        infos['img_path'] = img_path
        mask_dir = self.data_root + "_seg"

        mask_path = os.path.join(mask_dir, line.replace(".txt", ".jpg"))
        infos['mask_path'] = mask_path

        anno_path = os.path.join(self.data_root, line)
        
        with open(anno_path, 'r') as anno_file:
            data = [
                list(map(float, line.split()))
                for line in anno_file.readlines()
            ]


        lanes = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)
                  if lane[i] >= 0 and lane[i + 1] >= 0] for lane in data]
        
        lanes = [list(set(lane)) for lane in lanes]  # remove duplicated points
        lanes = [lane for lane in lanes
                 if len(lane) > 2]  # remove lanes with less than 2 points
        
        lanes = [sorted(lane, key=lambda x: x[1])
                 for lane in lanes]  # sort by y

        infos['lanes'] = lanes
        return infos

    def get_prediction_string(self, pred):
        ys = np.arange(1200, 1920, 8) / self.cfg.ori_img_h
        out = []
        for lane in pred:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * self.cfg.ori_img_w
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * self.cfg.ori_img_h
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            lane_str = ' '.join([
                '{:.5f} {:.5f}'.format(x, y) for x, y in zip(lane_xs, lane_ys)
            ])
            if lane_str != '':
                out.append(lane_str)

        return '\n'.join(out)

    def evaluate(self, predictions, output_basedir):
        loss_lines = [[], [], [], []]
        print('Generating prediction output...')
        for idx, pred in enumerate(predictions):
            output_filename = os.path.basename(
                self.data_infos[idx]['img_name']) + '.txt'
            
            os.makedirs(output_basedir, exist_ok=True)
            output = self.get_prediction_string(pred)

            with open(os.path.join(output_basedir, output_filename), 'w') as out_file:
                out_file.write(output)

        # result = nia_metric.eval_predictions(output_basedir,
        #                                         self.data_root,
        #                                         osp.join(self.data_root, LIST_FILE["val"]),
        #                                         iou_thresholds=[0.5],
        #                                         official=True)

        # print("output basedir", output_basedir)                               / work_dirs/clr/r101_nia/20240119_145713_lr_3e-04_b_12
        # print("data_root", self.data_root)                                    / ./data/NIA
        # print("testset path", osp.join(self.data_root, LIST_FILE["test"]))    / ./data/NIA/test.txt

        result = nia_metric.eval_predictions(output_basedir,
                                                self.data_root,
                                                osp.join(self.data_root, LIST_FILE["val"]),
                                                iou_thresholds=np.linspace(0.5, 0.95, 10),
                                                official=True)

        return result[0.5]['F1']
    
