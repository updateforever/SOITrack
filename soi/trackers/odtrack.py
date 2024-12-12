# import the necessary packages
import os

from .basetracker import Tracker
# from pytracking.tracker.base.basetracker import BaseTracker
from lib.test.evaluation.tracker import Tracker as PyTracker

class TrackerODTrack(Tracker):
    def __init__(self, dataset_name='lasot'):
        super(TrackerODTrack, self).__init__(name='odtrack-384', is_deterministic=True)
        pytracker = PyTracker('ostrack', 'vitb_384_mae_ce_32x4_ep300', dataset_name)
        params = pytracker.get_parameters()  # 获取相关先验参数
        self.tracker = pytracker.create_tracker(params)
        self.tracker_param = 'vitb_384_mae_ce_32x4_ep300'

    def init(self, image, box):
        # print(box)
        self.tracker.initialize(image, {'init_bbox': box})

    def update(self, image):
        out = self.tracker.track(image)
        out = out['target_bbox']

        return out