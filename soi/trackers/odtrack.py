# import the necessary packages
import os

from .basetracker import Tracker
from lib.test.evaluation.tracker import Tracker as ORGTracker

class TrackerODTrack(Tracker):
    def __init__(self, dataset_name='lasot'):
        super(TrackerODTrack, self).__init__(name='odtrack', is_deterministic=True)
        pytracker = ORGTracker('odtrack', 'baseline', dataset_name)
        params = pytracker.get_parameters()  # 获取相关先验参数
        params.debug = 0
        self.tracker = pytracker.create_tracker(params)
        self.tracker_param = 'baseline'

    def init(self, image, box):
        # print(box)
        self.tracker.initialize(image, {'init_bbox': box})

    def update(self, image):
        out = self.tracker.track(image)
        out = out['target_bbox']

        return out