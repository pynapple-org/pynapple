# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-10-31 18:34:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-10-31 19:24:48

import fastplotlib as fpl
import pynapple as nap
import numpy as np
import sys, os
sys.path.append(os.path.expanduser("~/fastplotlib-sfn2023"))
from _video import LazyVideo
from pathlib import Path
from ipywidgets import HBox

behavior_path = Path('/mnt/home/gviejo/fastplotlib-sfn2023/sample_data/M238Slc17a7_Chr2/20170824')

paths_side = sorted(behavior_path.glob("*side_v*.avi"))
paths_front = sorted(behavior_path.glob("*front_v*.avi"))


class Concat:
    def __init__(self, files):
        self.files = files
        self.videos = [LazyVideo(p) for p in self.files]
        self._nframes_per_video = [v.shape[0] for v in self.videos]
        self._cumsum = np.cumsum(self._nframes_per_video)
        self.nframes = sum(self._nframes_per_video)
        self.shape = (self.nframes, self.videos[0].shape[1], self.videos[0].shape[2])
        self.ndim = 3

        self.dtype = self.videos[0].dtype

    def __len__(self) -> int:
        return self.nframes

    def _get_vid_ix_sub_ix(self, key):
        vid_ix = np.searchsorted(self._cumsum, key)
        if vid_ix != 0:
            sub_ix = key - self._cumsum[vid_ix - 1]
        else:
            sub_ix = key

        return vid_ix, sub_ix

    def __getitem__(self, key)-> np.ndarray:        
        if isinstance(key, slice):
            start, stop = key.start, key.stop
            vid_ix, sub_ix0 = self._get_vid_ix_sub_ix(start)
            vid_ix, sub_ix1 = self._get_vid_ix_sub_ix(stop)
            return self.videos[vid_ix][sub_ix0:sub_ix1]
        elif isinstance(key, int):
            vid_ix, sub_ix0 = self._get_vid_ix_sub_ix(key)
            return self.videos[vid_ix][sub_ix0]

        

concat = Concat(paths_side)

# print(concat.videos)

t = np.linspace(0, concat.nframes / 500, concat.nframes)

tsd_video = nap.TsdTensor(t, concat)

v = LazyVideo(concat.files[0])

tsd = nap.TsdTensor(t=np.arange(0, len(v)), d=v)