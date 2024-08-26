"""
    Perform the dataset file for egohos dataset, override from basesegdataset.py.
"""
import copy
import os.path as osp
from typing import Callable, Dict, List, Optional, Sequence, Union

import mmengine
import mmengine.fileio as fileio
import numpy as np
from mmengine.dataset import BaseDataset, Compose

from mmseg.registry import DATASETS


@DATASETS.register_module()
class SeperateObjectEgohos(BaseDataset):
    """
    Seperate the left, right and two obj.
    Becuse the .npy file can not be used in computing the CE loss, so I load the GT of left obj and right obj seperately.
    """
    METAINFO = dict(
        class_hand = ( 'BG','Left_Hand', 'Right_Hand'),
        class_left_obj = (  'BG', 'Left_Object1'),
        class_right_obj = ( 'BG','Right_Object1'),
        class_two_obj = ( 'BG','Two_Object1'),
        class_cb = ( 'BG', 'ContactBoundary'),
        class_obj = ('Left_Object1', 'Right_Object1', 'Two_Object1'),
        classes=('Left_Hand', 'Right_Hand', 'Left_Object1', 'Right_Object1', 'Two_Object1'),

        palette_hand=[[0,0,0], [255, 0, 0], [0, 0, 255]],
        palette_left_obj=[[0,0,0], [255, 0, 255]],
        palette_right_obj=[[0,0,0], [0, 255, 255]],
        palette_two_obj = [[0,0,0], [0, 255, 0]],
        palette_obj = [[255, 0, 255], [0, 255, 255], [0, 255, 0]],
        palette_cb = [ [204, 255, 204]],
        palette=[[255, 0, 0], [0, 0, 255], [255, 0, 255], [0, 255, 255], [0, 255, 0]])

    def __init__(self,
                 ann_file_hand: str = '',
                 ann_file_left_obj: str='',
                 ann_file_right_obj: str='',
                 ann_file_two_obj: str='',
                 ann_filr_cb: str='',
                 ann_file: str='',
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img_path='', seg_map_path='',seg_map_path_hand='', seg_map_path_left_obj='', seg_map_path_right_obj='',seg_map_path_two_obj='',seg_map_path_cb=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 ignore_index: int = 255,
                 reduce_zero_label: bool = False,
                 backend_args: Optional[dict] = None) -> None:

        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.backend_args = backend_args.copy() if backend_args else None

        self.data_root = data_root
        self.data_prefix = copy.copy(data_prefix)
        self.ann_file = ann_file
        self.ann_file_hand = ann_file_hand
        self.ann_file_left_obj = ann_file_left_obj
        self.ann_file_right_obj = ann_file_right_obj
        self.ann_file_two_obj = ann_file_two_obj
        self.ann_file_cb = ann_filr_cb
        self.filter_cfg = copy.deepcopy(filter_cfg)
        self._indices = indices
        self.serialize_data = serialize_data
        self.test_mode = test_mode
        self.max_refetch = max_refetch
        self.data_list: List[dict] = []
        self.data_bytes: np.ndarray

        # Set meta information.
        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))

        # Get label map for custom classes
        new_classes = self._metainfo.get('classes', None)
        new_classes_hand = self._metainfo.get('classes_hand', None)
        new_classes_left_obj = self._metainfo.get('classes_left_obj')
        new_classes_right_obj = self.metainfo.get('classes_right_obj', None)
        new_classes_two_obj = self.metainfo.get('classes_two_obj', None)
        new_classes_cb = self._metainfo.get('classes_cb', None)
        # print(new_classes, new_classes_hand, new_classes_obj)
        self.label_map = self.get_label_map(new_classes)
        self.label_map_hand = self.get_label_map(new_classes_hand)
        self.label_map_left_obj = self.get_label_map(new_classes_left_obj)
        self.label_map_right_obj = self.get_label_map(new_classes_right_obj)
        self.label_map_two_obj = self.get_label_map(new_classes_two_obj)
        self.label_map_cb = self.get_label_map(new_classes_cb)
        # print(self.label_map)
        self._metainfo.update(
            dict(
                label_map=self.label_map,
                label_map_hand=self.label_map_hand,
                label_map_left_obj=self.label_map_left_obj,
                label_map_right_obj=self.label_map_right_obj,
                label_map_two_obj=self.label_map_two_obj,
                label_map_cb=self.label_map_cb,
                reduce_zero_label=self.reduce_zero_label))

        # Update palette based on label map or generate palette
        # if it is not defined
        updated_palette = self._update_palette()
        self._metainfo.update(dict(palette=updated_palette))

        # Join paths.
        if self.data_root is not None:
            self._join_prefix()

        # Build pipeline.
        self.pipeline = Compose(pipeline)
        # Full initialize the dataset.
        if not lazy_init:
            self.full_init()

        if test_mode:
            assert self._metainfo.get('classes') is not None, \
                'dataset metainfo `classes` should be specified when testing'

    @classmethod
    def get_label_map(cls,
                      new_classes: Optional[Sequence] = None
                      ) -> Union[Dict, None]:

        old_classes = cls.METAINFO.get('classes', None)
        if (new_classes is not None and old_classes is not None
                and list(new_classes) != list(old_classes)):

            label_map = {}
            if not set(new_classes).issubset(cls.METAINFO['classes']):
                raise ValueError(
                    f'new classes {new_classes} is not a '
                    f'subset of classes {old_classes} in METAINFO.')
            for i, c in enumerate(old_classes):
                if c not in new_classes:
                    label_map[i] = 255
                else:
                    label_map[i] = new_classes.index(c)
            return label_map
        else:
            return None

    def _update_palette(self) -> list:
        
        palette = self._metainfo.get('palette', [])
        classes = self._metainfo.get('classes', [])
        # palette does match classes yuejiao shi wo de laopo
        if len(palette) == len(classes):
            return palette

        if len(palette) == 0:
            # Get random state before set seed, and restore
            # random state later.
            # It will prevent loss of randomness, as the palette
            # may be different in each iteration if not specified.
            # See: https://github.com/open-mmlab/mmdetection/issues/5844
            state = np.random.get_state()
            np.random.seed(42)
            # random palette
            new_palette = np.random.randint(
                0, 255, size=(len(classes), 3)).tolist()
            np.random.set_state(state)
        elif len(palette) >= len(classes) and self.label_map is not None:
            new_palette = []
            # return subset of palette
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != 255:
                    new_palette.append(palette[old_id])
            new_palette = type(palette)(new_palette)
        else:
            raise ValueError('palette does not match classes '
                             f'as metainfo is {self._metainfo}.')
        return new_palette

    def load_data_list(self) -> List[dict]:

        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        ann_dir_hand = self.data_prefix.get('seg_map_path_hand', None)
        ann_dir_left_obj = self.data_prefix.get('seg_map_path_left_obj', None)
        ann_dir_right_obj = self.data_prefix.get('seg_map_path_right_obj', None)
        ann_dir_two_obj = self.data_prefix.get('seg_map_path_two_obj',None)
        ann_dir_cb = self.data_prefix.get('seg_map_path_cb', None)

        # print(ann_dir,ann_dir_cb,ann_dir_hand,ann_dir_left_right_obj,ann_dir_two_obj)
        # raise KeyError

        if osp.isfile(self.ann_file_hand):
            lines = mmengine.list_from_file(
                self.ann_file_hand, backend_args=self.backend_args)
            for line in lines:
                img_name = line.strip()
                data_info = dict(
                    img_path=osp.join(img_dir, img_name + self.img_suffix))
                # print(data_info)
                if ann_dir_hand is not None:
                    seg_map = img_name + self.seg_map_suffix
                    data_info['seg_map_path_hand'] = osp.join(ann_dir_hand, seg_map)
                    data_info['seg_map_path_left_obj'] = osp.join(ann_dir_left_obj, seg_map)
                    data_info['seg_map_path_right_obj'] = osp.join(ann_dir_right_obj, seg_map)
                    data_info['seg_map_path_two_obj'] = osp.join(ann_dir_two_obj, seg_map)
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                    data_info['seg_map_path_cb'] = osp.join(ann_dir_cb, seg_map)
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else:
            for img in fileio.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=True,
                    backend_args=self.backend_args):

                data_info = dict(img_path=osp.join(img_dir, img))

                if ann_dir_hand is not None and ann_dir_cb is not None and ann_dir_right_obj is not None \
                    and ann_dir_left_obj is not None and ann_dir_two_obj is not None:
                    seg_map = img.replace(self.img_suffix, self.seg_map_suffix) # turn the .jpg to .png
                    data_info['seg_map_path_hand'] = osp.join(ann_dir_hand, seg_map)
                    data_info['seg_map_path_left_obj'] = osp.join(ann_dir_left_obj, seg_map)
                    data_info['seg_map_path_right_obj'] = osp.join(ann_dir_right_obj, seg_map)
                    data_info['seg_map_path_two_obj'] = osp.join(ann_dir_two_obj, seg_map)
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                    data_info['seg_map_path_cb'] = osp.join(ann_dir_cb, seg_map)

                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = ['']
                data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x['img_path'])

        # print(data_list)
        # raise KeyError
        return data_list
    

