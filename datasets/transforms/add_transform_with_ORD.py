import copy
import inspect
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union, Iterable
import mmengine.fileio as fileio
from mmcv.image.geometric import _scale_size

import cv2
import mmcv
import mmengine
import numpy as np
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
from mmcv.transforms import to_tensor
from mmengine.utils import is_tuple_of
from mmengine.structures import PixelData
from numpy import random
from scipy.ndimage import gaussian_filter
import warnings

from mmseg.datasets.dataset_wrappers import MultiImageMixDataset
from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample

@TRANSFORMS.register_module()
class LoadSeperateTwoObjAnnotation(MMCV_LoadAnnotations):
    """
        Override from class LoadAnnotations()
    """

    def __init__(
        self,
        reduce_zero_label=None,
        backend_args=None,
        imdecode_backend='pillow',
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)
        self.reduce_zero_label = reduce_zero_label
        if self.reduce_zero_label is not None:
            warnings.warn('`reduce_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_zero_label=True` when dataset '
                          'initialized')
        self.imdecode_backend = imdecode_backend

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        img_bytes = fileio.get(results['seg_map_path'], backend_args=self.backend_args)
        img_bytes_hand = fileio.get(results['seg_map_path_hand'], backend_args=self.backend_args)
        img_bytes_left_obj = fileio.get(results['seg_map_path_left_obj'], backend_args=self.backend_args)
        img_bytes_right_obj = fileio.get(results['seg_map_path_right_obj'], backend_args=self.backend_args)
        img_bytes_two_obj = fileio.get(results['seg_map_path_two_obj'], backend_args=self.backend_args)
        img_bytes_cb = fileio.get(results['seg_map_path_cb'], backend_args=self.backend_args)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        gt_semantic_seg_hand = mmcv.imfrombytes(
            img_bytes_hand, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        gt_semantic_seg_left_obj = mmcv.imfrombytes(
            img_bytes_left_obj, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        gt_semantic_seg_right_obj = mmcv.imfrombytes(
            img_bytes_right_obj, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        gt_semantic_seg_two_obj = mmcv.imfrombytes(
            img_bytes_two_obj, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        gt_semantic_seg_cb = mmcv.imfrombytes(
            img_bytes_cb, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        
        # reduce zero_label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
            gt_semantic_seg_hand[gt_semantic_seg_hand == 0] = 255
            gt_semantic_seg_hand = gt_semantic_seg_hand - 1
            gt_semantic_seg_hand[gt_semantic_seg_hand == 254] = 255
            gt_semantic_seg_left_obj[gt_semantic_seg_left_obj == 0] = 255
            gt_semantic_seg_left_obj = gt_semantic_seg_left_obj - 1
            gt_semantic_seg_left_obj[gt_semantic_seg_left_obj == 254] = 255
            gt_semantic_seg_right_obj[gt_semantic_seg_right_obj == 0] = 255
            gt_semantic_seg_right_obj = gt_semantic_seg_right_obj - 1
            gt_semantic_seg_right_obj[gt_semantic_seg_right_obj == 254] = 255
            gt_semantic_seg_two_obj[gt_semantic_seg_two_obj == 0] = 255
            gt_semantic_seg_two_obj = gt_semantic_seg_two_obj - 1
            gt_semantic_seg_two_obj[gt_semantic_seg_two_obj == 254] = 255
            gt_semantic_seg_cb[gt_semantic_seg_cb == 0] = 255
            gt_semantic_seg_cb = gt_semantic_seg_cb - 1
            gt_semantic_seg_cb[gt_semantic_seg_cb==254] = 255

        results['gt_seg_map'] = gt_semantic_seg
        results['gt_seg_map_hand'] = gt_semantic_seg_hand
        results['gt_seg_map_left_obj'] = gt_semantic_seg_left_obj
        results['gt_seg_map_right_obj'] = gt_semantic_seg_right_obj
        results['gt_seg_map_two_obj'] = gt_semantic_seg_two_obj
        results['gt_seg_map_cb'] = gt_semantic_seg_cb

        results['seg_fields'].append('gt_seg_map')
        results['seg_fields'].append('gt_seg_map_hand')
        results['seg_fields'].append('gt_seg_map_left_obj')
        results['seg_fields'].append('gt_seg_map_right_obj')
        results['seg_fields'].append('gt_seg_map_two_obj')
        results['seg_fields'].append('gt_seg_map_cb')

        # with open('/home/suyuejiao/mmsegmentation/test_print.txt','a') as f:
        #     print("*****",results['gt_seg_map'],file=f)
        #     print("*****",results['gt_seg_map_hand'],file=f)
        #     print("------",results['gt_seg_map_left_obj'],file=f)
        #     print("------",results['gt_seg_map_right_obj'],file=f)
        # raise KeyError

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str


@TRANSFORMS.register_module()
class LabelResizeSeperateTwoObj(BaseTransform):
    """Random resize images & segmentation mask.

    Added Keys:
    - gt_seg_map_hand
    - gt_seg_map_obj
    - gt_seg_map_cb

    Original Keys:
    - img
    - gt_bboxes
    - gt_seg_map
    - gt_keypoints
    - scale
    - scale_factor
    - keep_ratio
   """

    def __init__(
        self,
        scale: Union[Tuple[int, int], Sequence[Tuple[int, int]]],
        ratio_range: Tuple[float, float] = None,
        keep_ratio: bool=False,
        resize_type: str = 'Resize',
        **resize_kwargs,
    ) -> None:

        self.scale = scale
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def _random_sample(scales: Sequence[Tuple[int, int]]) -> tuple:
        assert mmengine.is_list_of(scales, tuple) and len(scales) == 2
        scale_0 = [scales[0][0], scales[1][0]]
        scale_1 = [scales[0][1], scales[1][1]]
        edge_0 = np.random.randint(min(scale_0), max(scale_0) + 1)
        edge_1 = np.random.randint(min(scale_1), max(scale_1) + 1)
        scale = (edge_0, edge_1)
        return scale

    @staticmethod
    def _random_sample_ratio(scale: tuple, ratio_range: Tuple[float,
                                                              float]) -> tuple:

        assert isinstance(scale, tuple) and len(scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(scale[0] * ratio), int(scale[1] * ratio)
        return scale

    @cache_randomness
    def _random_scale(self) -> tuple:
        if mmengine.is_tuple_of(self.scale, int):
            assert self.ratio_range is not None and len(self.ratio_range) == 2
            scale = self._random_sample_ratio(
                self.scale,  # type: ignore
                self.ratio_range)
        elif mmengine.is_seq_of(self.scale, tuple):
            scale = self._random_sample(self.scale)  # type: ignore
        else:
            raise NotImplementedError('Do not support sampling function '
                                      f'for "{self.scale}"')

        return scale
    
    def transform(self, results: dict) -> dict:

        results['scale'] = self._random_scale()
        self.resize = ThreeLabelResizeSeperateTwoobj(scale=results['scale'], keep_ratio=self.keep_ratio)
        results = self.resize(results)
        return results


    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'resize_cfg={self.resize_cfg})'
        return repr_str
    
@TRANSFORMS.register_module()
class ThreeLabelResizeSeperateTwoobj(BaseTransform):

    def __init__(self,
                 scale: Optional[Union[int, Tuple[int, int]]] = None,
                 scale_factor: Optional[Union[float, Tuple[float,
                                                           float]]] = None,
                 keep_ratio: bool = False,
                 clip_object_border: bool = True,
                 backend: str = 'cv2',
                 interpolation='bilinear') -> None:
        assert scale is not None or scale_factor is not None, (
            '`scale` and'
            '`scale_factor` can not both be `None`')
        if scale is None:
            self.scale = None
        else:
            if isinstance(scale, int):
                self.scale = (scale, scale)
            else:
                self.scale = scale

        self.backend = backend
        self.interpolation = interpolation
        self.keep_ratio = keep_ratio
        self.clip_object_border = clip_object_border
        if scale_factor is None:
            self.scale_factor = None
        elif isinstance(scale_factor, float):
            self.scale_factor = (scale_factor, scale_factor)
        elif isinstance(scale_factor, tuple):
            assert (len(scale_factor)) == 2
            self.scale_factor = scale_factor
        else:
            raise TypeError(
                f'expect scale_factor is float or Tuple(float), but'
                f'get {type(scale_factor)}')

    def _resize_img(self, results: dict) -> None:
        """Resize images with ``results['scale']``."""

        if results.get('img', None) is not None:
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results['img'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results['img'].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results['img'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
            results['img'] = img
            results['img_shape'] = img.shape[:2]
            results['scale_factor'] = (w_scale, h_scale)
            results['keep_ratio'] = self.keep_ratio

    def _resize_seg(self, results: dict) -> None:
        """Resize semantic segmentation map with ``results['scale']``."""
        if results.get('gt_seg_map', None) is not None:
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results['gt_seg_map'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_seg = mmcv.imresize(
                    results['gt_seg_map'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results['gt_seg_map'] = gt_seg

        if results.get('gt_seg_map_hand', None) is not None:
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results['gt_seg_map_hand'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_seg = mmcv.imresize(
                    results['gt_seg_map_hand'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results['gt_seg_map_hand'] = gt_seg
        
        if results.get('gt_seg_map_left_obj', None) is not None:
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results['gt_seg_map_left_obj'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_seg = mmcv.imresize(
                    results['gt_seg_map_left_obj'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results['gt_seg_map_left_obj'] = gt_seg

        if results.get('gt_seg_map_right_obj', None) is not None:
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results['gt_seg_map_right_obj'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_seg = mmcv.imresize(
                    results['gt_seg_map_right_obj'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results['gt_seg_map_right_obj'] = gt_seg
        
        if results.get('gt_seg_map_two_obj', None) is not None:
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results['gt_seg_map_two_obj'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_seg = mmcv.imresize(
                    results['gt_seg_map_two_obj'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results['gt_seg_map_two_obj'] = gt_seg
        
        if results.get('gt_seg_map_cb', None) is not None:
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results['gt_seg_map_cb'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_seg = mmcv.imresize(
                    results['gt_seg_map_cb'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results['gt_seg_map_cb'] = gt_seg

    def transform(self, results: dict) -> dict:
       
        if self.scale:
            results['scale'] = self.scale
        else:
            img_shape = results['img'].shape[:2]
            results['scale'] = _scale_size(img_shape[::-1],
                                           self.scale_factor)  # type: ignore

        self._resize_img(results)      
        self._resize_seg(results)

        # with open('/home/suyuejiao/mmsegmentation/test_print.txt','a') as f:
        #     print("*****",results,file=f)
        #     print(results['img'].shape, file=f)
        #     print(results['gt_seg_map_left_obj'].shape, file=f)
        #     print(results['gt_seg_map_right_obj'].shape, file=f)
        #     print(results['gt_seg_map_hand'].shape, file=f)
        #     print(results['gt_seg_map_two_obj'].shape, file=f)
        #     print(results['gt_seg_map_cb'].shape, file=f)
        # raise KeyError
       
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, '
        repr_str += f'scale_factor={self.scale_factor}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'clip_object_border={self.clip_object_border}), '
        repr_str += f'backend={self.backend}), '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str
    
@TRANSFORMS.register_module()
class RandomSeperateObjectCrop(BaseTransform):

    def __init__(self,
                 crop_size: Union[int, Tuple[int, int]],
                 cat_max_ratio: float = 1.,
                 ignore_index: int = 255):
        super().__init__()
        assert isinstance(crop_size, int) or (
            isinstance(crop_size, tuple) and len(crop_size) == 2
        ), 'The expected crop_size is an integer, or a tuple containing two '
        'intergers'

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    @cache_randomness
    def crop_bbox(self, results: dict) -> tuple:
        def generate_crop_bbox(img: np.ndarray) -> tuple:
           
            margin_h = max(img.shape[0] - self.crop_size[0], 0)
            margin_w = max(img.shape[1] - self.crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

            return crop_y1, crop_y2, crop_x1, crop_x2

        img = results['img']
        crop_bbox = generate_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results['gt_seg_map'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]

                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = generate_crop_bbox(img)

        return crop_bbox

    def crop(self, img: np.ndarray, crop_bbox: tuple) -> np.ndarray:

        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def transform(self, results: dict) -> dict:
      
        img = results['img']

        crop_bbox = self.crop_bbox(results)

        # crop the image
        img = self.crop(img, crop_bbox)

        # crop semantic seg
        for key in results.get('seg_fields', []):
            if key != '':
                results[key] = self.crop(results[key], crop_bbox)

        results['img'] = img
        results['img_shape'] = img.shape[:2]

        # with open('/home/suyuejiao/mmsegmentation/test_print.txt','a') as f:
        #     print("*****",results,file=f)
        #     print(results['img'].shape, file=f)
        #     print(results['gt_seg_map_left_obj'].shape, file=f)
        #     print(results['gt_seg_map_right_obj'].shape, file=f)
        #     print(results['gt_seg_map_hand'].shape, file=f)
        #     print(results['gt_seg_map_two_obj'].shape, file=f)
        #     print(results['gt_seg_map_cb'].shape, file=f)
        # raise KeyError
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'
 

@TRANSFORMS.register_module()
class PackSeperateTwoObjLabelSegInputs(BaseTransform):
   
    def __init__(self,
                 meta_keys=('img_path', 'seg_map_path', 'seg_map_path_hand', 'seg_map_path_left_obj', 
                            'seg_map_path_right_obj', 'seg_map_path_two_obj',
                            'seg_map_path_cb','ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:

        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results['inputs'] = img
            # here visualize the images to see if the blue???
        data_sample = SegDataSample()
        if 'gt_seg_map' in results:
            if len(results['gt_seg_map'].shape) == 2:
                data = to_tensor(results['gt_seg_map'][None,
                                                       ...].astype(np.int64))
            if len(results['gt_seg_map_hand'].shape) == 2:
                data_hand = to_tensor(results['gt_seg_map_hand'][None,
                                                       ...].astype(np.int64))
            if len(results['gt_seg_map_left_obj'].shape) == 2:
                data_left_obj = to_tensor(results['gt_seg_map_left_obj'][None,
                                                       ...].astype(np.int64))
            if len(results['gt_seg_map_right_obj'].shape) == 2:
                data_right_obj = to_tensor(results['gt_seg_map_right_obj'][None,
                                                       ...].astype(np.int64))
            if len(results['gt_seg_map_two_obj'].shape) == 2:
                data_two_obj = to_tensor(results['gt_seg_map_two_obj'][None,
                                                       ...].astype(np.int64))
            if len(results['gt_seg_map_cb'].shape) == 2:
                data_cb = to_tensor(results['gt_seg_map_cb'][None,
                                                       ...].astype(np.int64))
            else:
                warnings.warn('Please pay attention your ground truth '
                              'segmentation map, usually the segmentation '
                              'map is 2D, but got '
                              f'{results["gt_seg_map"].shape}')
                data = to_tensor(results['gt_seg_map'].astype(np.int64))
                data_hand = to_tensor(results['gt_seg_map_hand'].astype(np.int64))
                data_left_obj = to_tensor(results['gt_seg_map_left_obj'].astype(np.int64))
                data_right_obj = to_tensor(results['gt_seg_map_right_obj'].astype(np.int64))
                data_two_obj = to_tensor(results['gt_seg_map_two_obj'].astype(np.int64))
                data_cb = to_tensor(results['gt_seg_map_cb'].astype(np.int64))
            gt_sem_seg_data = dict(data=data)
            gt_sem_seg_data_hand = dict(data=data_hand)
            gt_sem_seg_data_left_obj = dict(data=data_left_obj)
            gt_sem_seg_data_right_obj = dict(data=data_right_obj)
            gt_sem_seg_data_two_obj = dict(data=data_two_obj)
            gt_sem_seg_data_cb = dict(data=data_cb)
           
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)
            data_sample.gt_sem_seg_hand = PixelData(**gt_sem_seg_data_hand)
            data_sample.gt_sem_seg_left_obj = PixelData(**gt_sem_seg_data_left_obj)
            data_sample.gt_sem_seg_right_obj = PixelData(**gt_sem_seg_data_right_obj)
            data_sample.gt_sem_seg_two_obj = PixelData(**gt_sem_seg_data_two_obj)
            data_sample.gt_sem_seg_cb = PixelData(**gt_sem_seg_data_cb)

        if 'gt_edge_map' in results:
            gt_edge_data = dict(
                data=to_tensor(results['gt_edge_map'][None,
                                                      ...].astype(np.int64)))
            data_sample.set_data(dict(gt_edge_map=PixelData(**gt_edge_data)))

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample
        # with open('/home/suyuejiao/mmsegmentation/test_print.txt','a') as f:
        #     print("--------packed_results--------", packed_results, file=f)
        # ----------------this is blue-------------------------------------------
        # print("--------packed_results--------", packed_results['inputs'].shape)
        # img = np.array(packed_results['inputs'].permute(1,2,0))
        # import cv2
        # img_path = '/home/suyuejiao/mmsegmentation/example.jpg'
        # from PIL import Image
        # img = Image.fromarray(img)
        # img.save(img_path)
        # raise KeyError
        # print("--------packed_results-1-------", PixelData(**gt_sem_seg_data_hand))
        # print("--------packed_results-2-------", data_sample.gt_sem_seg_hand.data.shape)
        # print("--------packed_results-3-------", data_sample.gt_sem_seg_left_obj.data.shape)
        # print("--------packed_results-3-------", data_sample.gt_sem_seg_right_obj.data.shape)
        # print("--------packed_results-4-------", data_sample.gt_sem_seg_cb.data.shape)
        # print("--------packed_results--5------", data_sample.gt_sem_seg_two_obj.data.shape)

        # print("--------packed_results-1-------", packed_results['data_samples'].gt_sem_seg.data.shape)
        # print("--------packed_results-2-------", packed_results['data_samples'].gt_sem_seg_hand.data.shape)
        # print("--------packed_results-3-------", packed_results['data_samples'].gt_sem_seg_left_obj.data.shape)
        # print("--------packed_results-3-------", packed_results['data_samples'].gt_sem_seg_right_obj.data.shape)
        # print("--------packed_results-4-------", packed_results['data_samples'].gt_sem_seg_cb.data.shape)
        # print("--------packed_results--5------", packed_results['data_samples'].gt_sem_seg_two_obj.data.shape)
        # raise KeyError
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


