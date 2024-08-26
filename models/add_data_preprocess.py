# -*- coding: utf-8 -*-
"""
    Override from data_preprossor.py
"""
from numbers import Number
from typing import Any, Dict, List, Optional, Sequence, Union
from ..utils.typing_utils import SampleList

import torch
from mmengine.model import BaseDataPreprocessor
import numpy as np
import torch.nn.functional as F

from mmseg.registry import MODELS

@MODELS.register_module()
class SeperateTwoObjDataPreProcessor(BaseDataPreprocessor):
    def __init__(
        self,
        mean: Sequence[Number] = None,
        std: Sequence[Number] = None,
        size: Optional[tuple] = None,
        size_divisor: Optional[int] = None,
        pad_val: Number = 0,
        seg_pad_val: Number = 255,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        batch_augments: Optional[List[dict]] = None,
        test_cfg: dict = None,
    ):
        super().__init__()
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        self.channel_conversion = rgb_to_bgr or bgr_to_rgb

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                                    'preprocessing, please specify both ' \
                                    '`mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            self.register_buffer('mean',
                                 torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

        # TODO: support batch augmentations.
        self.batch_augments = batch_augments

        # Support different padding methods in testing
        self.test_cfg = test_cfg

    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)

        if self.channel_conversion and inputs[0].size(0) == 3:
            inputs = [_input[[2, 1, 0], ...] for _input in inputs]

        inputs = [_input.float() for _input in inputs]
        if self._enable_normalize:
            inputs = [(_input - self.mean) / self.std for _input in inputs]

        # print("----------######", ((data_samples[0].gt_sem_seg_hand.data)).shape) # tensor[1,360,379]
        # print("----------######", ((data_samples[0].gt_sem_seg.data)).shape) # tensor[1,360,379]
        # print("----------######", (inputs[0].shape)) # tensor[3,360,379]

        if training:
            assert data_samples is not None, ('During training, ',
                                              '`data_samples` must be define.')
            inputs, data_samples = stack_batch(
                inputs=inputs,
                data_samples=data_samples,
                size=self.size,
                size_divisor=self.size_divisor,
                pad_val=self.pad_val,
                seg_pad_val=self.seg_pad_val)

            if self.batch_augments is not None:
                inputs, data_samples = self.batch_augments(
                    inputs, data_samples)
        else:
            img_size = inputs[0].shape[1:]
            assert all(input_.shape[1:] == img_size for input_ in inputs),  \
                'The image size in a batch should be the same.'
            # pad images when testing
            if self.test_cfg:
                inputs, padded_samples = stack_batch(
                    inputs=inputs,
                    size=self.test_cfg.get('size', None),
                    size_divisor=self.test_cfg.get('size_divisor', None),
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val)
                for data_sample, pad_info in zip(data_samples, padded_samples):
                    data_sample.set_metainfo({**pad_info})
            else:
                inputs = torch.stack(inputs, dim=0)
        
        # print("==========",inputs.shape)
        # print("==============", dict(inputs=inputs, data_samples=data_samples))
        # print("----22------######", ((data_samples[0].gt_sem_seg_hand.data)).shape) # tensor[1,360,379]
        # print("----22------######", ((data_samples[0].gt_sem_seg.data)).shape) # tensor[1,360,379]
        # print("----22------######", (inputs[0].shape)) # tensor[3,360,379]

        return dict(inputs=inputs, data_samples=data_samples)

def stack_batch(inputs: List[torch.Tensor],
                data_samples: Optional[SampleList] = None,
                size: Optional[tuple] = None,
                size_divisor: Optional[int] = None,
                pad_val: Union[int, float] = 0,
                seg_pad_val: Union[int, float] = 255) -> torch.Tensor:

    assert isinstance(inputs, list), \
        f'Expected input type to be list, but got {type(inputs)}'
    assert len({tensor.ndim for tensor in inputs}) == 1, \
        f'Expected the dimensions of all inputs must be the same, ' \
        f'but got {[tensor.ndim for tensor in inputs]}'
    assert inputs[0].ndim == 3, f'Expected tensor dimension to be 3, ' \
        f'but got {inputs[0].ndim}'
    assert len({tensor.shape[0] for tensor in inputs}) == 1, \
        f'Expected the channels of all inputs must be the same, ' \
        f'but got {[tensor.shape[0] for tensor in inputs]}'

    # only one of size and size_divisor should be valid
    assert (size is not None) ^ (size_divisor is not None), \
        'only one of size and size_divisor should be valid'

    padded_inputs = []
    padded_samples = []
    inputs_sizes = [(img.shape[-2], img.shape[-1]) for img in inputs]
    # print("----------------",inputs_sizes)
    max_size = np.stack(inputs_sizes).max(0)
    if size_divisor is not None and size_divisor > 1:
        # the last two dims are H,W, both subject to divisibility requirement
        max_size = (max_size +
                    (size_divisor - 1)) // size_divisor * size_divisor

    for i in range(len(inputs)):
        tensor = inputs[i]
        if size is not None:
            # print("##", size)
            # print("###", tensor.shape)
            width = max(size[-1] - tensor.shape[-1], 0)
            height = max(size[-2] - tensor.shape[-2], 0)

            # (padding_left, padding_right, padding_top, padding_bottom)
            padding_size = (0, width, 0, height)
            # print("**", padding_size)
        elif size_divisor is not None:
            width = max(max_size[-1] - tensor.shape[-1], 0)
            height = max(max_size[-2] - tensor.shape[-2], 0)
            padding_size = (0, width, 0, height)
            # print("***", padding_size)
        else:
            padding_size = [0, 0, 0, 0]

        # pad img
        pad_img = F.pad(tensor, padding_size, value=pad_val)
        padded_inputs.append(pad_img)
        # pad gt_sem_seg
        if data_samples is not None:
            data_sample = data_samples[i]
            gt_sem_seg = data_sample.gt_sem_seg.data
            gt_sem_seg_hand = data_sample.gt_sem_seg_hand.data
            gt_sem_seg_left_obj = data_sample.gt_sem_seg_left_obj.data
            gt_sem_seg_right_obj = data_sample.gt_sem_seg_right_obj.data
            gt_sem_seg_two_obj = data_sample.gt_sem_seg_two_obj.data
            gt_sem_seg_cb = data_sample.gt_sem_seg_cb.data
            del data_sample.gt_sem_seg.data
            del data_sample.gt_sem_seg_hand.data
            del data_sample.gt_sem_seg_left_obj.data
            del data_sample.gt_sem_seg_right_obj.data
            del data_sample.gt_sem_seg_two_obj.data
            del data_sample.gt_sem_seg_cb.data
            data_sample.gt_sem_seg.data = F.pad(
                gt_sem_seg, padding_size, value=seg_pad_val)
            data_sample.gt_sem_seg_hand.data = F.pad(
                gt_sem_seg_hand, padding_size, value=seg_pad_val)
            data_sample.gt_sem_seg_left_obj.data = F.pad(
                gt_sem_seg_left_obj, padding_size, value=seg_pad_val)
            data_sample.gt_sem_seg_right_obj.data = F.pad(
                gt_sem_seg_right_obj, padding_size, value=seg_pad_val)
            data_sample.gt_sem_seg_two_obj.data = F.pad(
                gt_sem_seg_two_obj, padding_size, value=seg_pad_val)
            data_sample.gt_sem_seg_cb.data = F.pad(
                gt_sem_seg_cb, padding_size, value=seg_pad_val
            )
            
            if 'gt_edge_map' in data_sample:
                gt_edge_map = data_sample.gt_edge_map.data
                del data_sample.gt_edge_map.data
                data_sample.gt_edge_map.data = F.pad(
                    gt_edge_map, padding_size, value=seg_pad_val)
            data_sample.set_metainfo({
                'img_shape': tensor.shape[-2:],
                'pad_shape': data_sample.gt_sem_seg.shape,
                'padding_size': padding_size
            })
            padded_samples.append(data_sample)
        else:
            padded_samples.append(
                dict(
                    img_padding_size=padding_size,
                    pad_shape=pad_img.shape[-2:]))

    return torch.stack(padded_inputs, dim=0), padded_samples