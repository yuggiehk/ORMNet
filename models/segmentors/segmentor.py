# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor
from timm.models.layers import trunc_normal_
from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor
import torch
from mmengine.optim import OptimWrapper
from mmengine.utils import is_list_of
from typing import Dict, Union, Tuple
import copy
from collections import OrderedDict
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData
from ..utils import resize
from mmengine.runner import CheckpointLoader

import numpy as np

@MODELS.register_module()
class WithSeperateHeadsforObjCrossAttnSegmentor(BaseSegmentor):
   
    def __init__(self,
                 backbone: ConfigType,
                 decode_head1: ConfigType, # 1: hand
                 decode_head2: ConfigType, # 2: left obj
                 decode_head4: ConfigType, # 4: right obj
                 decode_head3: ConfigType, # 3: contact boundary
                 feature_cb_and_hand_to_obj: bool,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
            self.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        
        self._init_decode_head1(decode_head1, pretrained)
        self._init_decode_head2(decode_head2, pretrained)
        self._init_decode_head3(decode_head3, pretrained)
        self._init_decode_head4(decode_head4, pretrained)
        self._init_auxiliary_head(auxiliary_head)
        
        # self._init_weights()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.feature_cb_and_hand_to_obj = feature_cb_and_hand_to_obj
        # assert self.with_decode_head
    

    def _init_decode_head1(self, decode_head: ConfigType, pretrained) -> None:
        """Initialize ``decode_head``"""
        self.decode_head1 = MODELS.build(decode_head)
        self.num_classes_1 = self.decode_head1.num_classes
        self.align_corners1 = self.decode_head1.align_corners
        self.pretrained = pretrained

    def _init_decode_head2(self, decode_head: ConfigType, pretrained) -> None:
        """Initialize ``decode_head``"""
        self.decode_head2 = MODELS.build(decode_head)
        self.num_classes2 = self.decode_head2.num_classes
        self.align_corners2 = self.decode_head2.align_corners
        self.pretrained = pretrained

    def _init_decode_head3(self, decode_head: ConfigType, pretrained) -> None:
        """Initialize ``decode_head``"""
        self.decode_head3 = MODELS.build(decode_head)
        self.num_classes3 = self.decode_head3.num_classes
        self.align_corners3 = self.decode_head3.align_corners
        self.pretrained = pretrained

    def _init_decode_head4(self, decode_head: ConfigType, pretrained) -> None:
        """Initialize ``decode_head``"""
        self.decode_head4 = MODELS.build(decode_head)
        self.num_classes4 = self.decode_head4.num_classes
        self.align_corners4 = self.decode_head4.align_corners
        self.pretrained = pretrained

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(inputs)
        seg_logits_hand, feature_hand = self.decode_head1.predict(x, batch_img_metas, self.test_cfg)
        seg_logits_cb = self.decode_head3.predict(x, batch_img_metas, self.test_cfg)
        seg_logits_left_obj = self.decode_head2.predict(x,feature_hand, batch_img_metas, self.test_cfg)
        seg_logits_right_obj = self.decode_head4.predict(x, feature_hand, batch_img_metas, self.test_cfg)

        return seg_logits_hand, seg_logits_cb, seg_logits_left_obj, seg_logits_right_obj

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        losses = dict()
        data_samples_hand = copy.deepcopy(data_samples)
        data_samples_left_obj = copy.deepcopy(data_samples)
        data_samples_right_obj = copy.deepcopy(data_samples)
        data_samples_cb = copy.deepcopy(data_samples)
        for i in range(len(data_samples)):
            data_samples_hand[i].gt_sem_seg.data = data_samples_hand[i].gt_sem_seg_hand.data
            data_samples_left_obj[i].gt_sem_seg.data = data_samples_left_obj[i].gt_sem_seg_left_obj.data
            data_samples_right_obj[i].gt_sem_seg.data = data_samples_right_obj[i].gt_sem_seg_right_obj.data
            data_samples_cb[i].gt_sem_seg.data = data_samples_cb[i].gt_sem_seg_cb.data
            del data_samples_hand[i].gt_sem_seg_hand, data_samples_hand[i].gt_sem_seg_left_obj,data_samples_hand[i].gt_sem_seg_right_obj, data_samples_hand[i].gt_sem_seg_cb
            del data_samples_left_obj[i].gt_sem_seg_hand, data_samples_left_obj[i].gt_sem_seg_left_obj,data_samples_left_obj[i].gt_sem_seg_right_obj, data_samples_left_obj[i].gt_sem_seg_cb
            del data_samples_right_obj[i].gt_sem_seg_hand, data_samples_right_obj[i].gt_sem_seg_left_obj,data_samples_right_obj[i].gt_sem_seg_right_obj, data_samples_right_obj[i].gt_sem_seg_cb
            del data_samples_cb[i].gt_sem_seg_hand, data_samples_cb[i].gt_sem_seg_left_obj,data_samples_cb[i].gt_sem_seg_right_obj, data_samples_cb[i].gt_sem_seg_cb
           

        loss_decode_hand, feature_hand = self.decode_head1.loss(inputs, data_samples_hand, self.train_cfg)
        loss_decode_cb = self.decode_head3.loss(inputs, data_samples_cb, self.train_cfg)
        loss_decode_left_obj = self.decode_head2.loss(inputs, feature_hand, data_samples_left_obj, self.train_cfg)
        loss_decode_right_obj = self.decode_head4.loss(inputs, feature_hand, data_samples_right_obj, self.train_cfg)

        
        losses.update(add_prefix(loss_decode_hand, 'decode_hand'))
        losses.update(add_prefix(loss_decode_left_obj, 'decode_left_obj'))
        losses.update(add_prefix(loss_decode_right_obj, 'decode_right_obj'))
        losses.update(add_prefix(loss_decode_cb, 'decode_cb'))
        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        x = self.extract_feat(inputs)
        losses = dict()
        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logit= self.inference(inputs, batch_img_metas)

        return self.postprocess_result(seg_logit, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None,
                 mode='tensor') -> Tensor:
      
        if mode== 'tensor': # return: tensor / tuple
            pass
        elif mode=='predict': # return list
            self.predict(input, data_samples)
        elif mode=='loss': # return dict of losses
            self.loss(inputs, data_samples)

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:


        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:

        seg_logits_hand, seg_logits_cb, seg_logits_left_obj, seg_logits_right_obj= self.encode_decode(inputs, batch_img_metas)

        return seg_logits_hand, seg_logits_cb, seg_logits_left_obj, seg_logits_right_obj

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
       
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
            f'Only "slide" or "whole" test mode are supported, but got ' \
            f'{self.test_cfg["mode"]}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        if not all(_['ori_shape'] == ori_shape for _ in batch_img_metas):
            print_log(
                'Image shapes are different in the batch.',
                logger='current',
                level=logging.WARN)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)

        return seg_logit

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
    
    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Implements the default model training process including
        preprocessing, model forward propagation, loss calculation,
        optimization, and back-propagation.

        1. Calls ``self.data_processor(data, training=False)`` to collect
           batch_inputs and corresponding data_samples(labels).
        2. Calls ``self(batch_inputs, data_samples, mode='loss')`` to get raw
           loss
        3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
           backward and dict of loss tensor used to log messages.
        4. Calls ``optim_wrapper.update_params(loss)`` to update model.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def _run_forward(self, data: Union[dict, tuple, list],
                     mode: str) -> Union[Dict[str, torch.Tensor], list]:
        if mode=='loss':
            if isinstance(data, dict):
                results = self.loss(**data)
            elif isinstance(data, (list, tuple)):
                results = self.loss(*data)
            else:
                raise TypeError('Output of `data_preprocessor` in train should be '
                                f'list, tuple or dict, but got {type(data)}')
        elif mode=='predict':
            if isinstance(data, dict):
                results = self.predict(**data)
            elif isinstance(data, (list, tuple)):
                results = self.predict(*data)
            else:
                raise TypeError('Output of `data_preprocessor` in val should be '
                                f'list, tuple or dict, but got {type(data)}')
        elif mode=='tensor':
            pass
        return results
    
    def parse_losses(
            self, losses: Dict[str, torch.Tensor]
            ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        log_vars = []
        for loss_name, loss_value in losses.items():
            # print("==", loss_name, loss_value)
            if "decode_cb" in loss_name:
                # print(loss_value)
                loss_value = loss_value*0.0
                # print("==", loss_name, loss_value)
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars) 
        return loss, log_vars 
    
    def val_step(self, data: Union[tuple, dict, list]) -> list:
        data = self.data_preprocessor(data, True)
        return self._run_forward(data, mode='predict') 
    
    def postprocess_result(self,
                           seg_logits: Tensor,
                           data_samples: OptSampleList = None) -> SampleList:
        """ 
            seg_logits:[seg_logits_hand, seg_logits_obj]
        """
        batch_size, C_hand, H, W = seg_logits[0].shape
        seg_logits_hand = seg_logits[0]
        batch_size, C_cb, H, W = seg_logits[1].shape
        seg_logits_cb = seg_logits[1]
        batch_size, C_left_obj, H, W = seg_logits[2].shape
        seg_logits_left_obj = seg_logits[2]
        batch_size, C_right_obj, H, W = seg_logits[3].shape
        seg_logits_right_obj = seg_logits[3]

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                # remove padding area
                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom = padding_size

                i_seg_logits_hand = seg_logits_hand[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]
                i_seg_logits_left_obj = seg_logits_left_obj[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]
                i_seg_logits_right_obj = seg_logits_right_obj[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]
                i_seg_logits_cb = seg_logits_cb[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]
                
                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_seg_logits_hand = i_seg_logits_hand.flip(dims=(3,1))
                        i_seg_logits_left_obj = i_seg_logits_left_obj.flip(dims=(3,1))
                        i_seg_logits_right_obj = i_seg_logits_right_obj.flip(dims=(3,1))
                        i_seg_logits_cb = i_seg_logits_cb.flip(dims=(3,1))
                    else:
                        i_seg_logits_hand = i_seg_logits_hand.flip(dims=(2,1))
                        i_seg_logits_left_obj = i_seg_logits_left_obj.flip(dims=(2,1))
                        i_seg_logits_right_obj = i_seg_logits_right_obj.flip(dims=(3,1))
                        i_seg_logits_cb = i_seg_logits_cb.flip(dims=(2,1))
                # resize as original shape
                i_seg_logits_hand = resize(
                    i_seg_logits_hand,
                    # size=img_meta['ori_shape'],
                    size=img_meta['img_shape'],
                    mode='bilinear',
                    align_corners=self.align_corners1,
                    warning=False).squeeze(0)
                i_seg_logits_left_obj = resize(
                    i_seg_logits_left_obj,
                    # size=img_meta['ori_shape'],
                    size=img_meta['img_shape'],
                    mode='bilinear',
                    align_corners=self.align_corners2,
                    warning=False).squeeze(0)
                i_seg_logits_right_obj = resize(
                    i_seg_logits_right_obj,
                    # size=img_meta['ori_shape'],
                    size=img_meta['img_shape'],
                    mode='bilinear',
                    align_corners=self.align_corners4,
                    warning=False).squeeze(0)
                i_seg_logits_cb = resize(
                    i_seg_logits_cb,
                    size=img_meta['img_shape'],
                    mode='bilinear',
                    align_corners=self.align_corners3,
                    warning=False).squeeze(0)
            else:
                i_seg_logits_hand = seg_logits_hand[i]
                i_seg_logits_left_obj = seg_logits_left_obj[i]
                i_seg_logits_right_obj = seg_logits_right_obj[i]
                i_seg_logits_cb = seg_logits_cb[i]

            if C_hand > 1:
                i_seg_pred_hand = i_seg_logits_hand.argmax(dim=0, keepdim=True)
            else:
                i_seg_logits_hand = i_seg_logits_hand.sigmoid()
                i_seg_pred_hand = (i_seg_logits_hand >
                              self.decode_head1.threshold).to(i_seg_logits_hand)
                
            if C_left_obj > 1:
                i_seg_pred_left_obj = i_seg_logits_left_obj.argmax(dim=0, keepdim=True)
            else:
                i_seg_logits_left_obj = i_seg_logits_left_obj.sigmoid()
                i_seg_pred_left_obj = (i_seg_logits_left_obj >
                              self.decode_head2.threshold).to(i_seg_logits_left_obj)
                
            if C_right_obj > 1:
                i_seg_pred_right_obj = i_seg_logits_right_obj.argmax(dim=0, keepdim=True)
            else:
                i_seg_logits_right_obj = i_seg_logits_right_obj.sigmoid()
                i_seg_pred_right_obj = (i_seg_logits_right_obj >
                              self.decode_head4.threshold).to(i_seg_logits_right_obj)

            if C_cb > 1:
                i_seg_pred_cb = i_seg_logits_cb.argmax(dim=0, keepdim=True)
            else:          
                i_seg_logits_cb = i_seg_logits_cb.sigmod()
                i_seg_pred_cb = (i_seg_logits_cb >
                                 self.decode_head3.threshold).to(i_seg_logits_cb)
        
            # print("=", i_seg_pred_hand.shape)
            # print("==", i_seg_pred_right_obj.shape)
            # print("==", i_seg_pred_left_obj.shape)
            # import numpy as np
            # print("=",np.unique(np.array(i_seg_pred_hand.cpu())))
            # print("==",np.unique(np.array(i_seg_pred_right_obj.cpu())))
            # print("==",np.unique(np.array(i_seg_pred_left_obj.cpu())))
            # raise KeyError

            # ----------------add two obj prediction
            i_seg_pred_two_obj = np.zeros(i_seg_pred_left_obj.shape)
            plus = i_seg_pred_left_obj + i_seg_pred_right_obj
            mask = (plus>1).cpu().numpy()
            i_seg_pred_two_obj[mask] = 1
            i_seg_pred_two_obj = i_seg_pred_two_obj.astype(np.uint8)
            i_seg_pred_two_obj = torch.from_numpy((i_seg_pred_two_obj).astype(np.uint8)).to(i_seg_logits_left_obj)
            # print("==",np.unique(np.array(i_seg_pred_two_obj.cpu())))
            # print(i_seg_pred_two_obj.shape)

            data_samples[i].set_data({
                'pred_sem_seg_hand':
                PixelData(**{'data': i_seg_pred_hand}),
                'pred_sem_seg_left_obj':
                PixelData(**{'data': i_seg_pred_left_obj}),
                'pred_sem_seg_right_obj':
                PixelData(**{'data': i_seg_pred_right_obj}),
                'pred_sem_seg_two_obj':
                PixelData(**{'data': i_seg_pred_two_obj}),
                'pred_sem_seg_cb':
                PixelData(**{'data': i_seg_pred_cb})
            })

        # with open('/home/suyuejiao/mmsegmentation/test_print.txt','a') as f:
        #     print(data_samples, file=f)

        return data_samples
    
    def _init_weights(self):
            # with open('/home/suyuejiao/mmsegmentation/test.txt','a') as file:
            #     print("***********framework init weights",file=file)
            ckpt = CheckpointLoader.load_checkpoint(
                self.pretrained, logger=None, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict_1 = OrderedDict()
            state_dict_2 = OrderedDict()
            state_dict_3 = OrderedDict()
            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                # with open('/home/suyuejiao/mmsegmentation/test.txt','a') as file0:
                #     print("-------a------",k,file=file0)
                if k.startswith("decode_head1"):
                    state_dict_1[k[13:]] = v
                    # with open('/home/suyuejiao/mmsegmentation/test.txt','a') as file0:
                    #     print("===startwith=111==", k[13:],file=file0)
                elif k.startswith("decode_head2"):
                    state_dict_2[k[13:]] = v
                    # with open('/home/suyuejiao/mmsegmentation/test.txt','a') as file0:
                    #     print("===startwith=222==", k[13:],file=file0)
                elif k.startswith("decode_head3"):
                    state_dict_3[k[13:]] = v
                    # with open('/home/suyuejiao/mmsegmentation/test.txt','a') as file0:
                    #     print("===startwith=333==", k[13:],file=file0)
                else: #backbone
                    state_dict[k[9:]] = v
                    # with open('/home/suyuejiao/mmsegmentation/test.txt','a') as file0:
                    #     print("=no==startwith===", k[9:],file=file0)
            # with open('/home/suyuejiao/mmsegmentation/test.txt','a') as file:
            #     print("---11---", state_dict_1.keys(), file=file)
            #     print("---22---", state_dict_2.keys(), file=file)
            #     print("---33---", state_dict_3.keys(), file=file)
            #     print("---00---", state_dict.keys(), file=file)

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            
            # load state_dict
            self.decode_head1._init_weights_load_from_others(state_dict_1, strict=False)
            self.decode_head2._init_weights_load_from_others(state_dict_2, strict=False)
            self.decode_head3._init_weights_load_from_others(state_dict_3, strict=False)