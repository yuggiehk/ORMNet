import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mmseg.registry import MODELS
from mmcv.cnn import ConvModule
from ..utils import resize
from .decode_head import BaseDecodeHead
from .psp_head import PPM
import warnings
from abc import ABCMeta, abstractmethod
from typing import List, Tuple
from mmengine.model import BaseModule
from torch import Tensor
from mmseg.structures import build_pixel_sampler
from mmseg.utils import ConfigType, SampleList
from ..builder import build_loss
from ..losses import accuracy

from mmengine.runner import CheckpointLoader
from collections import OrderedDict

from .add_Unet_decoder_with_seperate_heads_obj import PatchExpand, PatchEmbed, FinalPatchExpand_X4, BasicLayer_up

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init
from einops import rearrange
import numbers
import math

def PhiTPhi_fun(x, PhiW):
    temp = F.conv2d(x, PhiW, padding=0,stride=32, bias=None)
    temp = F.conv_transpose2d(temp, PhiW, stride=32)
    return temp

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        print("@@@",x.shape)
        if len(x.shape) == 3:
            b,hw,c = x.shape
            return self.body(x)
        else:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)

class Atten(torch.nn.Module):
    def __init__(self, channels):
        super(Atten, self).__init__()
               
        self.channels = channels
        self.softmax = nn.Softmax(dim=-1)
        self.norm1 = LayerNorm(self.channels, 'WithBias')
        self.norm2 = LayerNorm(self.channels, 'WithBias')
        self.conv_q = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.conv_kv = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels*2, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels*2, self.channels*2, kernel_size=3, stride=1, padding=1, groups=self.channels*2, bias=True)
        )
        self.conv_out = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True)
        
    def forward(self, pre, cur):
            if cur.shape == pre.shape:
                b,hw,c = cur.shape
                h=w=int(math.sqrt(hw))
                cur = cur.view(b,h,w,c).permute(0,3,1,2)
            else:
                b,c,hw = cur.shape
                h=w=int(math.sqrt(hw))
                cur = cur.view(b,c,h,w)
            # cur-query, pre-key,value
            b,hw,c = pre.shape
            # print(pre.shape, cur.shape) #torch.Size([2, 196, 1024]) torch.Size([2, 1024, 36])
            h=w=int(math.sqrt(hw))
            pre = pre.view(b,h,w,c).permute(0,3,1,2)
            
            b, c, h, w = pre.shape
            # print(pre.shape) #2, 1024, 14, 14]
            # print(cur.shape) #[2, 1024, 6, 6]

            pre_ln = self.norm1(pre)
            cur_ln = self.norm2(cur)
            # print(pre.shape) #2, 1024, 14, 14]
            # print(cur.shape) #[2, 1024, 6, 6]

            q = self.conv_q(cur_ln)
            # print(q.shape) #[2, 1024, 6, 6]
            q = q.view(b, c, -1)
            # print(q.shape) #[2, 1024, 36]
            k,v = self.conv_kv(pre_ln).chunk(2, dim=1)
            k = k.view(b, c, -1)
            v = v.view(b, c, -1)
            print(k.shape, v.shape) #torch.Size([2, 1024, 196]) torch.Size([2, 1024, 196])
            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)
            print(q.shape, k.shape) #torch.Size([2, 1024, 36]) torch.Size([2, 1024, 196])
            att = torch.matmul(q, k.permute(0, 2, 1))
            att = self.softmax(att)
            out = torch.matmul(att, v).view(b, c, h, w)
            out = self.conv_out(out) + cur
            print(out.shape)
            b,c,h,w = out.shape
            out = out.view(b,c,-1).permute(0,2,1)
            print(out.shape)
        
            return out

@MODELS.register_module()
class UnetdecoderSeperateHeadsInputFeatureCrossAttn(BaseDecodeHead):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", align_corners=False,pretrained='',type_decode='',**kwargs):
        super().__init__(num_classes=num_classes,in_channels=in_chans,**kwargs)

        print("SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(depths,
        depths_decoder,drop_path_rate,num_classes))
        
        self.type_Decoder = type_decode

        self.align_corners = align_corners
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.pretrained = pretrained

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution


        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        
        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2*int(embed_dim*2**(self.num_layers-1-i_layer)),
            int(embed_dim*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer ==0 :
                layer_up = PatchExpand(input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))), dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)),
                                input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                                                    patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))),
                                depth=depths[(self.num_layers-1-i_layer)],
                                num_heads=num_heads[(self.num_layers-1-i_layer)],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])],
                                norm_layer=norm_layer,
                                upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm_up= norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            # self.up = FinalPatchExpand_X4(input_resolution=(img_size//patch_size,img_size//patch_size),dim_scale=4,dim=embed_dim)
            self.up = FinalPatchExpand_X4(input_resolution=(img_size[0]//patch_size,img_size[1]//patch_size),dim_scale=4,dim=embed_dim)
            self.output = nn.Conv2d(in_channels=embed_dim,out_channels=self.num_classes,kernel_size=1,bias=False)

        self.concat_fuse_conv = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=1,bias=False)
        self.concat_fuse_norm = norm_layer(128)
        self.concat_fuse_norm_before = norm_layer(256)
        self.cross_attn = Atten(channels=128)
        # self.apply(self._init_weights)
    
    def print_model_param_values(self):
        for name, param in self.named_parameters():
            print(name, param.data)

    def _init_weights_load_from_others(self, state_dict, strict=False):
        with open('/home/suyuejiao/mmsegmentation/test.txt','a') as file:
            print("*************decoder load init weights from others",file=file)
            print("=====",self.type_Decoder,file=file)
            for name, param in self.named_parameters():
                print("----1111111----",name, param.data,file=file)
            self.print_model_param_values()
            
        self.load_state_dict(state_dict, strict=strict)
        with open('/home/suyuejiao/mmsegmentation/test2.txt','a') as file:
            print("=====",self.type_Decoder,file=file)
            for name, param in self.named_parameters():
                
                print("----22222----",name, param.data,file=file)
            for i,m in enumerate(self.named_parameters()):
                print("-----33333-----", i,m,file=file)

    def _init_weights(self, m):
            with open('/home/suyuejiao/mmsegmentation/test.txt','a') as file:
                print("******decoder init weights, trunc_normal_",file=file)
        # if not self.pretrained:
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    #Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x,x_downsample[3-inx]],-1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)  # B L C
        return x

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H*W, "input features has wrong size"

        if self.final_upsample=="expand_first":
            x = self.up(x)
            # print("*******", x.shape)
            x = x.view(B,4*H,4*W,-1)
            x = x.permute(0,3,1,2) #B,C,H,W
            x = self.output(x)
        return x

    def forward(self,x, feature_hand):
        
        # torch.Size([16, 128, 112, 112])
        # torch.Size([16, 256, 56, 56])
        # torch.Size([16, 512, 28, 28])
        # torch.Size([16, 1024, 14, 14])
        
        for i in range(len(x)):
            # print("===",x[i].shape)
            if len(x[i].shape)==3:
                B, HW, C = x[i].shape
            elif len(x[i].shape)==4:
                B, C, H, W = x[i].shape
                HW = H * W
                x[i] = x[i].view(B, C, HW)
                x[i] = x[i].permute(0,2,1)
        x_downsample = x
        x = x_downsample[3]
        x = self.forward_up_features(x,x_downsample)
        # print('----',x.shape)# torch.Size([6, 12544, 128])

        # ----cross attn layer------
        x_cross = self.cross_attn(x, feature_hand)
        x = x + x_cross
        x = self.concat_fuse_norm(x)


        x = self.up_x4(x)
        # print(x.shape) # b,n,h,w
        return x
    
    def loss(self, inputs: Tuple[Tensor],feature_hand, batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        
        seg_logits = self.forward(inputs, feature_hand)
        loss_logits = self.loss_by_feat(seg_logits, batch_data_samples)
        return loss_logits
    
    def predict(self, inputs: Tuple[Tensor], feature_hand, batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tensor:
        
        seg_logits = self.forward(inputs, feature_hand)
        return self.predict_by_feat(seg_logits, batch_img_metas)
    
    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        for data_sample in batch_data_samples:
            if self.type_Decoder=='obj':
                # print("==", self.type_Decoder)
                import numpy as np
                data_np = np.unique(np.array(data_sample.gt_sem_seg.data.cpu()))
                # print("***",data_np)
        
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)
    
    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        
        loss = dict()
        seg_label  = self._stack_batch_gt(batch_data_samples)
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)      
        return loss

    def predict_by_feat(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        seg_logits = resize(
            input=seg_logits,
            size=batch_img_metas[0]['img_shape'],
            mode='bilinear',
            align_corners=self.align_corners)
        return seg_logits