import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_lib.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
from pointnet2_lib.pointnet2.pytorch_utils import SharedMLP
from lib.config import cfg
from torch.nn.functional import grid_sample
from lib.net.fusion_moudle import BasicBlock, ImgAttBlock, Fusion_Conv, IA_Layer, Atten_Fusion_Conv, AF_Block

BatchNorm2d = nn.BatchNorm2d

def conv3x3(in_planes, out_planes, stride = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride,
                     padding = 1, bias = False)


def Feature_Gather(feature_map, xy):
    """
    :param xy:(B,N,2)  normalize to [-1,1]
    :param feature_map:(B,C,H,W)
    :return:(B, C, N)
    """
    
    # use grid_sample for this.
    # xy(B,N,2)->(B,1,N,2)
    xy = xy.unsqueeze(1)

    interpolate_feature = grid_sample(feature_map, xy)  # (B,C,1,N)

    return interpolate_feature.squeeze(2) # (B,C,N)


def get_model(input_channels = 6, use_xyz = True):
    return Pointnet2MSG(input_channels = input_channels, use_xyz = use_xyz)


class Pointnet2MSG(nn.Module):
    def __init__(self, input_channels = 6, use_xyz = True):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        skip_channel_list = [input_channels]
        for k in range(cfg.RPN.SA_CONFIG.NPOINTS.__len__()):
            mlps = cfg.RPN.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                    PointnetSAModuleMSG(
                            npoint = cfg.RPN.SA_CONFIG.NPOINTS[k],
                            radii = cfg.RPN.SA_CONFIG.RADIUS[k],
                            nsamples = cfg.RPN.SA_CONFIG.NSAMPLE[k],
                            mlps = mlps,
                            use_xyz = use_xyz,
                            bn = cfg.RPN.USE_BN
                    )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        ##################
        if cfg.AF_FUSION.ENABLED:
            self.Img_Block = nn.ModuleList()
            self.Fusion_Conv = nn.ModuleList()
            self.DeConv = nn.ModuleList()
            self.AF_Blocks = nn.ModuleList()
            for i in range(len(cfg.AF_FUSION.IMG_CHANNELS) - 1):
                # add cy attention
                # self.Img_Block.append(BasicBlock(cfg.AF_FUSION.IMG_CHANNELS[i], cfg.AF_FUSION.IMG_CHANNELS[i+1], stride=1))
                if i == 0:
                    self.Img_Block.append(ImgAttBlock(cfg.AF_FUSION.IMG_CHANNELS[i], cfg.AF_FUSION.IMG_CHANNELS[i+1], stride=1))
                else:
                    self.Img_Block.append(BasicBlock(cfg.AF_FUSION.IMG_CHANNELS[i], cfg.AF_FUSION.IMG_CHANNELS[i+1], stride=1))
                # if cfg.AF_FUSION.ADD_Image_Attention:
                #     self.Fusion_Conv.append(
                #         Atten_Fusion_Conv(cfg.AF_FUSION.IMG_CHANNELS[i + 1], cfg.AF_FUSION.POINT_CHANNELS[i],
                #                           cfg.AF_FUSION.POINT_CHANNELS[i]))
                # else:
                self.Fusion_Conv.append(Fusion_Conv(cfg.AF_FUSION.IMG_CHANNELS[i + 1] + cfg.AF_FUSION.POINT_CHANNELS[i],
                                                    cfg.AF_FUSION.POINT_CHANNELS[i]))

                self.DeConv.append(nn.ConvTranspose2d(cfg.AF_FUSION.IMG_CHANNELS[i + 1], cfg.AF_FUSION.DeConv_Reduce[i],
                                                  kernel_size=cfg.AF_FUSION.DeConv_Kernels[i],
                                                  stride=cfg.AF_FUSION.DeConv_Kernels[i]))
                # self.AF_Block.append(SharedMLP([cfg.AF_FUSION.FUSION_CHANNELS[i], 64]))
                self.AF_Blocks.append(AF_Block(cfg.AF_FUSION.FUSION_CHANNELS[i], 1))
            #self.AF_Block.append(SharedMLP([160, 64]))
                

            self.image_fusion_conv = nn.Conv2d(sum(cfg.AF_FUSION.DeConv_Reduce), cfg.AF_FUSION.IMG_FEATURES_CHANNEL//4, kernel_size = 1)
            
            self.image_fusion_bn = torch.nn.BatchNorm2d(cfg.AF_FUSION.IMG_FEATURES_CHANNEL//4)
            self.image_fusion_conv2 = ImgAttBlock(cfg.AF_FUSION.IMG_FEATURES_CHANNEL//4, cfg.AF_FUSION.IMG_FEATURES_CHANNEL//4, stride=1)
            if cfg.AF_FUSION.ADD_Image_Attention:
                self.final_fusion_img_point = Atten_Fusion_Conv(cfg.AF_FUSION.IMG_FEATURES_CHANNEL//4, cfg.AF_FUSION.IMG_FEATURES_CHANNEL, cfg.AF_FUSION.IMG_FEATURES_CHANNEL)
            else:
                self.final_fusion_img_point = Fusion_Conv(cfg.AF_FUSION.IMG_FEATURES_CHANNEL + cfg.AF_FUSION.IMG_FEATURES_CHANNEL//4, cfg.AF_FUSION.IMG_FEATURES_CHANNEL)


        self.FP_modules = nn.ModuleList()

        for k in range(cfg.RPN.FP_MLPS.__len__()):
            pre_channel = cfg.RPN.FP_MLPS[k + 1][-1] if k + 1 < len(cfg.RPN.FP_MLPS) else channel_out
            self.FP_modules.append(
                    PointnetFPModule(mlp = [pre_channel + skip_channel_list[k]] + cfg.RPN.FP_MLPS[k])
            )
        self.sigmoid = nn.Sigmoid()

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous() #limit pointcloud N X 3
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features


    def forward(self, pointcloud: torch.cuda.FloatTensor, image=None, xy=None):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]

        if cfg.AF_FUSION.ENABLED:
            #### normalize xy to [-1,1]
            size_range = [1280.0, 384.0]
            xy[:, :, 0] = xy[:, :, 0] / (size_range[0] - 1.0) * 2.0 - 1.0
            xy[:, :, 1] = xy[:, :, 1] / (size_range[1] - 1.0) * 2.0 - 1.0  # = xy / (size_range - 1.) * 2 - 1.
            l_xy_cor = [xy]
            img = [image]

        for i in range(len(self.SA_modules)):
            li_xyz, li_features, li_index = self.SA_modules[i](l_xyz[i], l_features[i])

            if cfg.AF_FUSION.ENABLED:
                li_index = li_index.long().unsqueeze(-1).repeat(1,1,2)
                li_xy_cor = torch.gather(l_xy_cor[i],1,li_index) #find cor_xy lidar point
                image = self.Img_Block[i](img[i])
                #print(image.shape)
                img_gather_feature = Feature_Gather(image,li_xy_cor) #, scale= 2**(i+1))
                # fused_features = torch.cat([img_gather_feature, li_features], 1)
                li_features = self.Fusion_Conv[i](li_features,img_gather_feature)
                l_xy_cor.append(li_xy_cor)
                img.append(image)

            l_xyz.append(li_xyz)
            l_features.append(li_features)


        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                    l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        if cfg.AF_FUSION.ENABLED:
            #for i in range(1,len(img))
            DeConv = []
            for i in range(len(cfg.AF_FUSION.IMG_CHANNELS) - 1):
                DeConv.append(self.DeConv[i](img[i + 1]))
            de_concat = torch.cat(DeConv,dim=1)
            # fusion1 = self.image_fusion_conv(de_concat)
            # fusion2 = self.image_fusion_conv2(fusion1)
            img_fusion = F.relu(self.image_fusion_bn(de_concat), inplace=True)
            # img_fusion = F.relu(self.image_fusion_bn(self.image_fusion_conv(de_concat)), inplace=True)
            img_fusion_gather_feature = Feature_Gather(img_fusion, xy)
            # l_features[0] = self.final_fusion_img_point(l_features[0], img_fusion_gather_feature)
            final_fused_features = torch.cat([img_fusion_gather_feature, l_features[0]], 1)
            final_att = self.AF_Blocks[0](final_fused_features)
            # final_att = final_att.permute(0, 3, 2, 1)
            # final_att = F.max_pool2d(final_att, kernel_size = [1, final_att.size(3)])
            # final_att_mask = self.sigmoid(final_att)
            l_features[0] = l_features[0] * final_att
        return l_xyz[0], l_features[0]


class Pointnet2MSG_returnMiddleStages(Pointnet2MSG):
    def __init__(self, input_channels = 6, use_xyz = True):
        super().__init__(input_channels, use_xyz)

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        idxs = []
        for i in range(len(self.SA_modules)):
            li_xyz, li_features, idx = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            idxs.append(idx)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                    l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return l_xyz, l_features, idxs
