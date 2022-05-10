import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_lib.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
from lib.config import cfg
from torch.nn.functional import grid_sample


BatchNorm2d = nn.BatchNorm2d

def conv3x3(in_planes, out_planes, stride = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride,
                     padding = 1, bias = False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, outplanes, stride)
        self.bn1 = BatchNorm2d(outplanes )
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(outplanes, outplanes, 2*stride)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class ChannelAttention(nn.Module):
    def __init__(self, inplanes, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(inplanes, inplanes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(inplanes // ratio, inplanes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class ImgAttBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride = 1):
        super(ImgAttBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, outplanes, stride)
        self.bn1 = BatchNorm2d(outplanes )
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(outplanes, outplanes, 2*stride)
        
        self.ca = ChannelAttention(outplanes)
        self.sa = SpatialAttention()

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        #attention is begin
        residual = out
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out

class Fusion_Conv(nn.Module):
    def __init__(self, inplanes, outplanes):

        super(Fusion_Conv, self).__init__()

        self.conv1 = torch.nn.Conv1d(inplanes, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)

    def forward(self, point_features, img_features):
        #print(point_features.shape, img_features.shape)
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)), inplace=True)

        return fusion_features


#================addition attention (add)=======================#
#bcy
class AF_Block(nn.Module):
    '''
        input: img and points fusion features      (Batch_size, channels, num_points)
        out:   according fusion features get mask  (Batch_size, 1, numpoints)
    '''
    def __init__(self, inplanes, outplanes, global_feat = False):
        super(AF_Block, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(inplanes, 64, 1),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv1d(192, 64, 1),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv1d(64, outplanes, 1),
                                   nn.BatchNorm1d(outplanes),
                                   nn.ReLU(inplace=True))
        self.global_feat = global_feat
        self.sigmoid = nn.Sigmoid()

    def forward(self, fusion_features):
        num_pts = fusion_features.size(2)
        feature1 = self.conv1(fusion_features)
        pointfeat = feature1
        feature2 = self.conv2(feature1)
        max_feature = torch.max(feature2, 2, keepdim=True)[0]
        if self.global_feat:
            global_feat = max_feature.view(-1, max_feature.size(1)) # (Batch_size,128)
            return global_feat
        else:    
            global_feat = max_feature.repeat(1, 1, num_pts) # (Batch_size, 128, num_points)
            part_and_global = torch.cat([global_feat, pointfeat], 1) # (Batch_size, 128 + 64, num_points)
            decode1 = self.conv3(part_and_global)
            decode2 = self.conv4(decode1)
            att = self.sigmoid(decode2)
            return att

class IA_Layer(nn.Module):
    def __init__(self, channels):
        print('##############ADDITION ATTENTION(ADD)#########')
        super(IA_Layer, self).__init__()
        self.ic, self.pc = channels
        rc = self.pc // 4
        self.conv1 = nn.Sequential(nn.Conv1d(self.ic, self.pc, 1),
                                    nn.BatchNorm1d(self.pc),
                                    nn.ReLU(inplace=True))
        self.fc1 = nn.Linear(self.ic, rc)
        self.fc2 = nn.Linear(self.pc, rc)
        self.fc3 = nn.Linear(rc, 1)


    def forward(self, img_feas, point_feas):
        batch = img_feas.size(0)
        img_feas_f = img_feas.transpose(1,2).contiguous().view(-1, self.ic) #BCN->BNC->(BN)C
        point_feas_f = point_feas.transpose(1,2).contiguous().view(-1, self.pc) #BCN->BNC->(BN)C'
        # print(img_feas)
        ri = self.fc1(img_feas_f)
        rp = self.fc2(point_feas_f)
        att = F.sigmoid(self.fc3(F.tanh(ri + rp))) #BNx1
        att = att.squeeze(1)
        att = att.view(batch, 1, -1) #B1N
        # print(img_feas.size(), att.size())

        img_feas_new = self.conv1(img_feas)
        out = img_feas_new * att

        return out


class Atten_Fusion_Conv(nn.Module):
    def __init__(self, inplanes_I, inplanes_P, outplanes):
        super(Atten_Fusion_Conv, self).__init__()

        self.IA_Layer = IA_Layer(channels = [inplanes_I, inplanes_P])
        # self.conv1 = torch.nn.Conv1d(inplanes_P, outplanes, 1)
        self.conv1 = torch.nn.Conv1d(inplanes_P + inplanes_P, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)


    def forward(self, point_features, img_features):
        # print(point_features.shape, img_features.shape)

        img_features =  self.IA_Layer(img_features, point_features)
        #print("img_features:", img_features.shape)

        #fusion_features = img_features + point_features
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)), inplace=True)

        return fusion_features
