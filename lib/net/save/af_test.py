import torch
import torch.nn as nn
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

        
        
        att = self.shared_mlp(fusion_features)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)), inplace=True)
        att = att.permute(0, 3, 2, 1)
        att = F.max_pool2d(att, kernel_size = [1, att.size(3)])
        return torch.cat([x, pointfeat], 1)

if __name__ == "__main__":
    sim_data = torch.rand(1, 160, 4096)
    af = AF_Block(inplanes=160, outplanes=1)
    out = af(sim_data)
    print('point feat', out.size())