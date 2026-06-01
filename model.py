import torch
import torch.nn as nn
import torch.nn.init as init

class CrossAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(CrossAttentionBlock, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.scale = (in_channels // 2) ** -0.5
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feat, res):
        B, C, H, W = feat.size()
        
        q = self.query(feat).view(B, -1, H * W).permute(0, 2, 1)  # (B, H*W, C//2)
        k = self.key(res).view(B, -1, H * W)                      # (B, C//2, H*W)
        v = self.value(res).view(B, C, H * W)                     # (B, C, H*W)

        attn = self.softmax(torch.bmm(q, k) * self.scale)         # (B, H*W, H*W)
        
        out = torch.bmm(v, attn.permute(0, 2, 1))                 # (B, C, H*W)
        out = out.view(B, C, H, W)
        
        return out + feat
 
class X_NIRSNet(nn.Module):
    def __init__(self, channels=13, time_points=366, sampling_rate = 1/0.08192/2, num_classes=2, filter_size = 32, hidden_unit = [128, 128], 
                 share_cnn=True, share_resblock=True, share_branches=True, 
                 cross_fusion_type='sum'): # 'sum' or 'attention'
        
        super(X_NIRSNet, self).__init__()
        
        self.share_cnn = share_cnn
        self.share_resblock = share_resblock
        self.share_branches = share_branches
        self.cross_fusion_type = cross_fusion_type
        
        self.k_first  = int(round(sampling_rate * 10))
        self.k_pool   = int(sampling_rate // 3)
        self.k_second = int(round((sampling_rate * 5) / self.k_pool))

        def _build_cnn():
            return nn.Sequential(
                self._conv_block(1, filter_size, (1, self.k_first), stride=(1, 1)),
                nn.AvgPool2d(kernel_size=(1, self.k_pool), stride=(1, self.k_pool)),
                self._conv_block(filter_size, filter_size, (1, self.k_second), stride=(1, 1))
            )

        if self.share_cnn:
            self.cnn = _build_cnn()
        else:
            self.cnns = nn.ModuleList([_build_cnn() for _ in range(2)]) 

        def _build_resblock():
            res_layers = []
            for _ in range(4):
                res_layers.append(
                    self._conv_block(filter_size, filter_size, (channels, self.k_second), padding='same')
                )
            return nn.Sequential(*res_layers)

        if self.share_resblock:
            self.resblock = _build_resblock()
        else:
            self.resblocks = nn.ModuleList([_build_resblock() for _ in range(2)]) 

        self.late_activation = nn.ELU()
        
        temp_cnn = self.cnn if self.share_cnn else self.cnns[0]
        temp_res = self.resblock if self.share_resblock else self.resblocks[0]
        temp_late_conv = self._conv_block(filter_size, filter_size, kernel_size=(channels, self.k_second), stride=(1, self.k_pool))

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, channels, time_points)
            d_out = temp_cnn(dummy_input)
            d_res = temp_res(d_out) 
            d_add = d_out + d_res 
            d_final = temp_late_conv(d_add)
            self.flatten_dim = d_final.view(1, -1).shape[1]

        if self.cross_fusion_type == 'attention':
            self.attention_blocks = nn.ModuleList([CrossAttentionBlock(filter_size) for _ in range(4)])

        if self.share_branches:
            self.late_conv = self._conv_block(filter_size, filter_size, kernel_size=(channels, self.k_second), stride=(1, self.k_pool))
            self.fc_branch = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(self.flatten_dim, hidden_unit[0]),
                nn.ELU(),
            )
        else:
            self.late_convs = nn.ModuleList([
                self._conv_block(filter_size, filter_size, kernel_size=(channels, self.k_second), stride=(1, self.k_pool)) for _ in range(4)
            ])
            self.fc_branches = nn.ModuleList([
                nn.Sequential(
                    nn.Dropout(0.4),
                    nn.Linear(self.flatten_dim, hidden_unit[0]),
                    nn.ELU(),
                ) for _ in range(4)
            ])

        self.fc_fusion = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_unit[0] * 4, hidden_unit[1]),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_unit[1], num_classes),
        )
        
    def _conv_block(self, in_c, out_c, kernel_size, stride=(1,1), padding=0):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_c),
            nn.ELU()
        )
    
    def _process_branch(self, x, branch_idx):
        x = self.late_activation(x)
        if self.share_branches:
            x = self.late_conv(x)
            x = x.view(x.size(0), -1)
            x = self.fc_branch(x)
        else:
            x = self.late_convs[branch_idx](x)
            x = x.view(x.size(0), -1)
            x = self.fc_branches[branch_idx](x)
        return x
        
    def forward(self, x1, x2):
        
        hbo = x1.unsqueeze(1) if x1.dim() == 3 else x1
        hbr = x2.unsqueeze(1) if x2.dim() == 3 else x2
        
        if self.share_cnn:
            hbo_feat = self.cnn(hbo)
            hbr_feat = self.cnn(hbr)
        else:
            hbo_feat = self.cnns[0](hbo)
            hbr_feat = self.cnns[1](hbr)
        
        if self.share_resblock:
            hbo_res = self.resblock(hbo_feat)
            hbr_res = self.resblock(hbr_feat)
        else:
            hbo_res = self.resblocks[0](hbo_feat)
            hbr_res = self.resblocks[1](hbr_feat)
        
        if self.cross_fusion_type == 'attention':
            branches = [
                self.attention_blocks[0](hbo_feat, hbo_res), # hbo_feat(Q)가 hbo_res(K,V)를 탐색
                self.attention_blocks[1](hbr_feat, hbr_res), # hbr_feat(Q)가 hbr_res(K,V)를 탐색
                self.attention_blocks[2](hbo_feat, hbr_res), # hbo_feat(Q)가 hbr_res(K,V)를 탐색 (Cross!)
                self.attention_blocks[3](hbr_feat, hbo_res)  # hbr_feat(Q)가 hbo_res(K,V)를 탐색 (Cross!)
            ]
        else:
            branches = [
                hbo_feat + hbo_res,
                hbr_feat + hbr_res,
                hbo_feat + hbr_res,
                hbr_feat + hbo_res
            ]

        processed_branches = [self._process_branch(b, i) for i, b in enumerate(branches)]

        fused = torch.cat(processed_branches, dim=1)
        out = self.fc_fusion(fused)
        
        return out
