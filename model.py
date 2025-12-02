import torch
import torch.nn as nn

class X_NIRSNet(nn.Module):
    def __init__(self, channels=13, time_points=370, sampling_rate = 6.138, num_classes=2, filter_size = 32, hidden_unit = [256, 128]):
        super(X_NIRSNet, self).__init__()
        self.k_first  = int(round(sampling_rate * 5))
        self.k_second = int(round(sampling_rate * 2.5))
        self.k_pool   = int(sampling_rate // 2)

        self.cnn = nn.Sequential(
            self._conv_block(1, filter_size, (1, self.k_first), stride=(1, 1)),
            nn.AvgPool2d(kernel_size=(1, self.k_pool), stride=(1, self.k_pool)),
            self._conv_block(filter_size, filter_size, (1, self.k_second), stride=(1, 1))
        )

        res_layers = []
        for _ in range(4):
            res_layers.append(
                self._conv_block(filter_size, filter_size, (channels, self.k_second), padding='same')
            )
        self.resblock = nn.Sequential(*res_layers)

        self.late_activation = nn.ELU()
        self.late_conv = self._conv_block(
            filter_size, filter_size, 
            kernel_size=(channels, self.k_second), 
            stride=(1, self.k_pool)
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, channels, time_points)
            d_out = self.cnn(dummy_input)
            d_res = self.resblock(d_out) 
            d_add = d_out + d_res 
            d_final = self.late_conv(d_add)
            self.flatten_dim = d_final.view(1, -1).shape[1]


        self.fc_branch = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.flatten_dim, hidden_unit[0]),
            nn.ELU(),
        )

        self.fc_fusion = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_unit[0] * 4, hidden_unit[1]),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_unit[1], num_classes),
        )
        
    def _conv_block(self, in_c, out_c, kernel_size, stride=(1,1), padding=0):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_c),
            nn.ELU()
        )
    
    def _process_branch(self, x):
        x = self.late_activation(x)
        x = self.late_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc_branch(x)
        return x
    
    def forward(self, x1, x2):
        
        hbo = x1.unsqueeze(1) if x1.dim() == 3 else x1
        hbr = x2.unsqueeze(1) if x2.dim() == 3 else x2
        
        # 1. Feature Extraction
        hbo_feat = self.cnn(hbo)
        hbr_feat = self.cnn(hbr)
        
        # 2. Residual Features
        hbo_res = self.resblock(hbo_feat)
        hbr_res = self.resblock(hbr_feat)
        
        # 3. Cross-Residual Addition
        branches = [
            hbo_feat + hbo_res,
            hbr_feat + hbr_res,
            hbo_feat + hbr_res,
            hbr_feat + hbo_res
        ]

        processed_branches = [self._process_branch(b) for b in branches]

        fused = torch.cat(processed_branches, dim=1)
        out = self.fc_fusion(fused)
        
        return out