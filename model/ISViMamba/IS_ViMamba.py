# from models_mamba import VisionMamba,create_block
from model.ISViMamba.MambaIR import VSSBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VSS_Block(nn.Module):
    def __init__(self,hidden_dim):
        super(VSS_Block,self).__init__()
        self.vssBlock = VSSBlock(hidden_dim=hidden_dim, drop_path=0.1, attn_drop_rate=0.1, d_state=16, expand=2.0, is_light_sr=False)

    def forward(self,x):
        # print(x.shape)
        B,C,H,W = x.shape
        # x = x.view(B,-1,C)
        ###wo
        x = x.contiguous().view(B,-1,C)

        out = self.vssBlock.forward(x,(H,W))
        out = out.permute(0, 3, 1, 2)
        return out

class IS_ViMamba(nn.Module):

    '''
    结合mambe的红外小目标识别的模型，为参加比赛尝试的

    '''
    def __init__(self):
        super(IS_ViMamba,self).__init__()
        self.m = nn.Sequential(VSS_Block(hidden_dim=256),
                                VSS_Block(hidden_dim=256),
                                nn.MaxPool2d(kernel_size=2,stride=2),
                                VSS_Block(hidden_dim=128),
                                nn.MaxPool2d(kernel_size=2, stride=2),
                                VSS_Block(hidden_dim=64),
                                nn.MaxPool2d(kernel_size=2, stride=2),
                                VSS_Block(hidden_dim=32),
                                nn.Upsample(scale_factor=2,mode="nearest"),
                                VSS_Block(hidden_dim=64),
                                nn.Upsample(scale_factor=2,mode="nearest"),
                                VSS_Block(hidden_dim=128),
                                nn.Upsample(scale_factor=2, mode="nearest"),
                                VSS_Block(hidden_dim=256),
                                VSS_Block(hidden_dim=256),
                                )
        # self.vmamba_1 = VisionMamba(
        # patch_size=16, stride=8, embed_dim=192, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        # final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2",
        # if_cls_token=True, if_devide_out=True, use_middle_cls_token=True, pretrained=False)

        # self.vmamba_1 = create_block(d_model=16)
        # self.vmamba_1 = VSS_Block(hidden_dim=256)

        # self.m.add_module('vmamba_1',self.vmamba_1)


    def forward(self,x):
        # out  = self.m.get_submodule('vmamba_1').forward(x)
        out  = self.m.forward(x)
        # out  = out.permute(0,3,1,2)
        return out


if __name__ == '__main__':
    inputs = torch.randn(1, 16, 256, 256).to('cuda')
    B, C, H, W = inputs.shape
    print("输入特征维度：", inputs.shape)
    # inputs = inputs.view(B, C, H * W).permute(0, 2, 1)
    model = IS_ViMamba().to('cuda')
    out = model(inputs)
    # print(out)
    print(out.shape)
