import torch
import torch.nn as nn
import torch.nn.functional as F
from .visual_model.detr import build_detr
from .language_model.bert import build_bert
from .vl_transformer import build_vl_transformer
from .cross_module import cross_module


class TransVG_MLCMA(nn.Module):
    def __init__(self, args):
        super(TransVG_MLCMA, self).__init__()
        hidden_dim = args.vl_hidden_dim
        divisor = 16 if args.dilation else 32
        self.num_visu_token = int((args.imsize / divisor) ** 2)
        self.num_text_token = args.max_query_len

        self.visumodel = build_detr(args)
        self.textmodel = build_bert(args)
        # TODO: 多了这个cross_module
        self.cross_module = cross_module(args)

        num_total = self.num_visu_token + self.num_text_token + 1
        self.vl_pos_embed = nn.Embedding(num_total, hidden_dim)
        self.reg_token = nn.Embedding(1, hidden_dim)

        self.visu_proj = nn.Linear(self.visumodel.num_channels, hidden_dim)
        self.text_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)

        self.vl_transformer = build_vl_transformer(args)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, img_data, text_data):
        # img_data, text_data 都是 nested tensor，里面有两组数据，都已经在device上；
        # print("\n img_data shape: ", img_data.tensors.shape)  # torch.Size([128, 3, 640, 640])
        # print("\n img_data type: ", img_data.tensors.type())  # torch.cuda.FloatTensor
        # print("\n img_data mask shape: ", img_data.mask.shape)  # torch.Size([128, 640, 640])
        # print("\n text_data shape: ", text_data.tensors.shape)  # torch.Size([128, 20])
        # print("\n text_data mask shape: ", text_data.mask.shape)  # torch.Size([128, 20])
        # print("\n text_data is: ", text_data.tensors)

        bs = img_data.tensors.shape[0]

        # visual backbone
        visu_mask, visu_src = self.visumodel(img_data)
        # print("\nvisu_mask shape: ", visu_mask.size())  # torch.Size([128, 400])
        # visu_src = self.visu_proj(visu_src) # (N*B)xC

        # language bert
        text_fea = self.textmodel(text_data)
        text_src, text_mask = text_fea.decompose()
        # print("\ntext_src shape: ", text_src.shape)  # torch.Size([128, 20, 768])
        assert text_mask is not None

        # text_src = self.text_proj(text_src)
        # permute BxLenxC to LenxBxC
        #  text_src = text_src.permute(1, 0, 2)
        #  text_mask = text_mask.flatten(1)

        # target regression token
        tgt_src = self.reg_token.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt_mask = torch.zeros((bs, 1)).to(tgt_src.device).to(torch.bool)

        # TODO: 多了这个cross_module
        text_src, visu_src = self.cross_module(visu_src, text_src, visu_mask, text_mask)
        visu_src = self.visu_proj(visu_src)  # (N*B)xC
        text_src = self.text_proj(text_src)
        text_src = text_src.permute(1, 0, 2)
        text_mask = text_mask.flatten(1)

        vl_src = torch.cat([tgt_src, text_src, visu_src], dim=0)
        vl_mask = torch.cat([tgt_mask, text_mask, visu_mask], dim=1)
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        vg_hs = self.vl_transformer(vl_src, vl_mask, vl_pos)  # (1+L+N)xBxC
        vg_hs = vg_hs[0]
        pred_box = self.bbox_embed(vg_hs).sigmoid()

        return pred_box


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
