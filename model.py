from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn


# VIT 模型不能更改输入的图片的大小
class PatchEmbed(nn.Module):
    # patch_size: 是每一块的尺寸
    # in_c 是三通道的                D维是 embed_dim
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super(PatchEmbed, self).__init__()
        img_size = (img_size, img_size)  # 这是个矩阵
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        # 使图片的高宽达到预定的尺寸
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        # nn.Identity() 不做任何的操作
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # print(x)
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input img size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # flatten : [B, C, H, W] -> [B, C, HW]  将后两维展平
        # transpose : [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入的token的dim  是dk   应该是768
                 num_heads=8,  # 多头的个数
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads  # 每一个head对应的维度
        self.scale = qk_scale or head_dim ** -0.5  # 对应的下边的dk或者 根号下dk
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, patch_size+1, total_embed_dim]
        #   图片的个数     输入图片的尺寸14* 14   深度768
        B, N, C = x.shape

        # qkv() -> [batch-size, num_patch+1, 3*total_embed_size]
        # reshape -> [batch-size, num_patch+1, 3, num_heads,  ember_dim_per_head]
        # permute -> [3, batch-size, num_heads, num_patch+1, ember_dim_per_head]      .permute  实现一个维度的调换
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # @ 是矩阵相乘的操作符
        # transpose [batch_size, num_heads, ember_dim_per_head, num_patch+1]  为了进行最后两个维度 调换      一个转置
        # @ :multiply  [batch-size, num_heads, num_patch+1,num_patch+1]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # 这里的q  v 是针对的每个head进行的操作
        attn = attn.softmax(dim=-1)  # 最后一个维度的softmax操作   每一行的的layer的softmax的处理
        attn = self.attn_drop(attn)

        # @: multiply [batch-size, num_heads, num_patch+1, num_patch+1]
        # transpose [batch-size, num_patch+1, num_heads,num_patch+1] 维度的交换
        # reshape  [batch-size, num_heads, total_embed_dim]   经最后两层的信息的拼接
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super(MLP, self).__init__()
        hidden_features = hidden_features or in_features * 4
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act_layer = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_layer(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# encode block  层   每一个attention 之间使用的是droppath 层
class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 qk_scale=False,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


# 未看懂
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        x = drop_path(x, self.drop_prob, self.training)
        return x


# depth 是模块重复的次数
# mlt_ratio 是mlt模块线形层增加的倍数
# representation_size: 可表示层  pre_layer 是否显示，默认不行显示
# distilled=False,  VIT 模型中用不到
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None, act_layer=None):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1  # self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        #          embed_layer=PatchEmbed 这是个类
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # embed_dim 是channels 渠道的深度     class token 时的初始化
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = None
        # 位置的Position Embedding  的初始化参数   num_patches可能是 patch Embed 中的展开  14*14=196
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        # torch.linspace  返回一个1维张量，包含在区间start和end上均匀间隔的step个点。'
        # 每一个Transformer Encoder 实现的随机梯度的衰减
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule  随机梯度的衰减
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])

        # self.block1 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                     drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[0],
        #                     norm_layer=norm_layer, act_layer=act_layer)
        # self.block2 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                     drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[1],
        #                     norm_layer=norm_layer, act_layer=act_layer)
        # self.block3 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        #                     qk_scale=qk_scale,
        #                     drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[2],
        #                     norm_layer=norm_layer, act_layer=act_layer)
        # self.block4 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        #                     qk_scale=qk_scale,
        #                     drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[3],
        #                     norm_layer=norm_layer, act_layer=act_layer)
        # self.block5 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        #                     qk_scale=qk_scale,
        #                     drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[4],
        #                     norm_layer=norm_layer, act_layer=act_layer)
        # self.block6 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        #                     qk_scale=qk_scale,
        #                     drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[5],
        #                     norm_layer=norm_layer, act_layer=act_layer)
        # self.block7 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        #                     qk_scale=qk_scale,
        #                     drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[6],
        #                     norm_layer=norm_layer, act_layer=act_layer)
        # self.block8 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        #                     qk_scale=qk_scale,
        #                     drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[7],
        #                     norm_layer=norm_layer, act_layer=act_layer)
        # self.block9 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        #                     qk_scale=qk_scale,
        #                     drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[8],
        #                     norm_layer=norm_layer, act_layer=act_layer)
        # self.block10 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        #                      qk_scale=qk_scale,
        #                      drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[9],
        #                      norm_layer=norm_layer, act_layer=act_layer)
        # self.block11 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        #                      qk_scale=qk_scale,
        #                      drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[10],
        #                      norm_layer=norm_layer, act_layer=act_layer)
        # self.block12 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        #                      qk_scale=qk_scale,
        #                      drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[11],
        #                      norm_layer=norm_layer, act_layer=act_layer)

        self.norm = norm_layer(embed_dim)

        # 主要是这一层为真  Representation layer  马哥Pre_Logits 层
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            # OrderedDict 使字典有序
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None

        # Weight init  初始化  没有看呢
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]  num_patches = C*H
        # 在batch维度复制B 份
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]  在维度1 上的拼接
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        # x1 = self.blocks[0](x)
        # x1 = x + x1
        # x2 = self.blocks[1](x1)
        # x2 = x + x2
        # x3 = self.blocks[2](x2)
        # x3 = x + x3
        # x4 = self.blocks[3](x3)
        # x4 = x + x4
        # x5 = self.blocks[4](x4)
        # x5 = x + x5
        # x6 = self.blocks[5](x5)
        # x7 = self.blocks[6](x6)
        # x8 = self.blocks[7](x7)
        # x9 = self.blocks[8]()
        # x10 = self.blocks[9](x8)
        # x11 = self.blocks[10](x8)
        # x12 = self.blocks[11](x6)

        x = self.norm(x)
        if self.dist_token is None:
            # ,大都数执行这一步   batch维度的数据全取,  然后取索引为零的数据
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            # VIT中执行这一步  没有执行pre_logits这一步
            x = self.head(x)
        return x


# 测试使用
def vit_base_patch16_num10(num_classes=10, has_logits=None):
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch16_224_in21k(num_classes: int = 1000, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base16(num_classes=1000):
    model = VisionTransformer(img_size=224, patch_size=16,
                              embed_dim=768, depth=12,
                              num_heads=12, representation_size=768,
                              num_classes=num_classes)
    return model


def vit_large16(num_classes=1000, has_logits=False):
    model = VisionTransformer(img_size=224, patch_size=16,
                              embed_dim=1024, depth=24,
                              num_heads=16, representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_huge16(num_classes=1000, has_logits=False):
    model = VisionTransformer(img_size=224, patch_size=14,
                              embed_dim=1280, depth=32,
                              num_heads=16, representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
