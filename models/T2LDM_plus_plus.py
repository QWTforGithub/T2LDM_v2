# coding=utf-8
import copy
import os

import math
import torch
from torch import Tensor
from torch import nn, einsum
from einops import rearrange
import torch.nn.functional as F
from utils import common
from models.img_encoder.img_encoder import ImageEncoder

DOWNSAMPLE_STRIDE2KERNEL_DICT = {(1, 2): (3, 3), (1, 4): (3, 5), (2, 1): (3, 3), (2, 2): (3, 3)}
DOWNSAMPLE_STRIDE2PAD_DICT = {(1, 2): (0, 1, 1, 1), (1, 4): (1, 1, 1, 1), (2, 1): (1, 1, 1, 1), (2, 2): (0, 1, 0, 1)}
UNIFORM_KERNEL2PAD_DICT = {(3, 3): (1, 1, 1, 1), (1, 4): (1, 2, 0, 0)}
UPSAMPLE_STRIDE2KERNEL_DICT = {(1, 2): (1, 5), (1, 4): (1, 7), (2, 1): (5, 1), (2, 2): (3, 3)}
UPSAMPLE_STRIDE2PAD_DICT = {(1, 2): (2, 2, 0, 0), (1, 4): (3, 3, 0, 0), (2, 1): (0, 0, 2, 2), (2, 2): (1, 1, 1, 1)}
GROUP_NORM_NUM_CHANNELS = 16
LINEAR_ATTENTION_SCALE = True
ATTENTION_ROPE = True
ATTENTION_ZEOR_WEIGHT = True
RESBLOCK_ZEOR_WEIGHT = True
RESBLOCK_CIRCULARCONV_SHORTCUT = True


# ---- module ----
def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def make_attn(
        q_dim=64,
        kv_dim=None,
        heads=8,
        dim_head=8,
        attn_type="vanilla",
        norm_type="gn",
        use_norm=True,
        use_rope=ATTENTION_ROPE,
        use_res_connection=True,
        use_zero_weight=ATTENTION_ZEOR_WEIGHT,
        gate=False
):
    if attn_type == "vanilla":
        return AttentionBlock(
            q_dim=q_dim,
            kv_dim=kv_dim,
            heads=heads,
            dim_head=dim_head,
            norm_type=norm_type,
            use_norm=use_norm,
            use_rope=use_rope,
            use_res_connection=use_res_connection,
            use_zero_weight=use_zero_weight,
            gate=gate
        )
    elif attn_type == "nn":
        return Attention(
            q_dim=q_dim,
            kv_dim=kv_dim,
            heads=heads,
            dim_head=dim_head,
            norm_type=norm_type,
            use_norm=use_norm,
            use_rope=use_rope,
            use_res_connection=use_res_connection,
            use_zero_weight=use_zero_weight,
            gate=gate
        )
    elif attn_type == "linear":
        return LinearAttention(
            q_dim=q_dim,
            kv_dim=kv_dim,
            heads=heads,
            dim_head=dim_head,
            norm_type=norm_type,
            use_norm=use_norm,
            use_rope=use_rope,
            use_res_connection=use_res_connection,
            use_zero_weight=use_zero_weight,
            gate=gate
        )
    else:
        return nn.Identity(q_dim)


def make_norm(
        ch=64,
        norm_type="gn"  # "gn", "rmsn", "ln"
):
    if norm_type == "gn":
        return Normalize(in_channels=ch)
    elif norm_type == "rmsn":
        return RMSNorm(channels=ch)
    elif norm_type == "ln":
        return nn.LayerNorm(ch)
    else:
        return nn.Identity()


def get_module(
        name="CircularConv2D",
        **kwargs
):
    if name == "CircularConv2D":
        return CircularConv2D(**kwargs)
    elif name == "Conv2D":
        return torch.nn.Conv2d(**kwargs)
    elif name == "Conv2DSiLU":
        return Conv2DSiLU(**kwargs)
    elif name == "ResBlock":
        return ResnetBlock(**kwargs)
    elif name == "ResConv2DBlock":
        return ResnetConv2DBlock(**kwargs)
    elif name == "Upsample":
        return Upsample(**kwargs)
    elif name == "Downsample":
        return Downsample(**kwargs)
    elif name == "Attention":
        return make_attn(**kwargs)
# ---- module ----

# ---- Directional Position Encoding ----
class ThetaPhiPEInjector(nn.Module):
    """
    在每个 U-Net stage 入口调用一次即可：
        x = pe_injector(x)   # x: [B, C, Hs, Ws]
    作用：按 (Hs, Ws) 动态重建与投影一致的 (θ,φ) Fourier 位置编码，
         经 1×1 Conv 投到 C 通道后，残差注入：x <- x + α · Conv1x1(PE)

    参数
    ----
    in_ch:   当前 stage 的通道数 C
    K:       Fourier 频率数（总通道 = 4*K；默认 4 → 16 通道）
    fov_deg: (up, down)，单位“度”，与你投影的 fov 一致（默认 (3,-25)）
    mode:    'fov' 使用线性均匀俯仰角；'angles' 使用标定的逐线俯仰角
    vert_angles: 仅当 mode='angles' 时提供，形状 [H_full]（弧度）

    备注
    ----
    * 与你的投影一致：
        - θ：第0列≈+π，W/2≈0（列中心角：π - 2π·(c+0.5)/W）
        - φ：行中心角（若 'angles' 则对角度向量做 1D 线性采样到 Hs）
    * 门控 α 初始为 0，1×1 卷积零初始化 → 起步≈恒等映射。
    * 不同 stage 的 (Hs,Ws) 会自动缓存/重用，避免重复构建。
    """

    def __init__(self,
                 in_ch: int,
                 K: int = 4,
                 fov_deg=(3, -25),
                 mode: str = 'fov',
                 vert_angles: torch.Tensor | None = None):
        super().__init__()
        assert mode in ('fov', 'angles')
        self.in_ch = in_ch
        self.K = K
        self.fov_deg = fov_deg
        self.mode = mode

        if mode == 'angles':
            assert vert_angles is not None, "mode='angles' 需要提供 vert_angles（弧度）"
            # 存成 buffer（不随优化器更新）
            self.register_buffer('vert_angles', vert_angles.detach().float(), persistent=False)
        else:
            self.register_buffer('vert_angles', torch.empty(0), persistent=False)

        # 4K → C 的 1×1 映射 + 门控
        self.pe_proj = nn.Conv2d(4 * K, in_ch, kernel_size=1, bias=False)
        nn.init.zeros_(self.pe_proj.weight)  # 关键：零初始化，起步≈恒等
        self.alpha = nn.Parameter(torch.zeros(1))  # 可学习门控（初始0）

        # 缓存 (H, W, device_id, dtype) -> PE [1,4K,H,W]
        self._cache: dict[tuple, torch.Tensor] = {}

    @torch.no_grad()
    def _theta_center(self, W, device, dtype):
        # 与你的 grid_w 定义一致：第0列≈+π，W/2≈0（列中心）
        c = torch.arange(W, device=device, dtype=dtype)
        return math.pi - 2 * math.pi * (c + 0.5) / W  # [W]

    @torch.no_grad()
    def _phi_center(self, H, device, dtype):
        if self.mode == 'angles' and self.vert_angles.numel() > 0:
            va = self.vert_angles.to(device=device, dtype=dtype)  # [H_full]
            if va.numel() == H:
                return va  # [H]
            # 线性采样到 H（角度单调，线性安全）
            va_1d = va.view(1, 1, -1)  # [1,1,L]
            va_res = F.interpolate(va_1d, size=H, mode='linear', align_corners=True)[0, 0]
            return va_res  # [H]
        else:
            up, down = map(math.radians, self.fov_deg)  # 例如 +3°, -25°
            i = torch.arange(H, device=device, dtype=dtype)
            return up - (i + 0.5) * (up - down) / H  # [H] 行中心角

    @torch.no_grad()
    def _build_pe(self, H, W, device, dtype):
        theta = self._theta_center(W, device, dtype)  # [W]
        phi = self._phi_center(H, device, dtype)  # [H]

        TH = theta[None, None, None, :].expand(1, 1, H, W)  # [1,1,H,W]
        PH = phi[None, None, :, None].expand(1, 1, H, W)  # [1,1,H,W]

        feats = []
        # 频率：1,2,4,8,...（也可改为等差 1..K）
        for k in [2 ** i for i in range(self.K)]:
            feats += [torch.sin(k * TH), torch.cos(k * TH),
                      torch.sin(k * PH), torch.cos(k * PH)]
        pe = torch.cat(feats, dim=1).contiguous()  # [1,4K,H,W]
        return pe

    def clear_cache(self):
        self._cache.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, Hs, Ws]  —— 当前 stage 的特征
        返回：x + α · Conv1x1(PEθφ)
        """
        assert x.dim() == 4, "x 需为 [B,C,H,W]"
        B, C, Hs, Ws = x.shape
        assert C == self.in_ch, f"in_ch={self.in_ch} 与 x.shape[1]={C} 不一致"

        dev = x.device
        dev_id = dev.index if x.is_cuda else -1
        key = (Hs, Ws, dev_id, str(x.dtype))

        if key not in self._cache:
            with torch.no_grad():
                pe = self._build_pe(Hs, Ws, device=dev, dtype=x.dtype)  # [1,4K,Hs,Ws]
            self._cache[key] = pe

        pe = self._cache[key].expand(B, -1, Hs, Ws)  # [B,4K,Hs,Ws]
        return x + self.alpha * self.pe_proj(pe)
# ---- Directional Position Encoding ----

# ---- RoPE ----
class RoPECacheTheta:
    def __init__(self, base: float = 10000.0):
        self.base = base
        self.cache = {}  # (H,W,D_rot,device_id,azim_start,azim_range) -> (cos_fp32, sin_fp32)

    @torch.no_grad()
    def get(self, H, W, D_rot, device, dtype, azim_start=0.0, azim_range=2 * math.pi):
        assert D_rot % 2 == 0 and D_rot > 0
        dev = device if isinstance(device, torch.device) else torch.device(device)
        dev_id = (dev.index if dev.type == 'cuda' else -1)
        key = (H, W, D_rot, dev_id, round(azim_start, 7), round(azim_range, 7))
        if key in self.cache:
            cos, sin = self.cache[key]
            return cos.to(dtype=dtype), sin.to(dtype=dtype)

        cache_dtype = torch.float32
        N = H * W

        # 列中心角 φ_j = azim_start + (j+0.5)*Δφ, Δφ = azim_range / W
        j = torch.arange(W, device=dev, dtype=cache_dtype)
        dphi = azim_range / W
        phi = azim_start + (j + 0.5) * dphi  # [W]
        PH = phi.view(1, W).expand(H, W).reshape(N, 1)  # [N,1]（按行优先展开）

        inv = torch.exp(
            torch.arange(0, D_rot, 2, device=dev, dtype=cache_dtype)
            * (-math.log(self.base) / D_rot)
        )  # [D_rot/2]

        ang = PH @ inv.view(1, -1)  # [N, D_rot/2]
        cos = torch.cos(ang);
        sin = torch.sin(ang)  # [N, D_rot/2]
        cos = torch.stack([cos, cos], dim=-1).reshape(1, N, D_rot).contiguous()
        sin = torch.stack([sin, sin], dim=-1).reshape(1, N, D_rot).contiguous()

        self.cache[key] = (cos, sin)
        return cos.to(dtype=dtype), sin.to(dtype=dtype)

def _rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


def apply_rope_theta_seq(
        q: torch.Tensor,
        k: torch.Tensor = None,
        H: int = 64,
        W: int = 1024,  # 使 N=H*W
        rope_cache: RoPECacheTheta = None,
        rotary_dim: int = 64,  # 只旋转前 D_rot 维
        azim_start: float = 0.0,
        azim_range: float = 2 * math.pi,
        index_map: torch.Tensor | None = None  # 若你的 N 排列不是行优先，可提供重排索引 [N]
):
    """
    q,k: [B, N, C]（N 必须等于 H*W）
    返回：旋转后的 q,k（形状不变）
    """
    B, N, C = q.shape
    assert N == H * W, f"N={N} 必须等于 H*W={H * W}"
    D_rot = min(rotary_dim, C - (C % 2))
    if D_rot <= 0:
        return q, k

    cos, sin = rope_cache.get(
        H, W, D_rot, device=q.device, dtype=q.dtype,
        azim_start=azim_start, azim_range=azim_range
    )  # [1, N, D_rot]

    # 如果你的序列展平顺序不是行优先（i*W+j），提供 index_map 做对齐
    if index_map is not None:
        # index_map: [N]，把行优先的 [0..N-1] 重新排列成你实际的 token 顺序
        cos = cos[:, index_map, :]
        sin = sin[:, index_map, :]

    q1, q2 = q[..., :D_rot], q[..., D_rot:]
    q1 = q1 * cos + _rotate_half(q1) * sin

    if (k is not None):
        k1, k2 = k[..., :D_rot], k[..., D_rot:]
        k1 = k1 * cos + _rotate_half(k1) * sin

        return torch.cat([q1, q2], dim=-1), torch.cat([k1, k2], dim=-1)

    return torch.cat([q1, q2], dim=-1)
# ---- RoPE ----

# ---- ScaleLong ----
def universal_scalling(s_feat, s_factor=2 ** (-0.5)):
    return s_feat * s_factor


def exponentially_scalling(s_feat, k=0.8, i=1):
    return s_feat * k ** (i - 1)
# ---- ScaleLong ----

# ---- Normalization ----
class RMSNorm(nn.Module):
    """对 [B, C, H, W] 在通道维做 RMS 归一化"""

    def __init__(
            self,
            channels: int = 64,
            eps: float = 1e-6,
            affine: bool = True
    ):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        # rms over channel dim
        rms = x.pow(2).mean(dim=1, keepdim=True).add(self.eps).sqrt()
        y = x / rms
        if self.affine:
            y = y * self.weight + self.bias
        return y

def Normalize(in_channels, num_groups=None):
    if (num_groups is None):
        num_groups = in_channels // GROUP_NORM_NUM_CHANNELS
        if(num_groups <= 0):
            num_groups = 1
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
# ---- Normalization ----

# ---- Attention ----
class Attention(nn.Module):
    def __init__(
            self,
            q_dim=64,
            kv_dim=None,
            heads=8,
            dim_head=64,
            norm_type="ln",
            use_norm=True,
            use_res_connection=True,
            use_rope: bool = True,
            rope_base: float = 10000.0,
            use_zero_weight=True,
            gate=False
    ):
        super().__init__()

        self.use_res_connection = use_res_connection
        self.use_rope = use_rope
        self.rope_cache = RoPECacheTheta(base=rope_base) if use_rope else None

        if (kv_dim is None):
            kv_dim = q_dim

        self.heads = 1
        if (heads is not None):
            assert q_dim % heads == 0 and kv_dim % heads == 0, f"dim : {q_dim} % heads : {heads} != 0 or kv_dim : {kv_dim} % heads : {heads} != 0"
            self.heads = heads
            self.dim_head = q_dim // heads
        else:
            assert q_dim % dim_head == 0 and kv_dim % dim_head == 0, f"dim : {q_dim} % dim_head : {dim_head} != 0 or kv_dim : {kv_dim} % dim_head : {dim_head} != 0"
            self.heads = q_dim // dim_head
            self.dim_head = dim_head

        self.scale = self.dim_head ** -0.5
        self.heads = heads

        self.use_norm = use_norm
        if (self.use_norm):
            if(gate):
                self.norm_q = make_norm(norm_type=norm_type, ch=self.dim_head)
                self.norm_kv = make_norm(norm_type=norm_type, ch=self.dim_head)
            else:
                self.norm_q = make_norm(norm_type=norm_type, ch=q_dim)
                self.norm_kv = make_norm(norm_type=norm_type, ch=kv_dim)

        self.gate = gate
        if(gate == "head"):
            self.to_q = nn.Linear(q_dim, q_dim + self.heads, bias=False)
        elif(gate == "element"):
            self.to_q = nn.Linear(q_dim, q_dim * 2, bias=False)
        else:
            self.to_q = nn.Linear(q_dim, q_dim, bias=False)
        self.to_kv = nn.Linear(kv_dim, q_dim * 2, bias=False)
        self.to_out = nn.Linear(q_dim, q_dim, bias=False)

        if use_zero_weight:
            nn.init.zeros_(self.to_out.weight)
            if self.to_out.bias is not None:
                nn.init.zeros_(self.to_out.bias)

    def forward(self, x, kv=None):
        '''
        :param x: [batch_size, C, H, W]
        '''
        q = x

        if (len(x.shape) == 4):
            b, c, *spatial = x.shape
            q = q.permute(0, 2, 3, 1).reshape(b, -1, c)  # [b, n, c]

        if (kv is not None and len(kv.shape) == 2):
            B, C = kv.shape
            kv = kv.view(B, 1, C)
        elif (kv is None):
            kv = q

        h = self.heads

        # pre-norm
        if (self.use_norm and not self.gate):
            q = self.norm_q(q)
            kv = self.norm_kv(kv)

        q = self.to_q(q)
        k, v = self.to_kv(kv).chunk(2, dim=-1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        b,n,c = q.shape

        if(self.gate == "head"):
            query_states = q.view(b, n, h, -1) # [B, N, H, C+1]
            q, gate_score = torch.split(query_states,[
                self.dim_head,
                1
            ], dim=-1)
            # gate_score = gate_score.squeeze() # [B, N, H]
            gate_score = rearrange(gate_score, 'b n h d -> (b h) n d', h=h) # [B*H, N, 1]
            q = rearrange(q,'b n h d -> (b h) n d', h=h)
        elif(self.gate == "element"):
            query_states = q.view(b, n, h, -1)
            q, gate_score = torch.split(query_states, [
                self.dim_head,
                self.dim_head
            ], dim=-1)
            gate_score = rearrange(gate_score, 'b n h d -> (b h) n d', h=h) # [B*H, N, 1]
            q = rearrange(q,'b n h d -> (b h) n d', h=h)
        else:
            q = rearrange(q,'b n (h d) -> (b h) n d', h=h)

        k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (k, v))

        if (self.use_norm and self.gate):
            q = self.norm_q(q)
            k = self.norm_kv(k)

        # ===== RoPE（θ方向）=====
        if self.use_rope:
            _, _, H, W = x.shape
            if (q.shape == k.shape):
                q, k = apply_rope_theta_seq(
                    q=q, k=k, H=H, W=W,
                    rope_cache=self.rope_cache,
                    rotary_dim=64,  # 只旋转前 64 维，常见更稳
                    azim_start=0.0,  # 与投影起始角一致
                    azim_range=2 * math.pi,  # 与水平覆盖角一致（若裁剪/子FOV请对应修改）
                    index_map=None  # 若你的 N 排列不是行优先，传一个 [N] 的重排索引
                )
            else:
                q = apply_rope_theta_seq(
                    q=q, k=None, H=H, W=W,
                    rope_cache=self.rope_cache,
                    rotary_dim=64,  # 只旋转前 64 维，常见更稳
                    azim_start=0.0,  # 与投影起始角一致
                    azim_range=2 * math.pi,  # 与水平覆盖角一致（若裁剪/子FOV请对应修改）
                    index_map=None  # 若你的 N 排列不是行优先，传一个 [N] 的重排索引
                )

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)

        if(self.gate):
            out = out * torch.sigmoid(gate_score)

        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out = self.to_out(out)

        if (len(x.shape) == 4):
            b, c, *spatial = x.shape
            out = out.reshape(b, *spatial, c).permute(0, 3, 1, 2)

        if (self.use_res_connection):
            return x + out
        else:
            return out

class LinearAttention(nn.Module):
    def __init__(
            self,
            q_dim=64,
            kv_dim=None,
            heads=8,
            dim_head=8,
            norm_type="gn",
            norm_order="post-norm", # pre-norm
            use_norm=True,
            use_res_connection=True,
            use_rope: bool = True,
            rope_base: float = 10000.0,
            use_zero_weight=True,
            gate=False
    ):
        super().__init__()

        self.use_res_connection = use_res_connection
        self.use_rope = use_rope
        self.rope_cache = RoPECacheTheta(base=rope_base) if use_rope else None

        if kv_dim is None:
            kv_dim = q_dim
        self.heads = 1
        if heads is not None:
            self.heads = heads
            dim_head = q_dim // heads
        else:
            self.heads = q_dim // dim_head
        self.dim_head = dim_head

        self.scale = self.dim_head ** -0.5

        self.use_norm = use_norm
        self.norm_order = norm_order
        if self.use_norm:

            if(gate):
                self.norm_q = make_norm(norm_type=norm_type, ch=dim_head)
                self.norm_kv = make_norm(norm_type=norm_type, ch=dim_head)
            else:
                if(norm_order == "pre-norm"):
                    self.norm_q = make_norm(norm_type=norm_type, ch=q_dim)
                    self.norm_kv = make_norm(norm_type=norm_type, ch=q_dim)
                elif(norm_order == "post-norm"):
                    self.norm_q = make_norm(norm_type=norm_type, ch=q_dim)
                    self.norm_kv = make_norm(norm_type=norm_type, ch=q_dim*2)


        self.gate = gate
        if(gate == "head"):
            self.to_q = nn.Conv2d(q_dim, q_dim + heads, 1, bias=False)
        elif(gate == "element"):
            self.to_q = nn.Conv2d(q_dim, q_dim * 2, 1, bias=False)
        else:
            self.to_q = nn.Conv2d(q_dim, q_dim, 1, bias=False)
        self.to_kv = nn.Conv2d(kv_dim, q_dim * 2, 1, bias=False)
        self.to_out = nn.Conv2d(q_dim, q_dim, 1, bias=False)

        if use_zero_weight:
            nn.init.zeros_(self.to_out.weight)
            if self.to_out.bias is not None:
                nn.init.zeros_(self.to_out.bias)

    def forward(self, x, kv=None):
        """
        x  : [B, C, H, W] —— query 源
        kv : [B, C, H, W] 或 None —— key/value 源（None 则 self-attn）
        """
        q = x
        B, C, H, W = q.shape

        if kv is None:
            kv = q

        # pre norm
        if(self.use_norm and self.norm_order == "pre-norm" and not self.gate):
            q = self.norm_q(q)
            kv = self.norm_kv(kv)

        q = self.to_q(q)
        kv = self.to_kv(kv)

        # post norm
        if(self.use_norm and self.norm_order == "post-norm" and not self.gate):
            q = self.norm_q(q)
            kv = self.norm_kv(kv)

        # 投影
        if(self.gate == "head"):
            query_states = q.view(B, H, W, self.heads, -1) # [B, H, W, heads, C+1]
            q, gate_score = torch.split(query_states,[
                self.dim_head,
                1
            ], dim=-1)
            gate_score = rearrange(gate_score, 'b hgt wdt h 1 -> b h 1 (hgt wdt)', h=self.heads) # [B*H, N, 1]
            q = rearrange(q, 'b hgt wdt h c -> b h c (hgt wdt)', h=self.heads)

        elif(self.gate == "element"):
            query_states = q.view(B, H, W, self.heads, -1) # [B, H, W, heads, C+1]
            q, gate_score = torch.split(query_states,[
                self.dim_head,
                self.dim_head
            ], dim=-1)
            gate_score = rearrange(gate_score, 'b hgt wdt h c -> b h c (hgt wdt)', h=self.heads) # [B*H, N, 1]
            q = rearrange(q, 'b hgt wdt h c -> b h c (hgt wdt)', h=self.heads)
        else:
            q = rearrange(q, 'b (h c) hgt wdt -> b h c (hgt wdt)', h=self.heads)  # [B,h,Cd,N]
        q = q * (self.scale if (LINEAR_ATTENTION_SCALE) else 1.0)

        k, v = rearrange(kv, 'b (kv h c) hgt wdt -> kv b h c (hgt wdt)', kv=2, h=self.heads)

        if(self.use_norm and self.gate):
            # q : [B, H, C, N]
            q = rearrange(q, 'b h c n -> (b h) c n')
            k = rearrange(k, 'b h c n -> (b h) c n')

            q = self.norm_q(q)  # GroupNorm(ch=dim_head)
            k = self.norm_kv(k)

            q = rearrange(q, '(b h) c n -> b h c n', h=self.heads)
            k = rearrange(k, '(b h) c n -> b h c n', h=self.heads)

        # ===== RoPE（θ方向）=====
        if self.use_rope:
            qn = rearrange(q, 'b h c n -> (b h) c n', h=self.heads)
            qn = qn.transpose(2, 1).float()

            kn = rearrange(k, 'b h c n -> (b h) c n', h=self.heads)
            kn = kn.transpose(2, 1).float()

            if (qn.shape == kn.shape):
                qn, kn = apply_rope_theta_seq(
                    qn, kn, H=H, W=W, rope_cache=self.rope_cache,
                    rotary_dim=64,  # 只旋转前 64 维，常见更稳
                    azim_start=0.0,  # 与投影起始角一致
                    azim_range=2 * math.pi,  # 与水平覆盖角一致（若裁剪/子FOV请对应修改）
                    index_map=None  # 若你的 N 排列不是行优先，传一个 [N] 的重排索引
                )
            else:
                qn = apply_rope_theta_seq(
                    qn, None, H=H, W=W, rope_cache=self.rope_cache,
                    rotary_dim=64,  # 只旋转前 64 维，常见更稳
                    azim_start=0.0,  # 与投影起始角一致
                    azim_range=2 * math.pi,  # 与水平覆盖角一致（若裁剪/子FOV请对应修改）
                    index_map=None  # 若你的 N 排列不是行优先，传一个 [N] 的重排索引
                )

                kn = apply_rope_theta_seq(
                    kn, None, H=H, W=W, rope_cache=self.rope_cache,
                    rotary_dim=64,  # 只旋转前 64 维，常见更稳
                    azim_start=0.0,  # 与投影起始角一致
                    azim_range=2 * math.pi,  # 与水平覆盖角一致（若裁剪/子FOV请对应修改）
                    index_map=None  # 若你的 N 排列不是行优先，传一个 [N] 的重排索引
                )

            qn = qn.transpose(2, 1)
            q = rearrange(qn, '(b h) c n  -> b h c n', h=self.heads).to(q.dtype)

            kn = kn.transpose(2, 1)
            k = rearrange(kn, '(b h) c n  -> b h c n', h=self.heads).to(q.dtype)

        # 线性注意力（与你原实现一致）
        k = k.softmax(dim=-1)  # over spatial N
        context = torch.einsum('bhdn,bhen->bhde', k, v)  # Σ_n k(n)*v(n) (b,h,c,c)
        out = torch.einsum('bhde,bhdn->bhen', context, q)  # 乘回 query (b,h,c,c) * (b,h,c,n) -> (b,h,c,n)

        if(self.gate):
            out = out * torch.sigmoid(gate_score)

        # 还原回 [B,C,H,W]
        out = rearrange(out, 'b h c (hgt wdt) -> b (h c) hgt wdt', h=self.heads, hgt=H, wdt=W)
        out = self.to_out(out)

        if (self.use_res_connection):
            return x + out
        else:
            return out


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
            self,
            q_dim=64,
            kv_dim=None,
            heads=8,
            dim_head=8,
            norm_type="gn",
            use_new_attention_order=False,
            use_norm=True,
            use_res_connection=True,
            use_rope=True,
            rope_base: float = 10000.0,
            use_zero_weight=True,
            gate=False,
    ):
        super().__init__()

        self.channels = q_dim
        self.use_res_connection = use_res_connection
        self.use_rope = use_rope
        self.rope_cache = RoPECacheTheta(base=rope_base) if use_rope else None

        if (kv_dim is None):
            kv_dim = q_dim

        if (heads is not None):
            self.heads = heads
            self.dim_head = q_dim // heads
        else:
            self.heads = q_dim // dim_head
            self.dim_head = dim_head

        self.use_norm = use_norm
        if (self.use_norm):
            if(gate):
                self.norm_q = make_norm(norm_type=norm_type, ch=self.dim_head)
                self.norm_kv = make_norm(norm_type=norm_type, ch=self.dim_head)
            else:
                self.norm_q = make_norm(norm_type=norm_type, ch=q_dim)
                self.norm_kv = make_norm(norm_type=norm_type, ch=kv_dim)

        self.use_new_attention_order = use_new_attention_order

        self.gate = gate
        if(gate == "head"):
            self.to_q = conv_nd(1, q_dim, q_dim + self.heads, 1, bias=False)
        elif(gate == "element"):
            self.to_q = conv_nd(1, q_dim, q_dim * 2, 1, bias=False)
        else:
            self.to_q = conv_nd(1, q_dim, q_dim, 1, bias=False)
        self.to_kv = conv_nd(1, kv_dim, q_dim * 2, 1, bias=False)

        self.proj_out = conv_nd(1, q_dim, q_dim, 1, bias=False)

        if use_zero_weight:
            nn.init.zeros_(self.proj_out.weight)
            if self.proj_out.bias is not None:
                nn.init.zeros_(self.proj_out.bias)

    def qkv_attention_legacy(self, qkv, gate_score=None):
        bs, c_x3, n = qkv.shape
        assert c_x3 % (3 * self.heads) == 0
        ch = c_x3 // (3 * self.heads)
        q, k, v = qkv.reshape(bs * self.heads, ch * 3, n).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        if(gate_score is not None):
            a = a * gate_score
        return a.reshape(bs, -1, n)

    def qkv_attention(self, qkv, gate_score=None):
        bs, c_x3, n = qkv.shape
        assert c_x3 % (3 * self.heads) == 0
        ch = c_x3 // (3 * self.heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, n),
            (k * scale).view(bs * self.n_heads, ch, n),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.heads, ch, n))
        if(gate_score is not None):
            a = a * gate_score
        return a.reshape(bs, -1, n)

    def forward(self, q, kv=None):
        """
        q  : [b, c, *spatial]
        kv : [b, c, *spatial] or None
        """
        b, q_c, *q_spatial = q.shape

        if (kv is None):
            kv = q

        _, kv_c, *kv_spatial = kv.shape

        q = q.reshape(b, q_c, -1)  # [b, c, n]
        kv = kv.reshape(b, kv_c, -1)

        # pre-norm
        if (self.use_norm and not self.gate):
            q = self.norm_q(q)
            kv = self.norm_kv(kv)

        q_in = self.to_q(q)
        k_in, v_in = self.to_kv(kv).chunk(2, dim=1)

        gate_score = None
        if(self.gate == "head"):
            query_states = q_in.reshape(b, q_spatial[0] * q_spatial[1], self.heads, -1) # [B, N, H, C+1]
            q_in, gate_score = torch.split(query_states,[
                self.dim_head,
                1
            ], dim=-1)
            gate_score = rearrange(gate_score, 'b n h 1 -> (b h) 1 n', h=self.heads) # [B*H, N, 1]
            q_in = rearrange(q_in,'b n h c -> b (h c) n', h=self.heads)
        elif(self.gate == "element"):
            query_states = q_in.reshape(b, q_spatial[0] * q_spatial[1], self.heads, -1)
            q_in, gate_score = torch.split(query_states, [
                self.dim_head,
                self.dim_head
            ], dim=-1)
            gate_score = rearrange(gate_score, 'b n h c -> (b h) c n', h=self.heads) # [B*H, N, 1]
            q_in = rearrange(q_in,'b n h c -> b (h c) n', h=self.heads)

        if (self.use_norm and self.gate):
            q_in = q_in.reshape(b, self.heads, self.dim_head, -1)
            k_in = k_in.reshape(b, self.heads, self.dim_head, -1)

            q_in = q_in.reshape(b * self.heads, self.dim_head, -1)
            k_in = k_in.reshape(b * self.heads, self.dim_head, -1)

            q_in = self.norm_q(q_in)
            k_in = self.norm_kv(k_in)

            q_in = q_in.reshape(b, self.heads * self.dim_head, -1)
            k_in = k_in.reshape(b, self.heads * self.dim_head, -1)

        # ===== RoPE（θ方向）=====
        if self.use_rope:
            d_head = self.dim_head
            assert d_head % 2 == 0, "RoPE需要 head_dim 为偶数"

            # 改成 [B,h,N,d] 以匹配缓存
            qn = q_in.transpose(2, 1).float()  # [B,C,N] -> [B,N,C]
            qn = rearrange(qn, 'b n (h d) -> (b h) n d', h=self.heads)

            kn = k_in.transpose(2, 1).float()  # [B,C,N] -> [B,N,C]
            kn = rearrange(kn, 'b n (h d) -> (b h) n d', h=self.heads)

            if (qn.shape == kn.shape):
                qn, kn = apply_rope_theta_seq(
                    qn, kn, H=q_spatial[0], W=q_spatial[1], rope_cache=self.rope_cache,
                    rotary_dim=64,  # 只旋转前 64 维，常见更稳
                    azim_start=0.0,  # 与投影起始角一致
                    azim_range=2 * math.pi,  # 与水平覆盖角一致（若裁剪/子FOV请对应修改）
                    index_map=None  # 若你的 N 排列不是行优先，传一个 [N] 的重排索引
                )
            else:
                qn = apply_rope_theta_seq(
                    qn, None, H=q_spatial[0], W=q_spatial[1], rope_cache=self.rope_cache,
                    rotary_dim=64,  # 只旋转前 64 维，常见更稳
                    azim_start=0.0,  # 与投影起始角一致
                    azim_range=2 * math.pi,  # 与水平覆盖角一致（若裁剪/子FOV请对应修改）
                    index_map=None  # 若你的 N 排列不是行优先，传一个 [N] 的重排索引
                )
                kn = apply_rope_theta_seq(
                    kn, None, H=kv_spatial[0], W=kv_spatial[1], rope_cache=self.rope_cache,
                    rotary_dim=64,  # 只旋转前 64 维，常见更稳
                    azim_start=0.0,  # 与投影起始角一致
                    azim_range=2 * math.pi,  # 与水平覆盖角一致（若裁剪/子FOV请对应修改）
                    index_map=None  # 若你的 N 排列不是行优先，传一个 [N] 的重排索引
                )

            qn = rearrange(qn, '(b h) n d  -> b n (h d)', h=self.heads)
            q_in = qn.transpose(2, 1).to(q_in.dtype)

            kn = rearrange(kn, '(b h) n d  -> b n (h d)', h=self.heads)
            k_in = kn.transpose(2, 1).to(k_in.dtype)

        qkv = torch.cat([q_in, k_in, v_in], dim=1)

        if (self.use_new_attention_order):
            h = self.qkv_attention(qkv, gate_score)  # [b, c, n]
        else:
            h = self.qkv_attention_legacy(qkv, gate_score)

        h = self.proj_out(h)

        if (self.use_res_connection):
            return (q + h).reshape(b, q_c, *q_spatial)
        else:
            return h.reshape(b, q_c, *q_spatial)
# ---- Attention ----


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)

def get_timestep_embedding(timesteps, embedding_dim):
    """
        This matches the implementation in Denoising Diffusion Probabilistic Models:
        From Fairseq.
        Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        timesteps : [B]
        embedding_dim: dim
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class Conv2DSiLU(nn.Module):
    def __init__(
            self,
            *,
            in_channels,  # 64
            out_channels=None,  # 64
            kernel_size=3,
            stride=1,
            temb_channels=512,  # 0
    ):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels

        self.conv = torch.nn.Conv2d(in_channels,  # 64
                                     out_channels,  # 64
                                     kernel_size=kernel_size,  # (3,3)
                                     stride=stride,  # 1
                                     padding=1 if kernel_size == 3 else 0)  # (1, 1, 1, 1)

        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)


    def forward(self, x, temb=None):
        h = x

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        return nonlinearity(self.conv(h))


class ResnetBlock(nn.Module):
    def __init__(
            self,
            *,
            in_channels,  # 64
            out_channels=None,  # 64
            kernel_size=(3, 3),  # (3, 3)
            dropout=0.0,  # 0.0
            temb_channels=512,  # 0
            use_pe=False,  #
            use_zero_weight=RESBLOCK_ZEOR_WEIGHT,
            use_circularconv_shortcut=RESBLOCK_CIRCULARCONV_SHORTCUT,  # False
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_circularconv_shortcut = use_circularconv_shortcut
        pad = UNIFORM_KERNEL2PAD_DICT[kernel_size]  # (1, 1, 1, 1)

        self.norm1 = Normalize(in_channels)  # GroupNorm, 32 -> 64
        self.conv1 = CircularConv2D(in_channels,  # 64
                                    out_channels,  # 64
                                    kernel_size=kernel_size,  # (3,3)
                                    stride=1,  # 1
                                    padding=pad)  # (1, 1, 1, 1)

        self.pemb_proj = None
        self.use_pe = use_pe
        if self.use_pe:
            self.pemb_conv = CircularConv2D(out_channels,  # 64
                                            out_channels,  # 64
                                            kernel_size=kernel_size,  # (3,3)
                                            stride=1,  # 1
                                            padding=pad)
            self.pemb_mlp = nn.Linear(out_channels, out_channels)
            self.pemb_norm = Normalize(out_channels)

        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)

        self.norm2 = Normalize(out_channels)  # GroupNorm, 32 -> 64
        self.dropout = torch.nn.Dropout(dropout)  # 0.0
        self.conv2 = CircularConv2D(out_channels,  # 64
                                    out_channels,  # 64
                                    kernel_size=kernel_size,  # (3,3)
                                    stride=1,  # 1
                                    padding=pad)  # (1, 1, 1, 1)

        if (use_zero_weight):
            nn.init.zeros_(self.conv2.weight)
            if self.conv2.bias is not None:
                nn.init.zeros_(self.conv2.bias)

        if self.in_channels != self.out_channels:
            if self.use_circularconv_shortcut:
                self.conv_shortcut = CircularConv2D(in_channels,
                                                    out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=1,
                                                    padding=pad)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)  # Group Norm
        h = nonlinearity(h)  # x * torch.sigmoid(x)
        h = self.conv1(h)  # 第一次卷积

        if self.use_pe:
            pe = self.pemb_conv(h)
            pe = common.trans_mlp(pe, self.pemb_mlp)
            pe = self.pemb_norm(pe)
            h = h + pe

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)  # Group Norm
        h = nonlinearity(h)  # x * torch.sigmoid(x)
        h = self.dropout(h)  # 0.0
        h = self.conv2(h)  # 第二次卷积

        if self.in_channels != self.out_channels:
            if self.use_circularconv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h  # 残差连接

class ResnetConv2DBlock(nn.Module):
    def __init__(
            self,
            *,
            in_channels,  # 64
            out_channels=None,  # 64
            kernel_size=3,  # (3, 3)
            dropout=0.0,  # 0.0
            temb_channels=512,  # 0
            use_pe=False,  #
            use_zero_weight=RESBLOCK_ZEOR_WEIGHT,
            use_circularconv_shortcut=RESBLOCK_CIRCULARCONV_SHORTCUT,  # False
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_circularconv_shortcut = use_circularconv_shortcut

        self.norm1 = Normalize(in_channels)  # GroupNorm, 32 -> 64
        self.conv1 = torch.nn.Conv2d(in_channels,  # 64
                                    out_channels,  # 64
                                    kernel_size=kernel_size,  # (3,3)
                                    stride=1,  # 1
                                    padding=1 if kernel_size == 3 else 0)  # (1, 1, 1, 1)

        self.pemb_proj = None
        self.use_pe = use_pe
        if self.use_pe:
            self.pemb_conv = torch.nn.Conv2d(out_channels,  # 64
                                            out_channels,  # 64
                                            kernel_size=kernel_size,  # (3,3)
                                            stride=1,  # 1
                                            padding=1 if kernel_size == 3 else 0)
            self.pemb_mlp = nn.Linear(out_channels, out_channels)
            self.pemb_norm = Normalize(out_channels)

        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)

        self.norm2 = Normalize(out_channels)  # GroupNorm, 32 -> 64
        self.dropout = torch.nn.Dropout(dropout)  # 0.0
        self.conv2 = torch.nn.Conv2d(out_channels,  # 64
                                    out_channels,  # 64
                                    kernel_size=kernel_size,  # (3,3)
                                    stride=1,  # 1
                                    padding=1 if kernel_size == 3 else 0)  # (1, 1, 1, 1)

        if (use_zero_weight):
            nn.init.zeros_(self.conv2.weight)
            if self.conv2.bias is not None:
                nn.init.zeros_(self.conv2.bias)

        if self.in_channels != self.out_channels:
            if self.use_circularconv_shortcut:
                pad = UNIFORM_KERNEL2PAD_DICT[(3, 3)]  # (1, 1, 1, 1)
                self.conv_shortcut = CircularConv2D(in_channels,
                                                    out_channels,
                                                    kernel_size=(3,3),
                                                    stride=1,
                                                    padding=pad)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)  # Group Norm
        h = nonlinearity(h)  # x * torch.sigmoid(x)
        h = self.conv1(h)  # 第一次卷积

        if self.use_pe:
            pe = self.pemb_conv(h)
            pe = common.trans_mlp(pe, self.pemb_mlp)
            pe = self.pemb_norm(pe)
            h = h + pe

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)  # Group Norm
        h = nonlinearity(h)  # x * torch.sigmoid(x)
        h = self.dropout(h)  # 0.0
        h = self.conv2(h)  # 第二次卷积

        if self.in_channels != self.out_channels:
            if self.use_circularconv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h  # 残差连接

class ZeroConv2d(nn.Module):
    """
    1×1 convolution with both weight and bias initialized to zero.
    用于 ControlNet / Adapter 的零卷积。
    """

    def __init__(self, channels):
        super().__init__()
        # 定义 1×1 卷积
        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        # 初始化为 0
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)

class CircularConv2D(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        if 'padding' in kwargs:
            self.is_pad = True
            if isinstance(kwargs['padding'], int):
                h1 = h2 = v1 = v2 = kwargs['padding']
            elif isinstance(kwargs['padding'], tuple):
                h1, h2, v1, v2 = kwargs['padding']
            else:
                raise NotImplementedError
            self.h_pad, self.v_pad = (h1, h2, 0, 0), (0, 0, v1, v2)
            del kwargs['padding']
        else:
            self.is_pad = False

        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        if self.is_pad:
            if sum(self.h_pad) > 0:
                x = nn.functional.pad(x, self.h_pad, mode="circular")  # horizontal pad
            if sum(self.v_pad) > 0:
                x = nn.functional.pad(x, self.v_pad, mode="constant")  # vertical pad
        x = self._conv_forward(x, self.weight, self.bias)
        return x

class Downsample(nn.Module):
    def __init__(
            self,
            in_channels,
            stride,
            out_channels=-1,
            with_conv=False
    ):
        super().__init__()
        self.with_conv = with_conv
        self.stride = stride
        if(out_channels == -1):
            out_channels = in_channels
        if(self.with_conv == "CircluarConv2D"):
            k, p = DOWNSAMPLE_STRIDE2KERNEL_DICT[stride], DOWNSAMPLE_STRIDE2PAD_DICT[stride]
            self.conv = CircularConv2D(in_channels, out_channels, kernel_size=k, stride=stride, padding=p)
        elif(self.with_conv == "Conv2D"):
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=stride,
                stride=stride,
                padding=0
            )

    def forward(self, x):
        if self.with_conv:
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
        return x

class Upsample(nn.Module):
    def __init__(
            self,
            in_channels,
            stride,
            with_conv=False, # CircularConv2D, Conv2d, False
            out_channels=-1
    ):
        super().__init__()
        self.with_conv = with_conv
        self.stride = stride
        if(out_channels == -1):
            out_channels = in_channels
        if(self.with_conv == "CircularConv2D"):
            k, p = UPSAMPLE_STRIDE2KERNEL_DICT[stride], UPSAMPLE_STRIDE2PAD_DICT[stride]
            self.conv = CircularConv2D(in_channels, out_channels, kernel_size=k, padding=p)
        elif(self.with_conv == "Conv2D"):
            self.conv = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=self.stride, mode='bilinear', align_corners=True)
        if self.with_conv:
            x = self.conv(x)
        return x

'''
    Only using to draw the 9th figure in paper.
'''
# def get_encoder_deocder_gn_draw():
#
#
#     encoder = [
#         [
#             ["Conv2D"],                   # 0     2   -> 64
#             ["Downsample"]                # 1     64  -> 64
#         ],
#         [
#             ["Conv2DSiLU", "Downsample"]  # 2     128 -> 128
#         ],
#         [
#             ["Conv2DSiLU", "Downsample"]  # 3     256 -> 256
#         ],
#         [
#             ["Conv2DSiLU"]                # 4     256 -> 256
#         ]
#     ]
#
#
#
#     deocder = [
#         [
#             ["Conv2DSiLU"],                # 0     256 + 256 -> 256
#             ["Upsample"],                  # 1     256 + 256 -> 256
#         ],
#         [
#             ["Conv2DSiLU","Upsample"],     # 2     128 + 256 -> 256
#         ],
#         [
#             ["Conv2DSiLU", "Upsample"],    # 3     64  + 128 -> 128
#         ],
#         [
#             ["Conv2DSiLU"]                 # 4     64  + 128 -> 64
#         ]
#     ]
#
#
#     return encoder, deocder

'''
    Only using to draw the 9th figure in paper.
'''
# def get_encoder_deocder_dn_draw():
#
#     encoder = [
#         [
#             ["CircularConv2D"],                   # 0     2   -> 64
#             ["ResCircluarBlock", "LAB"],          # 1     64  -> 64
#             ["ResCircluarBlock", "LAB"],          # 2     64  -> 64
#             ["Downsample"]                        # 3     64  -> 64
#         ],
#         [
#             ["ResCircluarBlock", "LAB"],          # 4     64  -> 128
#             ["ResCircluarBlock", "LAB"],          # 5     128 -> 128
#             ["Downsample"]                        # 6     128 -> 128
#         ],
#         [
#             ["ResCircluarBlock", "LAB"],          # 7     128 -> 256
#             ["ResCircluarBlock", "LAB"],          # 8     256 -> 256
#             ["Downsample"]                        # 9     256 -> 256
#         ],
#         [
#             ["ResCircluarBlock"],                 # 10    256 -> 256
#             ["ResCircluarBlock"]                  # 11    256 -> 256
#         ]
#     ]
#
#     deocder = [
#         [
#             ["ResBlock"],                         # 0     256 + 256 -> 256
#             ["ResBlock"],                         # 1     256 + 256 -> 256
#             ["ResBlock", "Upsample"]              # 2     256 + 256 -> 256
#         ],
#         [
#             ["ResBlock", "LAB"],                  # 3     256 + 256 -> 256
#             ["ResBlock", "LAB"],                  # 4     256 + 256 -> 256
#             ["ResBlock", "LAB", "Upsample"]       # 5     128 + 256 -> 256
#         ],
#         [
#             ["ResBlock", "LAB"],                  # 6     128 + 256 -> 128
#             ["ResBlock", "LAB"],                  # 7     128 + 128 -> 128
#             ["ResBlock", "LAB", "Upsample"]       # 8     64  + 128 -> 128
#         ],
#         [
#             ["ResBlock", "LAB"],                  # 9     64  + 128 -> 64
#             ["ResBlock", "LAB"],                  # 10    64  + 64  -> 64
#             ["ResBlock", "LAB"]                   # 11    64  + 64  -> 64
#         ]
#     ]
#
#
#     middle = [ [          ["ResCircluarBlock"],               ["TAB"],               ["ResCircluarBlock"],          ] ]
#                         # 0 256 -> 512                     # 1 512 -> 512             # 2 512 -> 256
#
#     return encoder, deocder, middle


def get_encoder_deocder_gn():
    encoder = [
        [
            ["Conv2D"],
            ["Downsample"]
        ],
        [
            ["Conv2DSiLU", "Downsample"]
        ],
        [
            ["Conv2DSiLU", "Downsample"]
        ],
        [
            ["Conv2DSiLU"]
        ]
    ]

    deocder = [
        [
            ["Conv2DSiLU"],
            ["Upsample"],
        ],
        [
            ["Conv2DSiLU","Upsample"],
        ],
        [
            ["Conv2DSiLU", "Upsample"],
        ],
        [
            ["Conv2DSiLU"]
        ]
    ]

    return encoder, deocder

def get_encoder_deocder_dn():
    encoder = [
        [
            ["CircularConv2D"],                         # 0     2   -> 64
            ["ResBlock", "Attention"],                  # 1     64  -> 64
            ["ResBlock", "Attention"],                  # 2     64  -> 64
            ["Downsample"]                              # 3     64  -> 64
        ],
        [
            ["ResBlock", "Attention"],                  # 4     64  -> 128
            ["ResBlock", "Attention"],                  # 5     128 -> 128
            ["Downsample"]                              # 6     128 -> 128
        ],
        [
            ["ResBlock", "Attention"],                  # 7     128 -> 256
            ["ResBlock", "Attention"],                  # 8     256 -> 256
            ["Downsample"]                              # 9     256 -> 256
        ],
        [
            ["ResBlock"],                               # 10    256 -> 256
            ["ResBlock"]                                # 11    256 -> 256
        ]
    ]

    deocder = [
        [
            ["ResConv2DBlock"],                         # 0     256 + 256 -> 256
            ["ResConv2DBlock"],                         # 1     256 + 256 -> 256
            ["ResConv2DBlock", "Upsample"]              # 2     256 + 256 -> 256
        ],
        [
            ["ResConv2DBlock", "Attention"],            # 3     256 + 256 -> 256
            ["ResConv2DBlock", "Attention"],            # 4     256 + 256 -> 256
            ["ResConv2DBlock", "Attention", "Upsample"] # 5     128 + 256 -> 256
        ],
        [
            ["ResConv2DBlock", "Attention"],            # 6     128 + 256 -> 128
            ["ResConv2DBlock", "Attention"],            # 7     128 + 128 -> 128
            ["ResConv2DBlock", "Attention", "Upsample"] # 8     64  + 128 -> 128
        ],
        [
            ["ResConv2DBlock", "Attention"],            # 9     64  + 128 -> 64
            ["ResConv2DBlock", "Attention"],            # 10    64  + 64  -> 64
            ["ResConv2DBlock", "Attention"]             # 11    64  + 64  -> 64
        ]
    ]

    return encoder, deocder


class CircularUNet(nn.Module):
    # ---- DDPM UNet ----
    def __init__(
            self,
            *,                                                                  # 强制参数名传入
            in_channel=2,                                                       # 1
            out_channel=2,                                                      # 1
            control_channel=1,                                                  # 1

            n_base_channel=64,                                                  # 64
            n_channels=[1, 2, 4, 4],                                            # [1,2,4,8]
            n_strides=[[1, 2], [2, 2], [2, 2]],                                 # [[1, 2], [2, 2], [2, 2]]
            n_attn_types=["linear", "linear", "linear", "vanilla"],             # ["linear","linear","linear","vanilla"]
            n_norm_types=["gn", "gn", "gn", "gn"],                              # ["rmsn", "gn", "gn", "gn"]
            n_use_norm=[True, True, True, True],                                # [True, True, True, True]
            n_use_res_connection=[True, True, True, True],                      # [False, False, False, False]
            n_heads=[2, 4, 8, 8],                                               # [8, 8, 8, 8]

            n_midd_attn_type='vanilla',                                         # 'vanilla'
            n_midd_norm_type='gn',                                              # 'gn'
            n_midd_use_norm=True,                                               # 'gn'
            n_midd_use_res_connection=True,                                     # True
            n_midd_heads=8,                                                     # 8
            n_midd_channels=[4, 8, 4],                                          # True

            g_base_channel=64,                                                  # 32
            g_channels=[1, 2, 4, 4],                                            # [1,2,4,8]
            g_strides=[[1, 2], [2, 2], [2, 2]],                                 # [[1, 2], [2, 2], [2, 2]]
            g_attn_types=["linear", "linear", "linear", "vanilla"],             # ["linear","linear","linear","vanilla"]
            g_norm_types=["gn", "gn", "gn", "gn"],                              # ["rmsn", "gn", "gn", "gn"]
            g_use_norm=[True, True, True, True],                                # [True, True, True, True]
            g_use_res_connection=[True, True, True, True],                      # [False, False, False, False]
            g_heads=[2, 4, 8, 8],                                               # [8, 8, 8, 8]

            g_midd_attn_type='vanilla',                                         # 'vanilla'
            g_midd_norm_type='gn',                                              # 'gn'
            g_midd_use_norm=True,                                               # 'gn'
            g_midd_use_res_connection=True,                                     # True
            g_midd_heads=8,                                                     # 8
            g_midd_channels=[4, 8, 4],                                          # True

            dropout=0.0,                                                        # 0.0
            skip_connection_scale="sqrt(2)",                                    # equal, sqrt(2), scalelong
            freeze_guidence_net=False,
            attention_gate=False,

            resolution=[64, 1024],                                              # Range Map 大小
            fov=[3, -25],                                                       # Range Map 垂直视域

            text_channels=768,                                                  # text特征的维度
            t_channels=384,                                                     # t编码的维度，0意味着不使用t编码

            use_pe=False,                                                       # 是否使用位置编码
            use_rope=True,                                                      # 是否在attention中使用rope
            use_dpe_n=True,                                                     # DN是否使用DPE
            use_dpe_g=False,                                                    # GN是否使用DPE
            use_zero_weight=True,                                               # 是否在ResBlock和AttentionBlock中残差连接使用0初始化
            use_circularconv_shortcut=True,                                     # 是否在ResBlock中使用circularconv转换维度，否则使用conv2d

            use_guidence_net=False,                                             # 是否使用conditiona net (GN)
            use_multistage_feats=False,                                         # 是否GN与DN在多Stage交互(默认在Middle层交互)
            use_control_net=False,                                              # 是否使用control net (Non-Latent ControlNet)

            use_text=False,
            use_image_encoder=False,

            print_chnanels=True
    ):
        super().__init__()

        self.base_channel = n_base_channel
        self.in_channels = in_channel
        self.resolution = resolution
        self.skip_connection_scale = skip_connection_scale
        self.freeze_guidence_net = freeze_guidence_net
        self.use_text = use_text
        self.use_guidence_net = use_guidence_net
        self.use_multistage_feats = use_multistage_feats
        self.use_control_net = use_control_net
        self.midd_channels = n_midd_channels

        n_channels = list(n_channels)
        n_strides = list(n_strides)
        n_heads = list(n_heads)
        n_attn_types = list(n_attn_types)
        n_norm_types = list(n_norm_types)
        n_use_norm = list(n_use_norm)
        n_use_res_connection = list(n_use_res_connection)

        g_channels = list(g_channels)
        g_strides = list(g_strides)
        g_heads = list(g_heads)
        g_attn_types = list(g_attn_types)
        g_norm_types = list(g_norm_types)
        g_use_norm = list(g_use_norm)
        g_use_res_connection = list(g_use_res_connection)

        # ---- T embeding ----
        self.temb_ch = t_channels
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.base_channel, self.temb_ch),
            torch.nn.Linear(self.temb_ch, self.temb_ch)
        ])
        # ---- T embeding ----

        # ---- PE embeding ----
        self.pemb_ch = 0
        self.use_pe = use_pe
        if self.use_pe:
            self.pemb_ch = self.base_channel * 4
        # ---- PE embeding ----

        # ---- 获取encoder 和 decoder的框架 ----
        self.n_encoder_names, self.n_decoder_names = get_encoder_deocder_dn()
        if(self.use_guidence_net):
            self.g_encoder_names, self.g_decoder_names = get_encoder_deocder_gn()
        # ---- 获取encoder 和 decoder的框架 ----

        # ---- downsampling (曲面卷积，维度64) ----
        # ---- denoising encoder ----
        # ---- 获取encoder的输入维度的输出维度 ----
        n_encoder_block_ins = []
        for i_stage in range(len(self.n_encoder_names)):
            level_num = len(self.n_encoder_names[i_stage])
            for i_module in range(level_num):

                block_in = n_channels[i_stage] * n_base_channel  # stage中，第2个模块开始
                if (i_stage == 0 and i_module == 0):  # 初始输入通道
                    block_in = in_channel

                elif (i_module == 0):  # stage中，第1个模块
                    block_in = n_channels[i_stage - 1] * n_base_channel
                n_encoder_block_ins.append(block_in)
        n_encoder_block_outs = copy.deepcopy(n_encoder_block_ins)
        n_encoder_block_outs.append(n_channels[-1] * n_base_channel)
        n_encoder_block_outs = n_encoder_block_outs[1:]

        if(print_chnanels):
            print(f"--- denosing encoder block_ins : {n_encoder_block_ins}")
            print(f"--- denosing encoder block_outs : {n_encoder_block_outs}")

        self.encoder_stage_channels_for_control = n_encoder_block_outs
        # ---- 获取encoder的输入维度的输出维度 ----

        self.n_encoder_stage_channels_for_guidence = []
        n_encoder_stage_channels_for_DPE = []
        i_encoder = 0
        i_down = 0

        self.n_encoder = nn.ModuleList()
        for i_stage in range(len(self.n_encoder_names)):
            block_out = n_encoder_block_outs[i_encoder]
            level_num = len(self.n_encoder_names[i_stage])
            for i_module in range(level_num):
                block_in = n_encoder_block_ins[i_encoder]
                moudle_num = len(self.n_encoder_names[i_stage][i_module])
                down = nn.Module()
                for i_name in range(moudle_num):
                    name = self.n_encoder_names[i_stage][i_module][i_name]
                    if (name == "Conv2D"):
                        down.Conv2D = get_module(
                            name=name,
                            in_channels=block_in,
                            out_channels=block_out,
                            kernel_size=3,
                            stride=1,
                            padding=1
                        )
                        n_encoder_stage_channels_for_DPE.append(block_out)
                    elif (name == "CircularConv2D"):
                        down.CircularConv2D = get_module(
                            name=name,
                            in_channels=block_in,
                            out_channels=block_out,
                            kernel_size=3,
                            stride=1,
                            padding=1
                        )
                        n_encoder_stage_channels_for_DPE.append(block_out)
                    elif (name == "Conv2DSiLU"):
                        down.Conv2DSiLU = get_module(
                            name=name,
                            in_channels=block_in,
                            out_channels=block_out,
                            kernel_size=3,
                            stride=1,
                            temb_channels=self.temb_ch,
                        )
                        n_encoder_stage_channels_for_DPE.append(block_out)
                    elif (name == "ResBlock"):
                        down.ResBlock = get_module(
                            name=name,
                            in_channels=block_in,
                            out_channels=block_out,
                            temb_channels=self.temb_ch,
                            use_pe=self.use_pe,
                            dropout=dropout,
                            use_zero_weight=use_zero_weight,
                            use_circularconv_shortcut=True
                        )
                    elif (name == "ResConv2DBlock"):
                        down.ResConv2DBlock = get_module(
                            name=name,
                            in_channels=block_in,
                            out_channels=block_out,
                            temb_channels=self.temb_ch,
                            use_pe=self.use_pe,
                            dropout=dropout,
                            use_zero_weight=use_zero_weight,
                            use_circularconv_shortcut=False
                        )
                    elif (name == "Downsample"):
                        down.Downsample = get_module(
                            name=name,
                            in_channels=block_out,
                            out_channels=block_out,
                            with_conv="CircluarConv2D",
                            stride=tuple(n_strides[i_down])
                        )
                        n_encoder_stage_channels_for_DPE.append(block_out)
                        i_down = i_down + 1
                    elif (name == "Attention"):
                        down.Attention = get_module(
                            name=name,
                            q_dim=block_out,
                            kv_dim=text_channels if use_text else block_out,
                            heads=n_heads[i_stage],
                            attn_type=n_attn_types[i_stage],
                            norm_type=n_norm_types[i_stage],
                            use_norm=n_use_norm[i_stage],
                            use_res_connection=n_use_res_connection[i_stage],
                            use_rope=use_rope,
                            use_zero_weight=use_zero_weight,
                            gate=attention_gate
                        )
                i_encoder = i_encoder + 1
                self.n_encoder.append(down)
                if(i_module == level_num-1):
                    self.n_encoder_stage_channels_for_guidence.append(block_out)
                else:
                    self.n_encoder_stage_channels_for_guidence.append(0)

        if(print_chnanels):
            print(f"--- denosing encoder stage_channels : {self.n_encoder_stage_channels_for_guidence}")
        # ---- denoising encoder ----

        # ---- guidence encoder ----
        if (self.use_guidence_net):
            # ---- 获取encoder的输入维度的输出维度 ----
            g_encoder_block_ins = []
            for i_stage in range(len(self.g_encoder_names)):
                level_num = len(self.g_encoder_names[i_stage])
                for i_module in range(level_num):

                    block_in = g_channels[i_stage] * g_base_channel  # stage中，第2个模块开始
                    if (i_stage == 0 and i_module == 0):  # 初始输入通道
                        block_in = in_channel

                    elif (i_module == 0):  # stage中，第1个模块
                        block_in = g_channels[i_stage - 1] * g_base_channel
                    g_encoder_block_ins.append(block_in)
            g_encoder_block_outs = copy.deepcopy(g_encoder_block_ins)
            g_encoder_block_outs.append(g_channels[-1] * g_base_channel)
            g_encoder_block_outs = g_encoder_block_outs[1:]

            if(print_chnanels):
                print(f"--- guidence encoder block_ins : {g_encoder_block_ins}")
                print(f"--- guidence encoder block_out : {g_encoder_block_outs}")
            # ---- 获取encoder的输入维度的输出维度 ----

            self.g_encoder_stage_channels_for_guidence = []
            g_encoder_stage_channels_for_DPE = []
            i_encoder = 0
            i_down = 0

            self.g_encoder = nn.ModuleList()
            for i_stage in range(len(self.g_encoder_names)):
                block_out = g_encoder_block_outs[i_encoder]
                level_num = len(self.g_encoder_names[i_stage])
                for i_module in range(level_num):
                    block_in = g_encoder_block_ins[i_encoder]
                    moudle_num = len(self.g_encoder_names[i_stage][i_module])
                    down = nn.Module()
                    for i_name in range(moudle_num):
                        name = self.g_encoder_names[i_stage][i_module][i_name]
                        if (name == "Conv2D"):
                            down.Conv2D = get_module(
                                name=name,
                                in_channels=block_in,
                                out_channels=block_out,
                                kernel_size=3,
                                stride=1,
                                padding=1
                            )
                            g_encoder_stage_channels_for_DPE.append(block_out)
                        elif (name == "CircularConv2D"):
                            down.CircularConv2D = get_module(
                                name=name,
                                in_channels=block_in,
                                out_channels=block_out,
                                kernel_size=3,
                                stride=1,
                                padding=1
                            )
                            g_encoder_stage_channels_for_DPE.append(block_out)
                        elif (name == "Conv2DSiLU"):
                            down.Conv2DSiLU = get_module(
                                name=name,
                                in_channels=block_in,
                                out_channels=block_out,
                                kernel_size=3,
                                stride=1,
                                temb_channels=0,
                            )
                            n_encoder_stage_channels_for_DPE.append(block_out)
                        elif (name == "ResBlock"):
                            down.ResBlock = get_module(
                                name=name,
                                in_channels=block_in,
                                out_channels=block_out,
                                temb_channels=0,
                                use_pe=self.use_pe,
                                dropout=dropout,
                                use_zero_weight=use_zero_weight,
                                use_circularconv_shortcut=use_circularconv_shortcut,
                            )
                        elif (name == "ResConv2DBlock"):
                            down.ResConv2DBlock = get_module(
                                name=name,
                                in_channels=block_in,
                                out_channels=block_out,
                                temb_channels=0,
                                use_pe=self.use_pe,
                                dropout=dropout,
                                use_zero_weight=use_zero_weight,
                                use_circularconv_shortcut=False,
                            )
                        elif (name == "Downsample"):
                            down.Downsample = get_module(
                                name=name,
                                in_channels=block_out,
                                out_channels=block_out,
                                with_conv=False,
                                stride=tuple(n_strides[i_down])
                            )
                            g_encoder_stage_channels_for_DPE.append(block_out)
                            i_down = i_down + 1
                        elif (name == "Attention"):
                            down.Attention = get_module(
                                name=name,
                                q_dim=block_out,
                                kv_dim=n_encoder_block_outs[i_encoder],
                                heads=g_heads[i_stage],
                                attn_type=g_attn_types[i_stage],
                                norm_type=g_norm_types[i_stage],
                                use_norm=g_use_norm[i_stage],
                                use_res_connection=g_use_res_connection[i_stage],
                                use_rope=use_rope,
                                use_zero_weight=use_zero_weight,
                                gate=attention_gate
                            )
                    i_encoder = i_encoder + 1
                    self.g_encoder.append(down)
                    if (i_module == level_num - 1):
                        self.g_encoder_stage_channels_for_guidence.append(block_out)
                    else:
                        self.g_encoder_stage_channels_for_guidence.append(0)
            if(print_chnanels):
                print(f"--- guidance encoder stage_channels : {self.g_encoder_stage_channels_for_guidence}")
        # ---- guidence encoder ----

        # ---- control encoder ----
        if (use_control_net):
            self.l_encoder_zero_convs, self.l_midd_zero_convs = self.get_zero_conv()
            # ---- 获取encoder的输入维度的输出维度 ----
            l_encoder_block_ins = []
            for i_stage in range(len(self.n_encoder_names)):
                level_num = len(self.n_encoder_names[i_stage])
                for i_module in range(level_num):

                    block_in = n_channels[i_stage] * n_base_channel  # stage中，第2个模块开始
                    if (i_stage == 0 and i_module == 0):  # 初始输入通道
                        block_in = control_channel

                    elif (i_module == 0):  # stage中，第1个模块
                        block_in = n_channels[i_stage - 1] * n_base_channel
                    l_encoder_block_ins.append(block_in)
            l_encoder_block_outs = copy.deepcopy(l_encoder_block_ins)
            l_encoder_block_outs.append(n_channels[-1] * n_base_channel)
            l_encoder_block_outs = l_encoder_block_outs[1:]

            if(print_chnanels):
                print(f"--- control encoder block_ins : {l_encoder_block_ins}")
                print(f"--- control encoder block_outs : {l_encoder_block_outs}")
            # ---- 获取encoder的输入维度的输出维度 ----

            i_encoder = 0
            i_down = 0

            self.l_encoder = nn.ModuleList()
            for i_stage in range(len(self.n_encoder_names)):
                block_out = l_encoder_block_outs[i_encoder]
                level_num = len(self.n_encoder_names[i_stage])
                for i_module in range(level_num):
                    block_in = l_encoder_block_ins[i_encoder]
                    moudle_num = len(self.n_encoder_names[i_stage][i_module])
                    down = nn.Module()
                    for i_name in range(moudle_num):
                        name = self.n_encoder_names[i_stage][i_module][i_name]
                        if (name == "CircularConv2D"):
                            down.CircularConv2D = get_module(
                                name=name,
                                in_channels=block_in,
                                out_channels=block_out,
                                kernel_size=3,
                                stride=1,
                                padding=1
                            )
                        elif (name == "Conv2D"):
                            down.Conv2D = get_module(
                                name=name,
                                in_channels=block_in,
                                out_channels=block_out,
                                kernel_size=3,
                                stride=1,
                                padding=1
                            )
                        elif (name == "Conv2DSiLU"):
                            down.Conv2DSiLU = get_module(
                                name=name,
                                in_channels=block_in,
                                out_channels=block_out,
                                kernel_size=3,
                                stride=1,
                                temb_channels=self.temb_ch,
                            )
                            n_encoder_stage_channels_for_DPE.append(block_out)
                        elif (name == "ResBlock"):
                            down.ResBlock = get_module(
                                name=name,
                                in_channels=block_in,
                                out_channels=block_out,
                                temb_channels=self.temb_ch,
                                use_pe=self.use_pe,
                                dropout=dropout,
                                use_zero_weight=use_zero_weight,
                                use_circularconv_shortcut=True
                            )
                        elif (name == "ResConv2DBlock"):
                            down.ResConv2DBlock = get_module(
                                name=name,
                                in_channels=block_in,
                                out_channels=block_out,
                                temb_channels=self.temb_ch,
                                use_pe=self.use_pe,
                                dropout=dropout,
                                use_zero_weight=use_zero_weight,
                                use_circularconv_shortcut=False
                            )
                        elif (name == "Downsample"):
                            down.Downsample = get_module(
                                name=name,
                                in_channels=block_out,
                                out_channels=block_out,
                                with_conv="CircluarConv2D",
                                stride=tuple(n_strides[i_down])
                            )
                            i_down = i_down + 1
                        elif (name == "Attention"):
                            down.Attention = get_module(
                                name=name,
                                q_dim=block_out,
                                kv_dim=text_channels if use_text else block_out,
                                heads=n_heads[i_stage],
                                attn_type=n_attn_types[i_stage],
                                norm_type=n_norm_types[i_stage],
                                use_norm=n_use_norm[i_stage],
                                use_res_connection=n_use_res_connection[i_stage],
                                use_rope=use_rope,
                                use_zero_weight=use_zero_weight,
                                gate=attention_gate
                            )
                    i_encoder = i_encoder + 1
                    self.l_encoder.append(down)
        # ---- control encoder ----

        # ---- downsampling (曲面卷积，维度64) ----

        # ---- middle(曲面卷积+attention，512) ----
        # ---- denoising middle ----
        self.n_mid = nn.Module()
        self.n_mid.block_1 = ResnetBlock(
            in_channels=n_midd_channels[0] * n_base_channel,
            out_channels=n_midd_channels[1] * n_base_channel,
            temb_channels=self.temb_ch,
            use_pe=self.use_pe,
            dropout=dropout,
            use_zero_weight=use_zero_weight,
            use_circularconv_shortcut=True
        )
        self.n_mid.attn_1 = make_attn(
            q_dim=n_midd_channels[1] * n_base_channel,
            kv_dim=text_channels if use_text else n_midd_channels[1] * n_base_channel,
            heads=n_midd_heads,
            attn_type=n_midd_attn_type,
            norm_type=n_midd_norm_type,
            use_norm=n_midd_use_norm,
            use_res_connection=n_midd_use_res_connection,
            use_rope=use_rope,
            use_zero_weight=use_zero_weight,
            gate=attention_gate
        )
        self.n_mid.block_2 = ResnetBlock(
            in_channels=n_midd_channels[1] * n_base_channel,
            out_channels=n_midd_channels[2] * n_base_channel,
            temb_channels=self.temb_ch,
            use_pe=self.use_pe,
            dropout=dropout,
            use_zero_weight=use_zero_weight,
            use_circularconv_shortcut=use_circularconv_shortcut
        )
        # ---- denoising middle ----

        # ---- guidence middle ----
        if (use_guidence_net):
            self.g_mid = nn.Module()
            self.g_mid.block_1 = ResnetConv2DBlock(
                in_channels=g_midd_channels[0] * g_base_channel,
                out_channels=g_midd_channels[1] * g_base_channel,
                temb_channels=0,
                use_pe=self.use_pe,
                dropout=dropout,
                use_zero_weight=use_zero_weight,
                use_circularconv_shortcut=False
            )
            self.g_mid.attn_1 = make_attn(
                q_dim=g_midd_channels[1] * g_base_channel,
                kv_dim=n_midd_channels[1] * n_base_channel,
                heads=g_midd_heads,
                attn_type=g_midd_attn_type,
                norm_type=g_midd_norm_type,
                use_norm=g_midd_use_norm,
                use_res_connection=g_midd_use_res_connection,
                use_rope=use_rope,
                use_zero_weight=use_zero_weight,
                gate=False
            )
            self.g_mid.block_2 = ResnetConv2DBlock(
                in_channels=g_midd_channels[1] * g_base_channel,
                out_channels=g_midd_channels[2] * g_base_channel,
                temb_channels=0,
                use_pe=self.use_pe,
                dropout=dropout,
                use_zero_weight=use_zero_weight,
                use_circularconv_shortcut=False
            )
        # ---- guidence middle ----

        # ---- control middle ----
        if (use_control_net):
            self.l_mid = nn.Module()
            self.l_mid.block_1 = ResnetBlock(
                in_channels=n_midd_channels[0] * n_base_channel,
                out_channels=n_midd_channels[1] * n_base_channel,
                temb_channels=self.temb_ch,
                use_pe=self.use_pe,
                dropout=dropout,
                use_zero_weight=use_zero_weight,
                use_circularconv_shortcut=True
            )
            self.l_mid.attn_1 = make_attn(
                q_dim=n_midd_channels[1] * n_base_channel,
                kv_dim=text_channels if use_text else n_midd_channels[1] * n_base_channel,
                heads=n_midd_heads,
                attn_type=n_midd_attn_type,
                norm_type=n_midd_norm_type,
                use_norm=n_midd_use_norm,
                use_res_connection=n_midd_use_res_connection,
                use_rope=use_rope,
                use_zero_weight=use_zero_weight,
                gate=attention_gate
            )
            self.l_mid.block_2 = ResnetBlock(
                in_channels=n_midd_channels[1] * n_base_channel,
                out_channels=n_midd_channels[2] * n_base_channel,
                temb_channels=self.temb_ch,
                use_pe=self.use_pe,
                dropout=dropout,
                use_zero_weight=use_zero_weight,
                use_circularconv_shortcut=use_circularconv_shortcut
            )
        # ---- control middle ----
        # ---- middle(曲面卷积+attention，512) ----

        # ---- upsampling（上采样4层，2残差块+1上采样） ----
        # ---- denoising decoder ----
        n_encoder_block_ins.reverse()
        n_encoder_block_outs.reverse()

        n_channels.reverse()
        n_strides.reverse()
        n_heads.reverse()
        n_attn_types.reverse()
        n_norm_types.reverse()
        n_use_norm.reverse()
        n_use_res_connection.reverse()

        n_decoder_block_outs = []
        n_decoder_block_ins = []

        self.n_decoder_stage_channels_for_guidence = []

        for i_stage in range(len(self.n_decoder_names)):
            block_out = n_base_channel * n_channels[i_stage]
            level_num = len(self.n_decoder_names[i_stage])
            for i_module in range(level_num):
                if (i_stage == 0 and i_module == 0):
                    n_decoder_block_outs.append(n_midd_channels[-1] * n_base_channel)
                else:
                    n_decoder_block_outs.append(block_out)

        n_decoder_block_outs_temp = copy.deepcopy(n_decoder_block_outs)
        n_decoder_block_outs_temp.insert(0, n_midd_channels[-1] * n_base_channel)

        for i_encoder in range(len(n_encoder_block_outs)):
            n_decoder_block_ins.append(n_decoder_block_outs_temp[i_encoder] + n_encoder_block_outs[i_encoder])

        if(print_chnanels):
            print(f"--- denosing decoder block_ins : {n_decoder_block_ins}")
            print(f"--- denosing decoder block_out : {n_decoder_block_outs}")

        n_decoder_stage_channels_for_DPE = []
        n_decoder_stage_channels_for_DPE.append(n_midd_channels[-1] * n_base_channel)
        i_decoder = 0
        i_up = 0

        self.n_decoder = nn.ModuleList()
        for i_stage in range(len(self.n_decoder_names)):
            block_out = n_decoder_block_outs[i_decoder]
            level_num = len(self.n_decoder_names[i_stage])
            for i_module in range(level_num):
                block_in = n_decoder_block_ins[i_decoder]
                moudle_num = len(self.n_decoder_names[i_stage][i_module])
                up = nn.Module()
                for i_name in range(moudle_num):
                    name = self.n_decoder_names[i_stage][i_module][i_name]
                    if (name == "Conv2D"):
                        up.Conv2D = get_module(
                            name=name,
                            in_channels=in_channel,
                            out_channels=block_in,
                            kernel_size=3,
                            stride=1,
                            padding=1
                        )
                        n_decoder_stage_channels_for_DPE.append(block_out)
                    elif (name == "CircularConv2D"):
                        up.CircularConv2D = get_module(
                            name=name,
                            in_channels=in_channel,
                            out_channels=block_in,
                            kernel_size=3,
                            stride=1,
                            padding=1
                        )
                        n_decoder_stage_channels_for_DPE.append(block_out)
                    elif (name == "Conv2DSiLU"):
                        up.Conv2DSiLU = get_module(
                            name=name,
                            in_channels=block_in,
                            out_channels=block_out,
                            kernel_size=3,
                            stride=1,
                            temb_channels=self.temb_ch,
                        )
                        n_decoder_stage_channels_for_DPE.append(block_out)
                    elif (name == "ResConv2DBlock"):
                        up.ResConv2DBlock = get_module(
                            name=name,
                            in_channels=block_in,  # 残差块，2个CircularConv2D，2个Group Norm
                            out_channels=block_out,
                            temb_channels=self.temb_ch,
                            use_pe=self.use_pe,
                            dropout=dropout,
                            use_zero_weight=use_zero_weight,
                            use_circularconv_shortcut=use_circularconv_shortcut
                        )
                    elif (name == "ResDBlock"):
                        up.ResBlock = get_module(
                            name=name,
                            in_channels=block_in,  # 残差块，2个CircularConv2D，2个Group Norm
                            out_channels=block_out,
                            temb_channels=self.temb_ch,
                            use_pe=self.use_pe,
                            dropout=dropout,
                            use_zero_weight=use_zero_weight,
                            use_circularconv_shortcut=use_circularconv_shortcut
                        )
                    elif (name == "Upsample"):
                        up.Upsample = get_module(
                            name=name,
                            in_channels=block_out,
                            out_channels=block_out,
                            with_conv="Conv2D",
                            stride=tuple(n_strides[i_up])
                        )
                        n_decoder_stage_channels_for_DPE.append(block_out)
                        i_up = i_up + 1
                    elif (name == "Attention"):
                        up.Attention = get_module(
                            name=name,
                            q_dim=block_out,
                            kv_dim=text_channels if use_text else block_out,
                            heads=n_heads[i_stage],
                            attn_type=n_attn_types[i_stage],
                            norm_type=n_norm_types[i_stage],
                            use_norm=n_use_norm[i_stage],
                            use_res_connection=n_use_res_connection[i_stage],
                            use_rope=use_rope,
                            use_zero_weight=use_zero_weight,
                            gate=attention_gate
                        )

                self.n_decoder.append(up)
                i_decoder = i_decoder + 1
                if(i_module == level_num-1):
                    self.n_decoder_stage_channels_for_guidence.append(block_out)
                else:
                    self.n_decoder_stage_channels_for_guidence.append(0)

        if(print_chnanels):
            print(f"--- denosing decoder stage_channels : {self.n_decoder_stage_channels_for_guidence}")
        # ---- denoising decoder ----

        # ---- guidence decoder ----
        if (self.use_guidence_net):
            g_encoder_block_ins.reverse()
            g_encoder_block_outs.reverse()

            g_channels.reverse()
            g_strides.reverse()
            g_heads.reverse()
            g_attn_types.reverse()
            g_norm_types.reverse()
            g_use_norm.reverse()
            g_use_res_connection.reverse()

            g_decoder_block_outs = []
            g_decoder_block_ins = []

            self.g_decoder_stage_channels_for_guidence = []

            for i_stage in range(len(self.g_decoder_names)):
                block_out = g_base_channel * g_channels[i_stage]
                level_num = len(self.g_decoder_names[i_stage])
                for i_module in range(level_num):
                    if (i_stage == 0 and i_module == 0):
                        g_decoder_block_outs.append(g_midd_channels[-1] * g_base_channel)
                    else:
                        g_decoder_block_outs.append(block_out)

            g_decoder_block_outs_temp = copy.deepcopy(g_decoder_block_outs)
            g_decoder_block_outs_temp.insert(0, g_midd_channels[-1] * g_base_channel)

            for i_encoder in range(len(g_encoder_block_outs)):
                g_decoder_block_ins.append(g_decoder_block_outs_temp[i_encoder] + g_encoder_block_outs[i_encoder])

            g_decoder_stage_channels_for_DPE = []
            g_decoder_stage_channels_for_DPE.append(g_midd_channels[-1] * g_base_channel)
            i_decoder = 0
            i_up = 0

            self.g_decoder = nn.ModuleList()
            for i_stage in range(len(self.g_decoder_names)):
                block_out = g_decoder_block_outs[i_decoder]
                level_num = len(self.g_decoder_names[i_stage])
                for i_module in range(level_num):
                    block_in = g_decoder_block_ins[i_decoder]
                    moudle_num = len(self.g_decoder_names[i_stage][i_module])
                    up = nn.Module()
                    for i_name in range(moudle_num):
                        name = self.g_decoder_names[i_stage][i_module][i_name]
                        if (name == "Conv2D"):
                            up.Conv2D = get_module(
                                name=name,
                                in_channels=in_channel,
                                out_channels=block_in,
                                kernel_size=3,
                                stride=1,
                                padding=1
                            )
                            g_decoder_stage_channels_for_DPE.append(block_out)
                        elif (name == "CircleConv2D"):
                            up.CircularConv2D = get_module(
                                name=name,
                                in_channels=in_channel,
                                out_channels=block_in,
                                kernel_size=3,
                                stride=1,
                                padding=1
                            )
                            g_decoder_stage_channels_for_DPE.append(block_out)
                        elif (name == "Conv2DSiLU"):
                            up.Conv2DSiLU = get_module(
                                name=name,
                                in_channels=block_in,
                                out_channels=block_out,
                                kernel_size=3,
                                stride=1,
                                temb_channels=0,
                            )
                            g_decoder_stage_channels_for_DPE.append(block_out)
                        elif (name == "ResBlock"):
                            up.ResBlock = get_module(
                                name=name,
                                in_channels=block_in,
                                out_channels=block_out,
                                temb_channels=0,
                                use_pe=self.use_pe,
                                dropout=dropout,
                                use_zero_weight=use_zero_weight,
                                use_circularconv_shortcut=use_circularconv_shortcut
                            )
                        elif (name == "ResConv2DBlock"):
                            up.ResConv2DBlock = get_module(
                                name=name,
                                in_channels=block_in,
                                out_channels=block_out,
                                temb_channels=0,
                                use_pe=self.use_pe,
                                dropout=dropout,
                                use_zero_weight=use_zero_weight,
                                use_circularconv_shortcut=False
                            )
                        elif (name == "Upsample"):
                            up.Upsample = get_module(
                                name=name,
                                in_channels=block_out if i_name > 0 else block_in,
                                out_channels=block_out,
                                with_conv="Conv2D",
                                stride=tuple(g_strides[i_up])
                            )
                            g_decoder_stage_channels_for_DPE.append(block_out)
                            i_up = i_up + 1
                        elif (name == "Attention"):
                            up.Attention = get_module(
                                name=name,
                                q_dim=block_out,
                                kv_dim=n_decoder_block_outs[i_decoder],
                                heads=g_heads[i_stage],
                                attn_type=g_attn_types[i_stage],
                                norm_type=g_norm_types[i_stage],
                                use_norm=g_use_norm[i_stage],
                                use_res_connection=g_use_res_connection[i_stage],
                                use_rope=use_rope,
                                use_zero_weight=use_zero_weight,
                                gate=attention_gate
                            )

                    self.g_decoder.append(up)
                    i_decoder = i_decoder + 1
                    if (i_module == level_num - 1):
                        self.g_decoder_stage_channels_for_guidence.append(block_out)
                    else:
                        self.g_decoder_stage_channels_for_guidence.append(0)

            if(print_chnanels):
                print(f"--- guidence decoder block_ins : {g_decoder_block_ins}")
                print(f"--- guidence decoder block_out : {g_decoder_block_outs}")
            if (print_chnanels):
                print(f"--- guidance decoder stage_channels : {self.g_decoder_stage_channels_for_guidence}")
        # ---- guidence decoder ----
        # ---- upsampling（上采样4层，2残差块+1上采样） ----

        # ---- End ----
        # ---- denoising head ----
        self.n_norm_out = Normalize(n_decoder_block_outs[-1])
        self.n_conv_out = CircularConv2D(n_decoder_block_outs[-1],
                                         out_channel,
                                         kernel_size=(1, 4),
                                         stride=1,
                                         padding=(1, 2, 0, 0))
        # ---- denoising head ----

        # ---- guidence head ----
        if (self.use_guidence_net):
            self.g_norm_out = Normalize(g_decoder_block_outs[-1])
            self.g_conv_out = torch.nn.Conv2d(g_decoder_block_outs[-1],
                                             out_channel,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0)
        # ---- guidence head ----
        # ---- End ----

        self.n_encoder_names = self.list_flat(self.n_encoder_names)
        self.n_decoder_names = self.list_flat(self.n_decoder_names)

        if(self.use_guidence_net):
            self.g_encoder_names = self.list_flat(self.g_encoder_names)
            self.g_decoder_names = self.list_flat(self.g_decoder_names)

        # ---- Directional Position Encoding ----
        self.use_dpe_n = use_dpe_n
        if (self.use_dpe_n):

            self.n_dpe_encoder = nn.Sequential(
                ThetaPhiPEInjector(in_ch=n_encoder_stage_channels_for_DPE[0], fov_deg=fov),
                ThetaPhiPEInjector(in_ch=n_encoder_stage_channels_for_DPE[1], fov_deg=fov),
                ThetaPhiPEInjector(in_ch=n_encoder_stage_channels_for_DPE[2], fov_deg=fov),
                ThetaPhiPEInjector(in_ch=n_encoder_stage_channels_for_DPE[3], fov_deg=fov)
            )

            self.n_dpe_decoder = nn.Sequential(
                ThetaPhiPEInjector(in_ch=n_decoder_stage_channels_for_DPE[0], fov_deg=fov),
                ThetaPhiPEInjector(in_ch=n_decoder_stage_channels_for_DPE[1], fov_deg=fov),
                ThetaPhiPEInjector(in_ch=n_decoder_stage_channels_for_DPE[2], fov_deg=fov),
                ThetaPhiPEInjector(in_ch=n_decoder_stage_channels_for_DPE[3], fov_deg=fov),
            )

        self.use_dpe_g = use_dpe_g
        if (self.use_guidence_net and self.use_dpe_g):
            self.g_dpe_encoder = nn.Sequential(
                ThetaPhiPEInjector(in_ch=g_encoder_stage_channels_for_DPE[0], fov_deg=fov),
                ThetaPhiPEInjector(in_ch=g_encoder_stage_channels_for_DPE[1], fov_deg=fov),
                ThetaPhiPEInjector(in_ch=g_encoder_stage_channels_for_DPE[2], fov_deg=fov),
                ThetaPhiPEInjector(in_ch=g_encoder_stage_channels_for_DPE[3], fov_deg=fov)
            )

            self.g_dpe_decoder = nn.Sequential(
                ThetaPhiPEInjector(in_ch=g_decoder_stage_channels_for_DPE[0], fov_deg=fov),
                ThetaPhiPEInjector(in_ch=g_decoder_stage_channels_for_DPE[1], fov_deg=fov),
                ThetaPhiPEInjector(in_ch=g_decoder_stage_channels_for_DPE[2], fov_deg=fov),
                ThetaPhiPEInjector(in_ch=g_decoder_stage_channels_for_DPE[3], fov_deg=fov),
            )
        # ---- Directional Position Encoding ----

        # ---- Image Encoder ----
        self.use_camera = use_image_encoder
        if(self.use_camera):
            self.img_encoder = ImageEncoder(self.resolution)
        # ---- Image Encoder ----

        return

    def get_zero_conv(self):

        encoder_zero_convs = nn.Sequential()
        for channel in self.encoder_stage_channels_for_control:
            if (channel > 0):
                zero_conv = ZeroConv2d(channel)
            else:
                zero_conv = nn.Identity()

            encoder_zero_convs.append(zero_conv)

        midd_zero_convs = nn.Sequential()
        for i, channel in enumerate(self.midd_channels):
            if (i == 0):
                zero_conv = ZeroConv2d(self.base_channel * self.midd_channels[i + 1])
            else:
                zero_conv = ZeroConv2d(self.base_channel * channel)

            midd_zero_convs.append(zero_conv)

        return encoder_zero_convs, midd_zero_convs

    def list_flat(self, li):
        flat = [sublist for group in li for sublist in group]
        return flat

    def n_denosing(self, n_x, temb=None, cemb=None):

        if (cemb is not None and self.use_text):
            if (cemb.dtype == torch.float16):
                cemb = cemb.float()
        else:
            cemb = None

        # ---- downsampling ----， hs，保存残差连接特征
        n_hs = []
        n_h = n_x

        n_stage_feats = []

        dpe_i = 0
        encoder_attention_features = []
        for names, n_layers, stage_channel in zip(self.n_encoder_names, self.n_encoder, self.n_encoder_stage_channels_for_guidence):
            for name in names:
                n_layer = getattr(n_layers, name)
                if (name == "ResBlock" or name == "ResConv2DBlock"):
                    n_h = n_layer(n_h, temb)
                elif (name == "Attention"):
                    encoder_attention_features.append(n_h)
                    n_h = n_layer(n_h, cemb)
                else:
                    n_h = n_layer(n_h)
                    if (self.use_dpe_n):
                        n_h = self.n_dpe_encoder[dpe_i](n_h)
                        dpe_i = dpe_i + 1

            if (stage_channel > 0):
                n_stage_feats.append(n_h) # Downsample -> Downsample -> Downsample -> ResBlock(最后一层)
            n_hs.append(n_h)
        # ---- downsampling ----， hs，保存残差连接特征

        # ---- middle ----
        midd_attention_features = []
        n_h = self.n_mid.block_1(n_h, temb)  # 残差块，[2,256,16,128]
        midd_attention_features.append(n_h)
        n_h = self.n_mid.attn_1(n_h, cemb)  # 注意力机制，[2,256,16,128]
        n_h = self.n_mid.block_2(n_h, temb)  # 残差块，[2,256,16,128]
        n_stage_feats.append(n_h)
        # ---- middle ----

        # ---- upsampling ----
        if (self.use_dpe_n):
            n_h = self.n_dpe_decoder[0](n_h)
        dpe_i = 1
        decoder_attention_features = []
        for names, n_layers, stage_channel in zip(self.n_decoder_names, self.n_decoder, self.n_decoder_stage_channels_for_guidence):

            # ---- 跳跃连接 ----
            s_feat = n_hs.pop()
            if (self.skip_connection_scale == "sqrt(2)"):
                s_feat = universal_scalling(s_feat)
            n_h = torch.cat((n_h, s_feat), dim=1)
            # ---- 跳跃连接 ----

            for name in names:
                n_layer = getattr(n_layers, name)
                if (name == "ResBlock" or name == "ResConv2DBlock"):
                    n_h = n_layer(n_h, temb)
                elif (name == "Attention"):
                    decoder_attention_features.append(n_h)
                    n_h = n_layer(n_h, cemb)
                else:
                    n_h = n_layer(n_h)
                    if (self.use_dpe_n):
                        n_h = self.n_dpe_decoder[dpe_i](n_h)
                        dpe_i = dpe_i + 1

            if (stage_channel > 0):
                n_stage_feats.append(n_h) # Upsampling -> Upsampling -> Upsampling -> Attention(最后一层)
        # ---- upsampling ----

        # ---- End ----
        n_h = self.n_norm_out(n_h)
        n_h = nonlinearity(n_h)
        n_h = self.n_conv_out(n_h)
        # ---- End ----

        return n_h, n_stage_feats, (encoder_attention_features, midd_attention_features, decoder_attention_features)

    def g_guidence(self, g_x, attn_fs, n_stage_feats=None):

        # ---- downsampling ----， hs，保存残差连接特征
        encoder_attention_features, midd_attention_features, decoder_attention_features = attn_fs
        encoder_attention_features.reverse()
        midd_attention_features.reverse()
        decoder_attention_features.reverse()

        g_hs = []
        g_h = g_x

        g_stage_feats = []

        n_stage_feats_i = 0
        dpe_i = 0
        for names, g_layers, stage_channel in zip(self.g_encoder_names, self.g_encoder, self.g_encoder_stage_channels_for_guidence):
            for name in names:
                g_layer = getattr(g_layers, name)
                if (name == "ResBlock" or name == "ResConv2DBlock"):
                    g_h = g_layer(g_h)
                elif (name == "Attention"):
                    g_h = g_layer(g_h, encoder_attention_features.pop())
                else:
                    g_h = g_layer(g_h)
                    if (self.use_dpe_g):
                        g_h = self.g_dpe_encoder[dpe_i](g_h)
                        dpe_i = dpe_i + 1

            if (stage_channel > 0): # Downsample -> Downsample -> Downsample -> Conv2DSiLU
                g_stage_feats.append(g_h) # [2,64,64,512] -> [2,128,32,512] -> [2,256,16,128] -> [2,256,16,128]
                if(self.use_multistage_feats):
                    g_h = g_h + n_stage_feats[n_stage_feats_i]
                    n_stage_feats_i += 1
            g_hs.append(g_h)
        # ---- downsampling ----， hs，保存残差连接特征

        # ---- middle ----
        g_h = self.g_mid.block_1(g_h)  # 残差块，[2,256,16,128]
        g_h = self.g_mid.attn_1(g_h, midd_attention_features.pop())  # 注意力机制，[2,256,16,128]
        g_h = self.g_mid.block_2(g_h)  # 残差块，[2,256,16,128]
        g_stage_feats.append(g_h)
        if (self.use_multistage_feats):
            g_h = g_h + n_stage_feats[n_stage_feats_i]
            n_stage_feats_i += 1
        # ---- middle ----

        # # ---- upsampling ----
        if (self.use_dpe_g):
            g_h = self.g_dpe_decoder[0](g_h)
        dpe_i = 1

        for names, g_layers, stage_channel in zip(self.g_decoder_names, self.g_decoder, self.g_decoder_stage_channels_for_guidence):

            # ---- 跳跃连接 ----
            s_feat = g_hs.pop()
            g_h = torch.cat((g_h, s_feat), dim=1)
            # ---- 跳跃连接 ----

            for name in names:
                g_layer = getattr(g_layers, name)
                if (name == "ResBlock" or name == "ResConv2DBlock"):
                    g_h = g_layer(g_h)
                elif (name == "Attention"):
                    g_h = g_layer(g_h, decoder_attention_features.pop())
                else:
                    g_h = g_layer(g_h)
                    if (self.use_dpe_g):
                        g_h = self.g_dpe_decoder[dpe_i](g_h)
                        dpe_i = dpe_i + 1

            if (stage_channel > 0):
                g_stage_feats.append(g_h) # Upsampling -> Upsampling -> Upsampling -> Conv2DSiLU
                if (self.use_multistage_feats):
                    g_h = g_h + n_stage_feats[n_stage_feats_i]
                    n_stage_feats_i += 1
        # ---- upsampling ----

        # ---- End ----
        g_h = self.g_norm_out(g_h)
        g_h = nonlinearity(g_h)
        g_h = self.g_conv_out(g_h)
        # ---- End ----

        return g_h, g_stage_feats

    def l_control(self, n_x, l_x, temb=None):

        # ---- image encoder ----
        if(self.use_camera):
            l_x = self.img_encoder(l_x)
        # ---- image encoder ----

        # ---- denosing downsampling ----， hs，保存残差连接特征
        n_hs = []
        n_h = n_x
        dpe_i = 0

        with torch.no_grad():
            for names, n_layers in zip(self.n_encoder_names, self.n_encoder):
                for name in names:
                    n_layer = getattr(n_layers, name)
                    if (name == "ResBlock" or name == "ResConv2DBlock"):
                        n_h = n_layer(n_h, temb)
                    elif (name == "Attention"):
                        n_h = n_layer(n_h)
                    else:
                        n_h = n_layer(n_h)
                        if (self.use_dpe_n):
                            n_h = self.n_dpe_encoder[dpe_i](n_h)
                            dpe_i = dpe_i + 1

                n_hs.append(n_h)
        # ---- denosing downsampling ----， hs，保存残差连接特征

        # ---- control downsampling (移除DPE) ----， hs，保存残差连接特征
        l_h = l_x
        l_n_hs = []

        for names, l_layers, z_layer, n_h in zip(self.n_encoder_names, self.l_encoder, self.l_encoder_zero_convs, n_hs):
            for name in names:
                n_layer = getattr(l_layers, name)
                if (name == "ResBlock" or name == "ResConv2DBlock"):
                    l_h = n_layer(l_h, temb)
                elif (name == "Attention"):
                    l_h = n_layer(l_h)
                else:
                    l_h = n_layer(l_h)

            l_n_h = n_h + z_layer(l_h)
            l_n_hs.append(l_n_h)
        # ---- control downsampling (移除DPE) ----， hs，保存残差连接特征

        # ---- middle ----
        l_h = self.l_mid.block_1(l_h, temb)  # 残差块，[2,256,16,128]
        n_h = self.n_mid.block_1(n_h, temb)
        n_h = n_h + self.l_midd_zero_convs[0](l_h)

        l_h = self.l_mid.attn_1(l_h)  # 注意力机制，[2,256,16,128]
        n_h = self.n_mid.attn_1(n_h)  # 注意力机制，[2,256,16,128]
        n_h = n_h + self.l_midd_zero_convs[1](l_h)

        l_h = self.l_mid.block_2(l_h, temb)  # 残差块，[2,256,16,128]
        n_h = self.n_mid.block_2(n_h, temb)  # 残差块，[2,256,16,128]
        n_h = n_h + self.l_midd_zero_convs[2](l_h)
        # ---- middle ----

        # ---- denosing upsampling ----
        if (self.use_dpe_n):
            n_h = self.n_dpe_decoder[0](n_h)
        dpe_i = 1
        for names, n_layers, stage_channel in zip(self.n_decoder_names, self.n_decoder, self.n_decoder_stage_channels_for_guidence):

            # ---- 跳跃连接 ----
            s_feat = l_n_hs.pop()
            if (self.skip_connection_scale == "sqrt(2)"):
                s_feat = universal_scalling(s_feat)
            n_h = torch.cat((n_h, s_feat), dim=1)
            # ---- 跳跃连接 ----

            for name in names:
                n_layer = getattr(n_layers, name)
                if (name == "ResBlock" or name == "ResConv2DBlock"):
                    n_h = n_layer(n_h, temb)
                elif (name == "Attention"):
                    n_h = n_layer(n_h)
                else:
                    n_h = n_layer(n_h)
                    if (self.use_dpe_n):
                        n_h = self.n_dpe_decoder[dpe_i](n_h)
                        dpe_i = dpe_i + 1
        # ---- denosing upsampling ----

        # ---- End ----
        n_h = self.n_norm_out(n_h)
        n_h = nonlinearity(n_h)
        n_h = self.n_conv_out(n_h)
        # ---- End ----

        return n_h

    def forward(self, n_x, t=None, cemb=None, g_x=None, l_x=None):

        # timestep embedding
        if self.temb_ch > 0:
            # timestep embedding
            assert t is not None
            temb = get_timestep_embedding(t, self.base_channel)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        if (l_x is not None and self.use_control_net):
            # ---- Control Net ----
            n_h = self.l_control(n_x=n_x, l_x=l_x, temb=temb)
            # ---- Control Net ----
        else:
            # ---- Denosing Net ----
            n_h, n_stage_feats, attn_fs = self.n_denosing(n_x=n_x, temb=temb, cemb=cemb)
            # ---- Denosing Net ----

            # ---- Guidence Net ----
            if (g_x is not None and self.use_guidence_net):
                if (self.freeze_guidence_net):
                    with torch.no_grad():
                        g_h, g_stage_feats = self.g_guidence(g_x=g_x, attn_fs=attn_fs, n_stage_feats=n_stage_feats)
                else:
                    g_h, g_stage_feats = self.g_guidence(g_x=g_x, attn_fs=attn_fs, n_stage_feats=n_stage_feats)
            # ---- Guidence Net ----

            if (g_x is not None and self.use_guidence_net):
                # ---- only decoder features ----
                return n_h, g_h, n_stage_feats[5:], g_stage_feats[5:]

        return n_h


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    '''
        Text Denosing+Guidence: 33,500,426
            n_attn_types=["nn", "nn", "nn", "nn"],
            n_norm_types=["ln", "ln", "ln", "ln"],
            n_midd_attn_type="nn",
            n_midd_norm_type="ln",
            use_guidence_net=False,
            use_text=True,

        Text Denosing+Guidence: 43,845,516
            n_attn_types=["nn", "nn", "nn", "nn"],
            n_norm_types=["ln", "ln", "ln", "ln"],
            n_midd_attn_type="nn",
            n_midd_norm_type="ln",
            use_guidence_net=True,
            use_text=True,

        Denoising Net: 30,643,210

        Denoising+Guidence Net: 40,988,300
            use_guidence_net=True

        Denoising+Guidence+Contorl Net: 57,420,492
            use_guidence_net=True
            use_control_net=True

    '''

    # Text
    attn_types = ["nn", "nn", "nn", "nn"]
    norm_types = ["ln", "ln", "ln", "ln"]

    model = CircularUNet(
        # text
        # n_attn_types=attn_types,
        # n_norm_types=norm_types,
        # n_midd_attn_type="nn",
        # n_midd_norm_type="ln",

        control_channel=128,
        use_image_encoder=False,

        use_guidence_net=True,
        use_control_net=False,
        use_text=False,
        use_circularconv_shortcut=False,
        attention_gate=False,
        use_multistage_feats=False
    ).cuda()
    print(f"number of parameters: {common.count_parameters(model):,}")

    H = 32
    W = 64#1024

    x = torch.randn(size=(2, 2, H, W), dtype=torch.float32).cuda()
    g = torch.randn(size=(2, 2, H, W), dtype=torch.float32).cuda()
    l = torch.randn(size=(2, 3, H, W), dtype=torch.float32).cuda()
    t = torch.ones(size=(2,)).cuda()
    context = torch.randn(size=(2, 77, 768), dtype=torch.float32).cuda()

    print(x.shape)
    x = model(n_x=x, t=t, cemb=context, g_x=g, l_x=l)
    if (isinstance(x, tuple)):
        g_x, n_x, _, _ = x
        print(g_x.shape)
        print(n_x.shape)
    else:
        print(x.shape)

    # x = torch.randn(size=(2, 64, 16, 64), dtype=torch.float32).cuda()
    # # m = Attention(gate=False,heads=2).cuda()
    # # m = LinearAttention(gate=False,heads=2).cuda()
    # m = AttentionBlock(gate=False,heads=2).cuda()
    # x = m(x)
    #
    # print(f"number of parameters: {common.count_parameters(m):,}")
    # print(x.shape)

