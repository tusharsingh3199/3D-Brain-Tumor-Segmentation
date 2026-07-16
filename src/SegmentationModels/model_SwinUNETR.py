import numpy as np
import tensorflow as tf
from configs.config import *


def window_partition(x, window_size):
    B = tf.shape(x)[0]
    D, H, W, C = x.shape[1], x.shape[2], x.shape[3], x.shape[4]
    wd, wh, ww = window_size
    x = tf.reshape(x, [B, D // wd, wd, H // wh, wh, W // ww, ww, C])
    x = tf.transpose(x, [0, 1, 3, 5, 2, 4, 6, 7])
    return tf.reshape(x, [-1, wd, wh, ww, C])


def window_reverse(windows, window_size, D, H, W):
    wd, wh, ww = window_size
    C = windows.shape[-1]
    B = tf.shape(windows)[0] // ((D // wd) * (H // wh) * (W // ww))
    x = tf.reshape(windows, [B, D // wd, H // wh, W // ww, wd, wh, ww, C])
    x = tf.transpose(x, [0, 1, 4, 2, 5, 3, 6, 7])
    return tf.reshape(x, [B, D, H, W, C])


class WindowAttention3D(tf.keras.layers.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = tf.keras.layers.Dense(dim * 3, use_bias=qkv_bias)
        self.proj = tf.keras.layers.Dense(dim)

    def build(self, input_shape):
        wd, wh, ww = self.window_size
        self.num_tokens = wd * wh * ww
        self.relative_position_bias_table = self.add_weight(
            shape=((2 * wd - 1) * (2 * wh - 1) * (2 * ww - 1), self.num_heads),
            initializer="zeros", trainable=True, name="rel_pos_bias_table")

        coords = np.stack(np.meshgrid(np.arange(wd), np.arange(wh), np.arange(ww), indexing="ij"), axis=0)
        coords = coords.reshape(3, -1)
        rel = coords[:, :, None] - coords[:, None, :]
        rel = rel.transpose(1, 2, 0)
        rel[:, :, 0] += wd - 1
        rel[:, :, 1] += wh - 1
        rel[:, :, 2] += ww - 1
        rel[:, :, 0] *= (2 * wh - 1) * (2 * ww - 1)
        rel[:, :, 1] *= (2 * ww - 1)
        idx = rel.sum(-1).reshape(-1).astype(np.int32)
        self.relative_position_index = self.add_weight(
            shape=idx.shape, initializer=tf.keras.initializers.Constant(idx),
            trainable=False, dtype=tf.int32, name="rel_pos_index")
        super().build(input_shape)

    def call(self, x, mask=None):
        B_, N, C = tf.shape(x)[0], tf.shape(x)[1], self.dim
        qkv = tf.reshape(self.qkv(x), [B_, N, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = tf.matmul(q, k, transpose_b=True)

        bias = tf.gather(self.relative_position_bias_table, self.relative_position_index)
        bias = tf.transpose(tf.reshape(bias, [self.num_tokens, self.num_tokens, self.num_heads]), [2, 0, 1])
        attn = attn + bias[None, ...]

        if mask is not None:
            nW = tf.shape(mask)[0]
            attn = tf.reshape(attn, [B_ // nW, nW, self.num_heads, N, N]) + mask[None, :, None, :, :]
            attn = tf.reshape(attn, [-1, self.num_heads, N, N])

        attn = tf.nn.softmax(attn, axis=-1)
        out = tf.transpose(tf.matmul(attn, v), [0, 2, 1, 3])
        return self.proj(tf.reshape(out, [B_, N, C]))


class SwinTransformerBlock3D(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, window_size=(4, 4, 4), shift_size=(0, 0, 0), mlp_ratio=4.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention3D(dim, window_size, num_heads)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(int(dim * mlp_ratio), activation="gelu"),
            tf.keras.layers.Dense(dim),
        ])

    def get_attn_mask(self, D, H, W):
        wd, wh, ww = self.window_size
        sd, sh, sw = self.shift_size
        if sd == 0 and sh == 0 and sw == 0:
            return None
        d_s = [(0, D - wd), (D - wd, D - sd), (D - sd, D)]
        h_s = [(0, H - wh), (H - wh, H - sh), (H - sh, H)]
        w_s = [(0, W - ww), (W - ww, W - sw), (W - sw, W)]
        mask = np.zeros((1, D, H, W, 1), dtype=np.float32)
        cnt = 0
        for d0, d1 in d_s:
            for h0, h1 in h_s:
                for w0, w1 in w_s:
                    if d1 > d0 and h1 > h0 and w1 > w0:
                        mask[:, d0:d1, h0:h1, w0:w1, :] = cnt
                    cnt += 1
        mw = tf.reshape(window_partition(tf.constant(mask), self.window_size), [-1, wd * wh * ww])
        diff = mw[:, None, :] - mw[:, :, None]
        return tf.where(diff != 0, tf.constant(-100.0), tf.constant(0.0))

    def call(self, x, D=None, H=None, W=None):
        B = tf.shape(x)[0]
        C = self.dim
        shortcut = x
        x = tf.reshape(self.norm1(x), [B, D, H, W, C])

        wd, wh, ww = self.window_size
        sd, sh, sw = self.shift_size
        pad_d, pad_h, pad_w = (wd - D % wd) % wd, (wh - H % wh) % wh, (ww - W % ww) % ww
        x = tf.pad(x, [[0, 0], [0, pad_d], [0, pad_h], [0, pad_w], [0, 0]])
        Dp, Hp, Wp = D + pad_d, H + pad_h, W + pad_w

        if sd or sh or sw:
            shifted = tf.roll(x, shift=[-sd, -sh, -sw], axis=[1, 2, 3])
            attn_mask = self.get_attn_mask(Dp, Hp, Wp)
        else:
            shifted, attn_mask = x, None

        windows = tf.reshape(window_partition(shifted, self.window_size), [-1, wd * wh * ww, C])
        attn_out = self.attn(windows, mask=attn_mask)
        attn_out = tf.reshape(attn_out, [-1, wd, wh, ww, C])
        shifted = window_reverse(attn_out, self.window_size, Dp, Hp, Wp)

        x = tf.roll(shifted, shift=[sd, sh, sw], axis=[1, 2, 3]) if (sd or sh or sw) else shifted
        if pad_d or pad_h or pad_w:
            x = x[:, :D, :H, :W, :]

        x = shortcut + tf.reshape(x, [B, D * H * W, C])
        return x + self.mlp(self.norm2(x))


class PatchMerging3D(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.reduction = tf.keras.layers.Dense(2 * dim, use_bias=False)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, x, D=None, H=None, W=None):
        C = x.shape[-1]
        B = tf.shape(x)[0]
        x = tf.reshape(x, [B, D, H, W, C])
        parts = [x[:, i::2, j::2, k::2, :] for i in (0, 1) for j in (0, 1) for k in (0, 1)]
        x = tf.concat(parts, axis=-1)
        x = tf.reshape(x, [B, (D // 2) * (H // 2) * (W // 2), 8 * C])
        return self.reduction(self.norm(x))


def swin_stage(x, D, H, W, dim, depth, num_heads, window_size):
    shift = tuple(w // 2 for w in window_size)
    for i in range(depth):
        s = (0, 0, 0) if i % 2 == 0 else shift
        x = SwinTransformerBlock3D(dim, num_heads, window_size, s)(x, D=D, H=H, W=W)
    return x


def conv_block(x, filters, groups=8, dropout=0.0):
    x = tf.keras.layers.Conv3D(filters, 3, padding="same")(x)
    x = tf.keras.layers.GroupNormalization(groups=min(groups, filters))(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv3D(filters, 3, padding="same")(x)
    x = tf.keras.layers.GroupNormalization(groups=min(groups, filters))(x)
    x = tf.keras.layers.ReLU()(x)
    if dropout > 0:
        x = tf.keras.layers.SpatialDropout3D(dropout)(x)
    return x


def res_block(x, filters, groups=8):
    shortcut = x if x.shape[-1] == filters else tf.keras.layers.Conv3D(filters, 1, padding="same")(x)
    x = tf.keras.layers.Conv3D(filters, 3, padding="same")(x)
    x = tf.keras.layers.GroupNormalization(groups=min(groups, filters))(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv3D(filters, 3, padding="same")(x)
    x = tf.keras.layers.GroupNormalization(groups=min(groups, filters))(x)
    x = tf.keras.layers.Add()([x, shortcut])
    return tf.keras.layers.ReLU()(x)


def decoder_block(x, skip, filters):
    x = tf.keras.layers.Conv3DTranspose(filters, kernel_size=2, strides=2, padding="same")(x)
    x = tf.keras.layers.Concatenate()([x, skip])
    return res_block(x, filters)


class SwinEncoder(tf.keras.layers.Layer):
    def __init__(self, input_shape, patch_size, embed_dim, depths, num_heads, window_size, **kwargs):
        super().__init__(**kwargs)
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.dims = [embed_dim * (2 ** i) for i in range(len(depths))]
        H0, W0, D0, _ = input_shape
        self.D0, self.H0, self.W0 = H0 // patch_size, W0 // patch_size, D0 // patch_size
        self.patch_embed = tf.keras.layers.Conv3D(embed_dim, kernel_size=patch_size, strides=patch_size, padding="same")
        self.merges = [PatchMerging3D(self.dims[i]) for i in range(len(depths) - 1)]

    def call(self, inputs):
        x = self.patch_embed(inputs)
        B = tf.shape(x)[0]
        D, H, W = self.D0, self.H0, self.W0
        x_seq = tf.reshape(x, [B, D * H * W, self.embed_dim])

        skips = []
        for i, (depth, heads) in enumerate(zip(self.depths, self.num_heads)):
            x_seq = swin_stage(x_seq, D, H, W, self.dims[i], depth, heads, self.window_size)
            skips.append(tf.reshape(x_seq, [B, D, H, W, self.dims[i]]))
            if i < len(self.depths) - 1:
                x_seq = self.merges[i](x_seq, D=D, H=H, W=W)
                D, H, W = D // 2, H // 2, W // 2
        return skips


def Model_SwinUNETR(input_shape=(128, 128, 128, 4), num_classes=4, patch_size=2, embed_dim=24,
                    depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24), window_size=(4, 4, 4),):

    inputs = tf.keras.Input(input_shape)
    dims = [embed_dim * (2 ** i) for i in range(len(depths))]

    enc0 = conv_block(inputs, embed_dim)

    skips = SwinEncoder(input_shape, patch_size, embed_dim, depths, num_heads, window_size)(inputs)
    bottleneck = skips[-1]
    proj_skips = [conv_block(s, dims[i]) for i, s in enumerate(skips[:-1])]

    d = conv_block(bottleneck, dims[-1], dropout=0.3)
    for i in reversed(range(len(depths) - 1)):
        d = decoder_block(d, proj_skips[i], dims[i])
    d = decoder_block(d, enc0, embed_dim)

    outputs = tf.keras.layers.Conv3D(num_classes, kernel_size=1, activation="softmax")(d)
    return tf.keras.Model(inputs, outputs, name="Swin_UNETR")


callbacks = [
        tf.keras.callbacks.ModelCheckpoint("Swin_UNETR.keras", save_best_only=True, monitor="val_dice", mode="max"),
        tf.keras.callbacks.EarlyStopping(monitor="val_dice", mode="max", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, mode="min")
    ]
