"""Student implementation of the final project's image encoder.

Complete TODO-01 through TODO-06 using NumPy. Model parameters are supplied.
The public classes retain the final project's interface. helpers.py is provided.
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """TODO-02. Input [B,N,D], return [B,H,N,D/H].

        In this image model D=768, H=12 and D/H=64. Use reshape/transpose.
        Research the head organization independently. Preserve token order.
        """
        # TODO-02 contract: [B,N,D] to [B,H,N,D/H].
        # Image model: D=768, H=12, D/H=64.
        batch, tokens, width = x.shape
        # Split the last axis (features) into (heads, head_dim): [B,N,H,64].
        grouped = x.reshape(batch, tokens, self.heads, self.head_dim)
        # Move the head axis beside the batch axis, keeping token order: [B,H,N,64].
        return grouped.transpose(0, 2, 1, 3)

    def __call__(self, x, causal=False):
        # Given: stored linear weights use [out,in]. Parameters stay fixed.
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        # TODO-03a: query/key comparisons, scaled by sqrt(head_dim).
        # q,k,v: [B,H,N,64]. scores must have [B,H,N,N].
        # Every query row compares with every key row; only the last two axes swap.
        scale = np.float32(np.sqrt(float(self.head_dim)))
        scores = (q @ k.transpose(0, 1, 3, 2)) / scale

        # Given for later text-encoder compatibility. Images use causal=False.
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)  # supplied numerical helper

        # TODO-03b: mix value rows using coefficients; then join head outputs.
        # context: [B,H,N,64], merged: [B,N,D].
        # Each query row keeps a weighted average of the value rows: [B,H,N,64].
        context = coefficients @ v
        # Return the head axis next to the feature axis, then fuse it: [B,N,D].
        merged = context.transpose(0, 2, 1, 3).reshape(q.shape[0], q.shape[2], self.width)

        # TODO-03c: output projection, using out_weight and out_bias.
        # Return [B,N,D]. Remember the stored linear-weight orientation.
        # Stored as [out,in], so multiply by the transpose and broadcast the bias.
        return merged @ self.out_weight.T + self.out_bias


class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']  # [4D,D]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # [D,4D]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        # Given: normalize each token and calculate its attention update.
        update = self.attention(self.norm1(x), causal=causal)

        # TODO-04a: first residual addition. Name the result residual.
        # The residual path keeps the block input x, not the normalized copy.
        residual = x + update

        # Given: the second Norm and the exact checkpoint activation helper.
        u = self.norm2(residual)

        # TODO-04b: MLP = GELU(u @ W1.T + b1) @ W2.T + b2.
        # hidden: [B,N,3072], mlp_update: [B,N,768] for the image tower.
        # Return residual + mlp_update. Both residual paths retain [B,N,D].
        # Widen to 4D with the first linear layer, stored as [out,in].
        hidden = u @ self.fc1_weight.T + self.fc1_bias
        # Supplied exact checkpoint activation, applied per feature.
        activated = self.gelu(hidden)
        # Narrow back with the second linear layer: [B,N,768].
        mlp_update = activated @ self.fc2_weight.T + self.fc2_bias
        # Second residual addition on the residual path.
        return residual + mlp_update


class ImageEncoder:
    def __init__(self, weights):
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # [768,512], already [in,out]

    def patchify(self, images):
        """TODO-01. [B,3,224,224] -> [B,49,3072].

        Visit patches left to right, then top to bottom. Within a patch, flatten
        C,H,W in that order. Each 32x32 RGB patch contributes 3*32*32 values.
        Use reshape/transpose. Do not resize the already-prepared input.
        """
        # TODO-01: separate patch-grid axes; reorder; flatten.
        # Input [B,3,224,224], return [B,49,3072].
        batch = images.shape[0]
        # Separate the two 224 axes into (grid, patch) pairs: [B,3,7,32,7,32].
        grid = images.reshape(batch, 3, 7, 32, 7, 32)
        # Reorder to (B, row, column, C, patch_h, patch_w) so patch order is
        # left-to-right then top-to-bottom, with C,H,W flattened inside each patch.
        ordered = grid.transpose(0, 2, 4, 1, 3, 5)
        return ordered.reshape(batch, 49, 3072)

    def embed(self, images):
        patches = self.patchify(images)  # [B,49,3072]

        # TODO-05a: project patches to [B,49,768]. There is no patch bias.
        # The stored conv weight is [out,in] after the reshape, so transpose it.
        patch_tokens = patches @ self.patch_weight.T

        # Given: one supplied CLS row per image.
        cls = np.tile(self.cls, (images.shape[0], 1, 1))

        # TODO-05b: prepend CLS on the token axis, then add learned positions.
        # Return [B,50,768]. self.position broadcasts over B.
        # CLS sits at token index 0, ahead of the 49 patch rows.
        tokens = np.concatenate([cls, patch_tokens], axis=1)
        # Learned position table is [50,768]; broadcasting adds it per image.
        return tokens + self.position

    def encode_tokens(self, images):
        """Provided assembly of token input and the 12 blocks; [B,50,768]."""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """TODO-06. Return one normalized output token [B,512] per image."""
        tokens = self.encode_tokens(images)

        # Select CLS, apply post_norm, and project with self.projection.
        # [B,50,768] -> [B,768] -> [B,512]. visual.proj is already [in,out].
        # Name the projected array output. The final unit-vector helper is given.
        # Token index 0 is the CLS row; keep only it: [B,768].
        cls_token = tokens[:, 0, :]
        # Post-Norm runs on the selected CLS row alone.
        normalized = self.post_norm(cls_token)
        # visual.proj is already [in,out], so no transpose here: [B,512].
        output = normalized @ self.projection

        return unit_vector(output)
