import mlx.core as mx
import mlx.nn as nn
import math

def loss_fn(model: Transformer, x: mx.array, y: mx.array):
    logits = model(x)
    return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")

# mixed precision stuff
class BF16Linear(nn.Linear):
    def __call__(self, x):
        return x @ self.weight.astype(x.dtype).T

class BF16RMSNorm(nn.RMSNorm):
    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight.astype(x.dtype), self.eps)


def quantize_weights(W: mx.array):
    quantized_W = W.astype(mx.float32)  ### cast to fp32
    abs_mu = quantized_W.abs().mean(axis=-1, keepdims=True)  ### shape: (..., 1)
    abs_mu = mx.stop_gradient(mx.clip(abs_mu, 1e-5, None))
    quantized_W = quantized_W / abs_mu  ### drop the scale. each param is now "how many avg weights is this?"
    quantized_W = mx.round(quantized_W)
    quantized_W = mx.clip(quantized_W, -1, 1)
    return quantized_W * abs_mu

def quantize_activations(x: mx.array):
    quantized_x = x.astype(mx.float32)
    abs_max = quantized_x.abs().max(axis=-1, keepdims=True)  ### shape: (..., 1)
    abs_max = mx.stop_gradient(mx.clip(abs_max, 1e-5, None))
    scale = abs_max/127  ### scale: size of an integer step
    quantized_x = quantized_x / scale
    quantized_x = mx.round(quantized_x)
    quantized_x = mx.clip(quantized_x, -128, 127)
    return quantized_x * scale

class Bitlinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.rms = BF16RMSNorm(in_features)
        self.weight = mx.zeros((out_features, in_features))  ### immediately overwritten by _init_weights

    def __call__(self, x: mx.array):
        forward_x = self.rms(x)
        x_quantized = quantize_activations(forward_x)  ### x is quantized to int8, W to ternary
        W_quantized = quantize_weights(self.weight)
        W_quantized = self.weight + mx.stop_gradient(W_quantized - self.weight) ### forward uses quantized, backward uses latent weights
        x_quantized = forward_x + mx.stop_gradient(x_quantized - forward_x)
        y = x_quantized.astype(mx.bfloat16) @ W_quantized.astype(mx.bfloat16).T
        return y

class TransformerBlock(nn.Module):
    def __init__(self, D: int, H: int, ternary: bool = False):
        super().__init__()

        self.D_h = D // H
        assert self.D_h % 2 == 0, "dim should be divisible by number of heads"

        self.ternary = ternary

        if not ternary:
            self.Q_layer = BF16Linear(input_dims=D, output_dims=D, bias=False)  ### layernorm already acts like bias
            self.K_layer = BF16Linear(input_dims=D, output_dims=D, bias=False)  ### layernorm already acts like bias
            self.V_layer = BF16Linear(input_dims=D, output_dims=D, bias=False)  ### layernorm already acts like bias
            self.O_layer = BF16Linear(input_dims=D, output_dims=D, bias=False)
        else:
            self.Q_layer = Bitlinear(in_features=D, out_features=D)
            self.K_layer = Bitlinear(in_features=D, out_features=D)
            self.V_layer = Bitlinear(in_features=D, out_features=D)
            self.O_layer = Bitlinear(in_features=D, out_features=D)
    
        if not ternary:
            self.rms1 = BF16RMSNorm(dims=D)
            self.rms2 = BF16RMSNorm(dims=D)
            self.MLP = nn.Sequential(BF16Linear(input_dims=D, output_dims=4*D, bias=False), nn.GELU(), BF16Linear(input_dims=4*D, output_dims=D, bias=False))
        else:
            self.up_layer   = Bitlinear(in_features=D, out_features=4*D)
            self.down_layer = Bitlinear(in_features=4*D, out_features=D)


    def compute_qkv(self, X: mx.array) -> tuple:
        Q = self.Q_layer(X).reshape(X.shape[0], X.shape[1], -1, self.D_h).transpose(0, 2, 1, 3) ### (B, N, D)-->(B, N, D)-->(B, N, H, D_h)-->(B, H, N, D_h)
        K = self.K_layer(X).reshape(X.shape[0], X.shape[1], -1, self.D_h).transpose(0, 2, 1, 3) ### (B, N, D)-->(B, N, D)-->(B, N, H, D_h)-->(B, H, N, D_h)
        V = self.V_layer(X).reshape(X.shape[0], X.shape[1], -1, self.D_h).transpose(0, 2, 1, 3) ### (B, N, D)-->(B, N, D)-->(B, N, H, D_h)-->(B, H, N, D_h)
        return (mx.fast.rope(Q, dims=self.D_h, traditional=True, base=10000.00, scale=1.0, offset=0),
                 mx.fast.rope(K, dims=self.D_h, traditional=True, base=10000.00, scale=1.0, offset=0),
                 V)

    def __call__(self, X: mx.array):
        B, N = X.shape[0], X.shape[-2]
        Q, K, V = self.compute_qkv(self.rms1(X)) if not self.ternary else self.compute_qkv(X) ### pre-norm rmsnorm 1 before attention, this keeps softmax healthy
        sdpa_output = mx.fast.scaled_dot_product_attention(q=Q,k=K,v=V,mask="causal", scale=self.D_h ** -0.5)
        sdpa_output = sdpa_output.transpose(0,2,1,3).reshape(B, N, -1)  ### combine heads to make (B, N, D)
        output = self.O_layer(sdpa_output)  ### (B, N, D) = (B, N, D) @ (1, D, D)
        output = X + output  ### residual 1
        output = output + self.MLP(self.rms2(output)) if not self.ternary else output + self.down_layer(nn.gelu(self.up_layer(output))) ### rmsnorm 2 before MLP, and residual 2
        return output
        

class Transformer(nn.Module):
    def __init__(self, K: int, D: int, H: int, V: int, ternary: bool = False):
        super().__init__()
        assert D % H == 0

        self.embeddings = nn.Embedding(num_embeddings=V, dims=D)

        self.main = nn.Sequential(*[TransformerBlock(D, H, ternary) for _ in range(K)])

        self.rmsnorm_final = BF16RMSNorm(dims=D)

        self._init_weights(K)


    def _init_weights(self, K):
        normal = nn.init.normal(mean=0.0, std=0.02)

        def visit_module(name, module):
            if isinstance(module, (nn.RMSNorm, nn.LayerNorm)):
                return
            if isinstance(module, (nn.Linear, nn.Embedding, Bitlinear)):
                if name.endswith("O_layer") or name.endswith("MLP.layers.2") or name.endswith("down_layer"):
                    res_normal = nn.init.normal(mean=0.0, std=0.02 / math.sqrt(2 * K))
                    module.weight = res_normal(module.weight)
                else:
                    module.weight = normal(module.weight)

        self.apply_to_modules(visit_module)

    def __call__(self, X: mx.array, return_hidden: bool = False):
        embedded_X = self.embeddings(X).astype(mx.bfloat16)  ### (B,N) --> (B,N,D)
        intermediate = self.main(embedded_X)
        intermediate = self.rmsnorm_final(intermediate)
        if return_hidden:
            return intermediate
        return intermediate @ self.embeddings.weight.astype(intermediate.dtype).T