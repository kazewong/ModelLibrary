import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, Float, Bool
from typing import Union, Literal


# This model tries to mirror nanoGPT model from https://github.com/karpathy/nanoGPT/blob/master/model.py
class CausalSelfAttention(eqx.Module):

    c_attn: eqx.nn.Linear
    c_proj: eqx.nn.Linear
    attn_dropout: eqx.nn.Dropout
    resid_dropout: eqx.nn.Dropout
    mask: Float[Array, "block_size block_size"]

    n_head: int = 12
    n_embd: int = 768

    """
    TODO: Add flash attention support
    """

    def __init__(
        self,
        block_size: int = 1024,
        n_embd: int = 768,
        n_head: int = 12,
        dropout: float = 0.0,
        bias: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()
        assert n_embd % n_head == 0, "dimensions must be divisible by number of heads"
        self.n_embd = n_embd
        self.n_head = n_head
        key, key_attn, key_proj, key_init = jax.random.split(key, 4)
        # key, query, value projections for all heads, but in a batch
        self.c_attn = eqx.nn.Linear(n_embd, 3 * n_embd, use_bias=bias, key=key_attn)
        # output projection
        self.c_proj = eqx.nn.Linear(n_embd, n_embd, use_bias=bias, key=key_proj)
        # regularization
        self.attn_dropout = eqx.nn.Dropout(dropout)
        self.resid_dropout = eqx.nn.Dropout(dropout)
        self.mask = -jnp.inf * jnp.invert(jnp.tril(jnp.ones((block_size, block_size), dtype=bool)))
        self.mask = jnp.nan_to_num(self.mask, posinf=jnp.inf, neginf=-jnp.inf)

        # # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        # self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        # if not self.flash:
        #     print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        #     # causal mask to ensure that attention is only applied to the left in the input sequence
        #     self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
        #                                 .view(1, 1, block_size, block_size))

    def __call__(
        self, x: Float[Array, "n_seq n_embd"], key: PRNGKeyArray
    ) -> Float[Array, "n_seq n_embd"]:
        return self.forward(x, key=key)

    def forward(
        self, x: Float[Array, "n_seq n_embd"], key: PRNGKeyArray
    ) -> Float[Array, "n_seq n_embd"]:
        T, C = x.shape

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = jnp.split(jax.vmap(self.c_attn)(x), 3, axis=1)
        k = k.reshape((T, self.n_head, C // self.n_head)).swapaxes(
            0, 1
        )  # (n_head, n_seq, -1)
        q = q.reshape((T, self.n_head, C // self.n_head)).swapaxes(
            0, 1
        )  # (n_head, n_seq, -1)
        v = v.reshape((T, self.n_head, C // self.n_head)).swapaxes(
            0, 1
        )  # (n_head, n_seq, -1)

        att = (q @ k.swapaxes(-2, -1)) / jnp.sqrt(k.shape[-1])
        att +=  self.mask[:T, :T]
        att = jax.nn.softmax(att, axis=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.swapaxes(0, 1).reshape(T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(jax.vmap(self.c_proj)(y))
        return y


class MLP(eqx.Module):

    c_fc: eqx.nn.Linear
    c_proj: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        n_embd: int = 768,
        dropout: float = 0.0,
        bias: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()

        key, key_fc, key_proj = jax.random.split(key, 3)
        self.c_fc = eqx.nn.Linear(n_embd, 4 * n_embd, use_bias=bias, key=key_fc)
        self.c_proj = eqx.nn.Linear(4 * n_embd, n_embd, use_bias=bias, key=key_proj)
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(
        self, x: Float[Array, "n_seq n_embd"], key: PRNGKeyArray
    ) -> Float[Array, "n_seq n_embd"]:
        return self.forward(x, key=key)

    def forward(
        self, x: Float[Array, "n_seq n_embd"], *, key: PRNGKeyArray
    ) -> Float[Array, "n_seq n_embd"]:
        x = self.c_fc(x)
        x = jax.nn.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x, key=key)
        return x


class Block(eqx.Module):

    ln_1: eqx.nn.LayerNorm
    ln_2: eqx.nn.LayerNorm
    attn: CausalSelfAttention
    mlp: MLP

    def __init__(
        self,
        block_size: int = 1024,
        n_embd: int = 768,
        n_head: int = 12,
        dropout: float = 0.0,
        bias: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()
        key, key_att, key_mlp = jax.random.split(key, 3)
        self.ln_1 = eqx.nn.LayerNorm(n_embd, use_bias=bias)
        self.attn = CausalSelfAttention(
            block_size=block_size,
            n_embd=n_embd,
            n_head=n_head,
            dropout=dropout,
            bias=bias,
            key=key_att,
        )
        self.ln_2 = eqx.nn.LayerNorm(n_embd, use_bias=bias)
        self.mlp = MLP(n_embd=n_embd, dropout=dropout, bias=bias, key=key_mlp)

    def __call__(
        self, x: Float[Array, "n_seq n_embd"], key: PRNGKeyArray
    ) -> Float[Array, "n_seq n_embd"]:
        return self.forward(x, key=key)

    def forward(
        self, x: Float[Array, "n_seq n_embd"], key: PRNGKeyArray
    ) -> Float[Array, "n_seq n_embd"]:
        key, key_att, key_mlp = jax.random.split(key, 3)
        key_mlp = jax.random.split(key_mlp, x.shape[0])
        x = x + self.attn(jax.vmap(self.ln_1)(x), key=key_att)
        x = x + jax.vmap(self.mlp)(jax.vmap(self.ln_2)(x), key=key_mlp)
        return x


class GPT(eqx.Module):

    token_embedding: eqx.nn.Embedding
    position_embedding: eqx.nn.Embedding
    dropout: eqx.nn.Dropout
    blocks: list[Block]
    layer_norm: eqx.nn.LayerNorm
    lm_head: eqx.nn.Linear

    @property
    def n_layer(self) -> int:
        return len(self.blocks)

    @property
    def vocab_size(self) -> int:
        return self.token_embedding.num_embeddings

    @property
    def n_embed(self) -> int:
        return self.token_embedding.embedding_size

    @property
    def block_size(self) -> int:
        return self.position_embedding.num_embeddings

    def __init__(
        self,
        vocab_size: int = 50257,
        block_size: int = 1024,
        n_layer: int = 12,
        n_embd: int = 768,
        dropout: float = 0.0,
        bias: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()
        assert vocab_size is not None
        assert block_size is not None

        key, subkey = jax.random.split(key)
        self.token_embedding = eqx.nn.Embedding(vocab_size, n_embd, key=subkey)
        key, subkey = jax.random.split(key)
        self.position_embedding = eqx.nn.Embedding(block_size, n_embd, key=subkey)
        self.dropout = eqx.nn.Dropout(dropout)
        self.blocks = []
        for _ in range(n_layer):
            key, subkey = jax.random.split(key)
            self.blocks.append(Block(n_embd=n_embd, dropout=dropout, bias=bias, key=subkey))

        self.layer_norm = eqx.nn.LayerNorm(n_embd, use_bias=bias)
        key, subkey = jax.random.split(key)
        self.lm_head = eqx.nn.Linear(n_embd, vocab_size, key=subkey)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.token_embedding = eqx.tree_at(lambda l: l.weight, self.token_embedding,
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # initialize weights
        key, subkey = jax.random.split(key)
        new_model = self.init_weights(self, subkey)
        self.token_embedding = new_model.token_embedding
        self.position_embedding = new_model.position_embedding
        self.blocks = new_model.blocks
        self.layer_norm = new_model.layer_norm
        self.lm_head = new_model.lm_head

        # report number of parameters
        if jax.process_index() == 0:
            print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def __call__(self, x: Float[Array, "n_seq"], key: PRNGKeyArray) -> Float[Array, "n_seq n_embd"]:
        return self.forward(x, key=key)

    def get_num_params(self, non_embedding=True) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        arr, statics = eqx.partition(self, eqx.is_array)
        n_params = sum(jax.tree.leaves(jax.tree.map(lambda x: x.size, arr)))
        if non_embedding:
            n_params -= self.position_embedding.weight.size
        return n_params

    def init_weights(self, module: eqx.Module, key: PRNGKeyArray):
        is_linear = lambda x: isinstance(x, eqx.nn.Linear)
        is_embedding = lambda x: isinstance(x, eqx.nn.Embedding)
        is_causal_self_attention = lambda x: isinstance(x, CausalSelfAttention)
        get_weights = lambda m:[
            x.weight for x in jax.tree.leaves(m, is_leaf=is_linear) if is_linear(x)
        ]
        get_bias = lambda m:[
            x.bias for x in jax.tree.leaves(m, is_leaf=is_linear) if is_linear(x)
        ]
        get_embedding = lambda m:[
            x.weight for x in jax.tree.leaves(m, is_leaf=is_embedding) if is_embedding(x)
        ]
        get_projection = lambda m:[
            x.c_proj.weight for x in jax.tree_leaves(m, is_leaf=is_causal_self_attention) if is_causal_self_attention(x)
        ]
        new_weights = []
        new_biases = []
        new_embeddings = []
        new_projections = []
        for w in get_weights(module):
            key, subkey = jax.random.split(key)
            new_weights.append(jax.random.normal(subkey, w.shape)*0.02)
        for b in get_bias(module):
            key, subkey = jax.random.split(key)
            new_biases.append(jnp.zeros(b.shape))
        for e in get_embedding(module):
            key, subkey = jax.random.split(key)
            new_embeddings.append(jax.random.normal(subkey, e.shape)*0.02)
        for p in get_projection(module):
            key, subkey = jax.random.split(key)
            new_projections.append(jax.random.normal(subkey, p.shape)*0.02/jnp.sqrt(2 * self.n_layer))
        new_model = eqx.tree_at(get_weights, module, new_weights)
        new_model = eqx.tree_at(get_bias, new_model, new_biases)
        new_model = eqx.tree_at(get_embedding, new_model, new_embeddings)
        new_model = eqx.tree_at(get_projection, new_model, new_projections)
        return new_model

    def forward(self, x: Float[Array, "n_seq"], key: PRNGKeyArray) -> Float[Array, "n_seq n_embd"]:
        pos = jnp.arange(0, x.shape[0], dtype=x.dtype)

        # forward the GPT model itself
        tok_emb = jax.vmap(self.token_embedding)(x)  # token embeddings of shape (t, n_embd)
        pos_emb = jax.vmap(self.position_embedding)(pos)  # position embeddings of shape (t, n_embd)
        x = self.dropout(tok_emb + pos_emb)
        for block in self.blocks:
            key, subkey = jax.random.split(key)
            x = block(x, key=subkey)
        x = jax.vmap(self.layer_norm)(x)
        return jax.vmap(self.lm_head)(x)

        # if targets is not None:
        #     # if we are given some desired targets also calculate the loss
        #     logits = self.lm_head(x)
        #     loss = F.cross_entropy(
        #         logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        #     )
        # else:
        #     # inference-time mini-optimization: only forward the lm_head on the very last position
        #     logits = self.lm_head(
        #         x[:, [-1], :]
        #     )  # note: using list [-1] to preserve the time dim
        #     loss = None

        # return logits, loss

    # def crop_block_size(self, block_size):
    #     # model surgery to decrease the block size if necessary
    #     # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
    #     # but want to use a smaller block size for some smaller, simpler model
    #     assert block_size <= self.block_size
    #     self.block_size = block_size
    #     self.transformer.wpe.weight = nn.Parameter(
    #         self.transformer.wpe.weight[:block_size]
    #     )
    #     for block in self.transformer.h:
    #         if hasattr(block.attn, "bias"):
    #             block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    # @classmethod
    # def from_pretrained(cls, model_type, override_args=None):
    #     assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
    #     override_args = override_args or {}  # default to empty dict
    #     # only dropout can be overridden see more notes below
    #     assert all(k == "dropout" for k in override_args)
    #     from transformers import GPT2LMHeadModel

    #     print("loading weights from pretrained gpt: %s" % model_type)

    #     # n_layer, n_head and n_embd are determined from model_type
    #     config_args = {
    #         "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    #         "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
    #         "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
    #         "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
    #     }[model_type]
    #     print("forcing vocab_size=50257, block_size=1024, bias=True")
    #     config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
    #     config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
    #     config_args["bias"] = True  # always True for GPT model checkpoints
    #     # we can override the dropout rate, if desired
    #     if "dropout" in override_args:
    #         print(f"overriding dropout rate to {override_args['dropout']}")
    #         config_args["dropout"] = override_args["dropout"]
    #     # create a from-scratch initialized minGPT model
    #     config = GPTConfig(**config_args)
    #     model = GPT(config)
    #     sd = model.state_dict()
    #     sd_keys = sd.keys()
    #     sd_keys = [
    #         k for k in sd_keys if not k.endswith(".attn.bias")
    #     ]  # discard this mask / buffer, not a param

    #     # init a huggingface/transformers model
    #     model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    #     sd_hf = model_hf.state_dict()

    #     # copy while ensuring all of the parameters are aligned and match in names and shapes
    #     sd_keys_hf = sd_hf.keys()
    #     sd_keys_hf = [
    #         k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
    #     ]  # ignore these, just a buffer
    #     sd_keys_hf = [
    #         k for k in sd_keys_hf if not k.endswith(".attn.bias")
    #     ]  # same, just the mask (buffer)
    #     transposed = [
    #         "attn.c_attn.weight",
    #         "attn.c_proj.weight",
    #         "mlp.c_fc.weight",
    #         "mlp.c_proj.weight",
    #     ]
    #     # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
    #     # this means that we have to transpose these weights when we import them
    #     assert len(sd_keys_hf) == len(
    #         sd_keys
    #     ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    #     for k in sd_keys_hf:
    #         if any(k.endswith(w) for w in transposed):
    #             # special treatment for the Conv1D weights we need to transpose
    #             assert sd_hf[k].shape[::-1] == sd[k].shape
    #             with torch.no_grad():
    #                 sd[k].copy_(sd_hf[k].t())
    #         else:
    #             # vanilla copy over the other parameters
    #             assert sd_hf[k].shape == sd[k].shape
    #             with torch.no_grad():
    #                 sd[k].copy_(sd_hf[k])

    #     return model

    # def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    #     # start with all of the candidate parameters
    #     param_dict = {pn: p for pn, p in self.named_parameters()}
    #     # filter out those that do not require grad
    #     param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    #     # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    #     # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    #     decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    #     nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    #     optim_groups = [
    #         {"params": decay_params, "weight_decay": weight_decay},
    #         {"params": nodecay_params, "weight_decay": 0.0},
    #     ]
    #     num_decay_params = sum(p.numel() for p in decay_params)
    #     num_nodecay_params = sum(p.numel() for p in nodecay_params)
    #     print(
    #         f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    #     )
    #     print(
    #         f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    #     )
    #     # Create AdamW optimizer and use the fused version if it is available
    #     fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    #     use_fused = fused_available and device_type == "cuda"
    #     extra_args = dict(fused=True) if use_fused else dict()
    #     optimizer = torch.optim.AdamW(
    #         optim_groups, lr=learning_rate, betas=betas, **extra_args
    #     )
    #     print(f"using fused AdamW: {use_fused}")

    #     return optimizer

    # def estimate_mfu(self, fwdbwd_per_iter, dt):
    #     """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
    #     # first estimate the number of flops we do per iteration.
    #     # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
    #     N = self.get_num_params()
    #     cfg = self.config
    #     L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
    #     flops_per_token = 6 * N + 12 * L * H * Q * T
    #     flops_per_fwdbwd = flops_per_token * T
    #     flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    #     # express our flops throughput as ratio of A100 bfloat16 peak flops
    #     flops_achieved = flops_per_iter * (1.0 / dt)  # per second
    #     flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
    #     mfu = flops_achieved / flops_promised
    #     return mfu

    # @torch.no_grad()
    # def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    #     """
    #     Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    #     the sequence max_new_tokens times, feeding the predictions back into the model each time.
    #     Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    #     """
    #     for _ in range(max_new_tokens):
    #         # if the sequence context is growing too long we must crop it at block_size
    #         idx_cond = (
    #             idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
    #         )
    #         # forward the model to get the logits for the index in the sequence
    #         logits, _ = self(idx_cond)
    #         # pluck the logits at the final step and scale by desired temperature
    #         logits = logits[:, -1, :] / temperature
    #         # optionally crop the logits to only the top k options
    #         if top_k is not None:
    #             v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    #             logits[logits < v[:, [-1]]] = -float("Inf")
    #         # apply softmax to convert logits to (normalized) probabilities
    #         probs = F.softmax(logits, dim=-1)
    #         # sample from the distribution
    #         idx_next = torch.multinomial(probs, num_samples=1)
    #         # append sampled index to the running sequence and continue
    #         idx = torch.cat((idx, idx_next), dim=1)

    #     return idx
