import math
import torch
import torch.nn as nn
from unimodal_ebm import FlatEBM
from losses import compute_reconstruction_loss
class OptimusDecoderWrapper(nn.Module):
    """
    Wraps the Optimus GPT2ForLatentConnector decoder to fix the attention-mask
    size mismatch that occurs when a latent vector is injected as KV memory.
    When `past=z` is passed, the Optimus GPT-2 transformer expands it into
    1 virtual "past" token per layer (past_length = 1), making the effective
    key-sequence length  seq_len + 1.  The attention_mask, however, is still
    seq_len wide after the standard unsqueeze-broadcast in GPT2Model.forward,
    so the add  w + attention_mask  fails with a size mismatch.
    Fix: prepend a column of 1s ("attend to this slot") to the attention_mask
    along the sequence dimension before the call, so the mask covers
    seq_len + 1 positions and broadcasts correctly.
    """
    def __init__(self, optimus_decoder):
        super().__init__()
        self.decoder = optimus_decoder
    def forward(self, input_ids, attention_mask=None, past=None):
        if past is not None and attention_mask is not None:
            latent_ones = torch.ones(
                attention_mask.size(0), 1,
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            attention_mask = torch.cat([latent_ones, attention_mask], dim=1)
        return self.decoder(input_ids, past=past, attention_mask=attention_mask)
class SmallRandomDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, latent_dim=768, num_layers=4, num_heads=8, max_seq_len=512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.norm_combine = nn.LayerNorm(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4, 
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
    def forward(self, input_ids, attention_mask=None, past=None):
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        tok_embeds = self.token_emb(input_ids)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.pos_emb(positions)
        x = tok_embeds + pos_embeds
        if past is not None:
            z_proj = self.latent_proj(past)
            x = x + z_proj.unsqueeze(1)
            x = self.norm_combine(x)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.size(1) > seq_len:
                attention_mask = attention_mask[:, -seq_len:]
            key_padding_mask = (attention_mask == 0)
        h = self.transformer(
            x, 
            mask=causal_mask, 
            is_causal=True,
            src_key_padding_mask=key_padding_mask
        )
        h = self.final_norm(h)
        logits = self.lm_head(h)
        return (logits,)
class DecoderOnlyModelWithEBMPrior(nn.Module):
    """
    Model Wrapper that only uses the standalone generative model (Decoder)
    and an Energy Based Model Prior. It drops the amortized Inference Model.
    """
    def __init__(self, decoder, latent_dim=768):
        super().__init__()
        self.decoder = decoder
        self.ebm = FlatEBM(latent_dim=latent_dim)
def sample_langevin_prior(z, ebm, K_0=60, a_0=0.4):
    """
    Sample from the model prior using Unadjusted Langevin Algorithm (ULA).
    Targets: p_theta(z) \propto e^{-E_theta(z)} * p_0(z)
    (assuming p_0 is N(0, I))
    """
    z = z.clone().detach().requires_grad_(True)
    mcmc_log_table = []
    for i in range(K_0):
        en = ebm(z).squeeze(-1)
        z_grad = torch.autograd.grad(en.sum(), z)[0]
        grad_prior = z.clone()
        grad_total = z_grad + grad_prior
        noise = torch.randn_like(z)
        z.data = z.data - 0.5 * a_0 * a_0 * grad_total + a_0 * noise.data
        with torch.no_grad():
            mcmc_log_table.append({
                "step": i,
                "total_energy": (en + 0.5 * (z**2).sum(-1)).mean().item(),
                "ebm_energy": en.mean().item(),
                "grad_e_norm": z_grad.norm(dim=-1).mean().item(),
                "grad_prior_norm": grad_prior.norm(dim=-1).mean().item()
            })
    return z.detach(), mcmc_log_table
def sample_langevin_posterior(z, dec_input_ids, target_ids, dec_attention_mask, dec_word_mask, G, E, K_1=40, a_1=0.1, llhd_weight=1.0):
    """
    Sample from the true posterior using MCMC.
    Since we only have the Decoder, we backprop right through it to sample z!
    Targets: p(z|x) \propto p_theta(x|z) * p_theta(z)
    """
    z = z.clone().detach().requires_grad_(True)
    mcmc_log_table = []
    batch_size = z.size(0)
    for i in range(K_1):
        en = E(z).squeeze(-1)
        grad_e = torch.autograd.grad(en.sum(), z)[0]
        out = G(
            input_ids=dec_input_ids,
            attention_mask=dec_attention_mask,
            past=z
        )
        sentence_recon_loss, _ = compute_reconstruction_loss(out[0], target_ids, dec_word_mask)
        total_reco_loss_sum = sentence_recon_loss * batch_size 
        grad_g = torch.autograd.grad(total_reco_loss_sum, z)[0]
        grad_prior = z.clone()
        grad_total = (llhd_weight * grad_g) + grad_e + grad_prior
        noise = torch.randn_like(z)
        z.data = z.data - 0.5 * a_1 * a_1 * grad_total + a_1 * noise.data
        with torch.no_grad():
            mcmc_log_table.append({
                "step": i,
                "total_energy": ((llhd_weight * total_reco_loss_sum) + en + 0.5 * (z**2).sum(-1)).mean().item(),
                "ebm_energy": en.mean().item(),
                "recon_loss": sentence_recon_loss.item(),                                
                "grad_g_norm": (llhd_weight * grad_g).norm(dim=-1).mean().item(),
                "grad_e_norm": grad_e.norm(dim=-1).mean().item(),
                "grad_prior_norm": grad_prior.norm(dim=-1).mean().item()
            })
    return z.detach(), mcmc_log_table
