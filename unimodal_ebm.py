import math
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from losses import compute_reconstruction_loss
import torch.nn.functional as F
class ResidualBlock(nn.Module):
    """
    Residual block to allow deeper networks while preserving gradient flow.
    """
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(dim, dim)),
            nn.SiLU(),
            spectral_norm(nn.Linear(dim, dim)),
            nn.SiLU()
        )
    def forward(self, x):
        return x + self.net(x)
class FlatEBM(nn.Module):
    """
    The Energy-Based Prior (Flat EBM).
    It acts as a tilting function E_theta(z) on top of a base distribution p_0(z).
    The base distribution p_0(z) is a standard Gaussian N(0, I).
    The final EBM prior is p_theta(z) \propto p_0(z) \exp(-E_theta(z)).
    """
    def __init__(self, latent_dim=768, hidden_dim=1024, num_res_blocks=4):
        super().__init__()
        layers = [
            spectral_norm(nn.Linear(latent_dim, hidden_dim)),
            nn.SiLU(),
        ]
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(hidden_dim))
        layers.append(spectral_norm(nn.Linear(hidden_dim, 1)))
        self.net = nn.Sequential(*layers)
    def forward(self, z):
        return self.net(z)
class MCMC_Sampler:
    """
    Langevin Dynamics MCMC Sampler to sample from the EBM prior p_theta(z).
    """
    def __init__(self, ebm, latent_dim=768, step_size=0.3, num_steps=50, buffer_size=10000, replay_ratio=0.95):
        self.ebm = ebm
        self.latent_dim = latent_dim
        self.step_size = step_size
        self.num_steps = num_steps
        self.buffer_size = buffer_size
        self.replay_ratio = replay_ratio
        self.buffer = torch.randn(buffer_size, latent_dim)
    def sample(self, batch_size, device):
        num_fresh = int(batch_size * (1 - self.replay_ratio))
        num_replay = batch_size - num_fresh
        self.replay_indices = torch.randint(0, self.buffer_size, (num_replay,))
        z_replay = self.buffer[self.replay_indices].clone().to(device)
        z_fresh = torch.randn(num_fresh, self.latent_dim, device=device)
        z = torch.cat([z_replay, z_fresh], dim=0)
        z.requires_grad_(True)
        z_start = z.detach().clone()
        mcmc_log_table = []
        for step in range(self.num_steps):
            prior_energy = 0.5 * (z ** 2).sum(dim=-1)
            ebm_energy = self.ebm(z).squeeze(-1)
            grad_prior = z.clone()
            grad_ebm = torch.autograd.grad(ebm_energy.sum(), z)[0]
            grad_z = grad_prior + grad_ebm
            noise = torch.randn_like(z)
            update = -0.5 * self.step_size * grad_z
            z.data = z.data + update + math.sqrt(self.step_size) * noise
            with torch.no_grad():
                mcmc_log_table.append({
                    "step": step,
                    "total_energy": (prior_energy + ebm_energy).mean().item(),
                    "ebm_energy": ebm_energy.mean().item(),
                    "prior_energy": prior_energy.mean().item(),
                    "grad_z_norm": grad_z.norm(dim=-1).mean().item(),
                    "grad_prior_norm": grad_prior.norm(dim=-1).mean().item(),
                    "grad_ebm_norm": grad_ebm.norm(dim=-1).mean().item(),
                    "update_norm": update.norm(dim=-1).mean().item(),
                    "noise_norm": math.sqrt(self.step_size) * noise.norm(dim=-1).mean().item(),
                    "distance_from_start": (z - z_start).norm(dim=-1).mean().item()
                })
        z_final = z.detach()
        self.buffer[self.replay_indices] = z_final[:num_replay].cpu()
        fresh_replace_indices = torch.randint(0, self.buffer_size, (num_fresh,))
        self.buffer[fresh_replace_indices] = z_final[num_replay:].cpu()
        return z_final, mcmc_log_table
class UnimodalGenerativeModelWithEBMPrior(nn.Module):
    """
    The full Generative Model integrating the Inference Model (Encoder), 
    Generative Model (Decoder), and the EBM Prior.
    """
    def __init__(self, encoder, decoder, latent_dim=768):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.ebm = FlatEBM(latent_dim=latent_dim)
    def reparameterize(self, mu, logvar):
        """ The reparameterization trick to sample z ~ q(z|x) """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, dec_input_ids, dec_attention_mask, enc_input_ids, enc_attention_mask):
        outputs = self.encoder(enc_input_ids, enc_attention_mask)
        pooled_output = outputs[1]
        projection = self.encoder.linear(pooled_output)
        mu, logvar = projection.chunk(2, dim=-1)
        z_posterior = self.reparameterize(mu, logvar)
        prefix_ones = torch.ones((dec_input_ids.shape[0], 1), device=dec_input_ids.device, dtype=dec_input_ids.dtype)
        dec_attention_mask = torch.cat([prefix_ones, dec_attention_mask], dim=1)
        out = self.decoder(
            input_ids=dec_input_ids,
            attention_mask=dec_attention_mask,
            past=z_posterior,
        )
        reconstruction_logits = out[0]
        return reconstruction_logits, mu, logvar, z_posterior
def train_step_algorithm1(dec_input_ids, target_ids, dec_attention_mask, enc_input_ids, enc_attention_mask, model, optimizer_vae, optimizer_ebm, sampler, device, beta=1.0, max_norm=1.0):
    """
    Training script implementation according to Algorithm 1.
    We alternate between updating the VAE (Encoder and Decoder) and the EBM Prior.
    """
    model.train()
    reconstruction_logits, mu, logvar, z_posterior = model(dec_input_ids, dec_attention_mask, enc_input_ids, enc_attention_mask)
    reco_loss, token_loss = compute_reconstruction_loss(
        logits=reconstruction_logits,
        targets=target_ids,
        attention_mask=dec_attention_mask
    )
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_base = kl_per_dim.mean()
    free_bits = 0.0
    free_nats_kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    kl_loss_term = free_nats_kl_per_dim.sum(dim=-1).mean()
    ebm_energy_posterior = model.ebm(z_posterior).squeeze(-1).mean()
    total_kl = kl_base + ebm_energy_posterior
    vae_loss = 20*reco_loss + 1.0 * kl_loss_term + ebm_energy_posterior
    grad_reco_mu, grad_reco_logvar = torch.autograd.grad(20*reco_loss, [mu, logvar], retain_graph=True)
    grad_kl_mu, grad_kl_logvar = torch.autograd.grad(1.0 * kl_loss_term, [mu, logvar], retain_graph=True)
    grad_ebm_mu, grad_ebm_logvar = torch.autograd.grad(ebm_energy_posterior, [mu, logvar], retain_graph=True)
    force_reco = (grad_reco_mu.norm() + grad_reco_logvar.norm()).item()
    force_kl = (grad_kl_mu.norm() + grad_kl_logvar.norm()).item()
    force_ebm = (grad_ebm_mu.norm() + grad_ebm_logvar.norm()).item()
    optimizer_vae.zero_grad()
    vae_loss.backward()
    if max_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), max_norm)
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm)
    optimizer_vae.step()
    z_pos = z_posterior.detach()
    energy_pos = model.ebm(z_pos).squeeze(-1).mean()
    z_neg, mcmc_log_table = sampler.sample(batch_size=dec_input_ids.size(0), device=device)
    energy_neg = model.ebm(z_neg).squeeze(-1).mean()
    ebm_loss = energy_pos - energy_neg
    reg_loss = 1e-6 * ((energy_pos ** 2) + (energy_neg ** 2))
    total_ebm_loss = ebm_loss            
    optimizer_ebm.zero_grad()
    total_ebm_loss.backward()
    if max_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.ebm.parameters(), max_norm)
    optimizer_ebm.step()
    with torch.no_grad():
        sigma_posterior = torch.exp(0.5 * logvar)  # (batch, latent_dim)
        mu_mean = mu.abs().mean().item()
        sigma_mean = sigma_posterior.mean().item()
        z_post_distance = z_posterior.norm(dim=-1).mean().item()  # mean L2 norm across batch
    metrics = {
        "vae_loss": vae_loss.item(),
        "CD_loss": ebm_loss.item(),
        "reco_loss": reco_loss.item(),
        "token_loss": token_loss.item(),
        "kl_base": kl_base.item(),
        "ebm_energy_posterior": ebm_energy_posterior.item(),
        "total_kl": (total_kl).item(),
        "force_reco": force_reco,
        "force_kl": force_kl,
        "force_ebm": force_ebm,
        "posterior/mu_abs_mean": mu_mean,
        "posterior/sigma_mean": sigma_mean,
        "posterior/z_distance_from_origin": z_post_distance,
    }
    return metrics, mcmc_log_table
