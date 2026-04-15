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
        
        # Initial projection to hidden dim
        layers = [
            spectral_norm(nn.Linear(latent_dim, hidden_dim)),
            nn.SiLU(),
        ]
        
        # Add residual blocks for deep expressivity
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(hidden_dim))
            
        # Final projection to energy scalar with spectral norm
        # Added spectral norm to the final layer to enforce Lipschitz continuity everywhere.
        layers.append(spectral_norm(nn.Linear(hidden_dim, 1)))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, z):
        # Outputs the energy E_theta(z) for the given latent code
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
        
        # Initialize the replay buffer on CPU to save VRAM
        self.buffer = torch.randn(buffer_size, latent_dim)
        
    def sample(self, batch_size, device):
        # Determine how many samples to pull from buffer vs base distribution
        num_fresh = int(batch_size * (1 - self.replay_ratio))
        num_replay = batch_size - num_fresh
        
        # Draw samples from the replay buffer
        self.replay_indices = torch.randint(0, self.buffer_size, (num_replay,))
        z_replay = self.buffer[self.replay_indices].clone().to(device)
        
        # Draw fresh samples from the base distribution p_0(z) ~ N(0, I)
        z_fresh = torch.randn(num_fresh, self.latent_dim, device=device)
        
        # Combine
        z = torch.cat([z_replay, z_fresh], dim=0)
        z.requires_grad_(True)
        
        z_start = z.detach().clone()
        
        # Run Unadjusted Langevin Algorithm (ULA)
        mcmc_log_table = []
        for step in range(self.num_steps):
            # Target distribution energy: U(z) = -log p_0(z) + E_theta(z)
            # -log p_0(z) is 0.5 * ||z||^2 (ignoring the normalizing constant)
            prior_energy = 0.5 * (z ** 2).sum(dim=-1)
            ebm_energy = self.ebm(z).squeeze(-1)
            
            # Compute gradients separately to track their individual forces
            # The prior is basically N(0, I) (energy is 0.5 * ||z||^2), so d/dz is just z!
            # Removing autograd here speeds up MCMC significantly.
            grad_prior = z.clone()
            grad_ebm = torch.autograd.grad(ebm_energy.sum(), z)[0]
            
            # Total gradient is the sum
            grad_z = grad_prior + grad_ebm
            
            # Add Gaussian noise
            noise = torch.randn_like(z)
            
            # Gradient descent step on energy (equivalent to gradient ascent on log prob) + noise diffusion
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
            
        # Update the replay buffer with the refined samples (moved back to CPU)
        z_final = z.detach()
        self.buffer[self.replay_indices] = z_final[:num_replay].cpu()
        
        # Replace random other spots in the buffer with the completely fresh items
        # To ensure the buffer gradually introduces completely new paths explored by MCMC
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
        # Predefined Inference Model. MUST output parameters of the posterior distribution (mu, logvar)
        self.encoder = encoder
        
        # Predefined Generative Model. Takes z and produces the reconstruction.
        self.decoder = decoder
        
        # EBM modeling the prior
        self.ebm = FlatEBM(latent_dim=latent_dim)
        
    def reparameterize(self, mu, logvar):
        """ The reparameterization trick to sample z ~ q(z|x) """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, dec_input_ids, dec_attention_mask, enc_input_ids, enc_attention_mask):
        # 1. Inference: Obtain posterior distribution parameters from encoder
        outputs = self.encoder(enc_input_ids, enc_attention_mask)
        pooled_output = outputs[1]
        projection = self.encoder.linear(pooled_output)
        mu, logvar = projection.chunk(2, dim=-1)
        
        # 2. Sample a latent vector from the posterior using reparameterization
        z_posterior = self.reparameterize(mu, logvar)

        prefix_ones = torch.ones((dec_input_ids.shape[0], 1), device=dec_input_ids.device, dtype=dec_input_ids.dtype)

        dec_attention_mask = torch.cat([prefix_ones, dec_attention_mask], dim=1)
        
        
        out = self.decoder(
            input_ids=dec_input_ids,
            attention_mask=dec_attention_mask,
            past=z_posterior,
        )
        reconstruction_logits = out[0]
        
        # 3. Generation: Reconstruct data point from the sampled latent vector
        
        return reconstruction_logits, mu, logvar, z_posterior


def train_step_algorithm1(dec_input_ids, target_ids, dec_attention_mask, enc_input_ids, enc_attention_mask, model, optimizer_vae, optimizer_ebm, sampler, device, beta=1.0, max_norm=1.0):
    """
    Training script implementation according to Algorithm 1.
    We alternate between updating the VAE (Encoder and Decoder) and the EBM Prior.
    """
    model.train()
    
    # =========================================================
    # Phase 1: Update Inference Model (Encoder) & Generative Model (Decoder)
    # =========================================================
    
    reconstruction_logits, mu, logvar, z_posterior = model(dec_input_ids, dec_attention_mask, enc_input_ids, enc_attention_mask)
    
    # -- Compute Reconstruction Loss --
    reco_loss, token_loss = compute_reconstruction_loss(
        logits=reconstruction_logits,
        targets=target_ids,
        attention_mask=dec_attention_mask
    )
    
    # -- Compute KL Divergence Loss --
    # KL between inference posterior q_phi(z|x) and EBM prior p_theta(z)
    # KL(q_phi || p_theta) = KL(q_phi || p_0) + E_{q_phi}[E_theta(z)] + log Z_theta
    # (Since log Z_theta is constant wrt the encoder/decoder, we omit it from VAE loss calculation)
    
    # 1. KL divergence between q_phi and the base Gaussian p_0
    kl_base = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
    
    # 2. Expected energy of the posterior samples evaluated by the EBM
    # We detach the EBM parameters here because the EBM shouldn't be optimized 
    # to lower the energy of negative samples generated by an unoptimized encoder.
    ebm_energy_posterior = model.ebm(z_posterior).squeeze(-1).mean()
    
    # Total KL Divergence
    total_kl = kl_base + ebm_energy_posterior

    # Apply Hinge Loss (Free Bits) to kl_base only, softened with squaring
    # This creates a margin of 100. Below 100, the gradient is 0. 
    # Squaring it ensures the gradient starts small at 101 and grows smoothly.
    kl_base_clamped = F.softplus(kl_base - 100.0, beta=1.0)
    
    # Total VAE Loss (Using clamped kl_base, but keeping ebm_energy unrestrained)
    vae_loss = reco_loss + beta * (kl_base_clamped )+ ebm_energy_posterior
    
    # === COMPUTE FORCE NORMS ===
    # Calculate gradients of the three competing forces at the bottleneck (mu, logvar).
    # This perfectly represents their pull back into the encoder without needing 3 full backprops!
    grad_reco_mu, grad_reco_logvar = torch.autograd.grad(reco_loss, [mu, logvar], retain_graph=True)
    grad_kl_mu, grad_kl_logvar = torch.autograd.grad(beta * kl_base_clamped, [mu, logvar], retain_graph=True)
    grad_ebm_mu, grad_ebm_logvar = torch.autograd.grad(beta * ebm_energy_posterior, [mu, logvar], retain_graph=True)
    
    force_reco = (grad_reco_mu.norm() + grad_reco_logvar.norm()).item()
    force_kl = (grad_kl_mu.norm() + grad_kl_logvar.norm()).item()
    force_ebm = (grad_ebm_mu.norm() + grad_ebm_logvar.norm()).item()
    
    optimizer_vae.zero_grad()
    vae_loss.backward()
    if max_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), max_norm)
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm)
    optimizer_vae.step()
    
    
    # =========================================================
    # Phase 2: Update Energy-Based Model (EBM) Prior
    # =========================================================
    
    # 1. Sample from the posterior distribution using data (Positive Samples)
    # We detach z_posterior so the gradient doesn't flow back to the encoder.
    z_pos = z_posterior.detach()
    energy_pos = model.ebm(z_pos).squeeze(-1).mean()
    
    # 2. Sample from the EBM Prior using MCMC Langevin Dynamics (Negative Samples)
    z_neg, mcmc_log_table = sampler.sample(batch_size=dec_input_ids.size(0), device=device)
    energy_neg = model.ebm(z_neg).squeeze(-1).mean()
    
    # 3. Compute EBM Loss
    # We want to LOWER the energy of posterior samples and RAISE the energy of prior samples.
    # Allowing the EBM to dig freely without a margin to ensure it can match constantly shifting KL bases.
    ebm_loss = energy_pos - energy_neg
    
    # Vastly reduced L2 Regularization on energy magnitudes. 
    # Down to 1e-6 so the EBM isn't physically blocked from reaching deeply negative target numbers (-1500),
    # but still provides a tether against absolute numerical drift.
    reg_loss = 1e-6 * ((energy_pos ** 2) + (energy_neg ** 2))
    
    total_ebm_loss = ebm_loss #+ reg_loss
    
    optimizer_ebm.zero_grad()
    total_ebm_loss.backward()
    if max_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.ebm.parameters(), max_norm)
    optimizer_ebm.step()
    
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
    }
    
    return metrics, mcmc_log_table
