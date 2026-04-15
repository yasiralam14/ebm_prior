import sys
import torch
import wandb
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from losses import compute_reconstruction_loss
from algo2_ebm import DecoderOnlyModelWithEBMPrior, sample_langevin_prior, sample_langevin_posterior, SmallRandomDecoder, OptimusDecoderWrapper
def train_unimodal_ebm_algo2(decoder, train_loader, epochs=10, latent_dim=768, lr_e=0.00002, lr_g=0.0001, device='cuda'):
    """
    Implements the core training loops for the non-amortized Generator + EBM.
    """
    K_0, a_0, K_1, a_1 = 60, 0.4, 40, 0.1
    llhd_sigma = 0.3
    llhd_weight = 1.0 / (2.0 * llhd_sigma * llhd_sigma)
    model = DecoderOnlyModelWithEBMPrior(decoder=decoder, latent_dim=latent_dim).to(device)
    optG = torch.optim.Adam(model.decoder.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optE = torch.optim.Adam(model.ebm.parameters(), lr=lr_e, betas=(0.5, 0.999))
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        model.train()
        last_mcmc_prior_table = None
        last_mcmc_post_table = None
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(progress_bar):
            dec_input_ids = batch[0].to(device)
            dec_word_mask = batch[1].to(device)
            batch_size = dec_input_ids.size(0)
            z_e_0 = torch.randn(batch_size, latent_dim, device=device)
            z_g_0 = torch.randn(batch_size, latent_dim, device=device)
            z_e_k, prior_table = sample_langevin_prior(
                z_e_0, model.ebm, K_0=K_0, a_0=a_0
            )
            z_g_k, post_table = sample_langevin_posterior(
                z_g_0, dec_input_ids, dec_input_ids, dec_word_mask, dec_word_mask, 
                model.decoder, model.ebm, K_1=K_1, a_1=a_1, llhd_weight=llhd_weight
            )
            last_mcmc_prior_table = prior_table
            last_mcmc_post_table = post_table
            optG.zero_grad()
            out = model.decoder(
                input_ids=dec_input_ids,
                attention_mask=dec_word_mask,
                past=z_g_k.detach()
            )
            loss_g, token_loss = compute_reconstruction_loss(out[0], dec_input_ids, dec_word_mask)
            loss_g.backward()
            optG.step()
            optE.zero_grad()
            en_pos = model.ebm(z_g_k.detach()).mean()
            en_neg = model.ebm(z_e_k.detach()).mean()
            loss_e = en_pos - en_neg
            loss_e.backward()
            optE.step()
            if (batch_idx + 1) % 50 == 0:
                wandb.log({
                    "loss_g (recon)": loss_g.item(),
                    "token_loss": token_loss.item(),
                    "loss_e (CD)": loss_e.item(),
                    "en_pos": en_pos.item(),
                    "en_neg": en_neg.item(),
                })
            progress_bar.set_postfix({
                "L_G": f"{loss_g.item():.2f}",
                "L_E": f"{loss_e.item():.2f}"
            })
        if last_mcmc_prior_table is not None:
            df_prior = pd.DataFrame(last_mcmc_prior_table)
            df_post = pd.DataFrame(last_mcmc_post_table)
            wandb.log({
                f"Prior_MCMC_Epoch{epoch+1}": wandb.Table(dataframe=df_prior),
                f"Post_MCMC_Epoch{epoch+1}": wandb.Table(dataframe=df_post)
            })
    return model
inference_dir = "/home/salam4/hvae_project/Optimus"
sys.path.insert(0, inference_dir)
from pretrained_checkpoints.inference import load_model, InferenceArgs
from data.create_loaders import DualTokenizerDataset, make_dual_collate_fn
if __name__ == "__main__":
    CHECKPOINT_PATH = "/home/salam4/hvae_project/Optimus/pretrained_checkpoints/optimus-vae.pth" 
    args = InferenceArgs()
    print("\nLoading Model Checkpoints...")
    model_vae, enc_tok, dec_tok = load_model(CHECKPOINT_PATH, args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading Dataset...")
    df = pd.read_parquet("/home/salam4/hvae_project/Optimus/data/datasets/sentences_df.parquet")
    texts = df["sentence"].astype(str).tolist() 
    dataset = DualTokenizerDataset(texts, enc_tok, dec_tok)
    train_df, valid_df = train_test_split(dataset, test_size=0.85, random_state=42)
    collate_fn = make_dual_collate_fn(enc_tok, dec_tok, max_length=50)
    train_loader = DataLoader(train_df, batch_size=64, shuffle=True, collate_fn=collate_fn)
    wandb.init(
        project="Energy based prior model",
        name="Optimus decoder - fixed attn mask"
    )   
    print("\nStarting Decoder-Only EBM Training...")
    optimus_decoder = OptimusDecoderWrapper(model_vae.decoder).to(device)
    model = train_unimodal_ebm_algo2(
        decoder=optimus_decoder,
        train_loader=train_loader,
        epochs=2,
        latent_dim=768,
        lr_e=0.00002, 
        lr_g=0.00001,
        device=device
    )
    print("Training complete!")
    wandb.finish()
