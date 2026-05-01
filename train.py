import sys
import torch
import wandb
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from unimodal_ebm import UnimodalGenerativeModelWithEBMPrior, MCMC_Sampler, train_step_algorithm1
def get_cyclic_beta(batch_idx, total_batches, start_beta=0.005, max_beta=1.0):
    """
    Computes a cyclic beta value for KL/EBM annealing.
    - 0% to 50%: stays at start_beta
    - 50% to 75%: linearly increases to max_beta
    - 75% to 100%: stays at max_beta
    Resets instantly to start_beta at the start of the next epoch.
    """
    progress = batch_idx / total_batches
    if progress <= 0.5:
        return start_beta
    elif progress <= 0.75:
        increase = (progress - 0.5) / 0.25
        return start_beta + increase * (max_beta - start_beta)
    else:
        return max_beta
def train_unimodal_ebm(encoder, decoder, train_loader, epochs=10, latent_dim=768, lr=3e-4, beta=1.0, max_norm=1.0, device='cuda'):
    """
    Main training loop for the Unimodal Generative Model with EBM Prior.
    Assumes `encoder`, `decoder`, and `train_loader` are already constructed and passed in.
    """
    model = UnimodalGenerativeModelWithEBMPrior(
        encoder=encoder, 
        decoder=decoder, 
        latent_dim=latent_dim
    ).to(device)
    sampler = MCMC_Sampler(
        ebm=model.ebm, 
        latent_dim=latent_dim, 
        step_size=0.1, 
        num_steps=50,
        buffer_size=10000,
        replay_ratio=0
    )
    optimizer_vae = torch.optim.Adam([
        {'params': model.encoder.parameters(),'lr': 1e-4},
        {'params': model.decoder.parameters(),'lr': 2e-5}
    ], lr=lr)
    optimizer_ebm = torch.optim.Adam(model.ebm.parameters(), lr=1e-4*0.005)
    total_batches = len(train_loader)
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        model.train()
        last_mcmc_log_table = None
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(progress_bar):
            current_beta = get_cyclic_beta(
                batch_idx, 
                total_batches, 
                start_beta=0.005, 
                max_beta=beta
            )
            dec_input_ids = batch[0].to(device)
            dec_word_mask = batch[1].to(device)
            enc_input_ids = batch[2].to(device)
            enc_word_mask = batch[3].to(device)
            step_metrics, mcmc_log_table = train_step_algorithm1(
                dec_input_ids=dec_input_ids,
                target_ids=dec_input_ids,
                dec_attention_mask=dec_word_mask,
                enc_input_ids=enc_input_ids,
                enc_attention_mask=enc_word_mask,
                model=model, 
                optimizer_vae=optimizer_vae, 
                optimizer_ebm=optimizer_ebm, 
                sampler=sampler, 
                device=device,
                beta=1,
                max_norm=max_norm
            )
            if (batch_idx + 1) % 100 == 0:
                step_metrics["beta"] = 1
                wandb.log(step_metrics)
            last_mcmc_log_table = mcmc_log_table
            progress_bar.set_postfix({
                "VAE L": f"{step_metrics['vae_loss']:.3f}",
                "CD_loss": f"{step_metrics['CD_loss']:.3f}"
            })
        if last_mcmc_log_table is not None:
            table_columns = list(last_mcmc_log_table[0].keys())
            wandb_table = wandb.Table(columns=table_columns)
            for row in last_mcmc_log_table:
                wandb_table.add_data(*[row[col] for col in table_columns])
            wandb.log({f"MCMC_Dynamics_E{epoch+1}": wandb_table})
    return model
inference_dir = "/home/salam4/hvae_project/Optimus"
sys.path.insert(0, inference_dir)
from pretrained_checkpoints.inference import load_model, InferenceArgs
from data.create_loaders import DualTokenizerDataset, make_dual_collate_fn
CHECKPOINT_PATH = "/home/salam4/hvae_project/Optimus/pretrained_checkpoints/optimus-vae.pth" 
args = InferenceArgs()
try:
    model_vae, enc_tok, dec_tok = load_model(CHECKPOINT_PATH, args)
    print("Model loaded successfully.")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=== GPT-2 decoder tokenizer ===")
print("bos:", dec_tok.bos_token, "id:", dec_tok.bos_token_id)
print("eos:", dec_tok.eos_token, "id:", dec_tok.eos_token_id)
print("pad:", dec_tok.pad_token, "id:", dec_tok.pad_token_id)
print("\n=== BERT encoder tokenizer ===")
print("cls:", enc_tok.cls_token, "id:", enc_tok.cls_token_id)
print("sep:", enc_tok.sep_token, "id:", enc_tok.sep_token_id)
print("pad:", enc_tok.pad_token, "id:", enc_tok.pad_token_id)
HP_DICT = {}
HP_DICT["pad_idx"] = dec_tok.pad_token_id
HP_DICT["vocab_size"] = len(dec_tok)
df = pd.read_parquet("/home/salam4/hvae_project/Optimus/data/datasets/sentences_df.parquet")
texts = df["sentence"].astype(str).tolist() 
dataset = DualTokenizerDataset(texts, enc_tok, dec_tok)
train_df, valid_df = train_test_split(dataset, test_size=0.5, random_state=42)
collate_fn = make_dual_collate_fn(enc_tok, dec_tok, max_length=50)
train_loader = DataLoader(train_df, batch_size=128, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_df, batch_size=32, shuffle=True, collate_fn=collate_fn)
dec_ids, dec_mask, enc_ids, enc_mask = next(iter(train_loader))
print(dec_ids.shape, dec_mask.shape, enc_ids.shape, enc_mask.shape)               
print("dec pad id:", dec_tok.pad_token_id, "enc pad id:", enc_tok.pad_token_id)
import os
wandb.login(key=os.environ.get("WANDB_API_KEY"))                                         
wandb.init(
    project="EBM Prior",
    name = 'no reply buffer, 200x slower, 0.1 step size, 20x reco loss'
)   
print("\nStarting Unimodal EBM Training...")
model = train_unimodal_ebm(
    encoder=model_vae.encoder, 
    decoder=model_vae.decoder, 
    train_loader=train_loader, 
    epochs=3, 
    latent_dim=768, 
    lr=3e-5, 
    beta=1.0, 
    max_norm=1.0, 
    device=device
)
print("Training complete!")
