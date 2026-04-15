import torch
import torch.nn.functional as F

def compute_reconstruction_loss(logits, targets, attention_mask):
    """
    Computes sequence reconstruction Cross-Entropy loss properly masked.
    
    Args:
        logits (torch.Tensor): Output from the decoder of shape (batch_size, seq_len, vocab_size).
        targets (torch.Tensor): Ground truth token IDs of shape (batch_size, seq_len).
        attention_mask (torch.Tensor): Binary mask where 1 indicates a real token, 
                                       0 indicates a padding token (batch_size, seq_len).
    
    Returns:
        sentence_recon_loss (torch.Tensor): A scalar representing the average per-sentence sum of 
                                            reconstruction losses. This matches the dimensions of 
                                            KL and EBM energies which are also summed per sentence.
        token_loss_per_token (torch.Tensor): A scalar representing the traditional average 
                                             per-token cross entropy (for logging).
    """
    # Shift so that tokens < n predict n
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = targets[:, 1:].contiguous()
    shift_attention_mask = attention_mask[:, 1:].contiguous()
    
    batch_size, seq_len, vocab_size = shift_logits.shape
    
    # Calculate the unreduced Cross-Entropy loss
    # F.cross_entropy expects logits as (N, C) and targets as (N,)
    ce_loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size), 
        shift_targets.view(-1), 
        reduction='none'
    )
    
    # Reshape back to (batch_size, seq_len)
    ce_loss = ce_loss.view(batch_size, seq_len)
    
    # Mask out the padding tokens
    # Note: make sure attention_mask is float for multiplication
    shift_attention_mask = shift_attention_mask.to(ce_loss.dtype)
    masked_loss = ce_loss * shift_attention_mask
    
    # ========================================================
    # 1. Real Loss (Per-Sentence Sum, then Mean across batch)
    # ========================================================
    # Sum the losses over the sequence length axis for each sentence
    per_sentence_sum = masked_loss.sum(dim=-1)  # shape: (batch_size,)
    
    # Take the mean across the batch to get a stable scalar loss signal
    sentence_recon_loss = per_sentence_sum.mean()
    
    
    # ========================================================
    # 2. Logging Loss (Per-Token Mean)
    # ========================================================
    # Sum of all losses in the batch divided by the total number of non-padding tokens
    total_real_tokens = shift_attention_mask.sum().clamp_min(1.0) # Avoid divide by zero
    token_loss_per_token = masked_loss.sum() / total_real_tokens
    
    return sentence_recon_loss, token_loss_per_token
