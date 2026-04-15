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

                                        

    shift_logits = logits[:, :-1, :].contiguous()

    shift_targets = targets[:, 1:].contiguous()

    shift_attention_mask = attention_mask[:, 1:].contiguous()

    

    batch_size, seq_len, vocab_size = shift_logits.shape

    

                                                

                                                                  

    ce_loss = F.cross_entropy(

        shift_logits.view(-1, vocab_size), 

        shift_targets.view(-1), 

        reduction='none'

    )

    

                                           

    ce_loss = ce_loss.view(batch_size, seq_len)

    

                                 

                                                                

    shift_attention_mask = shift_attention_mask.to(ce_loss.dtype)

    masked_loss = ce_loss * shift_attention_mask

    

                                                              

                                                             

                                                              

                                                                    

    per_sentence_sum = masked_loss.sum(dim=-1)                        

    

                                                                       

    sentence_recon_loss = per_sentence_sum.mean()

    

    

                                                              

                                      

                                                              

                                                                                      

    total_real_tokens = shift_attention_mask.sum().clamp_min(1.0)                       

    token_loss_per_token = masked_loss.sum() / total_real_tokens

    

    return sentence_recon_loss, token_loss_per_token
