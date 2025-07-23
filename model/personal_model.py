import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Qwen2ForCausalLM, PretrainedConfig, AutoTokenizer

EMBED_SIZE = 1024
HIDDEN_SIZE = 512

class SparseAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size, dtype=torch.bfloat16),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size, dtype=torch.bfloat16),
            nn.GELU(),
        )
        self.rho = 0.05
        self.rho_hat = None

    def forward(self, x):
        z = self.encoder(x)
        self.rho_hat = z.mean(dim=1)
        x_recon = self.decoder(z)
        return z, x_recon
    
    def sae_loss(self, x, x_recon):
        eps = 1e-6
        rho_hat = torch.clamp(self.rho_hat, eps, 1 - eps)
        recon_loss = F.smooth_l1_loss(x, x_recon)
        kl_div = self.rho * torch.log(self.rho / rho_hat) + \
            (1 - self.rho) * torch.log((1 - self.rho) / (1 - rho_hat))
        rho_loss = kl_div.mean()
        return recon_loss, rho_loss

class DEPModel(Qwen2ForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sae = SparseAutoEncoder(EMBED_SIZE, HIDDEN_SIZE)
        self.sae.to_empty(device=device)
        self.align_mlp_his = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, config.hidden_size, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size, dtype=torch.bfloat16),
        )
        self.align_mlp_his.to_empty(device=device)
        self.align_mlp_diff = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, config.hidden_size, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size, dtype=torch.bfloat16),
        )
        self.align_mlp_diff.to_empty(device=device)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        his_diff_emb: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        inputs_embs = self.model.get_input_embeddings()(input_ids)
        his_diff_sparse_emb, his_diff_recon_emb = self.sae(his_diff_emb)
        his_emb = his_diff_sparse_emb[:, :8, :]
        diff_emb = his_diff_sparse_emb[:, 8:, :]

        his_new_tokens = [f"[HIS_TOKEN_{i}]" for i in range(8)]
        his_token_ids = self.llm_tokenizer.convert_tokens_to_ids(his_new_tokens)
        diff_new_tokens = [f"[DIFF_TOKEN_{i}]" for i in range(8)]
        diff_token_ids = self.llm_tokenizer.convert_tokens_to_ids(diff_new_tokens)
        
        his_emb = his_emb.to(inputs_embs.dtype)
        his_emb = self.align_mlp_his(his_emb)
        for bidx in range(inputs_embs.shape[0]):
            b_his_emb = his_emb[bidx]
            for i in range(8):
                pemb = b_his_emb[i, :]
                inputs_embs[bidx][input_ids[bidx] == his_token_ids[i]] = pemb.to(inputs_embs.dtype)

        diff_emb = diff_emb.to(inputs_embs.dtype)
        diff_emb = self.align_mlp_diff(diff_emb)
        for bidx in range(inputs_embs.shape[0]):
            b_diff_emb = diff_emb[bidx]
            for i in range(8):
                pemb = b_diff_emb[i, :]
                inputs_embs[bidx][input_ids[bidx] == diff_token_ids[i]] = pemb.to(inputs_embs.dtype)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embs,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        llm_loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        recon_loss, rho_loss = self.sae.sae_loss(his_diff_emb, his_diff_recon_emb)
        loss = llm_loss + (recon_loss + 1e-3 * rho_loss) * 100

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        training: bool = False,
        tokenizer: Optional[AutoTokenizer] = None,
        **kwargs
    ):
        model = super().from_pretrained(pretrained_model_name_or_path, 
                                        *model_args, 
                                        config=config, 
                                        cache_dir=cache_dir, 
                                        ignore_mismatched_sizes=ignore_mismatched_sizes, 
                                        force_download=force_download, 
                                        local_files_only=local_files_only, 
                                        token=token, revision=revision, 
                                        use_safetensors=use_safetensors,
                                        **kwargs)
        model.llm_tokenizer = tokenizer
        if training:
            for name, param in model.named_parameters():
                if not "align_mlp" in name and not "sae" in name:
                    param.requires_grad = False
        return model
