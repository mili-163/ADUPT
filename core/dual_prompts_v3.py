import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class DualInstancePromptsV3(nn.Module):
    def __init__(self, hidden_size: int, num_modalities: int, prompt_length: int, num_layers: int, config: Dict):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_modalities = num_modalities
        self.prompt_length = prompt_length
        self.num_layers = num_layers
        self.modality_names = config.get("modality_names", ["text", "image", "audio"])
        
        # Content aggregation: u = ψ(x) = LN([Pool(t^(1)) || ... || Pool(t^(M))])
        self.pool_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size)
        )
        
        # Specialization branch: P^sp = g_sp(u, P^mod(m))
        self.g_sp = nn.Sequential(
            nn.Linear(hidden_size + prompt_length * hidden_size * num_layers, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, prompt_length * hidden_size)
        )
        
        # Generalization branch: P^gn = g_gn(u, P^mod(m))
        self.g_gn = nn.Sequential(
            nn.Linear(hidden_size + prompt_length * hidden_size * num_layers, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, prompt_length * hidden_size)
        )
        
        # Gate module: ω = σ(w^T [Hash(m) || MLP(u)])
        self.mask_hash_embedding = nn.Embedding(2**num_modalities, hidden_size // 2)
        self.u_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU()
        )
        self.gate_mlp = nn.Linear(hidden_size, 1)
        
    def _pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        # Pool(t^(k)) aggregates tokens from modality k
        return tokens.mean(dim=1)  # (B, d)
    
    def forward(self, 
                modality_tokens: Dict[str, torch.Tensor], 
                modality_prompts: Dict[int, torch.Tensor],
                m: torch.Tensor,
                return_specialization_features: bool = False) -> Dict[str, torch.Tensor]:
        
        batch_size = m.shape[0]
        
        # 1. Compute content-dependent summary u
        # u = ψ(x) = LN([Pool(t^(1)) || ... || Pool(t^(M))])
        pooled_modality_features = []
        for i, modality_name in enumerate(self.modality_names):
            if m[:, i].any() and modality_name in modality_tokens:
                pooled_modality_features.append(self._pool_tokens(modality_tokens[modality_name]))
        
        if not pooled_modality_features:
            u = torch.zeros(batch_size, self.hidden_size, device=m.device)
        else:
            u = torch.stack(pooled_modality_features, dim=1).mean(dim=1)  # (B, d)
            u = self.pool_mlp(u)  # (B, d)
        
        # Flatten modality_prompts for conditioning
        modality_prompts_flat = torch.cat([p.view(batch_size, -1) for p in modality_prompts.values()], dim=1)
        
        # Condition both branches on u and P^mod(m)
        cond_input = torch.cat([u, modality_prompts_flat], dim=-1)
        
        # P^sp = g_sp(u, P^mod(m))
        P_sp_raw = self.g_sp(cond_input).view(batch_size, self.prompt_length, self.hidden_size)
        
        # P^gn = g_gn(u, P^mod(m))  
        P_gn_raw = self.g_gn(cond_input).view(batch_size, self.prompt_length, self.hidden_size)
        
        # Gate module: ω = σ(w^T [Hash(m) || MLP(u)])
        mask_hash_values = self._hash_mask(m)
        hashed_m_embed = self.mask_hash_embedding(mask_hash_values)
        u_mlp_out = self.u_mlp(u)
        
        gate_input = torch.cat([hashed_m_embed, u_mlp_out], dim=-1)
        omega = torch.sigmoid(self.gate_mlp(gate_input))  # (B, 1)
        
        # Mixed instance prompt: P̃^ins = ω P^sp + (1-ω) P^gn
        mixed_prompts = omega.unsqueeze(-1) * P_sp_raw + (1 - omega.unsqueeze(-1)) * P_gn_raw
        
        outputs = {"mixed_prompts": mixed_prompts, "omega": omega}
        if return_specialization_features:
            outputs["specialization_features"] = P_sp_raw.mean(dim=1)  # (B, d)
        
        return outputs
    
    def _hash_mask(self, m: torch.Tensor) -> torch.Tensor:
        # Convert binary mask (B, M) to unique integer hash (B,)
        powers_of_2 = 2**torch.arange(self.num_modalities - 1, -1, -1, device=m.device)
        hash_values = (m * powers_of_2).sum(dim=1)
        return hash_values.long()


class ContentAggregator(nn.Module):
    def __init__(self, hidden_size: int, num_modalities: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_modalities = num_modalities
        self.layer_norm = nn.LayerNorm(hidden_size * num_modalities)
        self.projection = nn.Linear(hidden_size * num_modalities, hidden_size)
    
    def forward(self, modality_tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        # u = ψ(x) = LN([Pool(t^(1)) || ... || Pool(t^(M))])
        pooled_tokens = []
        
        for modality, tokens in modality_tokens.items():
            pooled = tokens.mean(dim=1)  # Pool(t^(k))
            pooled_tokens.append(pooled)
        
        if pooled_tokens:
            combined = torch.cat(pooled_tokens, dim=-1)  # [Pool(t^(1)) || ... || Pool(t^(M))]
        else:
            batch_size = next(iter(modality_tokens.values())).shape[0]
            combined = torch.zeros(batch_size, self.hidden_size * self.num_modalities,
                                 device=next(iter(modality_tokens.values())).device)
        
        normalized = self.layer_norm(combined)  # LN([...])
        content_summary = self.projection(normalized)
        
        return content_summary
