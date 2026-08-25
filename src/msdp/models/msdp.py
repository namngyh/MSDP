from __future__ import annotations
import torch
from torch import nn
from .experts import ShortScaleExpert,MediumScaleExpert,LongScaleExpert,RangeVolatilityExpert
from .gate import ContextNetwork,HorizonGate
from .heads import MedianCenteredReturnQuantileHead,NegativeMonotonicMDDHead,AuxReturnHead

class MSDP(nn.Module):
    def __init__(self,n_features,horizons=(5,20,60),return_quantiles=(.05,.25,.5,.75,.95),mdd_quantiles=(.1,.5,.9),hidden_dim=64,latent_dim=48,n_blocks=2,dropout=.1,gate_temperature=1.,medium_pool=5,long_pool=10,range_indices=None,ablation="full"):
        super().__init__(); self.horizons=list(horizons); self.return_quantiles=list(return_quantiles); self.mdd_quantiles=list(mdd_quantiles); self.ablation=ablation
        specs=[(ShortScaleExpert,63,1,None),(MediumScaleExpert,126,medium_pool,None),(LongScaleExpert,252,long_pool,None),(RangeVolatilityExpert,126,3,range_indices)]
        if ablation=="no_range": specs=specs[:3]
        if ablation=="single": specs=specs[:1]
        self.experts=nn.ModuleList([cls(n_features,hidden_dim,latent_dim,w,p,n_blocks,dropout,ix) for cls,w,p,ix in specs]); k=len(specs)
        self.context=ContextNetwork(n_features,latent_dim,k,dropout); self.gate=HorizonGate(latent_dim,len(horizons),k,gate_temperature,dropout,ablation!="shared_gate"); self.fused_norm=nn.LayerNorm(latent_dim)
        self.ret=MedianCenteredReturnQuantileHead(latent_dim); self.mdd=NegativeMonotonicMDDHead(latent_dim); self.direction=nn.Linear(latent_dim,1); self.volatility=nn.Linear(latent_dim,1); self.aux=AuxReturnHead(latent_dim,dropout)
    def forward(self,x,gate_override=None):
        """`gate_override` lets the online tier replace the gate's softmax.

        The trained gate is a prior over experts; the online tier multiplies in
        the evidence collected since the last retrain (see `msdp.online.hedge`)
        and passes the posterior here. No weight of the network changes, and
        the output reports both `gate_prior` (what the network said) and
        `gate_weights` (what was actually fused).
        """
        lat=torch.stack([e(x) for e in self.experts],1); context=self.context(x[:,-1],lat); weights=self.gate(lat,context)
        if self.ablation=="equal": weights=torch.ones_like(weights)/weights.shape[-1]
        prior=weights
        if gate_override is not None:
            gate_override=torch.as_tensor(gate_override,dtype=weights.dtype,device=weights.device)
            if gate_override.shape!=weights.shape: raise ValueError(f"gate_override phải có shape {tuple(weights.shape)}, nhận {tuple(gate_override.shape)}")
            weights=gate_override
        horizon=self.gate.embedding.weight[None,:,:]; fused=self.fused_norm(torch.einsum("bhk,bkd->bhd",weights,lat)+context[:,None,:]+horizon)
        aux_input=lat[:,None,:,:]+horizon[:,:,None,:]; aux=self.aux(aux_input)
        return {"return_quantiles":self.ret(fused),"direction_logits":self.direction(fused).squeeze(-1),"direction_prob":torch.sigmoid(self.direction(fused).squeeze(-1)),"mdd_quantiles":self.mdd(fused),"volatility":self.volatility(fused).squeeze(-1),"gate_weights":weights,"gate_prior":prior,"aux_return_median":aux,"expert_disagreement":aux.std(-1,unbiased=False),"expert_latents":lat,"context":context}
