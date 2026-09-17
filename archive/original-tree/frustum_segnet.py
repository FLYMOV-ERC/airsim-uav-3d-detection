#!/usr/bin/env python3
"""FrustumSegNet (estilo F-PointNet): segmenta cada ponto como drone/fundo (aprendido),
e usa SÓ os pontos do drone (pooling ponderado pela máscara) para estimar cls/centro/size.
Substitui a heurística do depth-band por segmentação aprendida — robusto a oclusor."""
import torch, torch.nn as nn, torch.nn.functional as F


class FrustumSegNet(nn.Module):
    def __init__(self, in_xyz=3):
        super().__init__()
        # MLP por ponto (sem downsample → saída por ponto)
        self.mlp1 = nn.Sequential(nn.Conv1d(in_xyz,64,1), nn.BatchNorm1d(64), nn.ReLU(),
                                  nn.Conv1d(64,64,1), nn.BatchNorm1d(64), nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Conv1d(64,128,1), nn.BatchNorm1d(128), nn.ReLU(),
                                  nn.Conv1d(128,1024,1), nn.BatchNorm1d(1024), nn.ReLU())
        # cabeça de segmentação por ponto (concat local 64 + global 1024)
        self.seg = nn.Sequential(nn.Conv1d(64+1024,256,1), nn.BatchNorm1d(256), nn.ReLU(),
                                 nn.Conv1d(256,128,1), nn.BatchNorm1d(128), nn.ReLU(),
                                 nn.Conv1d(128,1,1))
        # cabeça de caixa: a partir da feature do OBJETO (pooling ponderado pela máscara)
        self.box = nn.Sequential(nn.Linear(1024,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
                                 nn.Linear(512,256), nn.ReLU())
        self.cls_head=nn.Linear(256,1); self.center_head=nn.Linear(256,3); self.size_head=nn.Linear(256,3)

    def forward(self, x, return_seg=False):
        # x: (B,N,3) normalizado
        B,N,_=x.shape
        p=x.transpose(1,2)                       # (B,3,N)
        l=self.mlp1(p)                            # (B,64,N) local
        g=self.mlp2(l)                            # (B,1024,N)
        gpool=g.max(dim=2,keepdim=True)[0]        # (B,1024,1) global
        seg_logit=self.seg(torch.cat([l, gpool.expand(-1,-1,N)],dim=1)).squeeze(1)  # (B,N)
        # pooling do OBJETO: pondera features globais-por-ponto pela prob de fg
        w=torch.sigmoid(seg_logit).unsqueeze(1)   # (B,1,N)
        obj=(g*w).sum(dim=2)/(w.sum(dim=2)+1e-6)  # (B,1024) feature do foreground
        h=self.box(obj)
        out=(self.cls_head(h).squeeze(-1), self.center_head(h), self.size_head(h))
        return (out+(seg_logit,)) if return_seg else out
