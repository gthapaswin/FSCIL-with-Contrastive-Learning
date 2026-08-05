import torch
import torch.nn as nn

from fscil.config import Config
from fscil.modules import (
    Projection, SupConHead, TopologyScorer, GATv2Layer, STIMemory,
    CosineClassifier, pool_prototypes, pool_class_adjacency,
    supcon_loss_view, graph_topology_loss, memory_stability_loss,
)


class StagStiModel(nn.Module):
    """Wraps every trainable module from Stage 3 onward. The backbone f_theta
    is handled separately (frozen, called before this model) per the spec."""

    def __init__(self):
        super().__init__()
        self.projection = Projection()                    # Stage 3 / Stage 8 query projection
        self.supcon_head = SupConHead()                       # Stage 3.5, operates on raw 512-d backbone feats
        self.topology = TopologyScorer()                        # Stage 4
        self.gat = GATv2Layer()                                    # Stage 5
        self.sti = STIMemory()                                       # Stage 7
        self.classifier = CosineClassifier()                          # Stage 8

    def supcon_on_raw_features(self, raw_feats, class_ids, view_ids):
        """
        Stage 3.5 — runs BEFORE shot-averaging, on the raw (N*M, d) backbone
        features (d=512), per the pipeline table.
        raw_feats: (N*M, 512); class_ids/view_ids: (N*M,)
        """
        z_supcon = self.supcon_head(raw_feats)
        loss = supcon_loss_view(z_supcon, class_ids, view_ids)
        return loss

    def encode_prototypes(self, w_cm, class_ids, view_ids):
        """
        Stage 3 (already shot-averaged outside) -> Stage 4 -> Stage 5.
        w_cm: (C_t*M, 512) shot-averaged fantasy prototypes
        class_ids/view_ids: (C_t*M,) at class-view granularity
        Returns h0 (K,d'), h_enriched (K,d'), A_soft (K,K)
        """
        h0 = self.projection(w_cm)                            # Stage 3: W_proj -> d'
        A_soft = self.topology(h0)                                # Stage 4
        h_enriched = self.gat(h0, A_soft, view_ids)              # Stage 5
        return h0, h_enriched, A_soft

    def build_prototypes(self, h_enriched, A_soft, class_ids, num_classes):
        P_t = pool_prototypes(h_enriched, class_ids, num_classes)        # Stage 6
        A_tilde = pool_class_adjacency(A_soft, class_ids, num_classes)
        return P_t, A_tilde

    def update_memory(self, H_prev, P_t, A_tilde, class_ages, session, new_class_mask):
        return self.sti(H_prev, P_t, A_tilde, class_ages, session, new_class_mask)  # Stage 7

    def classify_query(self, z_q, H_t):
        return self.classifier(z_q, H_t)                                              # Stage 8

    @staticmethod
    def graph_loss(A_soft, class_ids):
        return graph_topology_loss(A_soft, class_ids)

    @staticmethod
    def stability_loss(P_t, phi_p, R_gate):
        return memory_stability_loss(P_t, phi_p, R_gate)
