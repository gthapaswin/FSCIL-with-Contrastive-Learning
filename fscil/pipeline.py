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
        # Stage 3.5. Original build: on raw 512-d backbone feats (detached from
        # W_proj). Faithful (tech doc §2): on the projected d' nodes h^(0).
        self.supcon_head = SupConHead(in_dim=Config.proj_dim if Config.faithful else None)
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

    def supcon_on_nodes(self, h0, class_ids):
        """Tech doc §2: SupCon on the projected nodes h^(0) = W_proj w_{c,m}, so
        it pre-conditions W_proj. Positives P(i) are every other view of the
        same class, the denominator A(i) every other node (one shared view id
        turns the view-restricted loss into the standard one)."""
        z = self.supcon_head(h0)
        return supcon_loss_view(z, class_ids, torch.zeros_like(class_ids))

    def session_step(self, H_prev, w_active, class_ages, session):
        """Tech doc §4 over ALL active classes: re-run Stages 3-6 on the view
        means w_active (C, M, d) of every class seen so far (the first
        H_prev.shape[0] of them already have memory rows, in the same order),
        then gate the old rows and write the new ones. Returns H_t and the
        intermediates the training losses need."""
        C, M = w_active.shape[:2]
        dev = w_active.device
        cid = torch.arange(C, device=dev).unsqueeze(1).expand(C, M).reshape(-1)
        vid = torch.arange(M, device=dev).unsqueeze(0).expand(C, M).reshape(-1)
        h0, h_en, A_soft = self.encode_prototypes(w_active.reshape(C * M, -1), cid, vid)
        P, A_tilde = self.build_prototypes(h_en, A_soft, cid, C)
        n_prev = H_prev.shape[0]
        H_full = torch.cat([H_prev, P.new_zeros(C - n_prev, P.shape[1])], 0)
        new = torch.arange(C, device=dev) >= n_prev
        H, R_gate, phi_p = self.update_memory(H_full, P, A_tilde, class_ages, session, new)
        return H, {"h0": h0, "A": A_soft, "cid": cid, "P": P, "R_gate": R_gate, "phi": phi_p, "new": new}

    def encode_prototypes(self, w_cm, class_ids, view_ids):
        """
        Stage 3 (already shot-averaged outside) -> Stage 4 -> Stage 5.
        w_cm: (C_t*M, 512) shot-averaged fantasy prototypes
        class_ids/view_ids: (C_t*M,) at class-view granularity
        Returns h0 (K,d'), h_enriched (K,d'), A_soft (K,K)
        """
        h0 = self.projection(w_cm)                            # Stage 3: W_proj -> d'
        A_soft = self.topology(h0)                                # Stage 4
        if Config.ablate_topology:
            # Ablation: drop learned topology gating -> GAT attends over a
            # uniform adjacency (every class pair treated as equally related).
            A_soft = torch.ones_like(A_soft)
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
