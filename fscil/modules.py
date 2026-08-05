"""
IMPORTANT — READ THIS FIRST
============================
Three documents built this file up over time:
  1. The original master spec — prose + shapes only, equations stripped.
  2. A re-architected pipeline table — gave real formulas for Stages 4/5/6/8
     and the SupCon/STI structure, but Stage 1/2/3.5's inline formulas and
     the Age-Based Stability Prior / Combined Gating Factor / L_graph / L_stab
     formulas were still blank.
  3. The exact L_graph, L_stab, beta, and Gamma formulas — mostly explicit,
     but a few operators (the adjacency target, the gate anti-saturation
     term) were described in words/boundary-conditions rather than a literal
     equation. Comments below marked "# ASSUMPTION:" mark exactly those
     spots — everything else here is a direct implementation of what you
     provided across all three documents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fscil.config import Config


# ---------------------------------------------------------------------------
# Stage 3: Projection to Graph Hidden Space  (W_proj: d -> d')
# Applied to (a) the M-view class prototypes w_{c,m} before the graph stage,
# and (b) the query feature at inference: z_hat_q = W_proj f_theta(x_q).
# ---------------------------------------------------------------------------
class Projection(nn.Module):
    def __init__(self, in_dim=Config.backbone_out_dim, out_dim=Config.proj_dim):
        super().__init__()
        self.W_proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return self.W_proj(x)


# ---------------------------------------------------------------------------
# Stage 3.5: View-Preserving SupCon head g_proj: R^d -> S^(d_proj - 1)
# NOTE: per the pipeline table, this operates on the RAW backbone features
# z_{i,m} (dimension d=512), on the per-shot per-view embeddings, BEFORE
# Stage 3's shot-averaging — not on the projected/pooled prototypes.
# ---------------------------------------------------------------------------
class SupConHead(nn.Module):
    def __init__(self, in_dim=Config.backbone_out_dim, proj_dim=Config.supcon_proj_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1)  # unit hypersphere


def supcon_loss_view(z, class_ids, view_ids, temperature=Config.supcon_temperature):
    """
    View-Preserving Supervised Contrastive Loss, L_supcon_view.
    z: (N*M, d_proj) L2-normalized embeddings, one per (support-sample, view).
    class_ids: (N*M,) class label of each embedding.
    view_ids:  (N*M,) which fantasy transform m produced each embedding.

    P(i, m): positives are OTHER samples of the SAME class, under the SAME
             view m only (contrastive pull restricted within each view).
    A(i, m): denominator is restricted to the same view m as well (excludes
             self).
    This is what keeps "0deg dog" free to differ from "90deg dog" -- fantasy
    diversity across views is preserved, only intra-view collapse is
    penalized.
    """
    device = z.device
    K = z.shape[0]
    sim = torch.matmul(z, z.T) / temperature
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    class_ids = class_ids.view(-1, 1)
    view_ids = view_ids.view(-1, 1)
    same_class = torch.eq(class_ids, class_ids.T).float().to(device)
    same_view = torch.eq(view_ids, view_ids.T).float().to(device)
    self_mask = torch.eye(K, device=device)

    logit_mask = same_view * (1 - self_mask)                 # A(i, m)
    positive_mask = same_view * same_class * (1 - self_mask)   # P(i, m)

    exp_sim = torch.exp(sim) * logit_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    pos_count = positive_mask.sum(dim=1).clamp(min=1.0)
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / pos_count

    # nodes with zero same-view positives (e.g. shot=1) contribute 0 loss
    has_positive = (positive_mask.sum(dim=1) > 0).float()
    loss = -(mean_log_prob_pos * has_positive).sum() / has_positive.sum().clamp(min=1.0)
    return loss


# ---------------------------------------------------------------------------
# Stage 4: Dynamic Topology Scorer g_phi -> A_ij = sigma(g_phi([h_i || h_j]))
# ---------------------------------------------------------------------------
class TopologyScorer(nn.Module):
    def __init__(self, dim=Config.proj_dim, hidden=128, dropout_p=Config.dropout_p):
        super().__init__()
        self.W_phi = nn.Linear(2 * dim, hidden)
        self.b_phi = self.W_phi.bias  # named to match spec's b_phi
        self.dropout = nn.Dropout(dropout_p)
        self.out = nn.Linear(hidden, 1)

    def forward(self, h):
        """h: (K, d') -> returns A: (K, K) soft adjacency in [0,1]."""
        K = h.shape[0]
        hi = h.unsqueeze(1).expand(K, K, -1)
        hj = h.unsqueeze(0).expand(K, K, -1)
        pair = torch.cat([hi, hj], dim=-1)                 # (K,K,2d')
        hidden = self.dropout(F.relu(self.W_phi(pair)))
        A = torch.sigmoid(self.out(hidden)).squeeze(-1)     # (K,K)
        return A


def graph_topology_loss(A_soft, class_ids, eps=Config.graph_loss_eps):
    """
    L_graph: BCE(A, A*), supervising g_phi directly.
    # ASSUMPTION: A*_ij formula itself was blank in your doc, but the purpose
    # text is unambiguous: A*_ij = 1 for same-class node pairs (intra-class),
    # 0 otherwise (inter-class).
    """
    class_ids = class_ids.view(-1, 1)
    A_target = torch.eq(class_ids, class_ids.T).float().to(A_soft.device)
    A_clamped = A_soft.clamp(min=eps, max=1 - eps)
    bce = -(A_target * torch.log(A_clamped) + (1 - A_target) * torch.log(1 - A_clamped))
    return bce.mean()


# ---------------------------------------------------------------------------
# Stage 5: Fantasy-Aware GATv2 Attention Engine
# e_ij = a^T PReLU(W_gat [h_i || h_j || A_ij || R_{mi,mj}])
# R_{mi,mj}: fantasy compatibility / view-transform relation embedding
# (d_r) — ASSUMPTION: learned embedding lookup keyed by (view_i, view_j)
# pair, since no explicit construction for R was given.
# ---------------------------------------------------------------------------
class GATv2Layer(nn.Module):
    def __init__(self, dim=Config.proj_dim, attn_dim=Config.gat_hidden_dim,
                 view_rel_dim=Config.view_rel_dim, num_views=Config.num_views):
        super().__init__()
        self.dim = dim
        self.view_rel_dim = view_rel_dim
        self.view_rel_table = nn.Embedding(num_views * num_views, view_rel_dim)
        self.W_gat = nn.Linear(2 * dim + 1 + view_rel_dim, attn_dim)
        # PReLU INSIDE attention coefficient. nn.PReLU assumes the channel
        # dim sits at position 1, but our (K,K,attn_dim) tensor has channels
        # last -- so this is applied manually via _prelu_last_dim below.
        self.attn_alpha = nn.Parameter(torch.full((attn_dim,), 0.25))
        self.a = nn.Linear(attn_dim, 1, bias=False)
        self.W_v = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(Config.dropout_p)
        self.prelu_out = nn.PReLU(num_parameters=dim)          # per-channel slope alpha_d on aggregated output
        self.num_views = num_views

    def forward(self, h, A_soft, view_ids):
        """
        h: (K, d') node features
        A_soft: (K, K) topology scores from Stage 4
        view_ids: (K,) which fantasy-view index (0..M-1) produced each node
        returns h': (K, d') enriched node features
        """
        K = h.shape[0]
        hi = h.unsqueeze(1).expand(K, K, -1)
        hj = h.unsqueeze(0).expand(K, K, -1)

        vi = view_ids.unsqueeze(1).expand(K, K)
        vj = view_ids.unsqueeze(0).expand(K, K)
        rel_idx = (vi * self.num_views + vj).long()
        r_ij = self.view_rel_table(rel_idx)                 # (K,K,d_r)

        pair = torch.cat([hi, hj, A_soft.unsqueeze(-1), r_ij], dim=-1)  # (K,K,2d'+1+d_r)
        gat_out = self.W_gat(pair)                                       # (K,K,attn_dim)
        gat_act = torch.where(gat_out > 0, gat_out, self.attn_alpha * gat_out)  # PReLU, channel-last
        e = self.a(gat_act).squeeze(-1)                                    # (K,K)

        attn = F.softmax(e, dim=-1)                            # normalize over neighbors j
        v = self.W_v(h)                                          # (K, d')
        out = torch.matmul(attn, v)                              # (K, d')
        out = self.dropout(out)
        out = self.prelu_out(out)                                 # preserves negative activations (360deg geometry)
        return out


# ---------------------------------------------------------------------------
# Stage 7: Spatio-Temporal Integration (STI) Memory Module
# R      = sigma(H_{t-1} W_r1^T + P_t W_r2^T + b_r + A~ H_{t-1} W_n^T)
# beta_c = sigma(kappa*(t - tau_c) - kappa0)      [broadcast to d']
# Gamma  = beta (x) 1 + (1 - beta) (x) R           (affine blend, NOT R*beta)
# H_t    = Gamma (x) H_{t-1} + (1 - Gamma) (x) Phi(P_t)
# ---------------------------------------------------------------------------
class STIMemory(nn.Module):
    def __init__(self, dim=Config.proj_dim, kappa=Config.kappa, kappa0=Config.kappa0):
        super().__init__()
        self.Phi = nn.Linear(dim, dim)                     # memory adapter Phi(P_t) — distinct from W_n
        self.W_r1 = nn.Linear(dim, dim, bias=False)          # direct H_{t-1} term
        self.W_r2 = nn.Linear(dim, dim, bias=True)             # P_t term, carries shared bias b_r
        self.b_r = self.W_r2.bias
        self.W_n = nn.Linear(dim, dim, bias=False)               # graph-propagated (A~ H_{t-1}) term
        self.kappa = kappa
        self.kappa0 = kappa0

    def forward(self, H_prev, P_t, A_tilde, class_ages, session, new_class_mask):
        """
        H_prev: (C_t, d') memory from the previous session (rows for
                brand-new classes are meaningless and get overwritten below)
        P_t: (C_t, d') this session's fresh prototypes
        A_tilde: (C_t, C_t) view-reduced class-level adjacency
        class_ages: (C_t,) tau_c, session index each class was introduced
        session: int, current session t
        new_class_mask: (C_t,) bool, True for classes with no prior memory
        Returns: H_t, R_gate, phi_p  (R_gate & phi_p exposed for L_stab)
        """
        phi_p = self.Phi(P_t)                                                    # (C_t, d')
        propagated_old = torch.matmul(A_tilde, H_prev)                            # (C_t, d')

        R_gate = torch.sigmoid(
            self.W_r1(H_prev) + self.W_r2(P_t) + self.W_n(propagated_old)
        )                                                                            # (C_t, d')

        beta = torch.sigmoid(self.kappa * (session - class_ages) - self.kappa0)     # (C_t,)
        beta = beta.unsqueeze(-1).expand_as(R_gate)                                    # broadcast to d'

        Gamma = beta + (1 - beta) * R_gate                                              # affine blend, per Sec.4

        H_t = Gamma * H_prev + (1 - Gamma) * phi_p
        # brand-new classes have no valid H_prev -> just take the adapted prototype
        H_t = torch.where(new_class_mask.unsqueeze(-1), phi_p, H_t)
        return H_t, R_gate, phi_p


def memory_stability_loss(P_t, phi_p, R_gate, lambda_gate_sat=Config.lambda_gate_sat, eps=1e-7):
    """
    L_stab = L_align + lambda_sat * L_gate_sat

    L_align: cosine-distance penalty between raw prototypes P_t and
             memory-adapted prototypes Phi(P_t) -- keeps Phi a calibration
             layer rather than a destructive transform.
    L_gate_sat: # ASSUMPTION -- exact operator was blank; implemented as
             negative binary entropy of R_gate, the standard way to
             discourage a sigmoid gate from saturating at 0/1 (entropy is
             maximized at R=0.5 and minimized at the extremes, so minimizing
             -entropy pushes gates away from saturation), matching the
             stated purpose exactly.
    """
    cos_sim = F.cosine_similarity(P_t, phi_p, dim=-1)
    l_align = (1 - cos_sim).mean()

    R_clamped = R_gate.clamp(min=eps, max=1 - eps)
    neg_entropy = R_clamped * torch.log(R_clamped) + (1 - R_clamped) * torch.log(1 - R_clamped)
    l_gate_sat = neg_entropy.mean()

    return l_align + lambda_gate_sat * l_gate_sat


# ---------------------------------------------------------------------------
# Stage 8: Cosine classifier against persistent memory H_t
# p(y=c|x_q) = softmax_c( cos(z_hat_q, H_t,c) / tau )
# ---------------------------------------------------------------------------
class CosineClassifier(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        # learnable inverse-temperature (log-parameterized for stability),
        # equivalent to the fixed tau in the spec but lets the scale adapt.
        self.log_inv_tau = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(1.0 / temperature)))))

    def forward(self, z_q, H_t):
        z_q = F.normalize(z_q, dim=-1)
        H_norm = F.normalize(H_t, dim=-1)
        inv_tau = self.log_inv_tau.exp()
        logits = inv_tau * torch.matmul(z_q, H_norm.T)
        return logits


# ---------------------------------------------------------------------------
# Stage 6: Prototype View Pooling — plain mean over the M fantasy views
# belonging to each class (Monte Carlo approximation, no learnable params).
# ---------------------------------------------------------------------------
def pool_prototypes(h_enriched, class_ids, num_classes):
    """h_enriched: (K, d'), class_ids: (K,) -> P_t: (num_classes, d')"""
    dim = h_enriched.shape[-1]
    P_t = torch.zeros(num_classes, dim, device=h_enriched.device)
    counts = torch.zeros(num_classes, device=h_enriched.device)
    P_t.index_add_(0, class_ids, h_enriched)
    counts.index_add_(0, class_ids, torch.ones_like(class_ids, dtype=torch.float))
    counts = counts.clamp(min=1).unsqueeze(-1)
    return P_t / counts


def pool_class_adjacency(A_soft, class_ids, num_classes):
    """View-reduced class-level adjacency \tilde{A}: average soft-adjacency
    across all view-pairs belonging to each class pair."""
    onehot = F.one_hot(class_ids, num_classes).float()          # (K, C)
    class_sum = onehot.T @ A_soft @ onehot                        # (C, C)
    counts = onehot.sum(0)                                          # (C,)
    denom = torch.outer(counts, counts).clamp(min=1)
    return class_sum / denom
