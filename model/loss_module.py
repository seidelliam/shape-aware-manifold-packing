import torch
import torch.nn.functional as F
from torch import linalg as LA
from typing import Tuple
import torch.distributed as dist
class CrossEntropy:
    def __init__(self):
        self.loss_name = "cross_entropy_loss"
        self.loss = torch.nn.CrossEntropyLoss()
    def __call__(self,preds,labels):
        loss = self.loss(preds,labels)
        return loss

class InfoNCELoss:
    def __init__(self,n_views,batch_size,tau):
        self.loss_name = "info_nce_loss"
        self.n_views = n_views
        self.batch_size = batch_size  
        self.tau = tau  
        self.hyper_parameters = {"n_views":n_views,"batch_size":batch_size,"tau":tau}
    def __call__(self,preds,labels):
        sim = F.cosine_similarity(preds[:,None,:],preds[None,:,:],dim=-1)
        mask_self = torch.eye(preds.shape[0],dtype=torch.bool,device=sim.device)
        sim.masked_fill_(mask_self,0.0)
        positive_mask = mask_self.roll(shifts=self.batch_size,dims=0)
        sim /= self.tau
        ll = torch.mean(-sim[positive_mask] + torch.logsumexp(sim,dim=-1))
        return ll

class EllipsoidPackingLoss:
    def __init__(self,n_views:int,batch_size:int,lw0:float=1.0,lw1:float=1.0,lw2:float=1.0,
                 n_pow_iter:int=20,rs:float=2.0,pot_pow:float=2.0,record:bool = False):
        self.n_views = n_views
        self.batch_size = batch_size
        self.n_pow_iter = n_pow_iter # for power iteration
        self.rs = rs # scale of radii
        self.pot_pow = pot_pow # power for the repulsive potential
        self.lw0 = lw0 # loss weight for the ellipsoid size
        self.lw1 = lw1 # loss weight for the repulsion
        self.lw2 = lw2 # loss weight for the alignment
        self.loss_name = "ellipoids_packing_loss"
        self.hyper_parameters = {"n_views":n_views,"batch_size":batch_size,
                                "lw0":lw0,"lw1":lw1,"lw2":lw2,"n_pow_iter":n_pow_iter,"rs":rs}
        self.record = record
        if record:
            self.status = dict()
    def __call__(self,preds,labels):
        # preds is [(V*B),O] dimesional matrix
        com = torch.mean(preds,dim=0)
        # make the center of mass of pres locate at the origin
        preds -= com
        # normalize to make all the preds in the unit sphere
        std = torch.sqrt(torch.sum(preds*preds,dim=0)/(preds.shape[0] - 1.0) + 1e-12)
        preds = preds/std
        # reshape [(V*B),O] shape tensor to shape [V,B,O] 
        preds = torch.reshape(preds,(self.n_views,self.batch_size,preds.shape[-1]))
        # centers.shape = B*O for B ellipsoids
        centers = torch.mean(preds,dim=0)
        # correlation matrix 
        preds -= centers
        
        corr = torch.matmul(torch.permute(preds,(1,2,0)), torch.permute(preds,(1,0,2)))/self.n_views # size B*O*O
        # average radius for each ellipsoid
        # trace for each ellipsoids, t = torch.diagonal(corr,dim1=1,dim2=2).sum(dim=1)
        # t[i] = sum of eigenvalues for ellipsoid i, semi-axial length = sqrt(eigenvalue)
        # average radii = sqrt(sum of eigenvalues/rank) rank = min(n_views,output_dim) 
        # average radii.shape = (B,)
        radii = self.rs*torch.sqrt(torch.diagonal(corr,dim1=1,dim2=2).sum(dim=1)/min(preds.shape[-1],self.n_views)+ 1e-12)
        # loss 1: repulsive loss
        diff = centers[:, None, :] - centers[None, :, :]
        dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-12)
        #add 1e-6 to avoid dividing by zero
        sum_radii = radii[None,:] + radii[:,None] + 1e-6
        nbr_mask = dist_matrix < sum_radii
        self_mask = torch.eye(self.batch_size,dtype=bool,device=preds.device)
        mask = torch.logical_and(nbr_mask,torch.logical_not(self_mask))
        ll = 0.5*((1.0 - dist_matrix[mask]/sum_radii[mask])**self.pot_pow).sum()*self.lw1
        if abs(self.lw0) > 1e-6:
            # loss 0: minimize the size of each ellipsoids
            # to make sure radii =0 and dij = inf is not a valid state 
            ll += self.lw0*torch.sum(radii)
        if abs(self.lw2) > 1e-6:
            # calculate the largest eigenvectors by the [power iteration] method
            # devided by matrix norm to make sure |corr^n_power| not too small, and ~ 1
            corr_norm = torch.linalg.matrix_norm(corr,keepdim=True)
            normalized_corr = corr/(corr_norm + 1e-6).detach()
            corr_pow = torch.stack([torch.matrix_power(normalized_corr[i], self.n_pow_iter) for i in range(corr.shape[0])])
            b0 = torch.rand(preds.shape[-1],device=preds.device)
            eigens = torch.matmul(corr_pow,b0) # size = B*O
            eigens = eigens/(torch.norm(eigens,dim=1,keepdim=True) + 1e-6) 
            # loss 2: alignment loss (1 - cosine-similarity)
            sim = torch.matmul(eigens,eigens.transpose(0,1))**2
            ll += 0.5*(1.0 - torch.square(sim[mask])).sum()*self.lw2
        if self.record:
            self.status["corrs"] = corr.cpu().detach()
            self.status["centers"] = centers.cpu().detach()
            self.status["principle_vec"] = eigens.cpu().detach()
            self.status["preds"] = preds.cpu().detach()
        return ll
    
class MMCR_Loss(torch.nn.Module):
    def __init__(self, n_views: int,batch_size:int):
        super(MMCR_Loss, self).__init__()
        self.n_views = n_views
        self.batch_size = batch_size
        self.distribured = dist.is_available() and dist.is_initialized()

    def forward(self, z: torch.Tensor,labels:torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # z is [(V*B),O] dimesional matrix
        z = F.normalize(z, dim=-1)
        z = torch.reshape(z,(self.n_views,self.batch_size,z.shape[-1]))
        # gather across devices into list
        if self.distribured:
            ws = torch.distributed.get_world_size()
            z_list = [
                torch.zeros_like(z)
                for _ in range(ws)
            ]
            torch.distributed.all_gather(z_list, z, async_op=False)
            z_list[torch.distributed.get_rank()] = z
            # append all
            z = torch.cat(z_list)
        else:
            ws = 1
        centroids = torch.mean(z, dim=0)
        global_nuc = torch.linalg.svdvals(centroids).sum()
        loss = - global_nuc

        return loss

class LogRepulsiveEllipsoidPackingLossUnitNorm:
    def __init__(self,n_views:int,batch_size:int,lw0:float=1.0,lw1:float=1.0,
                 rs:float=2.0,pot_pow:float=2.0,min_margin:float=1e-3,max_range:float=1.5):
        self.n_views = n_views
        self.batch_size = batch_size
        self.rs = rs # scale of radii
        self.pot_pow = pot_pow # power for the repulsive potential
        self.lw0 = lw0 
        self.lw1 = lw1 # loss weight for the repulsion
        self.min_margin = min_margin
        self.max_range = max_range
        self.record = dict()
        self.hyper_parameters = {"n_views":n_views,"batch_size":batch_size,
                                "lw1":lw1,"rs":rs,"min_margin":min_margin}

    def __call__(self,preds,labels):   
        # reshape [(V*B),O] shape tensor to shape [V,B,O] 
        # V-number of views, B-batch size, O-output embedding dim
        preds_local = torch.reshape(preds,(self.n_views,self.batch_size,preds.shape[-1]))
        # get the embedings from all the processes(GPUs) if ddp
        if dist.is_available() and dist.is_initialized():
            ws = dist.get_world_size() # world size
            outputs = [torch.zeros_like(preds_local) for _ in range(ws)]
            dist.all_gather(outputs,preds_local,async_op=False)
            # it is important to set the outputs[rank] to local output, since computational graph is not 
            # copied through different gpus, preds_local preserves the local computational graphs
            outputs[dist.get_rank()] = preds_local
            # preds is now [V,(B*ws),O]
            preds = torch.cat(outputs,dim=1)
        else:
            preds = preds_local
            ws = 1
        # preds is [V,B*ws,O] dimesional matrix
        com = torch.mean(preds,dim=(0,1))
        # make the center of mass of pres locate at the origin
        preds -= com
        # normalize
        preds = torch.nn.functional.normalize(preds,dim=-1)
        # centers.shape = [B*ws,O] for B*ws ellipsoids
        centers = torch.mean(preds,dim=0)
        # correlation matrix 
        preds -= centers
        # traces[i] = 1/(n-1)*trace(pred[:,i,:]*pred[:,i,:]^T)
        traces = torch.sum(torch.permute(preds,(1,0,2))**2,dim=(1,2))/(self.n_views -1.0)
        # average radius for each ellipsoid
        # trace for each ellipsoids, t = torch.diagonal(corr,dim1=1,dim2=2).sum(dim=1)
        # t[i] = sum of eigenvalues for ellipsoid i, semi-axial length = sqrt(eigenvalue)
        # average radii = sqrt(sum of eigenvalues/rank) rank = min(n_views,output_dim) 
        # average radii.shape = (B,)
        radii = self.rs*torch.sqrt(traces/min(preds.shape[-1],self.n_views)+ 1e-12)
        # loss 1: repulsive loss
        diff = centers[:, None, :] - centers[None, :, :]
        dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-12)
        #add 1e-6 to avoid dividing by zero
        sum_radii = radii[None,:] + radii[:,None] + 1e-6
        sum_radii = torch.min(sum_radii,self.max_range*torch.ones_like(sum_radii,device=sum_radii.device))
        nbr_mask = torch.logical_and(dist_matrix < sum_radii,dist_matrix > self.min_margin)
        self_mask = torch.eye(self.batch_size*ws,dtype=bool,device=preds.device)
        mask = torch.logical_and(nbr_mask,torch.logical_not(self_mask))
        ll = 0.5*((1.0 - dist_matrix[mask]/sum_radii[mask])**self.pot_pow).sum()*self.lw1
        # loss 0: minimize the size of each ellipsoids
        ll += self.lw0*torch.sum(radii)
        self.record["radii"] =radii.detach()
        self.record["dist"] = dist_matrix[torch.logical_not(self_mask)].reshape((-1,)).detach()
        self.record["norm_center"] = torch.linalg.norm(centers,dim=-1).detach()
        return torch.log(ll + 1e-6)


# ---------------------------------------------------------------------------
# Helper: custom autograd function that attenuates gradients along a set of
# directions (eigenvectors of each cluster's scatter matrix) by factor gamma.
# gamma=1.0 → no attenuation (recovers original CLAMP behaviour)
# gamma=0.0 → fully removes gradient components along principal axes
# ---------------------------------------------------------------------------
class GradientAttenuator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, centers, eigvecs, gamma):
        # centers: [B, O]
        # eigvecs: [B, O, k]  — top-k right singular vectors (row-wise) of each cluster's data matrix
        # gamma:   scalar float in [0, 1]
        ctx.save_for_backward(eigvecs)
        ctx.gamma = gamma
        return centers.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: [B, O]
        eigvecs, = ctx.saved_tensors  # [B, O, k]
        gamma = ctx.gamma
        # Projection coefficients onto each principal direction
        proj = torch.einsum('bok,bo->bk', eigvecs, grad_output)  # [B, k]
        # Reconstruct the principal component of the gradient
        grad_principal = torch.einsum('bok,bk->bo', eigvecs, proj)  # [B, O]
        # Attenuated gradient: keep full grad for off-axis directions, scale by gamma on-axis
        grad_modified = grad_output - (1.0 - gamma) * grad_principal
        return grad_modified, None, None  # no gradient for eigvecs or gamma


# ---------------------------------------------------------------------------
# Vectorized helper: pairwise Mahalanobis distances (no Python loops)
#
# For each pair (i,j), the combined scatter matrix is Λi+Λj = [Xi;Xj]^T[Xi;Xj]
# where Xi = preds_T[i] * scale (shape [V,O]).  The squared Mahalanobis distance
# is  d²_ij = proj_ij^T  (gram_ij)^{-1}  proj_ij
# where  gram_ij = [Xi;Xj][Xi;Xj]^T  (2V×2V)  and  proj_ij = [Xi;Xj](ci-cj).
# All operations are batched over B² pairs using torch.linalg.solve.
# ---------------------------------------------------------------------------
def _mahalanobis_dist_matrix(centers, preds_T, scale, reg=1e-4):
    """Fully-vectorized pairwise Mahalanobis distance matrix.

    Args:
        centers:  [B, O] cluster centroids
        preds_T:  [B, V, O] zero-mean view embeddings per cluster
        scale:    float, typically 1/sqrt(V-1)
        reg:      diagonal regularization added to each gram matrix
    Returns:
        dist_matrix: [B, B] symmetric distance matrix
    """
    B, V, O = preds_T.shape
    device = centers.device
    dtype = centers.dtype

    # Cast to float32: eigh/matmul are numerically unstable in fp16/bf16
    X = preds_T.float() * float(scale)  # [B, V, O]
    centers_f = centers.float()         # [B, O]

    # All pairwise gram sub-blocks via a single flat matmul [B*V, O] @ [O, B*V]
    X_flat = X.reshape(B * V, O)
    cross_flat = X_flat @ X_flat.T                               # [B*V, B*V]
    # Reshape to [B, B, V, V]:  cross[i,j,k,l] = sum_d X[i,k,d]*X[j,l,d]
    cross = cross_flat.reshape(B, V, B, V).permute(0, 2, 1, 3)  # [B, B, V, V]
    # Per-cluster self-grams: gram_self[i] = cross[i,i]
    gram_self = cross.diagonal(dim1=0, dim2=1).permute(2, 0, 1) # [B, V, V]

    # Assemble 2V×2V block gram for every pair: [[Gi, Gij],[Gij^T, Gj]]
    gram_i   = gram_self[:, None].expand(B, B, V, V)            # [B, B, V, V]
    gram_j   = gram_self[None, :].expand(B, B, V, V)            # [B, B, V, V]
    gram_top = torch.cat([gram_i,                  cross       ], dim=3)  # [B, B, V, 2V]
    gram_bot = torch.cat([cross.transpose(-1, -2), gram_j      ], dim=3)  # [B, B, V, 2V]
    gram_full = torch.cat([gram_top, gram_bot], dim=2)           # [B, B, 2V, 2V]
    gram_full = gram_full + reg * torch.eye(2 * V, device=device, dtype=torch.float32)

    # Pairwise center-difference projections without materializing [B, B, O] diff.
    # XC[i,j,v] = sum_d X[i,v,d] * c[j,d]  →  [B, B, V]
    XC = torch.einsum('ivd,jd->ijv', X, centers_f)               # [B, B, V]
    XC_self = (X * centers_f[:, None, :]).sum(-1)                 # [B, V]
    proj_top  = XC_self[:, None, :] - XC                          # [B, B, V]
    proj_bot  = XC.permute(1, 0, 2) - XC_self[None, :, :]        # [B, B, V]
    proj_full = torch.cat([proj_top, proj_bot], dim=2)            # [B, B, 2V]

    # Eigendecomposition of symmetric PSD gram matrices (stable for PSD inputs)
    # eigvals: [B*B, 2V] (ascending), eigvecs: [B*B, 2V, 2V] (columns = eigenvectors)
    gram_flat = gram_full.reshape(B * B, 2 * V, 2 * V)
    eigvals, eigvecs = torch.linalg.eigh(gram_flat)

    # Project each diff-vector onto eigenbasis:  coords[n,k] = eigvec_k · proj[n]
    proj_flat = proj_full.reshape(B * B, 2 * V)                  # [B*B, 2V]
    coords = torch.bmm(eigvecs.mT, proj_flat.unsqueeze(-1)).squeeze(-1)  # [B*B, 2V]

    # Mahalanobis²: d²=Σ_k (coords_k / λ_k)² = proj^T gram^{-2} proj
    # (correctly accounts for both orientation and scale of each eigendirection)
    d2 = (coords / eigvals.clamp(min=reg)).pow(2).sum(dim=-1).reshape(B, B)

    return torch.sqrt(torch.clamp(d2, min=0.0) + 1e-12).to(dtype)  # [B, B]


# ---------------------------------------------------------------------------
# Extension 1 (standalone): Anisotropic Overlap Detection
# Replaces Euclidean centroid distance with Mahalanobis distance using the
# low-rank pseudoinverse of (Lambda_i + Lambda_j).
# ---------------------------------------------------------------------------
class AnisotropicLogRepulsiveEllipsoidPackingLoss(LogRepulsiveEllipsoidPackingLossUnitNorm):
    """Anisotropic overlap detection via Mahalanobis distance.

    Inherits all behaviour from LogRepulsiveEllipsoidPackingLossUnitNorm and
    only overrides the distance computation: Euclidean ||ci - cj|| is replaced
    by the Mahalanobis distance sqrt((ci-cj)^T (Li+Lj)^+ (ci-cj)) where
    Li = (1/(V-1)) * Xi^T Xi is the scatter matrix of cluster i (rank <= V << O).
    All pairwise distances are computed in a single batched GPU operation.
    """

    def __call__(self, preds, labels):
        import math as _math
        preds_local = torch.reshape(preds, (self.n_views, self.batch_size, preds.shape[-1]))
        if dist.is_available() and dist.is_initialized():
            ws = dist.get_world_size()
            outputs = [torch.zeros_like(preds_local) for _ in range(ws)]
            dist.all_gather(outputs, preds_local, async_op=False)
            outputs[dist.get_rank()] = preds_local
            preds = torch.cat(outputs, dim=1)
        else:
            preds = preds_local
            ws = 1

        com = torch.mean(preds, dim=(0, 1))
        preds -= com
        preds = torch.nn.functional.normalize(preds, dim=-1)
        centers = torch.mean(preds, dim=0)   # [B*ws, O]
        preds -= centers

        traces = torch.sum(torch.permute(preds, (1, 0, 2)) ** 2, dim=(1, 2)) / (self.n_views - 1.0)
        radii = self.rs * torch.sqrt(traces / min(preds.shape[-1], self.n_views) + 1e-12)

        B_total = self.batch_size * ws
        V = self.n_views
        scale = 1.0 / _math.sqrt(max(V - 1, 1))
        preds_T = preds.permute(1, 0, 2)  # [B_total, V, O]

        dist_matrix = _mahalanobis_dist_matrix(centers, preds_T, scale)

        sum_radii = radii[None, :] + radii[:, None] + 1e-6
        sum_radii = torch.min(sum_radii, self.max_range * torch.ones_like(sum_radii, device=sum_radii.device))
        nbr_mask = torch.logical_and(dist_matrix < sum_radii, dist_matrix > self.min_margin)
        self_mask = torch.eye(B_total, dtype=bool, device=preds.device)
        mask = torch.logical_and(nbr_mask, torch.logical_not(self_mask))
        ll = 0.5 * ((1.0 - dist_matrix[mask] / sum_radii[mask]) ** self.pot_pow).sum() * self.lw1
        ll += self.lw0 * torch.sum(radii)
        self.record["radii"] = radii.detach()
        self.record["dist"] = dist_matrix[torch.logical_not(self_mask)].reshape((-1,)).detach()
        self.record["norm_center"] = torch.linalg.norm(centers, dim=-1).detach()
        return torch.log(ll + 1e-6)


# ---------------------------------------------------------------------------
# Extension 2 (combined): Anisotropic Overlap + Orientation-Aware Repulsion
# Adds gamma parameter on top of AnisotropicLogRepulsiveEllipsoidPackingLoss.
# gamma=1.0 recovers anisotropic overlap without gradient attenuation.
# ---------------------------------------------------------------------------
class SAMPLoss(AnisotropicLogRepulsiveEllipsoidPackingLoss):
    """Shape-Aware Manifold Packing loss.

    Combines:
      1. Anisotropic overlap detection (Mahalanobis distance)
      2. Asymmetric orientation-aware repulsion (gamma gradient attenuation)

    gamma in [0, 1]:
      gamma=1.0 → identical to AnisotropicLogRepulsiveEllipsoidPackingLoss
      gamma=0.0 → gradient component along each cluster's principal axes zeroed
    """

    def __init__(self, n_views: int, batch_size: int, lw0: float = 1.0,
                 lw1: float = 1.0, rs: float = 2.0, pot_pow: float = 2.0,
                 min_margin: float = 1e-3, max_range: float = 1.5,
                 gamma: float = 1.0):
        super().__init__(n_views, batch_size, lw0, lw1, rs, pot_pow, min_margin, max_range)
        self.gamma = gamma
        self.hyper_parameters["gamma"] = gamma

    def __call__(self, preds, labels):
        import math as _math
        preds_local = torch.reshape(preds, (self.n_views, self.batch_size, preds.shape[-1]))
        if dist.is_available() and dist.is_initialized():
            ws = dist.get_world_size()
            outputs = [torch.zeros_like(preds_local) for _ in range(ws)]
            dist.all_gather(outputs, preds_local, async_op=False)
            outputs[dist.get_rank()] = preds_local
            preds = torch.cat(outputs, dim=1)
        else:
            preds = preds_local
            ws = 1

        com = torch.mean(preds, dim=(0, 1))
        preds -= com
        preds = torch.nn.functional.normalize(preds, dim=-1)
        centers = torch.mean(preds, dim=0)   # [B*ws, O]
        preds -= centers

        traces = torch.sum(torch.permute(preds, (1, 0, 2)) ** 2, dim=(1, 2)) / (self.n_views - 1.0)
        radii = self.rs * torch.sqrt(traces / min(preds.shape[-1], self.n_views) + 1e-12)

        B_total = self.batch_size * ws
        V = self.n_views
        scale = 1.0 / _math.sqrt(max(V - 1, 1))
        preds_T = preds.permute(1, 0, 2)  # [B_total, V, O]

        # Batched SVD for per-cluster principal directions: [B_total, V, O]
        _, _, Vh = torch.linalg.svd(preds_T, full_matrices=False)
        eigvecs = Vh.permute(0, 2, 1)  # [B_total, O, V]

        # Apply gradient attenuation to centers (identity forward, attenuates backward)
        centers_att = GradientAttenuator.apply(centers, eigvecs, self.gamma)

        dist_matrix = _mahalanobis_dist_matrix(centers_att, preds_T, scale)

        sum_radii = radii[None, :] + radii[:, None] + 1e-6
        sum_radii = torch.min(sum_radii, self.max_range * torch.ones_like(sum_radii, device=sum_radii.device))
        nbr_mask = torch.logical_and(dist_matrix < sum_radii, dist_matrix > self.min_margin)
        self_mask = torch.eye(B_total, dtype=bool, device=preds.device)
        mask = torch.logical_and(nbr_mask, torch.logical_not(self_mask))
        ll = 0.5 * ((1.0 - dist_matrix[mask] / sum_radii[mask]) ** self.pot_pow).sum() * self.lw1
        ll += self.lw0 * torch.sum(radii)
        self.record["radii"] = radii.detach()
        self.record["dist"] = dist_matrix[torch.logical_not(self_mask)].reshape((-1,)).detach()
        self.record["norm_center"] = torch.linalg.norm(centers, dim=-1).detach()
        return torch.log(ll + 1e-6)
