import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from hadamard import matmul_hadU_cuda, matmul_hadUt_cuda, get_hadK
import math


class TrainableDecompLinear(nn.Module):
    def __init__(self, A, B, truncation_rank, bias=None, train_frac_beta=0.2):
        """
        Y=WX~=ABX
        """
        super().__init__()
        train_rank = int(B.size(0) * train_frac_beta)
        self.truncation_rank = (truncation_rank - train_rank, train_rank)

        # Separate the weights without sharing
        no_train_rank = B.size(0) - train_rank
        self.B_no_train = B[:no_train_rank]
        self.B_train = B[no_train_rank:]
        self.A_no_train = A[:, :no_train_rank]
        self.A_train = A[:, no_train_rank:]

        self.BLinear_no_train = nn.Linear(self.B_no_train.size(1), self.B_no_train.size(0), bias=False)
        self.BLinear_train = nn.Linear(self.B_train.size(1), self.B_train.size(0), bias=False)
        self.ALinear_no_train = nn.Linear(self.A_no_train.size(1), self.A_no_train.size(0), bias=False)
        self.ALinear_train = nn.Linear(self.A_train.size(1), self.A_train.size(0), bias=bias is not None)

        # Initialize weights without sharing
        self.BLinear_no_train.weight.data = self.B_no_train.contiguous()
        self.BLinear_train.weight.data = self.B_train.contiguous()
        self.ALinear_no_train.weight.data = self.A_no_train.contiguous()
        self.ALinear_train.weight.data = self.A_train.contiguous()

        # Gradients for no-train weights should be disabled
        self.BLinear_no_train.weight.requires_grad = False
        self.ALinear_no_train.weight.requires_grad = False

        if bias is not None:
            # self.ALinear_no_train.bias = nn.Parameter(bias.clone())
            self.ALinear_train.bias.data = bias

    def forward(self, inp):
        y_no_train = self.BLinear_no_train(inp)
        y_train = self.BLinear_train(inp)
        y_no_train = self.ALinear_no_train(y_no_train)
        y_train = self.ALinear_train(y_train)
        y = y_no_train + y_train
        return y


class SVDLinear(nn.Module):
    def __init__(self, A, B, param_ratio, bias=None) -> None:
        super().__init__()
        self.ALinear = nn.Linear(A.size(1), A.size(0), bias=bias is not None)
        if bias is not None:
            self.ALinear.bias.data = bias
        self.BLinear = nn.Linear(B.size(1), B.size(0), bias=False)
        self.truncation_rank = B.size(0)
        self.ALinear.weight.data = A.contiguous()
        self.BLinear.weight.data = B.contiguous()
        self.param_ratio = param_ratio

    @staticmethod
    def from_linear(
        linear: nn.Linear,
        param_ratio: float,
        act_aware=False,
        ic_split=1,
        oc_split=1,
        alpha=1,
        sigma_fuse="UV",
        rank_align=1,
    ):
        if hasattr(linear, "whitening_matrix"):
            W = linear.weight.data
            # dtype = W.dtype

            num_s_after_trunc = int(W.shape[0] * W.shape[1] * param_ratio / (W.shape[0] + W.shape[1]))

            if getattr(linear, "is_calibration_stage", False) and hasattr(linear, "cached_svd"):
                U, S, VT, scaling_matrix_inv = linear.cached_svd
            else:
                W = W.float()
                scaling_diag_matrix = linear.whitening_matrix.to(W.device)
                try:
                    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                except Exception as e:
                    print("Warning: scaling_diag_matrix is not full rank!")
                    print(scaling_diag_matrix)
                    print(f"max: {(scaling_diag_matrix).max()}")
                    scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0], device=W.device)
                    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                scaling_diag_matrix = scaling_diag_matrix.float()
                scaling_matrix_inv = scaling_matrix_inv.float()
                if hasattr(linear, "scaling_diag_matrix2"):
                    scaling_diag_matrix2 = linear.scaling_diag_matrix2.to(W.device)
                    print("size of W: ", W.size())
                    print("size of scaling_diag_matrix2: ", scaling_diag_matrix2.size())
                    W = W / scaling_diag_matrix2.unsqueeze(1)
                W_scale = torch.matmul(W, scaling_diag_matrix)
                U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
                if getattr(linear, "is_calibration_stage", False):
                    linear.cached_svd = (U, S, VT, scaling_matrix_inv)
            S = S[:num_s_after_trunc]
            U = U[:, :num_s_after_trunc]
            V = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv).T

            # truc_sigma = torch.diag(truc_s)
            #### Replace Attn, MLP ####
            # sqrtSigma = torch.sqrt(truc_sigma)
            # svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(dtype)
            # svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(dtype)

            Us = [U]
            Ss = [S]
            Vs = [V]

            if linear.bias is not None:
                bias = linear.bias.data
            else:
                bias = None

            # nan or inf check
            for S in Ss:
                if (S != S).any():
                    print("nan in S")
                    return (
                        nn.Linear(linear.in_features, linear.out_features)
                        .to(linear.weight.dtype)
                        .to(linear.weight.device)
                    )
            for U in Us:
                if (U != U).any():
                    print("nan in U")
                    return (
                        nn.Linear(linear.in_features, linear.out_features)
                        .to(linear.weight.dtype)
                        .to(linear.weight.device)
                    )
            for V in Vs:
                if (V != V).any():
                    print("nan in V")
                    return (
                        nn.Linear(linear.in_features, linear.out_features)
                        .to(linear.weight.dtype)
                        .to(linear.weight.device)
                    )

            assert len(Us) == len(Ss) == len(Vs) == 1
            U, S, V = Us[0], Ss[0], Vs[0]
            if sigma_fuse == "UV":
                A = U.mul(S.sqrt()).contiguous()
                if hasattr(linear, "scaling_diag_matrix2"):
                    scaling_diag_matrix2 = linear.scaling_diag_matrix2.to(W.device)
                    A = A * scaling_diag_matrix2.unsqueeze(1)
                    if (A != A).any():
                        print("nan in A!")
                        print("max of scaling_diag_matrix2: ", scaling_diag_matrix2.max())
                B = V.t().mul(S.sqrt().view(-1, 1)).contiguous()
            elif sigma_fuse == "U":
                A = U.mul(S).contiguous()
                B = V.t().contiguous()
            elif sigma_fuse == "V":
                A = U.contiguous()
                B = V.t().mul(S.view(-1, 1)).contiguous()

            new_linear = SVDLinear(A, B, param_ratio, bias)
            new_linear.to(linear.weight.dtype)
            if not getattr(linear, "is_calibration_stage", False):
                linear.whitening_matrix.to("cpu")
            return new_linear
        elif hasattr(linear, "scaling_diag_matrix") or hasattr(linear, "fisher_info"):
            # if param_ratio >= 1:
            #     return linear
            n_params = linear.weight.numel()
            compressed_params = int(n_params * param_ratio)
            assert ic_split == 1 or oc_split == 1
            rank = compressed_params // (linear.in_features + linear.out_features)
            # rank align
            rank = int(np.ceil(rank / rank_align) * rank_align)

            # print("rank", rank)
            w = linear.weight.data.float()
            if act_aware:
                scaling_diag_matrix = 1  # avoid zero division
                if hasattr(linear, "scaling_diag_matrix"):
                    # print("WARNING: scaling_diag_matrix is used")
                    scaling_diag_matrix *= linear.scaling_diag_matrix**alpha
                    # scaling_diag_matrix *= linear.scaling_diag_matrix**0.5
                if hasattr(linear, "fisher_info"):
                    scaling_diag_matrix *= linear.fisher_info**alpha
                    # scaling_diag_matrix *= linear.fisher_info**1
                # if not (scaling_diag_matrix == scaling_diag_matrix).all():
                #     breakpoint()
                scaling_diag_matrix += 1e-6  # avoid zero division
                w = w * scaling_diag_matrix.view(1, -1)
            Us = []
            Ss = []
            Vs = []
            try:
                U, S, V = torch.svd_lowrank(w, q=rank)
            except:
                print(f"svd failed for {linear}, disable act_aware")
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
                )
            if act_aware:
                V = V / scaling_diag_matrix.view(-1, 1)
            Us = [U]
            Ss = [S]
            Vs = [V]

            if linear.bias is not None:
                bias = linear.bias.data
            else:
                bias = None

            # nan or inf check
            for S in Ss:
                if (S != S).any():
                    print("nan in S")
                    return (
                        nn.Linear(linear.in_features, linear.out_features)
                        .to(linear.weight.dtype)
                        .to(linear.weight.device)
                    )
            for U in Us:
                if (U != U).any():
                    print("nan in U")
                    return (
                        nn.Linear(linear.in_features, linear.out_features)
                        .to(linear.weight.dtype)
                        .to(linear.weight.device)
                    )
            for V in Vs:
                if (V != V).any():
                    print("nan in V")
                    return (
                        nn.Linear(linear.in_features, linear.out_features)
                        .to(linear.weight.dtype)
                        .to(linear.weight.device)
                    )

            assert len(Us) == len(Ss) == len(Vs) == 1
            U, S, V = Us[0], Ss[0], Vs[0]
            if sigma_fuse == "UV":
                A = U.mul(S.sqrt()).contiguous()
                B = V.t().mul(S.sqrt().view(-1, 1)).contiguous()
            elif sigma_fuse == "U":
                A = U.mul(S).contiguous()
                B = V.t().contiguous()
            elif sigma_fuse == "V":
                A = U.contiguous()
                B = V.t().mul(S.view(-1, 1)).contiguous()
            new_linear = SVDLinear(A, B, param_ratio, bias)
            new_linear.to(linear.weight.dtype)
            return new_linear
        else:
            print("Cannot find scaling_diag_matrix or fisher_info, disable act_aware")
            return linear
    
    @staticmethod
    def from_trainable_decomp_linear(
        trainableDecompLinear: TrainableDecompLinear,
    ):
        A = torch.concat([trainableDecompLinear.ALinear_no_train.weight.data, trainableDecompLinear.ALinear_train.weight], dim=1)
        B = torch.concat([trainableDecompLinear.BLinear_no_train.weight.data, trainableDecompLinear.BLinear_train.weight], dim=0)
        if trainableDecompLinear.ALinear_train.bias is not None:
            bias = trainableDecompLinear.ALinear_train.bias.data
        else:
            bias = None
        new_linear = SVDLinear(A, B, 1, bias)
        return new_linear

    def change_param_ratio(
        self,
        new_param_ratio: float,
    ):
        BLinear_new_weight = self.BLinear.weight.data
        old_rank = self.truncation_rank
        old_param_ratio = self.param_ratio
        assert new_param_ratio <= old_param_ratio, "new_param_ratio should be smaller than old_param_ratio"
        new_rank = round(old_rank * (new_param_ratio / old_param_ratio))
        BLinear_new_weight = self.BLinear.weight.data[:new_rank]
        ALinear_new_weight = self.ALinear.weight.data[:, :new_rank]
        return SVDLinear(ALinear_new_weight, BLinear_new_weight, new_param_ratio, self.ALinear.bias)

    def forward(self, inp):
        # compute USV^Tx + b
        y = self.BLinear(inp)
        # y=torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        y = self.ALinear(y)
        # y=torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return y


def gen_hadamard_matrix(n):
    pass

class WhiteningSVDLinear(SVDLinear):
    @staticmethod
    def from_linear(
        linear: nn.Linear,
        param_ratio: float,
        act_aware=False,
        ic_split=1,
        oc_split=1,
        alpha=1,
        sigma_fuse="UV",
        rank_align=1,
    ):
        W = linear.weight.data.float()
        # dtype = W.dtype
        scaling_diag_matrix = linear.scaling_diag_matrix
        try:
            scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
        except Exception as e:
            print("Warning: scaling_diag_matrix is not full rank!")
            scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0], device=W.device)
            scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
        scaling_diag_matrix = scaling_diag_matrix.float()
        scaling_matrix_inv = scaling_matrix_inv.float()
        W_scale = torch.matmul(W, scaling_diag_matrix)
        U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
        num_s_after_trunc = int(W.shape[0] * W.shape[1] * param_ratio / (W.shape[0] + W.shape[1]))
        truc_s = S[:num_s_after_trunc]
        truc_u = U[:, :num_s_after_trunc]
        truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
        # truc_sigma = torch.diag(truc_s)
        #### Replace Attn, MLP ####
        # sqrtSigma = torch.sqrt(truc_sigma)
        # svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(dtype)
        # svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(dtype)
        new_linear = SVDLinear(truc_u, truc_s, truc_v, linear.bias, sigma_fuse)
        new_linear.to(linear.weight.dtype)
        return new_linear


class GradSVDLinear(nn.Module):
    def __init__(self, weight, scale, bias, rank) -> None:
        super().__init__()
        self.weight = weight
        self.scale = nn.Parameter(scale)
        self.bias = bias
        self.rank = rank

    @staticmethod
    def from_linear(
        linear: nn.Linear, param_ratio: float, act_aware=False, ic_split=1, oc_split=1, alpha=1, sigma_fuse="UV"
    ):
        if param_ratio >= 1:
            return linear
        n_params = linear.weight.numel()
        compressed_params = int(n_params * param_ratio)
        assert ic_split == 1 or oc_split == 1
        rank = compressed_params // (linear.in_features + linear.out_features)
        # print("rank", rank)
        w = linear.weight.data.float()
        if act_aware:
            scaling_diag_matrix = 1  # avoid zero division
            if hasattr(linear, "scaling_diag_matrix"):
                # print("WARNING: scaling_diag_matrix is used")
                scaling_diag_matrix *= linear.scaling_diag_matrix**alpha
                # scaling_diag_matrix *= linear.scaling_diag_matrix**0.5
            if hasattr(linear, "fisher_info"):
                scaling_diag_matrix *= linear.fisher_info**alpha
                # scaling_diag_matrix *= linear.fisher_info**1
            # if not (scaling_diag_matrix == scaling_diag_matrix).all():
            #     breakpoint()
            scaling_diag_matrix += 1e-6  # avoid zero division

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None
        return GradSVDLinear(w, scaling_diag_matrix, bias, rank)

    def forward(self, inp):
        w = self.weight * self.scale.view(1, -1)
        U, S, V = torch.svd_lowrank(w, q=self.rank)
        new_w = U.mul(S).mm(V.t())
        y = F.linear(inp, new_w, self.bias)
        return y

class SVDLinearIter(nn.Module):
    def __init__(self, A, B, param_ratio, bias=None) -> None:
        super().__init__()
        self.ALinear = nn.Linear(A.size(1), A.size(0), bias=bias is not None)
        if bias is not None:
            self.ALinear.bias.data = bias
        self.BLinear = nn.Linear(B.size(1), B.size(0), bias=False)
        self.truncation_rank = B.size(0)
        self.ALinear.weight.data = A.contiguous()
        self.BLinear.weight.data = B.contiguous()
        self.param_ratio = param_ratio
        self.A_train = nn.Linear()
        self.B_train = nn.Linear()

    @staticmethod
    def from_linear(
        linear: nn.Linear,
        param_ratio: float,
        act_aware=False,
        ic_split=1,
        oc_split=1,
        alpha=1,
        sigma_fuse="UV",
        rank_align=1,
    ):
        if hasattr(linear, "whitening_matrix"):
            W = linear.weight.data
            # dtype = W.dtype

            num_s_after_trunc = int(W.shape[0] * W.shape[1] * param_ratio / (W.shape[0] + W.shape[1]))

            if getattr(linear, "is_calibration_stage", False) and hasattr(linear, "cached_svd"):
                U, S, VT, scaling_matrix_inv = linear.cached_svd
            else:
                W = W.float()
                scaling_diag_matrix = linear.whitening_matrix.to(W.device)
                try:
                    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                except Exception as e:
                    print("Warning: scaling_diag_matrix is not full rank!")
                    print(scaling_diag_matrix)
                    print(f"max: {(scaling_diag_matrix).max()}")
                    scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0], device=W.device)
                    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                scaling_diag_matrix = scaling_diag_matrix.float()
                scaling_matrix_inv = scaling_matrix_inv.float()
                if hasattr(linear, "scaling_diag_matrix2"):
                    scaling_diag_matrix2 = linear.scaling_diag_matrix2.to(W.device)
                    print("size of W: ", W.size())
                    print("size of scaling_diag_matrix2: ", scaling_diag_matrix2.size())
                    W = W / scaling_diag_matrix2.unsqueeze(1)
                W_scale = torch.matmul(W, scaling_diag_matrix)
                U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
                if getattr(linear, "is_calibration_stage", False):
                    linear.cached_svd = (U, S, VT, scaling_matrix_inv)
            S = S[:num_s_after_trunc]
            U = U[:, :num_s_after_trunc]
            V = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv).T

            # truc_sigma = torch.diag(truc_s)
            #### Replace Attn, MLP ####
            # sqrtSigma = torch.sqrt(truc_sigma)
            # svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(dtype)
            # svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(dtype)

            Us = [U]
            Ss = [S]
            Vs = [V]

            if linear.bias is not None:
                bias = linear.bias.data
            else:
                bias = None

            # nan or inf check
            for S in Ss:
                if (S != S).any():
                    print("nan in S")
                    return (
                        nn.Linear(linear.in_features, linear.out_features)
                        .to(linear.weight.dtype)
                        .to(linear.weight.device)
                    )
            for U in Us:
                if (U != U).any():
                    print("nan in U")
                    return (
                        nn.Linear(linear.in_features, linear.out_features)
                        .to(linear.weight.dtype)
                        .to(linear.weight.device)
                    )
            for V in Vs:
                if (V != V).any():
                    print("nan in V")
                    return (
                        nn.Linear(linear.in_features, linear.out_features)
                        .to(linear.weight.dtype)
                        .to(linear.weight.device)
                    )

            assert len(Us) == len(Ss) == len(Vs) == 1
            U, S, V = Us[0], Ss[0], Vs[0]
            if sigma_fuse == "UV":
                A = U.mul(S.sqrt()).contiguous()
                if hasattr(linear, "scaling_diag_matrix2"):
                    scaling_diag_matrix2 = linear.scaling_diag_matrix2.to(W.device)
                    A = A * scaling_diag_matrix2.unsqueeze(1)
                    if (A != A).any():
                        print("nan in A!")
                        print("max of scaling_diag_matrix2: ", scaling_diag_matrix2.max())
                B = V.t().mul(S.sqrt().view(-1, 1)).contiguous()
            elif sigma_fuse == "U":
                A = U.mul(S).contiguous()
                B = V.t().contiguous()
            elif sigma_fuse == "V":
                A = U.contiguous()
                B = V.t().mul(S.view(-1, 1)).contiguous()

            new_linear = SVDLinear(A, B, param_ratio, bias)
            new_linear.to(linear.weight.dtype)
            if not getattr(linear, "is_calibration_stage", False):
                linear.whitening_matrix.to("cpu")
            return new_linear
        elif hasattr(linear, "scaling_diag_matrix") or hasattr(linear, "fisher_info"):
            # if param_ratio >= 1:
            #     return linear
            n_params = linear.weight.numel()
            compressed_params = int(n_params * param_ratio)
            assert ic_split == 1 or oc_split == 1
            rank = compressed_params // (linear.in_features + linear.out_features)
            # rank align
            rank = int(np.ceil(rank / rank_align) * rank_align)

            # print("rank", rank)
            w = linear.weight.data.float()
            if act_aware:
                scaling_diag_matrix = 1  # avoid zero division
                if hasattr(linear, "scaling_diag_matrix"):
                    # print("WARNING: scaling_diag_matrix is used")
                    scaling_diag_matrix *= linear.scaling_diag_matrix**alpha
                    # scaling_diag_matrix *= linear.scaling_diag_matrix**0.5
                if hasattr(linear, "fisher_info"):
                    scaling_diag_matrix *= linear.fisher_info**alpha
                    # scaling_diag_matrix *= linear.fisher_info**1
                # if not (scaling_diag_matrix == scaling_diag_matrix).all():
                #     breakpoint()
                scaling_diag_matrix += 1e-6  # avoid zero division
                w = w * scaling_diag_matrix.view(1, -1)
            Us = []
            Ss = []
            Vs = []
            try:
                U, S, V = torch.svd_lowrank(w, q=rank)
            except:
                print(f"svd failed for {linear}, disable act_aware")
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
                )
            if act_aware:
                V = V / scaling_diag_matrix.view(-1, 1)
            Us = [U]
            Ss = [S]
            Vs = [V]

            if linear.bias is not None:
                bias = linear.bias.data
            else:
                bias = None

            # nan or inf check
            for S in Ss:
                if (S != S).any():
                    print("nan in S")
                    return (
                        nn.Linear(linear.in_features, linear.out_features)
                        .to(linear.weight.dtype)
                        .to(linear.weight.device)
                    )
            for U in Us:
                if (U != U).any():
                    print("nan in U")
                    return (
                        nn.Linear(linear.in_features, linear.out_features)
                        .to(linear.weight.dtype)
                        .to(linear.weight.device)
                    )
            for V in Vs:
                if (V != V).any():
                    print("nan in V")
                    return (
                        nn.Linear(linear.in_features, linear.out_features)
                        .to(linear.weight.dtype)
                        .to(linear.weight.device)
                    )

            assert len(Us) == len(Ss) == len(Vs) == 1
            U, S, V = Us[0], Ss[0], Vs[0]
            if sigma_fuse == "UV":
                A = U.mul(S.sqrt()).contiguous()
                B = V.t().mul(S.sqrt().view(-1, 1)).contiguous()
            elif sigma_fuse == "U":
                A = U.mul(S).contiguous()
                B = V.t().contiguous()
            elif sigma_fuse == "V":
                A = U.contiguous()
                B = V.t().mul(S.view(-1, 1)).contiguous()
            new_linear = SVDLinear(A, B, param_ratio, bias)
            new_linear.to(linear.weight.dtype)
            return new_linear
        else:
            print("Cannot find scaling_diag_matrix or fisher_info, disable act_aware")
            return linear
    
    @staticmethod
    def from_trainable_decomp_linear(
        trainableDecompLinear: TrainableDecompLinear,
    ):
        A = torch.concat([trainableDecompLinear.ALinear_no_train.weight.data, trainableDecompLinear.ALinear_train.weight], dim=1)
        B = torch.concat([trainableDecompLinear.BLinear_no_train.weight.data, trainableDecompLinear.BLinear_train.weight], dim=0)
        if trainableDecompLinear.ALinear_train.bias is not None:
            bias = trainableDecompLinear.ALinear_train.bias.data
        else:
            bias = None
        new_linear = SVDLinear(A, B, 1, bias)
        return new_linear

    def change_param_ratio(
        self,
        new_param_ratio: float,
    ):
        BLinear_new_weight = self.BLinear.weight.data
        old_rank = self.truncation_rank
        old_param_ratio = self.param_ratio
        assert new_param_ratio <= old_param_ratio, "new_param_ratio should be smaller than old_param_ratio"
        new_rank = round(old_rank * (new_param_ratio / old_param_ratio))
        BLinear_new_weight = self.BLinear.weight.data[:new_rank]
        ALinear_new_weight = self.ALinear.weight.data[:, :new_rank]
        return SVDLinear(ALinear_new_weight, BLinear_new_weight, new_param_ratio, self.ALinear.bias)

    def forward(self, inp):
        # compute USV^Tx + b
        y = self.BLinear(inp)
        # y=torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        y = self.ALinear(y)
        # y=torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return y