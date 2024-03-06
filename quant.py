# Modified from https://github.com/Cornell-RelaxML/quip-sharp/blob/main/lib/algo/quip.py
import math

import scipy
import torch
from safetensors.torch import load_file

had_tensors = load_file("hadamard.safetensors")


def next_power_of_2(n):
    if n == 0:
        return 1
    return 2**math.ceil(math.log(n, 2))


def get_power_of_2(n):
    """Returns the highest power of 2 that divides n."""
    k = 0
    while n % 2 == 0:
        n //= 2
        k += 1
    return k, n


def get_hadK(n, use_rand=True):
    exp, base = get_power_of_2(n)
    if base == 1:
        return None, 1, n
    if use_rand:
        rand_mat = torch.tensor(scipy.stats.special_ortho_group.rvs(base)).to(torch.float32)
        return rand_mat, base, n

    # Use hadamad only and add padding if cannot find one
    pad_n = next_power_of_2(n)
    if exp < 2 or str(base * 4) not in had_tensors:
        return None, 1, pad_n
    base_mat = had_tensors[str(base * 4)]/math.sqrt(base * 4)
    return base_mat, base * 4, n


def matmul_hadU(X, hadK, K, padN, transpose=False):
    n = X.shape[-1]
    #hadK, K, padN = get_hadK(n, transpose, use_rand)
    if padN != n:
        input = torch.nn.functional.pad(X, (0, padN - n)).view(-1, padN, 1)
    else:
        input = X.clone().view(-1, n, 1)
    output = input.clone()
    while input.shape[1] > K:
        input = input.view(input.shape[0], input.shape[1] // 2, 2,
                           input.shape[2])
        output = output.view(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.view(input.shape[0], input.shape[1], -1)
        (input, output) = (output, input)
    del output
    if K > 1:
        if transpose:
            hadK = hadK.T
        input = torch.bmm(
            hadK.repeat(len(input), 1, 1).to(input.device).to(input.dtype),
            input)
    return input.view(*X.shape[:-1], padN) / torch.tensor(padN / K).sqrt()


def matmul_hadUt(X, hadK, K, padN):
    return matmul_hadU(X, hadK, K, padN, transpose=True)


def matmul_hadU_cuda(X, hadK, K, n, scale=None, transpose=False):
    if n != X.shape[-1]:
        X = torch.nn.functional.pad(X, (0, n - X.shape[-1]))
    had_scale = 1 / math.sqrt(n // K) if scale is None else scale / math.sqrt(n // K)

    if K == 1:
        return torch.ops.quip_lib.hadamard(X, had_scale)
    if transpose:
        hadK = hadK.T.contiguous()
    input = X.view(-1, K, n // K)
    input = torch.ops.quip_lib.hadamard(input, had_scale)
    input = hadK @ input
    return input.reshape(X.shape)


def matmul_hadUt_cuda(X, hadK, K, n, scale=None):
    return matmul_hadU_cuda(X, hadK, K, n, scale=scale, transpose=True)


def block_LDL(L, b):
    n = L.shape[0]
    assert (n % b == 0)
    m = n // b
    DL = torch.diagonal(L.reshape(m, b, m, b), dim1=0, dim2=2).permute(2, 0, 1)
    DL = torch.linalg.inv(DL)
    L = L.view(n, m, b)
    for i in range(m):
        L[:, i, :] = L[:, i, :] @ DL[i, :, :]
    if L.isnan().any():
        raise ValueError("Hessian is not invertible")
    L = L.reshape(n, n)
    return L


def LDLQ(Wr, Hr, L, cb, quip_tune_iters):
    '''
    want eta = (Wr - hatWr) @ L
    want hatWr + eta = Wr + (Wr - hatWr) @ (L - I)
    want hatWr = Q( Wr + (Wr - hatWr) @ (L - I) )
    '''
    (m, n) = Wr.shape
    L = block_LDL(L, cb.codesz)
    hatWr = torch.zeros(m, n, dtype=Hr.dtype, device=Hr.device)
    Qidxs = torch.zeros(m,
                        n // cb.codesz,
                        dtype=cb.idx_dtype,
                        device=Hr.device)
    for k in reversed(range(n // cb.codesz)):
        WXWX = Wr[:, (cb.codesz * k):(cb.codesz * (k + 1))] + \
            (Wr[:, (cb.codesz * (k + 1)):n] - hatWr[:, (cb.codesz * (k + 1)):n]) @ \
            L[(cb.codesz * (k + 1)):n, (cb.codesz * k):(cb.codesz * (k + 1))]
        hatWr[:, (cb.codesz * k):(cb.codesz * (k + 1))], Qidxs[:, k] = \
            cb.quantize(WXWX)
    for _ in range(quip_tune_iters):
        for k in reversed(range(n // cb.codesz)):
            WXWX = hatWr[:, (cb.codesz * k):(cb.codesz * (k + 1))] + (Wr - hatWr) @ \
                Hr[:, (cb.codesz * k):(cb.codesz * (k + 1))] @ \
                torch.linalg.inv(Hr[(cb.codesz * k):(cb.codesz * (k + 1)),
                                    (cb.codesz * k):(cb.codesz * (k + 1))])
            hatWr[:, (cb.codesz * k):(cb.codesz *
                                      (k + 1))], Qidxs[:,
                                                       k] = cb.quantize(WXWX)

    return hatWr, Qidxs


def LDLQ_buffered(Wr, Hr, L, cb, quip_tune_iters, buf_cols=128):
    '''
    reduce overhead of memory r/w
    buffer size is in groups of codesz (4) columns (for D4)
    '''
    (m, n) = Wr.shape
    assert buf_cols % cb.codesz == 0
    assert n % buf_cols == 0
    buf_size = buf_cols // cb.codesz

    L = block_LDL(L, cb.codesz)
    hatWr_T = torch.zeros(n, m, dtype=Hr.dtype, device=Hr.device)
    Qidxs_T = torch.zeros(n // cb.codesz,
                          m,
                          dtype=cb.idx_dtype,
                          device=Hr.device)

    Wr_T = Wr.T.contiguous()
    Wr = Wr.cpu()
    Hr_T = Hr.T.contiguous()
    Hr = Hr.cpu()
    torch.cuda.empty_cache()

    # quip
    prod_cache = torch.zeros(n, m, dtype=Wr_T.dtype, device=Wr_T.device)
    for cur_col in range(n // cb.codesz, 0, -buf_size):
        b_Wr_T = Wr_T[cb.codesz * (cur_col - buf_size):cb.codesz * cur_col]
        b_hatWr_T = hatWr_T[cb.codesz * (cur_col - buf_size):cb.codesz *
                            cur_col]
        b_L = L[cb.codesz * (cur_col - buf_size):cb.codesz *
                cur_col].contiguous()
        b_prod = prod_cache[cb.codesz * (cur_col - buf_size):cb.codesz *
                            cur_col]
        b_Qidxs_T = Qidxs_T[cur_col - buf_size:cur_col]
        L_offset = cb.codesz * (cur_col - buf_size)
        for i in reversed(range(buf_size)):
            WXWX = b_Wr_T[cb.codesz * i : cb.codesz * (i + 1)] + \
                b_L[cb.codesz * (i + 1):, L_offset + cb.codesz * i : L_offset + cb.codesz * (i + 1)].T @ \
                (b_Wr_T[cb.codesz * (i + 1):] - b_hatWr_T[cb.codesz * (i + 1):]) + \
                b_prod[cb.codesz * i : cb.codesz * (i + 1)]
            q_out = cb.quantize(WXWX.T)
            b_hatWr_T[cb.codesz * i:cb.codesz * (i + 1)] = q_out[0].T
            b_Qidxs_T[i] = q_out[1]

        prod_cache += b_L.T @ (b_Wr_T - b_hatWr_T)
        hatWr_T[cb.codesz * (cur_col - buf_size):cb.codesz *
                cur_col] = b_hatWr_T

    del b_Wr_T, b_hatWr_T, b_L, b_prod, L_offset, prod_cache
    torch.cuda.empty_cache()

    # tune
    for ie in range(quip_tune_iters):
        # recompute delta to minimize errors
        delta_T = Wr_T - hatWr_T
        for cur_col in range(n // cb.codesz, 0, -buf_size):
            b_hatWr_T = hatWr_T[cb.codesz * (cur_col - buf_size):cb.codesz *
                                cur_col]
            b_Hr_T = Hr_T[cb.codesz * (cur_col - buf_size):cb.codesz * cur_col]
            b_delta_T = delta_T[cb.codesz * (cur_col - buf_size):cb.codesz *
                                cur_col]
            b_Qidxs_T = Qidxs_T[cur_col - buf_size:cur_col]
            Hr_offset = cb.codesz * (cur_col - buf_size)
            for i in reversed(range(buf_size)):
                if cb.codesz > 1:
                    WXWX = b_hatWr_T[cb.codesz * i : cb.codesz * (i + 1)] + \
                        torch.linalg.inv(b_Hr_T[cb.codesz * i : cb.codesz * (i + 1), Hr_offset + cb.codesz * i : Hr_offset + cb.codesz * (i + 1)].T).T @ b_Hr_T[cb.codesz * i : cb.codesz * (i + 1)] @ delta_T
                else:
                    WXWX = b_hatWr_T[cb.codesz * i : cb.codesz * (i + 1)] + \
                        (1/b_Hr_T[i, Hr_offset + i]) * b_Hr_T[cb.codesz * i : cb.codesz * (i + 1)] @ delta_T
                b_delta_T[cb.codesz * i:cb.codesz *
                          (i + 1)] += b_hatWr_T[cb.codesz * i:cb.codesz *
                                                (i + 1)]

                if ie < quip_tune_iters - 1:
                    b_hatWr_T[cb.codesz * i:cb.codesz * (i + 1)] = cb.quantize(
                        WXWX.T, False).T
                else:
                    q_out = cb.quantize(WXWX.T)
                    b_hatWr_T[cb.codesz * i:cb.codesz * (i + 1)] = q_out[0].T
                    b_Qidxs_T[i] = q_out[1]

                b_delta_T[cb.codesz * i:cb.codesz *
                          (i + 1)] -= b_hatWr_T[cb.codesz * i:cb.codesz *
                                                (i + 1)]
            hatWr_T[cb.codesz * (cur_col - buf_size):cb.codesz *
                    cur_col] = b_hatWr_T
            Qidxs_T[cur_col - buf_size:cur_col] = b_Qidxs_T

        del delta_T, b_hatWr_T, b_Hr_T, b_delta_T, b_Qidxs_T, Hr_offset
        torch.cuda.empty_cache()

    return hatWr_T.T.contiguous(), Qidxs_T.T.contiguous()
