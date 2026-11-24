"""Extraction of MACE-Field models (https://github.com/mdi-group/mace-field) for MACEFoundation (CUDA).

MACE-Field adds, after every interaction but the last, a coupling to a uniform electric field eps (irrep 1o):

    h' = h - field_linear[i](field_feats[i](h, eps))

where h are the node features after the product (h = M0 @ W0 + sc0 for the first layer).  The coupling is
bilinear in (h, eps), so it is stored as a sum of terms

    Delta_{l_out}[k, m_o] = sum_t W_t[k, k'] cg_t[m_o, m_i, m_f] h_{l_in}[k', m_i] eps_{m_f}

found by probing the torch modules with basis vectors and factorizing each (l_in, l_out) block by SVD; nothing
about the paths of the tensor product is assumed.  eps is used exactly as the model's ``electric_field`` input
(Cartesian components in the order given), and the polarization returned by Symmetrix is
P = -dE/d eps (e*Angstrom for the whole cell), as ``mace.modules.utils.get_polarization``.

Because the products[0].linear weights W0 are not invertible, the field mode keeps h explicit: the JSON holds
W0, the bias sc0 and the downstream weights acting on h (instead of on M0, as in the plain extraction).
"""

import logging

import numpy as np
import torch

from .extract_mace_data import extract_mace_data


def _probe_terms(model, tol=1e-9):
    ff = model.field_feats[0]
    fl = model.field_linear[0]
    irreps = ff.irreps_in1
    d = irreps.dim
    slices = {}
    for (mul, ir), sl in zip(irreps, irreps.slices()):
        if ir.l in slices:
            raise RuntimeError("repeated l in hidden irreps is not supported")
        slices[ir.l] = (sl.start, mul)
    eye = torch.eye(d, dtype=torch.float64)
    with torch.no_grad():
        out = []
        for mf in range(3):
            e = torch.zeros(d, 3, dtype=torch.float64)
            e[:, mf] = 1
            out.append(fl(ff(eye, e)))
        T = torch.stack(out).numpy()  # [mf, i_in, i_out]
    terms = []
    for li, (si, Ci) in slices.items():
        for lo, (so, Co) in slices.items():
            ni, no = 2 * li + 1, 2 * lo + 1
            # X[k_out, mo, k_in, mi, mf]
            blk = T[:, si : si + Ci * ni, so : so + Co * no].reshape(3, Ci, ni, Co, no)
            X = blk.transpose(3, 1, 4, 2, 0)  # k_out, k_in, mo, mi, mf
            Mx = X.reshape(Co * Ci, no * ni * 3)
            if not np.any(Mx):
                continue
            U, s, Vt = np.linalg.svd(Mx, full_matrices=False)
            keep = s > tol * s[0]
            for c in np.nonzero(keep)[0]:
                terms.append(
                    {
                        "l_in": li,
                        "l_out": lo,
                        "W": (U[:, c] * s[c]).tolist(),  # [k_out][k_in]
                        "cg": Vt[c].tolist(),  # [mo][mi][mf]
                    }
                )
            logging.info(
                f"field block l_in={li} l_out={lo}: rank {keep.sum()} (s={s[:3]})"
            )
    # validate
    rng = np.random.default_rng(0)
    h = torch.tensor(rng.normal(size=(5, d)))
    e = torch.tensor(rng.normal(size=(5, 3)))
    with torch.no_grad():
        ref = fl(ff(h, e)).numpy()
    rec = np.zeros_like(ref)
    for t in terms:
        li, lo = t["l_in"], t["l_out"]
        si, Ci = slices[li]
        so, Co = slices[lo]
        ni, no = 2 * li + 1, 2 * lo + 1
        W = np.reshape(t["W"], (Co, Ci))
        cg = np.reshape(t["cg"], (no, ni, 3))
        hl = h.numpy()[:, si : si + Ci * ni].reshape(5, Ci, ni)
        v = np.einsum("oim,bki,bm->bko", cg, hl, e.numpy())
        rec[:, so : so + Co * no] += np.einsum("jk,bko->bjo", W, v).reshape(5, Co * no)
    err = np.abs(rec - ref).max() / np.abs(ref).max()
    logging.warning(
        f"field coupling: {len(terms)} terms, relative reconstruction error {err:.2e}"
    )
    if err > 1e-10:
        raise RuntimeError(f"field coupling reconstruction failed ({err})")
    return terms, err


def extract_mace_field_data(model_file, species, **kwargs):
    output = extract_mace_data(model_file, species, **kwargs)
    model = torch.load(model_file, map_location="cpu", weights_only=False).to(
        dtype=torch.float64
    )
    model.eval()
    if type(model).__name__ != "MACEField":
        raise RuntimeError(f"{type(model).__name__} is not a MACEField model")
    if len(model.field_feats) != 1:
        raise RuntimeError("only two-layer MACE-Field models are supported")
    from mace.modules.blocks import RealAgnosticResidualInteractionBlock

    C = output["num_channels"]
    CN = output.get("num_channels_node", C)
    if CN != C:
        raise RuntimeError("field mode needs num_channels_node == num_channels")
    L_max = output["L_max"]
    Tm = len(model.atomic_numbers)
    atomic_numbers = output["atomic_numbers"]
    indices = [model.atomic_numbers.tolist().index(a) for a in atomic_numbers]

    W0 = np.reshape(
        model.products[0].linear.weight.numpy(force=True), [L_max + 1, CN, C]
    ) / np.sqrt(CN)
    W1 = np.reshape(
        model.interactions[1].linear_up.weight.numpy(force=True), [L_max + 1, C, C]
    ) / np.sqrt(C)
    readout_1 = model.readouts[0].linear.weight.numpy(force=True) / np.sqrt(C)
    scale = model.scale_shift.scale.item()

    first_residual = isinstance(
        model.interactions[0], RealAgnosticResidualInteractionBlock
    )
    if first_residual:
        with torch.no_grad():
            eye = torch.eye(Tm, dtype=torch.float64)
            sc0 = (
                model.interactions[0]
                .skip_tp(model.node_embedding(eye), eye)[:, :C]
                .numpy(force=True)
            )
        output["atomic_energies"] = [
            e - scale * float(sc0[i] @ readout_1)
            for e, i in zip(output["atomic_energies"], indices)
        ]
        h_bias = [sc0[i].tolist() for i in indices]
    else:
        h_bias = [[0.0] * C for _ in indices]
    output.pop("H1_bias", None)
    output.pop("H2_bias", None)

    output["H1_weights"] = W1.flatten().tolist()
    output["readout_1_weights_for_M0"] = (readout_1 * scale).tolist()
    H2 = []
    for i in indices:
        w = (
            model.interactions[1]
            .skip_tp.weight_view_for_instruction(0)[:, i, :]
            .numpy(force=True)
        )
        H2.append((w / np.sqrt(Tm * C)).flatten().tolist())
    output["H2_weights_for_M0"] = H2

    terms, err = _probe_terms(model)
    # Symmetrix uses r_ij = x_j - x_i, the opposite of MACE's edge vectors, so its features of degree l are
    # (-1)^l times those of e3nn: a term coupling l_in to l_out changes sign by (-1)^(l_in + l_out).
    for t in terms:
        if (t["l_in"] + t["l_out"]) % 2:
            t["cg"] = [-c for c in t["cg"]]
    output["field_coupling"] = {
        "W0": W0.flatten().tolist(),  # [L][k_M0][k_h]
        "h_bias": h_bias,  # [element][k_h], added to the l=0 part of h
        "terms": terms,
        "reconstruction_error": err,
        "convention": "h' = h - sum_t W_t cg_t h eps; P = -dE/d eps (e*Angstrom, whole cell)",
    }
    return output
