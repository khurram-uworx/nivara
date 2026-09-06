"""Generate reference tensors for each NN layer type using PyTorch.

Saves raw float32 binary + JSON manifest for C# unit tests to compare against.
Test fixtures go to samples/data/torch-comparison/.

Reproducibility:
  - Verified with Python 3.12, torch 2.13.0+cpu, numpy 1.26.
  - Fixtures are RNG-derived. Both the global torch RNG (consumed by nn.*
    module weight initialization) and a dedicated torch.Generator are seeded
    with 42 (see run()), and every random draw uses generator=rng. CPU-only
    execution is required for bit-stable output. Add a new draw to the end of
    a case, never insert one mid-stream, or every subsequent fixture changes
    and the C# manifest must be regenerated together.
  - Regenerate after upgrading torch/numpy: run `python gen_reference.py` and
    commit the full samples/data/torch-comparison/ tree as one unit.

Usage: python gen_reference.py
"""
import os, json, struct, sys, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEST_DIR = os.path.join(REPO_ROOT, "samples", "data", "torch-comparison")


def save_tensor(path, tensor):
    """Save a torch tensor as raw float32 binary."""
    arr = tensor.detach().cpu().contiguous().numpy().astype(np.float32)
    arr.tofile(path)
    return arr.shape


def run():
    os.makedirs(TEST_DIR, exist_ok=True)
    manifest = {}

    print(f"torch {torch.__version__} | numpy {np.__version__} | python {sys.version.split()[0]}")
    print(f"RNG seed: 42 (global torch RNG + dedicated per-case generators)\n")

    # Deterministic RNG
    torch.manual_seed(42)
    rng = torch.Generator()
    rng.manual_seed(42)

    # =========================================================================
    # Conv2d tests
    # =========================================================================
    cases = [
        # (name, in_ch, out_ch, k, s, p, groups, input_shape)
        ("conv2d_3x3_s1_p1",    3,  16, 3, 1, 1, 1,  (1, 3, 7, 7)),
        ("conv2d_1x1_s1_p0",    3,  32, 1, 1, 0, 1,  (1, 3, 7, 7)),
        ("conv2d_depthwise",   16,  16, 3, 1, 1, 16, (1, 16, 5, 5)),
        ("conv2d_stride2",      3,  32, 3, 2, 1, 1,  (1, 3, 14, 14)),
        ("conv2d_with_bias",    3,   8, 3, 1, 1, 1,  (1, 3, 4, 4)),
    ]

    for name, in_ch, out_ch, k, s, p, groups, inp_shape in cases:
        conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, groups=groups, bias=True)
        inp = torch.randn(inp_shape, generator=rng)

        with torch.no_grad():
            out = conv(inp)

        inp_np = inp.numpy().astype(np.float32)
        w_np = conv.weight.data.cpu().numpy().astype(np.float32)
        b_np = conv.bias.data.cpu().numpy().astype(np.float32)
        out_np = out.numpy().astype(np.float32)

        inp_np.tofile(os.path.join(TEST_DIR, f"{name}_input.bin"))
        w_np.tofile(os.path.join(TEST_DIR, f"{name}_weight.bin"))
        b_np.tofile(os.path.join(TEST_DIR, f"{name}_bias.bin"))
        out_np.tofile(os.path.join(TEST_DIR, f"{name}_output.bin"))

        manifest[name] = {
            "layer": "Conv2d",
            "input_shape": list(inp_shape),
            "weight_shape": list(w_np.shape),
            "bias_shape": list(b_np.shape),
            "output_shape": list(out_np.shape),
            "params": {"in_channels": in_ch, "out_channels": out_ch,
                       "kernel_size": k, "stride": s, "padding": p, "groups": groups},
        }
        print(f"  {name}: input={inp_shape} weight={w_np.shape} output={out_np.shape}")

    # =========================================================================
    # Conv1d tests
    # =========================================================================
    conv1d_cases = [
        # (name, in_ch, out_ch, k, s, p, input_shape)
        ("conv1d_k3",    8,  8, 3, 1, 1, (1, 8, 16)),
        ("conv1d_k5",    8, 16, 5, 1, 2, (1, 8, 16)),
        ("conv1d_k7",    4,  8, 7, 1, 3, (1, 4, 32)),
        ("conv1d_s2",    8, 16, 3, 2, 1, (1, 8, 16)),
    ]

    for name, in_ch, out_ch, k, s, p, inp_shape in conv1d_cases:
        conv = nn.Conv1d(in_ch, out_ch, k, stride=s, padding=p, bias=True)
        inp = torch.randn(inp_shape, generator=rng)

        with torch.no_grad():
            out = conv(inp)

        inp_np = inp.numpy().astype(np.float32)
        w_np = conv.weight.data.cpu().numpy().astype(np.float32)
        b_np = conv.bias.data.cpu().numpy().astype(np.float32)
        out_np = out.numpy().astype(np.float32)

        inp_np.tofile(os.path.join(TEST_DIR, f"{name}_input.bin"))
        w_np.tofile(os.path.join(TEST_DIR, f"{name}_weight.bin"))
        b_np.tofile(os.path.join(TEST_DIR, f"{name}_bias.bin"))
        out_np.tofile(os.path.join(TEST_DIR, f"{name}_output.bin"))

        manifest[name] = {
            "layer": "Conv1d",
            "input_shape": list(inp_shape),
            "weight_shape": list(w_np.shape),
            "bias_shape": list(b_np.shape),
            "output_shape": list(out_np.shape),
            "params": {"in_channels": in_ch, "out_channels": out_ch,
                       "kernel_size": k, "stride": s, "padding": p},
        }
        print(f"  {name}: input={inp_shape} weight={w_np.shape} output={out_np.shape}")

    # =========================================================================
    # BatchNorm2d tests (eval mode with running stats)
    # =========================================================================
    bn_cases = [
        # (name, num_features, input_shape)
        ("bn2d_16ch",   16, (1, 16, 5, 5)),
        ("bn2d_3ch",     3, (1, 3, 7, 7)),
        ("bn2d_batch4", 16, (4, 16, 8, 8)),  # batch > 1 to verify running stats != batch stats
    ]

    for name, nf, inp_shape in bn_cases:
        bn = nn.BatchNorm2d(nf)
        inp = torch.randn(inp_shape, generator=rng)

        # Set known running stats
        bn.running_mean.copy_(torch.randn(nf, generator=rng) * 0.5)
        bn.running_var.copy_(torch.rand(nf, generator=rng) + 0.5)
        bn.weight.data.copy_(torch.randn(nf, generator=rng) * 0.1 + 1.0)
        bn.bias.data.copy_(torch.randn(nf, generator=rng) * 0.1)

        bn.eval()

        with torch.no_grad():
            out = bn(inp)

        inp_np = inp.numpy().astype(np.float32)
        gamma_np = bn.weight.data.cpu().numpy().astype(np.float32)
        beta_np = bn.bias.data.cpu().numpy().astype(np.float32)
        rm_np = bn.running_mean.cpu().numpy().astype(np.float32)
        rv_np = bn.running_var.cpu().numpy().astype(np.float32)
        out_np = out.numpy().astype(np.float32)

        inp_np.tofile(os.path.join(TEST_DIR, f"{name}_input.bin"))
        gamma_np.tofile(os.path.join(TEST_DIR, f"{name}_gamma.bin"))
        beta_np.tofile(os.path.join(TEST_DIR, f"{name}_beta.bin"))
        rm_np.tofile(os.path.join(TEST_DIR, f"{name}_running_mean.bin"))
        rv_np.tofile(os.path.join(TEST_DIR, f"{name}_running_var.bin"))
        out_np.tofile(os.path.join(TEST_DIR, f"{name}_output.bin"))

        manifest[name] = {
            "layer": "BatchNorm2d",
            "input_shape": list(inp_shape),
            "gamma_shape": list(gamma_np.shape),
            "beta_shape": list(beta_np.shape),
            "running_mean_shape": list(rm_np.shape),
            "running_var_shape": list(rv_np.shape),
            "output_shape": list(out_np.shape),
            "params": {"num_features": nf, "eps": 1e-5},
        }
        print(f"  {name}: input={inp_shape} running_mean={rm_np.shape} output={out_np.shape}")

    # Also save what the batch-stats-only result would be (the bug case)
    for name, nf, inp_shape in bn_cases:
        bug_name = name + "_batch_stats"
        bn = nn.BatchNorm2d(nf)
        inp = torch.randn(inp_shape, generator=rng)

        # Set known running stats but also compute in TRAIN mode to show difference
        bn.running_mean.copy_(torch.randn(nf, generator=rng) * 0.5)
        bn.running_var.copy_(torch.rand(nf, generator=rng) + 0.5)
        bn.weight.data.copy_(torch.randn(nf, generator=rng) * 0.1 + 1.0)
        bn.bias.data.copy_(torch.randn(nf, generator=rng) * 0.1)

        # Use train mode -> batch stats (this is what Nivara currently does wrong)
        bn.train()

        with torch.no_grad():
            out_bug = bn(inp)

        inp_np = inp.numpy().astype(np.float32)
        out_bug_np = out_bug.numpy().astype(np.float32)
        inp_np.tofile(os.path.join(TEST_DIR, f"{bug_name}_input.bin"))
        out_bug_np.tofile(os.path.join(TEST_DIR, f"{bug_name}_output.bin"))

        manifest[bug_name] = {
            "layer": "BatchNorm2d",
            "note": "batch_stats_only (no running stats) - the bug case",
            "input_shape": list(inp_shape),
            "output_shape": list(out_bug_np.shape),
        }
        print(f"  {bug_name}: input={inp_shape} output={out_bug_np.shape}")

    # =========================================================================
    # BatchNorm1d tests (eval mode with running stats)
    # =========================================================================
    bn1d_cases = [
        # (name, num_features, input_shape)
        ("bn1d_2d",  16, (4, 16)),
        ("bn1d_3d",   8, (2, 8, 20)),
    ]

    for name, nf, inp_shape in bn1d_cases:
        bn = nn.BatchNorm1d(nf)
        inp = torch.randn(inp_shape, generator=rng)

        bn.running_mean.copy_(torch.randn(nf, generator=rng) * 0.5)
        bn.running_var.copy_(torch.rand(nf, generator=rng) + 0.5)
        bn.weight.data.copy_(torch.randn(nf, generator=rng) * 0.1 + 1.0)
        bn.bias.data.copy_(torch.randn(nf, generator=rng) * 0.1)

        bn.eval()

        with torch.no_grad():
            out = bn(inp)

        inp_np = inp.numpy().astype(np.float32)
        gamma_np = bn.weight.data.cpu().numpy().astype(np.float32)
        beta_np = bn.bias.data.cpu().numpy().astype(np.float32)
        rm_np = bn.running_mean.cpu().numpy().astype(np.float32)
        rv_np = bn.running_var.cpu().numpy().astype(np.float32)
        out_np = out.numpy().astype(np.float32)

        inp_np.tofile(os.path.join(TEST_DIR, f"{name}_input.bin"))
        gamma_np.tofile(os.path.join(TEST_DIR, f"{name}_gamma.bin"))
        beta_np.tofile(os.path.join(TEST_DIR, f"{name}_beta.bin"))
        rm_np.tofile(os.path.join(TEST_DIR, f"{name}_running_mean.bin"))
        rv_np.tofile(os.path.join(TEST_DIR, f"{name}_running_var.bin"))
        out_np.tofile(os.path.join(TEST_DIR, f"{name}_output.bin"))

        manifest[name] = {
            "layer": "BatchNorm1d",
            "input_shape": list(inp_shape),
            "gamma_shape": list(gamma_np.shape),
            "beta_shape": list(beta_np.shape),
            "running_mean_shape": list(rm_np.shape),
            "running_var_shape": list(rv_np.shape),
            "output_shape": list(out_np.shape),
            "params": {"num_features": nf, "eps": 1e-5},
        }
        print(f"  {name}: input={inp_shape} output={out_np.shape}")

    # =========================================================================
    # ReLU / ReLU6 tests
    # =========================================================================
    relu_cases = [
        ("relu_1d",  (32,)),
        ("relu_4d",  (1, 16, 8, 8)),
    ]

    for name, inp_shape in relu_cases:
        inp = torch.randn(inp_shape, generator=rng)
        out_relu = torch.relu(inp)
        out_relu6 = torch.nn.functional.relu6(inp)

        inp_np = inp.numpy().astype(np.float32)
        out_relu_np = out_relu.numpy().astype(np.float32)
        out_relu6_np = out_relu6.numpy().astype(np.float32)

        inp_np.tofile(os.path.join(TEST_DIR, f"{name}_input.bin"))
        out_relu_np.tofile(os.path.join(TEST_DIR, f"{name}_relu_output.bin"))
        out_relu6_np.tofile(os.path.join(TEST_DIR, f"{name}_relu6_output.bin"))

        manifest[name] = {
            "layer": "ReLU/ReLU6",
            "input_shape": list(inp_shape),
            "relu_output_shape": list(out_relu_np.shape),
            "relu6_output_shape": list(out_relu6_np.shape),
        }
        print(f"  {name}: input={inp_shape}")

    # =========================================================================
    # LeakyReLU tests
    # =========================================================================
    leaky_cases = [
        ("leaky_relu_1d", (32,)),
        ("leaky_relu_4d", (1, 8, 4, 4)),
    ]

    for name, inp_shape in leaky_cases:
        inp = torch.randn(inp_shape, generator=rng)
        out = F.leaky_relu(inp, negative_slope=0.01)

        inp_np = inp.numpy().astype(np.float32)
        out_np = out.numpy().astype(np.float32)

        inp_np.tofile(os.path.join(TEST_DIR, f"{name}_input.bin"))
        out_np.tofile(os.path.join(TEST_DIR, f"{name}_output.bin"))

        manifest[name] = {
            "layer": "LeakyReLU",
            "input_shape": list(inp_shape),
            "output_shape": list(out_np.shape),
            "params": {"negative_slope": 0.01},
        }
        print(f"  {name}: input={inp_shape} output={out_np.shape}")

    # =========================================================================
    # Sigmoid tests
    # =========================================================================
    sigmoid_cases = [
        ("sigmoid_1d", (32,)),
        ("sigmoid_4d", (1, 8, 4, 4)),
    ]

    for name, inp_shape in sigmoid_cases:
        inp = torch.randn(inp_shape, generator=rng)
        out = torch.sigmoid(inp)

        inp_np = inp.numpy().astype(np.float32)
        out_np = out.numpy().astype(np.float32)

        inp_np.tofile(os.path.join(TEST_DIR, f"{name}_input.bin"))
        out_np.tofile(os.path.join(TEST_DIR, f"{name}_output.bin"))

        manifest[name] = {
            "layer": "Sigmoid",
            "input_shape": list(inp_shape),
            "output_shape": list(out_np.shape),
        }
        print(f"  {name}: input={inp_shape} output={out_np.shape}")

    # =========================================================================
    # Tanh tests
    # =========================================================================
    tanh_cases = [
        ("tanh_1d", (32,)),
        ("tanh_4d", (1, 8, 4, 4)),
    ]

    for name, inp_shape in tanh_cases:
        inp = torch.randn(inp_shape, generator=rng)
        out = torch.tanh(inp)

        inp_np = inp.numpy().astype(np.float32)
        out_np = out.numpy().astype(np.float32)

        inp_np.tofile(os.path.join(TEST_DIR, f"{name}_input.bin"))
        out_np.tofile(os.path.join(TEST_DIR, f"{name}_output.bin"))

        manifest[name] = {
            "layer": "Tanh",
            "input_shape": list(inp_shape),
            "output_shape": list(out_np.shape),
        }
        print(f"  {name}: input={inp_shape} output={out_np.shape}")

    # =========================================================================
    # GELU tests
    # "gelu_*" fixtures use the tanh approximation (PyTorch F.gelu approximate="tanh"),
    # "gelu_exact_*" fixtures use the exact erf-based GELU (F.gelu default).
    # The exact cases use a dedicated RNG so the main rng stream (and therefore all
    # other fixtures) is unaffected.
    # =========================================================================
    gelu_cases = [
        ("gelu_1d", (32,)),
        ("gelu_4d", (1, 8, 4, 4)),
    ]

    for name, inp_shape in gelu_cases:
        inp = torch.randn(inp_shape, generator=rng)
        out = F.gelu(inp, approximate="tanh")

        inp_np = inp.numpy().astype(np.float32)
        out_np = out.numpy().astype(np.float32)

        inp_np.tofile(os.path.join(TEST_DIR, f"{name}_input.bin"))
        out_np.tofile(os.path.join(TEST_DIR, f"{name}_output.bin"))

        manifest[name] = {
            "layer": "GELU (tanh)",
            "input_shape": list(inp_shape),
            "output_shape": list(out_np.shape),
        }
        print(f"  {name}: input={inp_shape} output={out_np.shape}")

    gelu_exact_rng = torch.Generator().manual_seed(101)
    gelu_exact_cases = [
        ("gelu_exact_1d", (32,)),
        ("gelu_exact_4d", (1, 8, 4, 4)),
    ]

    for name, inp_shape in gelu_exact_cases:
        inp = torch.randn(inp_shape, generator=gelu_exact_rng)
        out = F.gelu(inp)

        inp_np = inp.numpy().astype(np.float32)
        out_np = out.numpy().astype(np.float32)

        inp_np.tofile(os.path.join(TEST_DIR, f"{name}_input.bin"))
        out_np.tofile(os.path.join(TEST_DIR, f"{name}_output.bin"))

        manifest[name] = {
            "layer": "GELU (exact)",
            "input_shape": list(inp_shape),
            "output_shape": list(out_np.shape),
        }
        print(f"  {name}: input={inp_shape} output={out_np.shape}")

    # =========================================================================
    # MaxPool2d tests
    # =========================================================================
    pool_cases = [
        ("maxpool_3x3_s2_p1",  (1, 16, 14, 14), 3, 2, 1),
        ("maxpool_2x2_s2_p0",  (1, 32, 28, 28), 2, 2, 0),
    ]

    for name, inp_shape, k, s, p in pool_cases:
        pool = nn.MaxPool2d(k, stride=s, padding=p)
        inp = torch.randn(inp_shape, generator=rng)

        with torch.no_grad():
            out = pool(inp)

        inp_np = inp.numpy().astype(np.float32)
        out_np = out.numpy().astype(np.float32)

        inp_np.tofile(os.path.join(TEST_DIR, f"{name}_input.bin"))
        out_np.tofile(os.path.join(TEST_DIR, f"{name}_output.bin"))

        manifest[name] = {
            "layer": "MaxPool2d",
            "input_shape": list(inp_shape),
            "output_shape": list(out_np.shape),
            "params": {"kernel_size": k, "stride": s, "padding": p},
        }
        print(f"  {name}: input={inp_shape} output={out_np.shape}")

    # =========================================================================
    # AdaptiveAvgPool2d tests
    # =========================================================================
    aap_cases = [
        ("adaptiveavgpool_1x1", (1, 512, 7, 7), 1),
        ("adaptiveavgpool_1x1_sm", (1, 32, 14, 14), 1),
    ]

    for name, inp_shape, out_size in aap_cases:
        pool = nn.AdaptiveAvgPool2d(out_size)
        inp = torch.randn(inp_shape, generator=rng)

        with torch.no_grad():
            out = pool(inp)

        inp_np = inp.numpy().astype(np.float32)
        out_np = out.numpy().astype(np.float32)

        inp_np.tofile(os.path.join(TEST_DIR, f"{name}_input.bin"))
        out_np.tofile(os.path.join(TEST_DIR, f"{name}_output.bin"))

        manifest[name] = {
            "layer": "AdaptiveAvgPool2d",
            "input_shape": list(inp_shape),
            "output_shape": list(out_np.shape),
            "params": {"output_size": out_size},
        }
        print(f"  {name}: input={inp_shape} output={out_np.shape}")

    # =========================================================================
    # Linear tests
    # =========================================================================
    linear_cases = [
        ("linear_128_64",     128, 64),
        ("linear_512_1000",   512, 1000),
    ]

    for name, in_f, out_f in linear_cases:
        lin = nn.Linear(in_f, out_f, bias=True)
        inp = torch.randn(1, in_f, generator=rng)

        with torch.no_grad():
            out = lin(inp)

        inp_np = inp.numpy().astype(np.float32)
        w_np = lin.weight.data.cpu().numpy().astype(np.float32)
        b_np = lin.bias.data.cpu().numpy().astype(np.float32)
        out_np = out.numpy().astype(np.float32)

        inp_np.tofile(os.path.join(TEST_DIR, f"{name}_input.bin"))
        w_np.tofile(os.path.join(TEST_DIR, f"{name}_weight.bin"))
        b_np.tofile(os.path.join(TEST_DIR, f"{name}_bias.bin"))
        out_np.tofile(os.path.join(TEST_DIR, f"{name}_output.bin"))

        manifest[name] = {
            "layer": "Linear",
            "input_shape": list(inp.shape),
            "weight_shape": list(w_np.shape),
            "bias_shape": list(b_np.shape),
            "output_shape": list(out_np.shape),
            "params": {"in_features": in_f, "out_features": out_f},
        }
        print(f"  {name}: input={inp.shape} weight={w_np.shape} output={out_np.shape}")

    # =========================================================================
    # Embedding tests
    # =========================================================================
    emb_vocab = 100
    emb_dim = 16
    emb_weight = torch.randn(emb_vocab, emb_dim, generator=rng)

    # Single token lookup
    single_idx = torch.tensor([42])
    single_out = emb_weight[42]

    single_idx.numpy().astype(np.int32).tofile(os.path.join(TEST_DIR, "emb_single_input.bin"))
    emb_weight.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "emb_single_weight.bin"))
    single_out.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "emb_single_output.bin"))

    manifest["emb_single"] = {
        "layer": "Embedding",
        "input_shape": [1],
        "weight_shape": list(emb_weight.shape),
        "output_shape": list(single_out.shape),
        "params": {"num_embeddings": emb_vocab, "embedding_dim": emb_dim},
    }
    print(f"  emb_single: input=[1] weight={emb_weight.shape} output={single_out.shape}")

    # Batch lookup
    batch_idx = torch.tensor([0, 13, 42, 99])
    batch_out = emb_weight[batch_idx]

    batch_idx.numpy().astype(np.int32).tofile(os.path.join(TEST_DIR, "emb_batch_input.bin"))
    emb_weight.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "emb_batch_weight.bin"))
    batch_out.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "emb_batch_output.bin"))

    manifest["emb_batch"] = {
        "layer": "Embedding",
        "input_shape": [4],
        "weight_shape": list(emb_weight.shape),
        "output_shape": list(batch_out.shape),
        "params": {"num_embeddings": emb_vocab, "embedding_dim": emb_dim},
    }
    print(f"  emb_batch: input=[4] weight={emb_weight.shape} output={batch_out.shape}")

    # =========================================================================
    # Dropout tests (eval mode = passthrough)
    # =========================================================================
    drop_inp = torch.randn(4, 32, generator=rng)
    drop_out = F.dropout(drop_inp, p=0.5, training=False)

    drop_inp.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "dropout_eval_input.bin"))
    drop_out.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "dropout_eval_output.bin"))

    manifest["dropout_eval"] = {
        "layer": "Dropout",
        "input_shape": list(drop_inp.shape),
        "output_shape": list(drop_out.shape),
        "params": {"p": 0.5, "training": False},
    }
    print(f"  dropout_eval: input={drop_inp.shape} output={drop_out.shape}")

    # =========================================================================
    # RMSNorm test (op-level, no affine gamma — the op normalizes only)
    # The affine-gamma RMSNorm<T> module is exercised separately below via
    # nn.RMSNorm in "rmsnorm_module_2d".
    # =========================================================================
    rms_cases = [
        ("rmsnorm_2d", (4, 32)),
        ("rmsnorm_3d", (2, 4, 32)),
    ]

    for name, inp_shape in rms_cases:
        inp = torch.randn(inp_shape, generator=rng)
        eps = 1e-5
        rms = torch.sqrt(torch.mean(inp ** 2, dim=-1, keepdim=True) + eps)
        out = inp / rms

        inp_np = inp.numpy().astype(np.float32)
        out_np = out.numpy().astype(np.float32)

        inp_np.tofile(os.path.join(TEST_DIR, f"{name}_input.bin"))
        out_np.tofile(os.path.join(TEST_DIR, f"{name}_output.bin"))

        manifest[name] = {
            "layer": "RMSNorm",
            "input_shape": list(inp_shape),
            "output_shape": list(out_np.shape),
            "params": {"eps": eps},
        }
        print(f"  {name}: input={inp_shape} output={out_np.shape}")

    # =========================================================================
    # LayerNorm tests (with learnable params)
    # =========================================================================
    ln_cases = [
        ("layernorm_2d", (4, 32), 32),
        ("layernorm_3d", (2, 4, 32), 32),
    ]

    for name, inp_shape, norm_shape in ln_cases:
        ln = nn.LayerNorm(norm_shape)
        inp = torch.randn(inp_shape, generator=rng)

        with torch.no_grad():
            out = ln(inp)

        inp_np = inp.numpy().astype(np.float32)
        gamma_np = ln.weight.data.cpu().numpy().astype(np.float32)
        beta_np = ln.bias.data.cpu().numpy().astype(np.float32)
        out_np = out.numpy().astype(np.float32)

        inp_np.tofile(os.path.join(TEST_DIR, f"{name}_input.bin"))
        gamma_np.tofile(os.path.join(TEST_DIR, f"{name}_gamma.bin"))
        beta_np.tofile(os.path.join(TEST_DIR, f"{name}_beta.bin"))
        out_np.tofile(os.path.join(TEST_DIR, f"{name}_output.bin"))

        manifest[name] = {
            "layer": "LayerNorm",
            "input_shape": list(inp_shape),
            "gamma_shape": list(gamma_np.shape),
            "beta_shape": list(beta_np.shape),
            "output_shape": list(out_np.shape),
            "params": {"normalized_shape": norm_shape, "eps": 1e-5},
        }
        print(f"  {name}: input={inp_shape} output={out_np.shape}")

    # =========================================================================
    # Softmax tests
    # =========================================================================
    softmax_inp = torch.randn(4, 10, generator=rng)
    softmax_out = F.softmax(softmax_inp, dim=1)

    softmax_inp.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "softmax_input.bin"))
    softmax_out.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "softmax_output.bin"))

    manifest["softmax"] = {
        "layer": "Softmax",
        "input_shape": list(softmax_inp.shape),
        "output_shape": list(softmax_out.shape),
        "params": {"dim": 1},
    }
    print(f"  softmax: input={softmax_inp.shape} output={softmax_out.shape}")

    # =========================================================================
    # LogSoftmax tests
    # =========================================================================
    logsoftmax_inp = torch.randn(4, 10, generator=rng)
    logsoftmax_out = F.log_softmax(logsoftmax_inp, dim=1)

    logsoftmax_inp.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "log_softmax_input.bin"))
    logsoftmax_out.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "log_softmax_output.bin"))

    manifest["log_softmax"] = {
        "layer": "LogSoftmax",
        "input_shape": list(logsoftmax_inp.shape),
        "output_shape": list(logsoftmax_out.shape),
        "params": {"dim": 1},
    }
    print(f"  log_softmax: input={logsoftmax_inp.shape} output={logsoftmax_out.shape}")

    # =========================================================================
    # MatMul tests
    # =========================================================================
    matmul_a = torch.randn(4, 8, generator=rng)
    matmul_b = torch.randn(8, 16, generator=rng)
    matmul_out = torch.matmul(matmul_a, matmul_b)

    matmul_a.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "matmul_a.bin"))
    matmul_b.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "matmul_b.bin"))
    matmul_out.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "matmul_output.bin"))

    manifest["matmul"] = {
        "layer": "MatMul",
        "a_shape": list(matmul_a.shape),
        "b_shape": list(matmul_b.shape),
        "output_shape": list(matmul_out.shape),
    }
    print(f"  matmul: a={matmul_a.shape} b={matmul_b.shape} output={matmul_out.shape}")

    # =========================================================================
    # BCEWithLogitsLoss tests
    # =========================================================================
    bce_inp = torch.randn(4, 10, generator=rng)
    bce_target = torch.rand(4, 10, generator=rng)

    bce_sum = F.binary_cross_entropy_with_logits(bce_inp, bce_target, reduction='sum')
    bce_mean = F.binary_cross_entropy_with_logits(bce_inp, bce_target, reduction='mean')
    bce_none = F.binary_cross_entropy_with_logits(bce_inp, bce_target, reduction='none')

    bce_inp.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "bce_with_logits_input.bin"))
    bce_target.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "bce_with_logits_target.bin"))
    torch.tensor([bce_sum.item()]).numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "bce_with_logits_sum_output.bin"))
    torch.tensor([bce_mean.item()]).numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "bce_with_logits_mean_output.bin"))
    bce_none.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "bce_with_logits_none_output.bin"))

    manifest["bce_with_logits"] = {
        "layer": "BCEWithLogitsLoss",
        "input_shape": list(bce_inp.shape),
        "target_shape": list(bce_target.shape),
        "sum_output_shape": [1],
        "mean_output_shape": [1],
        "none_output_shape": list(bce_none.shape),
    }
    print(f"  bce_with_logits: input={bce_inp.shape} sum={bce_sum.item():.6f} mean={bce_mean.item():.6f}")

    # =========================================================================
    # CrossEntropyLoss tests
    # =========================================================================
    ce_inp = torch.randn(4, 10, generator=rng)
    ce_target = torch.tensor([0, 3, 7, 2])

    ce_out = F.cross_entropy(ce_inp, ce_target)
    ce_none = F.cross_entropy(ce_inp, ce_target, reduction='none')

    ce_inp.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "cross_entropy_input.bin"))
    ce_target.numpy().astype(np.int64).tofile(os.path.join(TEST_DIR, "cross_entropy_target.bin"))
    torch.tensor([ce_out.item()]).numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "cross_entropy_output.bin"))
    ce_none.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "cross_entropy_none_output.bin"))

    manifest["cross_entropy"] = {
        "layer": "CrossEntropyLoss",
        "input_shape": list(ce_inp.shape),
        "target_shape": list(ce_target.shape),
        "output_shape": [1],
        "none_output_shape": list(ce_none.shape),
    }
    print(f"  cross_entropy: input={ce_inp.shape} target={ce_target.shape} loss={ce_out.item():.6f}")

    # =========================================================================
    # MSELoss tests
    # =========================================================================
    mse_pred = torch.randn(4, 10, generator=rng)
    mse_target = torch.randn(4, 10, generator=rng)

    mse_sum = F.mse_loss(mse_pred, mse_target, reduction='sum')
    mse_mean = F.mse_loss(mse_pred, mse_target, reduction='mean')
    mse_none = F.mse_loss(mse_pred, mse_target, reduction='none')

    mse_pred.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "mse_loss_pred.bin"))
    mse_target.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "mse_loss_target.bin"))
    torch.tensor([mse_sum.item()]).numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "mse_loss_sum_output.bin"))
    torch.tensor([mse_mean.item()]).numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "mse_loss_mean_output.bin"))
    mse_none.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "mse_loss_none_output.bin"))

    manifest["mse_loss"] = {
        "layer": "MSELoss",
        "pred_shape": list(mse_pred.shape),
        "target_shape": list(mse_target.shape),
        "sum_output_shape": [1],
        "mean_output_shape": [1],
        "none_output_shape": list(mse_none.shape),
    }
    print(f"  mse_loss: pred={mse_pred.shape} sum={mse_sum.item():.6f} mean={mse_mean.item():.6f}")

    # =========================================================================
    # L1Loss tests
    # =========================================================================
    l1_pred = torch.randn(4, 10, generator=rng)
    l1_target = torch.randn(4, 10, generator=rng)

    l1_out = F.l1_loss(l1_pred, l1_target, reduction='sum')
    l1_none = F.l1_loss(l1_pred, l1_target, reduction='none')

    l1_pred.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "l1_loss_pred.bin"))
    l1_target.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "l1_loss_target.bin"))
    torch.tensor([l1_out.item()]).numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "l1_loss_output.bin"))
    l1_none.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "l1_loss_none_output.bin"))

    manifest["l1_loss"] = {
        "layer": "L1Loss",
        "pred_shape": list(l1_pred.shape),
        "target_shape": list(l1_target.shape),
        "output_shape": [1],
        "none_output_shape": list(l1_none.shape),
    }
    print(f"  l1_loss: pred={l1_pred.shape} sum={l1_out.item():.6f}")

    # =========================================================================
    # AddBias tests (row-broadcast bias addition, linear bias op)
    # Uses a dedicated RNG so the main stream (and every other fixture) is
    # bit-stable. Computes a + b where b is broadcast across rows.
    # =========================================================================
    ops_rng = torch.Generator().manual_seed(202)

    add_bias_a = torch.randn(4, 16, generator=ops_rng)
    add_bias_b = torch.randn(16, generator=ops_rng)
    add_bias_out = add_bias_a + add_bias_b

    add_bias_a.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "add_bias_a.bin"))
    add_bias_b.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "add_bias_b.bin"))
    add_bias_out.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "add_bias_output.bin"))

    manifest["add_bias"] = {
        "layer": "AddBias",
        "a_shape": list(add_bias_a.shape),
        "bias_shape": list(add_bias_b.shape),
        "output_shape": list(add_bias_out.shape),
    }
    print(f"  add_bias: a={add_bias_a.shape} bias={add_bias_b.shape} output={add_bias_out.shape}")

    # =========================================================================
    # MatMulTransposedB tests (inference a @ b^T, linear weight layout)
    # b is saved in [N, K] row-major — the raw nn.Linear weight layout the
    # kernel consumes without a transpose. Same dedicated RNG.
    # =========================================================================
    mmtb_a = torch.randn(4, 8, generator=ops_rng)
    mmtb_b = torch.randn(16, 8, generator=ops_rng)
    mmtb_out = torch.matmul(mmtb_a, mmtb_b.t())

    mmtb_a.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "matmul_transposed_b_a.bin"))
    mmtb_b.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "matmul_transposed_b_b.bin"))
    mmtb_out.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "matmul_transposed_b_output.bin"))

    manifest["matmul_transposed_b"] = {
        "layer": "MatMulTransposedB",
        "a_shape": list(mmtb_a.shape),
        "b_shape": list(mmtb_b.shape),
        "output_shape": list(mmtb_out.shape),
    }
    print(f"  matmul_transposed_b: a={mmtb_a.shape} b={mmtb_b.shape} output={mmtb_out.shape}")

    # =========================================================================
    # Pow tests (scalar-exponent element-wise pow, reverse-mode Pow op)
    # Dedicated RNG keeps the main and ops_rng streams bit-stable. Exponent 2.0
    # over randn input avoids the NaN edge cases of fractional exponents.
    # Saves forward output + input gradient (of the sum) for backward parity.
    # =========================================================================
    pow_rng = torch.Generator().manual_seed(404)

    pow_inp = torch.randn(8, generator=pow_rng)
    pow_out = pow_inp.pow(2.0)
    pow_inp_grad = pow_inp.detach().requires_grad_(True)
    pow_inp_grad.pow(2.0).sum().backward()

    pow_inp.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "pow_input.bin"))
    pow_out.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "pow_output.bin"))
    pow_inp_grad.grad.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "pow_grad.bin"))

    manifest["pow"] = {
        "layer": "Pow",
        "input_shape": list(pow_inp.shape),
        "output_shape": list(pow_out.shape),
        "grad_shape": list(pow_inp_grad.grad.shape),
        "params": {"exponent": 2.0},
    }
    print(f"  pow: input={pow_inp.shape} output={pow_out.shape}")

    # =========================================================================
    # Fused multi-head attention tests (ReverseGradOperations.MultiHeadAttention)
    # Dedicated RNG keeps the main stream bit-stable. scale = 1/sqrt(headDim).
    # =========================================================================
    attn_rng = torch.Generator().manual_seed(303)

    def save_attn_case(name, q, k, v, scale, mask, dout, num_heads=4):
        q = q.detach().requires_grad_(True)
        k = k.detach().requires_grad_(True)
        v = v.detach().requires_grad_(True)
        d = q.shape[1]
        head_dim = d // num_heads
        heads = []
        for h in range(num_heads):
            qh = q[:, h * head_dim:(h + 1) * head_dim]
            kh = k[:, h * head_dim:(h + 1) * head_dim]
            vh = v[:, h * head_dim:(h + 1) * head_dim]
            scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
            if mask is not None:
                scores = scores + mask
            p = torch.softmax(scores, dim=-1)
            heads.append(torch.matmul(p, vh))
        out = torch.cat(heads, dim=-1)
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), grad_outputs=dout)

        q.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_q.bin"))
        k.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_k.bin"))
        v.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_v.bin"))
        out.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_output.bin"))
        if mask is not None:
            mask.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_mask.bin"))
        dout.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_dout.bin"))
        dq.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_dq.bin"))
        dk.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_dk.bin"))
        dv.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_dv.bin"))
        manifest[name] = {
            "layer": "MultiHeadAttention",
            "q_shape": list(q.shape),
            "k_shape": list(k.shape),
            "v_shape": list(v.shape),
            "num_heads": num_heads,
            "scale": float(scale),
            "masked": mask is not None,
            "output_shape": list(out.shape),
        }
        print(f"  {name}: q={list(q.shape)} k={list(k.shape)} v={list(v.shape)} output={list(out.shape)}")

    # Self-attention with a causal additive mask (qLen == kvLen == 4, D == 16).
    attn_q = torch.randn(4, 16, generator=attn_rng)
    attn_k = torch.randn(4, 16, generator=attn_rng)
    attn_v = torch.randn(4, 16, generator=attn_rng)
    attn_scale = 1.0 / math.sqrt(4)  # headDim = 16 / 4
    attn_mask = torch.triu(torch.full((4, 4), float("-inf")), diagonal=1)
    attn_dout = torch.randn(4, 16, generator=attn_rng)
    save_attn_case("attn_self_causal", attn_q, attn_k, attn_v, attn_scale, attn_mask, attn_dout)

    # Self-attention without a mask.
    save_attn_case("attn_self", attn_q, attn_k, attn_v, attn_scale, None, attn_dout)

    # Cross-attention (qLen != kvLen), last key/value is padding.
    attn_cq = torch.randn(3, 8, generator=attn_rng)
    attn_ck = torch.randn(5, 8, generator=attn_rng)
    attn_cv = torch.randn(5, 8, generator=attn_rng)
    attn_cscale = 1.0 / math.sqrt(4)  # headDim = 8 / 2
    attn_cmask = torch.zeros(3, 5)
    attn_cmask[:, 4] = float("-inf")
    attn_cdout = torch.randn(3, 8, generator=attn_rng)
    save_attn_case("attn_cross", attn_cq, attn_ck, attn_cv, attn_cscale, attn_cmask, attn_cdout, num_heads=2)

    # =========================================================================
    # Batched fused multi-head attention tests
    # (ReverseGradOperations.BatchedMultiHeadAttention, inputs are [B, L, D])
    # Same semantics as the single-sequence cases above but with a leading batch
    # dimension and a per-batch-element [B, qLen, kvLen] additive mask.
    # Appended at the END of the generation stream so the shared attn_rng stream
    # is untouched and every existing fixture stays bit-identical.
    # =========================================================================
    def save_batched_attn_case(name, q, k, v, scale, mask, dout, num_heads=4):
        q = q.detach().requires_grad_(True)
        k = k.detach().requires_grad_(True)
        v = v.detach().requires_grad_(True)
        d = q.shape[2]
        head_dim = d // num_heads
        heads = []
        for h in range(num_heads):
            qh = q[:, :, h * head_dim:(h + 1) * head_dim]
            kh = k[:, :, h * head_dim:(h + 1) * head_dim]
            vh = v[:, :, h * head_dim:(h + 1) * head_dim]
            scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
            if mask is not None:
                scores = scores + mask
            p = torch.softmax(scores, dim=-1)
            heads.append(torch.matmul(p, vh))
        out = torch.cat(heads, dim=-1)
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), grad_outputs=dout)

        q.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_q.bin"))
        k.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_k.bin"))
        v.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_v.bin"))
        out.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_output.bin"))
        if mask is not None:
            mask.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_mask.bin"))
        dout.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_dout.bin"))
        dq.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_dq.bin"))
        dk.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_dk.bin"))
        dv.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_dv.bin"))
        manifest[name] = {
            "layer": "BatchedMultiHeadAttention",
            "q_shape": list(q.shape),
            "k_shape": list(k.shape),
            "v_shape": list(v.shape),
            "num_heads": num_heads,
            "scale": float(scale),
            "masked": mask is not None,
            "output_shape": list(out.shape),
        }
        print(f"  {name}: q={list(q.shape)} k={list(k.shape)} v={list(v.shape)} output={list(out.shape)}")

    # Batched self-attention with a per-batch causal mask (B=2, L=4, D=16, H=4).
    bat_q = torch.randn(2, 4, 16, generator=attn_rng)
    bat_k = torch.randn(2, 4, 16, generator=attn_rng)
    bat_v = torch.randn(2, 4, 16, generator=attn_rng)
    bat_scale = 1.0 / math.sqrt(4)  # headDim = 16 / 4
    bat_mask = torch.triu(torch.full((1, 4, 4), float("-inf")), diagonal=1).repeat(2, 1, 1)
    bat_dout = torch.randn(2, 4, 16, generator=attn_rng)
    save_batched_attn_case("batched_attn_causal", bat_q, bat_k, bat_v, bat_scale, bat_mask, bat_dout)

    # Batched cross-attention (B=2, qLen=3, kvLen=5, D=8, H=2), last key padded.
    bat_cq = torch.randn(2, 3, 8, generator=attn_rng)
    bat_ck = torch.randn(2, 5, 8, generator=attn_rng)
    bat_cv = torch.randn(2, 5, 8, generator=attn_rng)
    bat_cscale = 1.0 / math.sqrt(4)  # headDim = 8 / 2
    bat_cmask = torch.zeros(2, 3, 5)
    bat_cmask[:, :, 4] = float("-inf")
    bat_cdout = torch.randn(2, 3, 8, generator=attn_rng)
    save_batched_attn_case("batched_attn_cross", bat_cq, bat_ck, bat_cv, bat_cscale, bat_cmask, bat_cdout, num_heads=2)

    # =========================================================================
    # SiLU tests (Activation.Silu / ReverseGradOperations.Silu)
    # Element-wise silu(x) = x * sigmoid(x). Dedicated RNG keeps the main and
    # all earlier per-case streams bit-identical. Saves forward output + the
    # input gradient of the sum (sum-backward parity).
    # =========================================================================
    silu_rng = torch.Generator().manual_seed(505)

    for name, inp_shape in [("silu_1d", (32,)), ("silu_4d", (1, 8, 4, 4))]:
        inp = torch.randn(inp_shape, generator=silu_rng).requires_grad_(True)
        out = F.silu(inp)
        out.sum().backward()

        inp_np = inp.detach().numpy().astype(np.float32)
        out_np = out.detach().numpy().astype(np.float32)
        grad_np = inp.grad.detach().numpy().astype(np.float32)

        inp_np.tofile(os.path.join(TEST_DIR, f"{name}_input.bin"))
        out_np.tofile(os.path.join(TEST_DIR, f"{name}_output.bin"))
        grad_np.tofile(os.path.join(TEST_DIR, f"{name}_grad.bin"))

        manifest[name] = {
            "layer": "SiLU",
            "input_shape": list(inp_shape),
            "output_shape": list(out_np.shape),
            "grad_shape": list(grad_np.shape),
        }
        print(f"  {name}: input={inp_shape} output={out_np.shape}")

    # =========================================================================
    # RotaryEmbedding (RoPE) tests — Llama-family half-split rotate_half layout.
    # No parameters. Saves forward output + input gradient of the sum. Uses its
    # own dedicated RNG and pure-tensor math (no nn.* modules), so the global
    # torch RNG stream is untouched.
    # =========================================================================
    rope_rng = torch.Generator().manual_seed(606)

    def build_rope_cache(head_dim, seq_len, theta):
        half = head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)  # [L, half]
        return torch.cos(freqs), torch.sin(freqs)

    def apply_rope(x, cos, sin):
        # x: [L, width], cos/sin: [L, headDim/2]; rotate every contiguous headDim
        # block using the half-split rotate_half layout:
        #   out[i]      = x[i] * c[i] - x[half+i] * s[i]
        #   out[half+i] = x[i] * s[i] + x[half+i] * c[i]
        L, width = x.shape
        half = cos.shape[-1]
        head_dim = 2 * half
        xr = x.reshape(L, -1, head_dim)  # [L, blocks, headDim]
        x1 = xr[..., :half]
        x2 = xr[..., half:]
        c = cos.reshape(L, 1, half)
        s = sin.reshape(L, 1, half)
        out = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
        return out.reshape(L, width)

    for name, head_dim, seq_len, width, theta in [
        ("rope_1head", 8, 8, 8, 10000.0),
        ("rope_2head", 8, 8, 16, 10000.0),
    ]:
        inp = torch.randn(seq_len, width, generator=rope_rng).requires_grad_(True)
        cos, sin = build_rope_cache(head_dim, seq_len, theta)
        out = apply_rope(inp, cos, sin)
        out.sum().backward()

        inp_np = inp.detach().numpy().astype(np.float32)
        out_np = out.detach().numpy().astype(np.float32)
        grad_np = inp.grad.detach().numpy().astype(np.float32)

        inp_np.tofile(os.path.join(TEST_DIR, f"{name}_input.bin"))
        out_np.tofile(os.path.join(TEST_DIR, f"{name}_output.bin"))
        grad_np.tofile(os.path.join(TEST_DIR, f"{name}_grad.bin"))

        manifest[name] = {
            "layer": "RotaryEmbedding",
            "input_shape": [seq_len, width],
            "output_shape": list(out_np.shape),
            "grad_shape": list(grad_np.shape),
            "params": {"head_dim": head_dim, "max_position_embeddings": seq_len, "rope_theta": theta},
        }
        print(f"  {name}: input=[{seq_len},{width}] head_dim={head_dim} output={out_np.shape}")

    # =========================================================================
    # RMSNorm module tests (elementwise affine gamma)
    # PyTorch nn.RMSNorm initializes gamma to ones (no RNG draw), so the global
    # stream is untouched. Saves input, gamma, output, input grad AND gamma grad
    # (all from a single sum-backward).
    # =========================================================================
    rms_rng = torch.Generator().manual_seed(707)

    rn = nn.RMSNorm(32, eps=1e-5)
    rn.weight.data.copy_(torch.randn(32, generator=rms_rng) * 0.1 + 1.0)
    rms_inp = torch.randn(4, 32, generator=rms_rng).requires_grad_(True)
    rms_out = rn(rms_inp)
    rms_out.sum().backward()

    rms_name = "rmsnorm_module_2d"
    rms_inp.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{rms_name}_input.bin"))
    rn.weight.data.cpu().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{rms_name}_gamma.bin"))
    rms_out.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{rms_name}_output.bin"))
    rms_inp.grad.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{rms_name}_input_grad.bin"))
    rn.weight.grad.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{rms_name}_gamma_grad.bin"))

    manifest[rms_name] = {
        "layer": "RMSNormModule",
        "input_shape": list(rms_inp.shape),
        "gamma_shape": list(rn.weight.data.shape),
        "output_shape": list(rms_out.shape),
        "input_grad_shape": list(rms_inp.grad.shape),
        "gamma_grad_shape": list(rn.weight.grad.shape),
        "params": {"normalized_shape": 32, "eps": 1e-5},
    }
    print(f"  {rms_name}: input={list(rms_inp.shape)} gamma={list(rn.weight.data.shape)} output={list(rms_out.shape)}")

    # =========================================================================
    # LlamaCausalAttention tests (GQA + RoPE + causal mask, fused MHA)
    # Reference: bias-less Linear projections -> RoPE -> consecutive KV-head
    # repeat (repeat_interleave per head) -> per-head scaled dot product with an
    # additive causal mask -> concat heads -> output projection. Saves input, the
    # four Linear weights ([out, in] row-major, matching the raw nn.Linear
    # layout), output, and the input gradient of the sum.
    # =========================================================================
    llama_rng = torch.Generator().manual_seed(808)

    def gqa_mha(q, k, v, num_heads, num_kv_heads, mask, scale):
        L, width = q.shape
        head_dim = width // num_heads
        repeat = num_heads // num_kv_heads
        kh = k.reshape(L, num_kv_heads, head_dim).repeat_interleave(repeat, dim=1).reshape(L, width)
        vh = v.reshape(L, num_kv_heads, head_dim).repeat_interleave(repeat, dim=1).reshape(L, width)
        heads = []
        for h in range(num_heads):
            qq = q[:, h * head_dim:(h + 1) * head_dim]
            kk = kh[:, h * head_dim:(h + 1) * head_dim]
            vv = vh[:, h * head_dim:(h + 1) * head_dim]
            scores = torch.matmul(qq, kk.transpose(-2, -1)) * scale
            if mask is not None:
                scores = scores + mask
            p = torch.softmax(scores, dim=-1)
            heads.append(torch.matmul(p, vv))
        return torch.cat(heads, dim=-1)

    attn_hidden = 64
    attn_heads = 4
    attn_kv_heads = 2
    attn_head_dim = 16
    attn_seq = 5

    wq = torch.randn(attn_heads * attn_head_dim, attn_hidden, generator=llama_rng)
    wk = torch.randn(attn_kv_heads * attn_head_dim, attn_hidden, generator=llama_rng)
    wv = torch.randn(attn_kv_heads * attn_head_dim, attn_hidden, generator=llama_rng)
    wo = torch.randn(attn_hidden, attn_hidden, generator=llama_rng)
    attn_inp = torch.randn(attn_seq, attn_hidden, generator=llama_rng).requires_grad_(True)

    q = attn_inp @ wq.t()
    k = attn_inp @ wk.t()
    v = attn_inp @ wv.t()
    cos, sin = build_rope_cache(attn_head_dim, attn_seq, 10000.0)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    attn_mask = torch.triu(torch.full((attn_seq, attn_seq), float("-inf")), diagonal=1)
    attn_scale = 1.0 / math.sqrt(attn_head_dim)
    attn_out = gqa_mha(q, k, v, attn_heads, attn_kv_heads, attn_mask, attn_scale) @ wo.t()
    attn_out.sum().backward()

    attn_name = "llama_attn"
    attn_inp.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{attn_name}_input.bin"))
    wq.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{attn_name}_qw.bin"))
    wk.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{attn_name}_kw.bin"))
    wv.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{attn_name}_vw.bin"))
    wo.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{attn_name}_ow.bin"))
    attn_out.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{attn_name}_output.bin"))
    attn_inp.grad.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{attn_name}_input_grad.bin"))

    manifest[attn_name] = {
        "layer": "LlamaCausalAttention",
        "input_shape": [attn_seq, attn_hidden],
        "q_weight_shape": list(wq.shape),
        "k_weight_shape": list(wk.shape),
        "v_weight_shape": list(wv.shape),
        "o_weight_shape": list(wo.shape),
        "output_shape": list(attn_out.shape),
        "input_grad_shape": list(attn_inp.grad.shape),
        "params": {"hidden_size": attn_hidden, "num_heads": attn_heads,
                   "num_key_value_heads": attn_kv_heads, "max_position_embeddings": 16,
                   "rope_theta": 10000.0},
    }
    print(f"  {attn_name}: input=[{attn_seq},{attn_hidden}] q={list(wq.shape)} k={list(wk.shape)} output={list(attn_out.shape)}")

    # =========================================================================
    # LlamaDecoderBlock tests (pre-norm GQA attention + gated SiLU FFN, residuals)
    # Reference: RMSNorm(affine) -> Llama attention -> residual; then RMSNorm
    # (affine) -> silu(gate(h)) * up(h) -> down -> residual. Saves input, every
    # learnable weight ([out, in] row-major), output, and the input gradient.
    # =========================================================================
    dec_rng = torch.Generator().manual_seed(909)

    dec_hidden = 32
    dec_heads = 4
    dec_kv_heads = 2
    dec_head_dim = 8
    dec_seq = 4
    dec_inter = 48
    dec_eps = 1e-5

    def rms_norm_affine(x, gamma, eps):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return x / rms * gamma

    dec_in_gamma = torch.randn(dec_hidden, generator=dec_rng) * 0.1 + 1.0
    dec_post_gamma = torch.randn(dec_hidden, generator=dec_rng) * 0.1 + 1.0
    dec_wq = torch.randn(dec_heads * dec_head_dim, dec_hidden, generator=dec_rng)
    dec_wk = torch.randn(dec_kv_heads * dec_head_dim, dec_hidden, generator=dec_rng)
    dec_wv = torch.randn(dec_kv_heads * dec_head_dim, dec_hidden, generator=dec_rng)
    dec_wo = torch.randn(dec_hidden, dec_hidden, generator=dec_rng)
    dec_gate = torch.randn(dec_inter, dec_hidden, generator=dec_rng)
    dec_up = torch.randn(dec_inter, dec_hidden, generator=dec_rng)
    dec_down = torch.randn(dec_hidden, dec_inter, generator=dec_rng)
    dec_inp = torch.randn(dec_seq, dec_hidden, generator=dec_rng).requires_grad_(True)

    h = rms_norm_affine(dec_inp, dec_in_gamma, dec_eps)
    qq = h @ dec_wq.t()
    kk = h @ dec_wk.t()
    vv = h @ dec_wv.t()
    cos, sin = build_rope_cache(dec_head_dim, dec_seq, 10000.0)
    qq = apply_rope(qq, cos, sin)
    kk = apply_rope(kk, cos, sin)
    dec_mask = torch.triu(torch.full((dec_seq, dec_seq), float("-inf")), diagonal=1)
    attn_h = gqa_mha(qq, kk, vv, dec_heads, dec_kv_heads, dec_mask, 1.0 / math.sqrt(dec_head_dim)) @ dec_wo.t()
    h = dec_inp + attn_h

    ffn_in = rms_norm_affine(h, dec_post_gamma, dec_eps)
    gate_h = F.silu(ffn_in @ dec_gate.t())
    up_h = ffn_in @ dec_up.t()
    mlp_h = (gate_h * up_h) @ dec_down.t()
    dec_out = h + mlp_h
    dec_out.sum().backward()

    dec_name = "llama_decoder"
    dec_inp.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dec_name}_input.bin"))
    dec_in_gamma.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dec_name}_in_gamma.bin"))
    dec_post_gamma.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dec_name}_post_gamma.bin"))
    dec_wq.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dec_name}_qw.bin"))
    dec_wk.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dec_name}_kw.bin"))
    dec_wv.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dec_name}_vw.bin"))
    dec_wo.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dec_name}_ow.bin"))
    dec_gate.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dec_name}_gatew.bin"))
    dec_up.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dec_name}_upw.bin"))
    dec_down.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dec_name}_downw.bin"))
    dec_out.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dec_name}_output.bin"))
    dec_inp.grad.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dec_name}_input_grad.bin"))

    manifest[dec_name] = {
        "layer": "LlamaDecoderBlock",
        "input_shape": [dec_seq, dec_hidden],
        "in_gamma_shape": list(dec_in_gamma.shape),
        "post_gamma_shape": list(dec_post_gamma.shape),
        "q_weight_shape": list(dec_wq.shape),
        "k_weight_shape": list(dec_wk.shape),
        "v_weight_shape": list(dec_wv.shape),
        "o_weight_shape": list(dec_wo.shape),
        "gate_weight_shape": list(dec_gate.shape),
        "up_weight_shape": list(dec_up.shape),
        "down_weight_shape": list(dec_down.shape),
        "output_shape": list(dec_out.shape),
        "input_grad_shape": list(dec_inp.grad.shape),
        "params": {"hidden_size": dec_hidden, "num_heads": dec_heads,
                   "num_key_value_heads": dec_kv_heads, "intermediate_size": dec_inter,
                   "max_position_embeddings": 16, "rope_theta": 10000.0, "rms_norm_eps": dec_eps},
    }
    print(f"  {dec_name}: input=[{dec_seq},{dec_hidden}] gate={list(dec_gate.shape)} output={list(dec_out.shape)}")

    # =========================================================================
    # DepthwiseSeparableConv2d tests
    # Reference: depthwise Conv2d (groups=inCh) -> ReLU -> 1x1 pointwise Conv2d.
    # Uses F.conv2d with explicit weight tensors (no nn.Conv2d module) so the
    # global torch RNG stream is untouched. Saves input, depthwise and pointwise
    # weights, pointwise bias, output, and the input gradient of the sum.
    # =========================================================================
    dsc_rng = torch.Generator().manual_seed(1010)

    dsc_in_c = 4
    dsc_out_c = 8
    dsc_k = 3
    dsc_inp = torch.randn(1, dsc_in_c, 8, 8, generator=dsc_rng).requires_grad_(True)

    dw_weight = torch.randn(dsc_in_c, 1, dsc_k, dsc_k, generator=dsc_rng)
    pw_weight = torch.randn(dsc_out_c, dsc_in_c, 1, 1, generator=dsc_rng)
    pw_bias = torch.randn(dsc_out_c, generator=dsc_rng)

    dsc_h = torch.relu(F.conv2d(dsc_inp, dw_weight, stride=1, padding=1, groups=dsc_in_c))
    dsc_out = F.conv2d(dsc_h, pw_weight, pw_bias, stride=1, padding=0)
    dsc_out.sum().backward()

    dsc_name = "dsc"
    dsc_inp.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dsc_name}_input.bin"))
    dw_weight.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dsc_name}_dw_weight.bin"))
    pw_weight.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dsc_name}_pw_weight.bin"))
    pw_bias.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dsc_name}_pw_bias.bin"))
    dsc_out.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dsc_name}_output.bin"))
    dsc_inp.grad.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dsc_name}_input_grad.bin"))

    manifest[dsc_name] = {
        "layer": "DepthwiseSeparableConv2d",
        "input_shape": list(dsc_inp.shape),
        "dw_weight_shape": list(dw_weight.shape),
        "pw_weight_shape": list(pw_weight.shape),
        "pw_bias_shape": list(pw_bias.shape),
        "output_shape": list(dsc_out.shape),
        "input_grad_shape": list(dsc_inp.grad.shape),
        "params": {"in_channels": dsc_in_c, "out_channels": dsc_out_c,
                   "kernel_size": dsc_k, "stride": 1, "padding": 1},
    }
    print(f"  {dsc_name}: input={list(dsc_inp.shape)} dw={list(dw_weight.shape)} output={list(dsc_out.shape)}")

    # =========================================================================
    # TransformerBlock tests (pre-norm GELU MLP, causal MHA; RMSNorm + LayerNorm)
    # Reference: pre-norm (no-affine RMSNorm or LayerNorm) -> Q/K/V project ->
    # per-head causal scaled-dot-product -> O project + residual -> pre-norm ->
    # GELU(tanh) MLP -> residual. Saves input, all six Linear weights ([out, in]
    # row-major), output, and the input gradient of the sum.
    # =========================================================================
    tb_rng = torch.Generator().manual_seed(1111)
    tb_ln_rng = torch.Generator().manual_seed(1212)

    def rms_norm_no_affine(x, eps=1e-5):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    def transformer_block_forward(x, qw, kw, vw, ow, f1w, f2w, n_embd, n_head, norm_fn, eps=1e-5):
        L, D = x.shape
        head_dim = D // n_head
        scale = 1.0 / math.sqrt(head_dim)
        mask = torch.triu(torch.full((L, L), float("-inf")), diagonal=1)
        h = norm_fn(x, eps)
        q = h @ qw.t()
        k = h @ kw.t()
        v = h @ vw.t()
        attn = gqa_mha(q, k, v, n_head, n_head, mask, scale)  # repeat=1 when kv==heads
        x = x + attn @ ow.t()
        h = norm_fn(x, eps)
        mlp_h = F.gelu(h @ f1w.t(), approximate="tanh") @ f2w.t()
        return x + mlp_h

    tb_embd = 32
    tb_heads = 4
    tb_seq = 6

    def tb_layer_norm_no_affine(x, eps):
        return F.layer_norm(x, (tb_embd,), None, None, eps)

    for tb_name, tb_rng_used, tb_norm_fn in [
        ("transformer_block_rms", tb_rng, rms_norm_no_affine),
        ("transformer_block_ln", tb_ln_rng, tb_layer_norm_no_affine),
    ]:
        tb_qw = torch.randn(tb_embd, tb_embd, generator=tb_rng_used)
        tb_kw = torch.randn(tb_embd, tb_embd, generator=tb_rng_used)
        tb_vw = torch.randn(tb_embd, tb_embd, generator=tb_rng_used)
        tb_ow = torch.randn(tb_embd, tb_embd, generator=tb_rng_used)
        tb_f1w = torch.randn(4 * tb_embd, tb_embd, generator=tb_rng_used)
        tb_f2w = torch.randn(tb_embd, 4 * tb_embd, generator=tb_rng_used)
        tb_inp = torch.randn(tb_seq, tb_embd, generator=tb_rng_used).requires_grad_(True)

        tb_out = transformer_block_forward(
            tb_inp, tb_qw, tb_kw, tb_vw, tb_ow, tb_f1w, tb_f2w,
            tb_embd, tb_heads, tb_norm_fn)
        tb_out.sum().backward()

        tb_inp.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{tb_name}_input.bin"))
        tb_qw.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{tb_name}_qw.bin"))
        tb_kw.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{tb_name}_kw.bin"))
        tb_vw.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{tb_name}_vw.bin"))
        tb_ow.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{tb_name}_ow.bin"))
        tb_f1w.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{tb_name}_f1w.bin"))
        tb_f2w.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{tb_name}_f2w.bin"))
        tb_out.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{tb_name}_output.bin"))
        tb_inp.grad.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{tb_name}_input_grad.bin"))

        manifest[tb_name] = {
            "layer": "TransformerBlock",
            "input_shape": [tb_seq, tb_embd],
            "q_weight_shape": list(tb_qw.shape),
            "k_weight_shape": list(tb_kw.shape),
            "v_weight_shape": list(tb_vw.shape),
            "o_weight_shape": list(tb_ow.shape),
            "f1_weight_shape": list(tb_f1w.shape),
            "f2_weight_shape": list(tb_f2w.shape),
            "output_shape": list(tb_out.shape),
            "input_grad_shape": list(tb_inp.grad.shape),
            "params": {"n_embd": tb_embd, "n_head": tb_heads, "dropout": 0.0, "eps": 1e-5},
        }
        print(f"  {tb_name}: input=[{tb_seq},{tb_embd}] f1={list(tb_f1w.shape)} output={list(tb_out.shape)}")

    # =========================================================================
    # SparseEmbedding tests (sum-mode embedding bag with padding indices skipped)
    # Reference: per-batch sum of selected embedding rows; entries equal to the
    # padding index are ignored. Saves weight, float-cast index input, output,
    # and the weight gradient of the sum.
    # =========================================================================
    sse_rng = torch.Generator().manual_seed(1313)

    sse_num_emb = 20
    sse_emb_dim = 8
    sse_batch = 4
    sse_max_active = 5
    sse_padding = -1

    sse_weight = torch.randn(sse_num_emb, sse_emb_dim, generator=sse_rng).requires_grad_(True)
    sse_idx = torch.randint(0, 12, (sse_batch, sse_max_active), generator=sse_rng).long()
    sse_idx[0, 1] = sse_padding
    sse_idx[2, 3] = sse_padding

    sse_out = torch.zeros(sse_batch, sse_emb_dim)
    for b in range(sse_batch):
        active = sse_idx[b][sse_idx[b] != sse_padding]
        sse_out[b] = sse_weight.index_select(0, active).sum(0)
    sse_out.sum().backward()

    sse_name = "sparse_embedding"
    sse_weight.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{sse_name}_weight.bin"))
    sse_idx.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{sse_name}_input.bin"))
    sse_out.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{sse_name}_output.bin"))
    sse_weight.grad.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{sse_name}_weight_grad.bin"))

    manifest[sse_name] = {
        "layer": "SparseEmbedding",
        "input_shape": [sse_batch, sse_max_active],
        "weight_shape": [sse_num_emb, sse_emb_dim],
        "output_shape": list(sse_out.shape),
        "weight_grad_shape": list(sse_weight.grad.shape),
        "params": {"num_embeddings": sse_num_emb, "embedding_dim": sse_emb_dim,
                   "padding_index": sse_padding},
    }
    print(f"  {sse_name}: input=[{sse_batch},{sse_max_active}] weight=[{sse_num_emb},{sse_emb_dim}] output={list(sse_out.shape)}")

    # =========================================================================
    # LlamaCausalAttention with QKV bias (qkvBias=true) tests — the Qwen2 variant
    # Reference: biased Linear projections (input @ W^T + b) -> RoPE -> GQA repeat ->
    # per-head scaled dot product with an additive causal mask -> concat -> output
    # projection. Appended at the END of the generation stream so all earlier
    # fixtures stay bit-identical. Saves input, the four Linear weights ([out, in]
    # row-major), the Q/K/V biases, output, and the input + bias gradients of the
    # sum.
    # =========================================================================
    attn_bias_rng = torch.Generator().manual_seed(1313)

    attn_b_hidden = 64
    attn_b_heads = 4
    attn_b_kv_heads = 2
    attn_b_head_dim = 16
    attn_b_seq = 5

    wq_b = torch.randn(attn_b_heads * attn_b_head_dim, attn_b_hidden, generator=attn_bias_rng)
    wk_b = torch.randn(attn_b_kv_heads * attn_b_head_dim, attn_b_hidden, generator=attn_bias_rng)
    wv_b = torch.randn(attn_b_kv_heads * attn_b_head_dim, attn_b_hidden, generator=attn_bias_rng)
    wo_b = torch.randn(attn_b_hidden, attn_b_hidden, generator=attn_bias_rng)
    bq_b = torch.randn(attn_b_heads * attn_b_head_dim, generator=attn_bias_rng).requires_grad_(True)
    bk_b = torch.randn(attn_b_kv_heads * attn_b_head_dim, generator=attn_bias_rng).requires_grad_(True)
    bv_b = torch.randn(attn_b_kv_heads * attn_b_head_dim, generator=attn_bias_rng).requires_grad_(True)
    attn_b_inp = torch.randn(attn_b_seq, attn_b_hidden, generator=attn_bias_rng).requires_grad_(True)

    q_b = attn_b_inp @ wq_b.t() + bq_b
    k_b = attn_b_inp @ wk_b.t() + bk_b
    v_b = attn_b_inp @ wv_b.t() + bv_b
    cos_b, sin_b = build_rope_cache(attn_b_head_dim, attn_b_seq, 10000.0)
    q_b = apply_rope(q_b, cos_b, sin_b)
    k_b = apply_rope(k_b, cos_b, sin_b)
    attn_b_mask = torch.triu(torch.full((attn_b_seq, attn_b_seq), float("-inf")), diagonal=1)
    attn_b_scale = 1.0 / math.sqrt(attn_b_head_dim)
    attn_b_out = gqa_mha(q_b, k_b, v_b, attn_b_heads, attn_b_kv_heads, attn_b_mask, attn_b_scale) @ wo_b.t()
    attn_b_out.sum().backward()

    attn_b_name = "llama_attn_bias"
    attn_b_inp.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{attn_b_name}_input.bin"))
    wq_b.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{attn_b_name}_qw.bin"))
    wk_b.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{attn_b_name}_kw.bin"))
    wv_b.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{attn_b_name}_vw.bin"))
    wo_b.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{attn_b_name}_ow.bin"))
    bq_b.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{attn_b_name}_qb.bin"))
    bk_b.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{attn_b_name}_kb.bin"))
    bv_b.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{attn_b_name}_vb.bin"))
    attn_b_out.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{attn_b_name}_output.bin"))
    attn_b_inp.grad.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{attn_b_name}_input_grad.bin"))
    bq_b.grad.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{attn_b_name}_q_bias_grad.bin"))
    bk_b.grad.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{attn_b_name}_k_bias_grad.bin"))
    bv_b.grad.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{attn_b_name}_v_bias_grad.bin"))

    manifest[attn_b_name] = {
        "layer": "LlamaCausalAttention",
        "qkv_bias": True,
        "input_shape": [attn_b_seq, attn_b_hidden],
        "q_weight_shape": list(wq_b.shape),
        "k_weight_shape": list(wk_b.shape),
        "v_weight_shape": list(wv_b.shape),
        "o_weight_shape": list(wo_b.shape),
        "q_bias_shape": list(bq_b.shape),
        "k_bias_shape": list(bk_b.shape),
        "v_bias_shape": list(bv_b.shape),
        "output_shape": list(attn_b_out.shape),
        "input_grad_shape": list(attn_b_inp.grad.shape),
        "params": {"hidden_size": attn_b_hidden, "num_heads": attn_b_heads,
                   "num_key_value_heads": attn_b_kv_heads, "max_position_embeddings": 16,
                   "rope_theta": 10000.0, "qkv_bias": True},
    }
    print(f"  {attn_b_name}: input=[{attn_b_seq},{attn_b_hidden}] q_bias={list(bq_b.shape)} output={list(attn_b_out.shape)}")

    # =========================================================================
    # LlamaDecoderBlock with QKV bias (qkvBias=true) tests — the Qwen2 variant
    # Reference: RMSNorm(affine) -> biased Llama attention -> residual; then
    # RMSNorm(affine) -> silu(gate(h)) * up(h) -> down -> residual. Appended at the
    # END of the generation stream with a dedicated RNG so all earlier fixtures stay
    # bit-identical. Saves input, every learnable weight, the Q/K/V biases, output,
    # and the input + bias gradients of the sum.
    # =========================================================================
    dec_bias_rng = torch.Generator().manual_seed(1414)

    dec_b_hidden = 32
    dec_b_heads = 4
    dec_b_kv_heads = 2
    dec_b_head_dim = 8
    dec_b_seq = 4
    dec_b_inter = 48
    dec_b_eps = 1e-5

    dec_b_in_gamma = torch.randn(dec_b_hidden, generator=dec_bias_rng) * 0.1 + 1.0
    dec_b_post_gamma = torch.randn(dec_b_hidden, generator=dec_bias_rng) * 0.1 + 1.0
    dec_b_wq = torch.randn(dec_b_heads * dec_b_head_dim, dec_b_hidden, generator=dec_bias_rng)
    dec_b_wk = torch.randn(dec_b_kv_heads * dec_b_head_dim, dec_b_hidden, generator=dec_bias_rng)
    dec_b_wv = torch.randn(dec_b_kv_heads * dec_b_head_dim, dec_b_hidden, generator=dec_bias_rng)
    dec_b_wo = torch.randn(dec_b_hidden, dec_b_hidden, generator=dec_bias_rng)
    dec_b_bq = torch.randn(dec_b_heads * dec_b_head_dim, generator=dec_bias_rng).requires_grad_(True)
    dec_b_bk = torch.randn(dec_b_kv_heads * dec_b_head_dim, generator=dec_bias_rng).requires_grad_(True)
    dec_b_bv = torch.randn(dec_b_kv_heads * dec_b_head_dim, generator=dec_bias_rng).requires_grad_(True)
    dec_b_gate = torch.randn(dec_b_inter, dec_b_hidden, generator=dec_bias_rng)
    dec_b_up = torch.randn(dec_b_inter, dec_b_hidden, generator=dec_bias_rng)
    dec_b_down = torch.randn(dec_b_hidden, dec_b_inter, generator=dec_bias_rng)
    dec_b_inp = torch.randn(dec_b_seq, dec_b_hidden, generator=dec_bias_rng).requires_grad_(True)

    h_b = rms_norm_affine(dec_b_inp, dec_b_in_gamma, dec_b_eps)
    qq_b = h_b @ dec_b_wq.t() + dec_b_bq
    kk_b = h_b @ dec_b_wk.t() + dec_b_bk
    vv_b = h_b @ dec_b_wv.t() + dec_b_bv
    cos_b, sin_b = build_rope_cache(dec_b_head_dim, dec_b_seq, 10000.0)
    qq_b = apply_rope(qq_b, cos_b, sin_b)
    kk_b = apply_rope(kk_b, cos_b, sin_b)
    dec_b_mask = torch.triu(torch.full((dec_b_seq, dec_b_seq), float("-inf")), diagonal=1)
    attn_b_h = gqa_mha(qq_b, kk_b, vv_b, dec_b_heads, dec_b_kv_heads, dec_b_mask, 1.0 / math.sqrt(dec_b_head_dim)) @ dec_b_wo.t()
    h_b = dec_b_inp + attn_b_h

    ffn_b_in = rms_norm_affine(h_b, dec_b_post_gamma, dec_b_eps)
    gate_b_h = F.silu(ffn_b_in @ dec_b_gate.t())
    up_b_h = ffn_b_in @ dec_b_up.t()
    mlp_b_h = (gate_b_h * up_b_h) @ dec_b_down.t()
    dec_b_out = h_b + mlp_b_h
    dec_b_out.sum().backward()

    dec_b_name = "llama_decoder_bias"
    dec_b_inp.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dec_b_name}_input.bin"))
    dec_b_in_gamma.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dec_b_name}_in_gamma.bin"))
    dec_b_post_gamma.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dec_b_name}_post_gamma.bin"))
    dec_b_wq.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dec_b_name}_qw.bin"))
    dec_b_wk.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dec_b_name}_kw.bin"))
    dec_b_wv.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dec_b_name}_vw.bin"))
    dec_b_wo.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dec_b_name}_ow.bin"))
    dec_b_bq.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dec_b_name}_qb.bin"))
    dec_b_bk.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dec_b_name}_kb.bin"))
    dec_b_bv.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dec_b_name}_vb.bin"))
    dec_b_gate.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dec_b_name}_gatew.bin"))
    dec_b_up.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dec_b_name}_upw.bin"))
    dec_b_down.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dec_b_name}_downw.bin"))
    dec_b_out.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dec_b_name}_output.bin"))
    dec_b_inp.grad.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dec_b_name}_input_grad.bin"))
    dec_b_bq.grad.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dec_b_name}_q_bias_grad.bin"))
    dec_b_bk.grad.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dec_b_name}_k_bias_grad.bin"))
    dec_b_bv.grad.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{dec_b_name}_v_bias_grad.bin"))

    manifest[dec_b_name] = {
        "layer": "LlamaDecoderBlock",
        "qkv_bias": True,
        "input_shape": [dec_b_seq, dec_b_hidden],
        "in_gamma_shape": list(dec_b_in_gamma.shape),
        "post_gamma_shape": list(dec_b_post_gamma.shape),
        "q_weight_shape": list(dec_b_wq.shape),
        "k_weight_shape": list(dec_b_wk.shape),
        "v_weight_shape": list(dec_b_wv.shape),
        "o_weight_shape": list(dec_b_wo.shape),
        "q_bias_shape": list(dec_b_bq.shape),
        "k_bias_shape": list(dec_b_bk.shape),
        "v_bias_shape": list(dec_b_bv.shape),
        "gate_weight_shape": list(dec_b_gate.shape),
        "up_weight_shape": list(dec_b_up.shape),
        "down_weight_shape": list(dec_b_down.shape),
        "output_shape": list(dec_b_out.shape),
        "input_grad_shape": list(dec_b_inp.grad.shape),
        "params": {"hidden_size": dec_b_hidden, "num_heads": dec_b_heads,
                   "num_key_value_heads": dec_b_kv_heads, "intermediate_size": dec_b_inter,
                   "max_position_embeddings": 16, "rope_theta": 10000.0, "rms_norm_eps": dec_b_eps,
                   "qkv_bias": True},
    }
    print(f"  {dec_b_name}: input=[{dec_b_seq},{dec_b_hidden}] q_bias={list(dec_b_bq.shape)} output={list(dec_b_out.shape)}")

    # =========================================================================
    # Write manifest
    # =========================================================================
    manifest_path = os.path.join(TEST_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")
    print(f"Total test cases: {len(manifest)}")


if __name__ == "__main__":
    run()
