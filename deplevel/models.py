"""Model factory (braindecode implementations) and complexity measures."""
import warnings

import torch
from torch.utils.flop_counter import FlopCounterMode

MODEL_NAMES = ("EEGNet", "ShallowFBCSPNet", "Deep4Net")


def build_model(name, n_chans=2, n_times=128, n_outputs=2):
    """All models return raw logits (no log-softmax) of shape (batch, n_outputs)."""
    from braindecode.models import Deep4Net, EEGNetv4, ShallowFBCSPNet

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name == "EEGNet":
            # Default EEGNet-8,2 hyper-parameters; kernel_length=64 is half a second at 128 Hz,
            # matching the original paper's sampling rate.
            return EEGNetv4(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times)
        if name == "ShallowFBCSPNet":
            return ShallowFBCSPNet(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times,
                                   final_conv_length="auto", add_log_softmax=False)
        if name == "Deep4Net":
            # The default Deep4Net (kernel 10, pool 3/3 in four blocks) needs > 128 samples.
            # Kernels and pools are scaled to the 1-s / 128-Hz window
            # (roughly 128/250 of the original 250-Hz design): 128 -> 62 -> 29 -> 12 -> 4.
            return Deep4Net(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times,
                            final_conv_length="auto", add_log_softmax=False,
                            filter_time_length=5, filter_length_2=5, filter_length_3=5,
                            filter_length_4=5, pool_time_length=2, pool_time_stride=2)
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def complexity(model, n_chans=2, n_times=128):
    """Parameters, model size and FLOPs for one input window.

    FLOPs are counted by torch's FlopCounterMode (convolutions and matmuls,
    one multiply-add = 2 FLOPs); element-wise ops such as BN, activations and
    pooling are not included, as in most EEG papers.
    """
    model = model.eval().cpu()
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_bytes = sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers()))
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(torch.zeros(1, n_chans, n_times))
    flops = fc.get_total_flops()
    return {
        "params": n_params,
        "trainable_params": n_trainable,
        "size_kb": size_bytes / 1024,
        "flops": flops,
        "mflops": flops / 1e6,
    }
