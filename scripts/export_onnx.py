# (Author: Hu Liang)

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

import kaldiio
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from torch import nn


def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()


class FireRedVad(nn.Module):
    @classmethod
    def from_pretrained(cls, model_path):
        checkpoint_path = os.path.join(model_path, "model.pth.tar")
        package = torch.load(
            checkpoint_path,
            map_location=lambda storage, loc: storage,
            weights_only=False,
        )
        cmvn_path = os.path.join(model_path, "cmvn.ark")
        model = cls(package["args"], cmvn_path)
        model.load_state_dict(package["model_state_dict"], strict=True)
        model.eval()
        return model

    def __init__(self, args, cmvn_path):
        super().__init__()
        self.cmvn = CMVN(cmvn_path)
        if self.cmvn.dim != args.idim:
            raise ValueError(
                f"CMVN dimension {self.cmvn.dim} does not match model input dimension {args.idim}"
            )
        self.dfsmn = DFSMN(
            args.idim,
            args.R,
            args.M,
            args.H,
            args.P,
            args.N1,
            args.S1,
            args.N2,
            args.S2,
            args.dropout,
        )
        self.out = torch.nn.Linear(args.H, args.odim)

    @torch.no_grad()
    def forward(
        self,
        feat: torch.Tensor,
        cache1: torch.Tensor,
        cache2: torch.Tensor,
        cache3: torch.Tensor,
        cache4: torch.Tensor,
        cache5: torch.Tensor,
        cache6: torch.Tensor,
        cache7: torch.Tensor,
        cache8: torch.Tensor,
    ):
        """Accept raw FP32 Fbank features [B, T, D], without external CMVN."""
        feat = self.cmvn(feat)
        outputs: list[torch.Tensor] = self.dfsmn(
            feat, cache1, cache2, cache3, cache4, cache5, cache6, cache7, cache8
        )
        logits = self.out(outputs[0])
        probs = torch.sigmoid(logits)
        probs = probs.squeeze(2)
        return probs, *outputs[1:]


class DFSMN(nn.Module):
    def __init__(self, D, R, M, H, P, N1, S1, N2=0, S2=0, dropout=0.1):
        """
        DFSMN config: Rx[H-P(N1,N2,S1,S2)]-MxH
        Args:
            D: input dimension
            R: number of DFSMN blocks
            M: number of DNN layers
            H: hidden size
            P: projection size
            N1: lookback order
            S1: lookback stride
            N2: lookahead order
            S2: lookahead stride
        """
        super().__init__()
        # Components
        # 1st FSMN block connecting input layer, without skip connection
        self.fc1 = nn.Sequential(
            nn.Linear(D, H, bias=True), nn.ReLU(), nn.Dropout(dropout)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(H, P, bias=True), nn.ReLU(), nn.Dropout(dropout)
        )
        self.fsmn1 = FSMN(P, N1, S1, N2, S2)
        # N-1 DFSMN blocks
        self.fsmns = nn.ModuleList(
            [DFSMNBlock(H, P, N1, S1, N2, S2, dropout) for _ in range(R - 1)]
        )
        # M DNN layers
        dnn = [nn.Linear(P, H, bias=True), nn.ReLU(), nn.Dropout(dropout)]
        for l in range(M - 1):
            dnn += [nn.Linear(H, H, bias=True), nn.ReLU(), nn.Dropout(dropout)]
        self.dnns = nn.Sequential(*dnn)

        print(
            f"DFSMN D={D}, R={R}, M={M}, H={H}, P={P}, N1={N1}, S1={S1}, N2={N2}, S2={S2}"
        )

    def forward(
        self,
        inputs,
        cache1: torch.Tensor,
        cache2: torch.Tensor,
        cache3: torch.Tensor,
        cache4: torch.Tensor,
        cache5: torch.Tensor,
        cache6: torch.Tensor,
        cache7: torch.Tensor,
        cache8: torch.Tensor,
        input_lengths=None,
    ):
        """
        Args:
            inputs: [N, T, D], padded, T is sequence length, D is input dim
            mask: processing padding issue, masked position is 1
                tensor.masked_fill(mask, value) will fill elements of tensor with value where mask is one.
        Returns:
            output: [N, T, P]
        """
        if input_lengths is None:
            mask = None
        else:
            mask = get_mask_from_lengths(input_lengths)

        caches = [cache1, cache2, cache3, cache4, cache5, cache6, cache7, cache8]

        # 1st FSMN
        h = self.fc1(inputs)
        p = self.fc2(h)
        new_caches = []
        memory, new_cache = self.fsmn1(p, mask=mask, cache=caches[0])
        new_caches.append(new_cache)

        # R-1 FSMN
        i = 1
        for fsmn in self.fsmns:
            memory, new_cache = fsmn(memory, mask=mask, cache=caches[i])
            new_caches.append(new_cache)
            i += 1
        # M DNN
        output = self.dnns(memory)
        return output, *new_caches


def get_mask_from_lengths(lengths):
    """Mask position is set to 1 for Tensor.masked_fill(mask, value)
    Args:
        lengths: (N, )
    Return:
        mask: (N, T)
    """
    N = lengths.size(0)
    T = torch.max(lengths).item()
    mask = torch.zeros(N, T).to(lengths.device)
    for i in range(N):
        mask[i, lengths[i] :] = 1
    return mask.to(torch.uint8)


class DFSMNBlock(nn.Module):
    def __init__(self, H, P, N1, S1, N2=0, S2=0, dropout=0.1):
        """
        DFSMNBlock = [input -> Affine+ReLU -> Affine -> vFSMN -> output]
                        |                                           ^
                        |-------------------------------------------|
                                 (skip connection)
        Args:
            H: hidden size
            P: projection size
            N1: lookback order
            S1: lookback stride
            N2: lookahead order
            S2: lookahead stride
        """
        super().__init__()
        # Hyper-parameter
        self.H, self.P, self.N1, self.S1, self.N2, self.S2 = H, P, N1, S1, N2, S2
        # Components
        # step1. \hat{P}^{l-1} -> H^{l}, nonlinear affine transform
        self.fc1 = nn.Sequential(
            nn.Linear(P, H, bias=True), nn.ReLU(), nn.Dropout(dropout)
        )
        # Step2. H^{l} -> P^{l}, linear affine transform
        self.fc2 = nn.Linear(H, P, bias=False)
        # Step3. P^{l}-> \hat{P}^{l}, fsmn layer
        self.fsmn = FSMN(P, N1, S1, N2, S2)

    def forward(self, inputs, mask=None, cache=None):
        """
        Args:
            inputs: [N, T, P], padded, T is sequence length, P is projection size
            mask: processing padding issue, masked position is 1
                tensor.masked_fill(mask, value) will fill elements of tensor with value where mask is one.
        Returns:
            output: [N, T, P]
        """
        residual = inputs
        # step1. \hat{P}^{l-1} -> H^{l}, nonlinear affine transform
        h = self.fc1(inputs)
        # Step2. H^{l} -> P^{l}, linear affine transform
        p = self.fc2(h)
        # Step3. P^{l}-> \hat{P}^{l}, fsmn layer
        memory, new_cache = self.fsmn(p, mask=mask, cache=cache)
        # Step4. skip connection
        output = memory + residual
        return output, new_cache


class FSMN(nn.Module):
    def __init__(self, P, N1, S1, N2=0, S2=0):
        """
        Args:
            P: projection size
            N1: lookback order
            S1: lookback stride
            N2: lookahead order
            S2: lookahead stride
        """
        super().__init__()
        # Hyper-parameter
        assert N1 >= 1
        self.N1, self.S1, self.N2, self.S2 = N1, S1, N2, S2
        # Components
        # P^{l}-> \hat{P}^{l}
        self.lookback_padding = (N1 - 1) * S1
        self.lookback_filter = nn.Conv1d(
            in_channels=P,
            out_channels=P,
            kernel_size=N1,
            stride=1,
            padding=self.lookback_padding,
            dilation=S1,
            groups=P,
            bias=False,
        )
        if self.N2 > 0:
            self.lookahead_filter = nn.Conv1d(
                in_channels=P,
                out_channels=P,
                kernel_size=N2,
                stride=1,
                padding=(N2 - 1) * S2,
                dilation=S2,
                groups=P,
                bias=False,
            )
        else:
            self.lookahead_filter = nn.Identity()

        print(f"FSMN: P={P}, N1={N1}, S1={S1}, N2={N2}, S2={S2}")

    def forward(self, inputs, mask=None, cache=None):
        """
        Args:
            inputs: [N, T, P], padded, T is sequence length, P is projection size
            mask: processing padding issue, masked position is 1
                tensor.masked_fill(mask, value) will fill elements of tensor with value where mask is one.
        Returns:
            memory: [N, T, P]
        """
        T = inputs.size(1)
        if mask is not None:
            mask = mask.unsqueeze(-1)  # [N, T, 1]
            inputs = inputs.masked_fill(mask, 0.0)

        inputs = inputs.permute((0, 2, 1)).contiguous()  # [N, T, P] -> [N, P, T]
        residual = inputs

        if cache is not None:
            inputs = torch.cat((cache, inputs), dim=2)  # (N, P, C+T)
        new_cache = inputs[:, :, -self.lookback_padding :]  # (N, P, Co)

        # P^{l}-> \hat{P}^{l}, fsmn layer
        lookback = self.lookback_filter(inputs)
        if self.N1 > 1:
            lookback = lookback[:, :, : -(self.N1 - 1) * self.S1]
            if cache is not None:
                start = cache.size(2)
                lookback = lookback[:, :, start:]
            memory = residual + lookback
        else:
            memory = residual + lookback

        if self.N2 > 0 and T > 1:
            lookahead = self.lookahead_filter(inputs)
            memory += F.pad(lookahead[:, :, self.N2 * self.S2 :], (0, self.S2))
        memory = memory.permute((0, 2, 1)).contiguous()  # [N, P, T] -> [N, T, P]

        if mask is not None:
            memory = memory.masked_fill(mask, 0.0)
        return memory, new_cache


class CMVN(nn.Module):
    def __init__(self, kaldi_cmvn_file):
        super().__init__()

        # 加载 CMVN 统计信息并注册为 buffer
        dim, means, inverse_std_variances = self._read_kaldi_cmvn(kaldi_cmvn_file)

        # 随模型迁移设备并导出为常量，不向原始 checkpoint 增加状态键。
        self.register_buffer(
            "means", torch.tensor(means, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "inverse_std_variances",
            torch.tensor(inverse_std_variances, dtype=torch.float32),
            persistent=False,
        )
        self.dim = dim

    def forward(self, x):
        """
        Args:
            x: 输入特征张量，形状为 [..., feature_dim]
        """
        # 确保最后一个维度是特征维度
        assert x.shape[-1] == self.dim, "CMVN dim mismatch"

        # 执行 CMVN 归一化
        out = x - self.means
        out = out * self.inverse_std_variances
        return out

    def _read_kaldi_cmvn(self, kaldi_cmvn_file):
        if not os.path.isfile(kaldi_cmvn_file):
            raise FileNotFoundError(f"CMVN file not found: {kaldi_cmvn_file}")
        stats = kaldiio.load_mat(os.fspath(kaldi_cmvn_file))
        assert stats.shape[0] == 2
        dim = stats.shape[-1] - 1
        count = stats[0, dim]
        assert count >= 1
        floor = 1e-20
        means = []
        inverse_std_variances = []
        for d in range(dim):
            mean = stats[0, d] / count
            means.append(mean.item())
            variance = (stats[1, d] / count) - mean * mean
            if variance < floor:
                variance = floor
            istd = 1.0 / math.sqrt(variance)
            inverse_std_variances.append(istd)
        return dim, np.array(means), np.array(inverse_std_variances)


def add_onnx_metadata(input, output):
    meta_data = {
        "model_type": "fire-red-vad",
        "maintainer": "Hu Liang",
        "email": "coolhuhu@foxmail.com",
        "version": "1.0",
        "description": "The input is fbank features, and CMVN is computed inside the ONNX model.",
    }
    model = onnx.load(input)
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = value
    onnx.save(model, output)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export streaming FireRedVAD with built-in CMVN and eight cache inputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Output ONNX file path"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        type=Path,
        help="Directory containing model.pth.tar and cmvn.ark",
    )
    parser.add_argument(
        "--opset-version", type=int, default=13, help="ONNX opset version (>=13)"
    )

    return parser.parse_args()


def main(argv=None):
    args = parse_args()
    fireredvad = FireRedVad.from_pretrained(args.model_path)
    # Architecture dimensions come from the checkpoint, rather than CLI overrides.
    if len(fireredvad.dfsmn.fsmns) != 7 or fireredvad.dfsmn.fsmn1.N2 != 0:
        raise ValueError(
            "This exporter requires a streaming checkpoint with eight FSMN layers"
        )
    feature_dim = fireredvad.cmvn.dim
    projection_dim = fireredvad.dfsmn.fsmn1.lookback_filter.in_channels
    torch.manual_seed(0)
    caches = [torch.zeros(1, projection_dim, 0) for _ in range(8)]
    # Raw FP32 Fbank without external CMVN; feature dimension is fixed by the checkpoint.
    feat = torch.rand(1, 10, feature_dim, dtype=torch.float32)
    input_names = ["feat"] + [f"input_cache{i}" for i in range(1, 9)]
    output_names = ["logits"] + [f"output_cache{i}" for i in range(1, 9)]
    dynamic_axes = {"feat": {1: "T"}, "logits": {1: "T"}}
    dynamic_axes.update({name: {2: "input_cache_length"} for name in input_names[1:]})
    dynamic_axes.update({name: {2: "output_cache_length"} for name in output_names[1:]})
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        tmp_onnx = tmpdir_path / "firered_stream_vad.onnx"
        torch.onnx.export(
            fireredvad,
            (feat, *caches),
            tmp_onnx,
            export_params=True,
            opset_version=args.opset_version,
            verbose=False,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
        add_onnx_metadata(tmp_onnx, args.output)

    onnx.checker.check_model(str(args.output))
    ort_session = ort.InferenceSession(
        str(args.output), providers=["CPUExecutionProvider"]
    )
    # Validate a new input and carry all eight cache outputs into the next call.
    ort_caches = [to_numpy(cache) for cache in caches]
    for _ in range(2):
        feat = torch.rand(1, 10, feature_dim, dtype=torch.float32)
        torch_model_output = fireredvad(feat, *caches)
        ort_inputs = dict(zip(input_names, [to_numpy(feat), *ort_caches]))
        ort_outputs = ort_session.run(None, ort_inputs)
        for i, (expected, actual) in enumerate(zip(torch_model_output, ort_outputs)):
            np.testing.assert_allclose(
                to_numpy(expected),
                actual,
                rtol=1e-4,
                atol=1e-5,
                err_msg=f"ONNX output {i} does not match PyTorch",
            )
        caches = torch_model_output[1:]
        ort_caches = ort_outputs[1:]
    print(f"Exported and validated ONNX model: {args.output}")


if __name__ == "__main__":
    main()
