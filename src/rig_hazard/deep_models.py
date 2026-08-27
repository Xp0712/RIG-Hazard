from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from .torch_runtime import nn, torch


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        if kernel_size < 1 or dilation < 1:
            raise ValueError("kernel_size and dilation must be positive")
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.nn.functional.pad(values, (self.left_padding, 0)))


class ChannelLayerNorm(nn.Module):
    """Layer normalization across channels at each time step, preserving causality."""

    def __init__(self, channels: int):
        super().__init__()
        self.normalization = nn.LayerNorm(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.normalization(values.transpose(1, 2)).transpose(1, 2)


class TCNResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)
        self.output_normalization = ChannelLayerNorm(out_channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.residual(values)
        hidden = self.dropout(self.activation(self.conv1(values)))
        hidden = self.dropout(self.activation(self.conv2(hidden)))
        return self.activation(self.output_normalization(hidden + residual))


class TCNEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        channels: int = 32,
        kernel_size: int = 3,
        dilations: list[int] | tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.2,
    ):
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = input_size
        for dilation in dilations:
            blocks.append(TCNResidualBlock(in_channels, channels, kernel_size, int(dilation), dropout))
            in_channels = channels
        self.network = nn.Sequential(*blocks)
        self.output_size = channels
        self.receptive_field_steps = 1 + 2 * (kernel_size - 1) * sum(int(value) for value in dilations)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3:
            raise ValueError("history must have shape [batch, time, feature]")
        hidden = self.network(history.transpose(1, 2))
        return hidden[:, :, -1]


class GRUEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output_size = hidden_size

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3:
            raise ValueError("history must have shape [batch, time, feature]")
        _, hidden = self.gru(history)
        return self.dropout(hidden[-1])


class FeatureSubsetGRUEncoder(nn.Module):
    """GRU over an explicit input field subset, with no dormant masked weights."""

    def __init__(
        self,
        input_size: int,
        feature_indices: list[int] | tuple[int, ...],
        hidden_size: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        indices = sorted({int(value) for value in feature_indices})
        if not indices or min(indices) < 0 or max(indices) >= input_size:
            raise ValueError("Feature-subset GRU indices must address the input")
        self.register_buffer(
            "feature_index", torch.tensor(indices, dtype=torch.long), persistent=False
        )
        self.gru = GRUEncoder(len(indices), hidden_size, dropout)
        self.output_size = self.gru.output_size

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3:
            raise ValueError("history must have shape [batch, time, feature]")
        return self.gru(history.index_select(2, self.feature_index))


class RecurrentDualTimeEncoder(nn.Module):
    """Low-capacity fast weather branch plus a current recurrence-state branch."""

    def __init__(
        self,
        input_size: int,
        history_steps: int,
        fast_feature_indices: list[int] | tuple[int, ...],
        slow_feature_indices: list[int] | tuple[int, ...],
        fast_steps: int = 36,
        fast_hidden_size: int = 16,
        slow_hidden_size: int = 8,
        output_size: int = 24,
        dropout: float = 0.1,
    ):
        super().__init__()
        fast = sorted({int(value) for value in fast_feature_indices})
        slow = sorted({int(value) for value in slow_feature_indices})
        if not fast or not slow:
            raise ValueError("Both recurrent dual-time branches require at least one feature")
        if min(fast + slow) < 0 or max(fast + slow) >= input_size:
            raise ValueError("Recurrent dual-time feature index is outside the input")
        if history_steps < 1 or fast_steps < 1:
            raise ValueError("history_steps and fast_steps must be positive")
        self.input_size = int(input_size)
        self.history_steps = int(history_steps)
        self.fast_steps = min(int(fast_steps), int(history_steps))
        self.fast_feature_indices = fast
        self.slow_feature_indices = slow
        self.register_buffer("fast_index", torch.tensor(fast, dtype=torch.long), persistent=False)
        self.register_buffer("slow_index", torch.tensor(slow, dtype=torch.long), persistent=False)
        self.fast_encoder = GRUEncoder(len(fast), int(fast_hidden_size), dropout)
        self.slow_encoder = nn.Sequential(
            nn.Linear(len(slow), int(slow_hidden_size)),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(int(fast_hidden_size) + int(slow_hidden_size), int(output_size)),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_size = int(output_size)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[1:] != (self.history_steps, self.input_size):
            raise ValueError(
                f"history must have shape [batch, {self.history_steps}, {self.input_size}]"
            )
        fast_history = history[:, -self.fast_steps :, :].index_select(2, self.fast_index)
        current_state = history[:, -1, :].index_select(1, self.slow_index)
        fast_representation = self.fast_encoder(fast_history)
        slow_representation = self.slow_encoder(current_state)
        return self.fusion(torch.cat([fast_representation, slow_representation], dim=1))


class DualResolutionWeatherEncoder(nn.Module):
    """Causal fast/slow weather encoder with optional recurrent-history injection.

    The fast branch reads the most recent high-resolution bins.  The slow branch
    causally averages non-overlapping bins over the complete history before a
    separate GRU.  Both branches receive exactly the same weather variables.
    Explicit recurrent-event variables are only available to the optional
    ``direct`` and ``gate`` modes.
    """

    def __init__(
        self,
        input_size: int,
        history_steps: int,
        weather_feature_indices: list[int] | tuple[int, ...],
        recurrence_feature_indices: list[int] | tuple[int, ...] = (),
        mode: str = "dual",
        fast_steps: int = 18,
        slow_bucket_steps: int = 3,
        fast_hidden_size: int = 8,
        slow_hidden_size: int = 8,
        output_size: int = 16,
        dropout: float = 0.1,
        recurrence_mode: str = "none",
        recurrence_hidden_size: int = 8,
        gate_penalty_weight: float = 0.0,
    ):
        super().__init__()
        weather = sorted({int(value) for value in weather_feature_indices})
        recurrence = sorted({int(value) for value in recurrence_feature_indices})
        if not weather:
            raise ValueError("Dual-resolution weather encoder requires weather features")
        if min(weather + recurrence) < 0 or max(weather + recurrence) >= input_size:
            raise ValueError("Dual-resolution feature index is outside the input")
        if mode not in {"fast", "slow", "dual"}:
            raise ValueError("mode must be fast, slow, or dual")
        if recurrence_mode not in {"none", "direct", "gate"}:
            raise ValueError("recurrence_mode must be none, direct, or gate")
        if recurrence_mode != "none" and not recurrence:
            raise ValueError("A recurrent-history mode requires recurrence features")
        if history_steps < 1 or fast_steps < 1 or slow_bucket_steps < 1:
            raise ValueError("History and aggregation sizes must be positive")
        if history_steps % slow_bucket_steps != 0:
            raise ValueError("history_steps must be divisible by slow_bucket_steps")

        self.input_size = int(input_size)
        self.history_steps = int(history_steps)
        self.mode = str(mode)
        self.fast_steps = min(int(fast_steps), self.history_steps)
        self.slow_bucket_steps = int(slow_bucket_steps)
        self.slow_steps = self.history_steps // self.slow_bucket_steps
        self.recurrence_mode = str(recurrence_mode)
        self.gate_penalty_weight = float(gate_penalty_weight)
        self.register_buffer(
            "weather_index", torch.tensor(weather, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "recurrence_index", torch.tensor(recurrence, dtype=torch.long), persistent=False
        )

        feature_count = len(weather)
        self.fast_encoder = (
            GRUEncoder(feature_count, int(fast_hidden_size), dropout)
            if self.mode in {"fast", "dual"}
            else None
        )
        self.slow_encoder = (
            GRUEncoder(feature_count, int(slow_hidden_size), dropout)
            if self.mode in {"slow", "dual"}
            else None
        )
        if self.mode == "fast":
            assert self.fast_encoder is not None
            self.single_projection = nn.Sequential(
                nn.Linear(self.fast_encoder.output_size, int(output_size)),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        elif self.mode == "slow":
            assert self.slow_encoder is not None
            self.single_projection = nn.Sequential(
                nn.Linear(self.slow_encoder.output_size, int(output_size)),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.fast_projection = nn.Linear(int(fast_hidden_size), int(output_size))
            self.slow_projection = nn.Linear(int(slow_hidden_size), int(output_size))
            self.weather_gate = nn.Linear(int(output_size) * 2, int(output_size))
            self.weather_normalization = nn.LayerNorm(int(output_size))

        if self.recurrence_mode != "none":
            self.recurrence_projection = nn.Sequential(
                nn.Linear(len(recurrence), int(recurrence_hidden_size)),
                nn.GELU(),
                nn.Linear(int(recurrence_hidden_size), int(output_size)),
            )
            self.recurrence_normalization = nn.LayerNorm(int(output_size))
        if self.recurrence_mode == "gate":
            self.recurrence_gate = nn.Sequential(
                nn.Linear(int(output_size) + len(recurrence), int(recurrence_hidden_size)),
                nn.GELU(),
                nn.Linear(int(recurrence_hidden_size), 1),
                nn.Sigmoid(),
            )
        self.output_size = int(output_size)
        self._last_rec_gate: torch.Tensor | None = None

    def _slow_history(self, weather_history: torch.Tensor) -> torch.Tensor:
        batch, _, features = weather_history.shape
        return weather_history.reshape(
            batch, self.slow_steps, self.slow_bucket_steps, features
        ).mean(dim=2)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[1:] != (self.history_steps, self.input_size):
            raise ValueError(
                f"history must have shape [batch, {self.history_steps}, {self.input_size}]"
            )
        weather_history = history.index_select(2, self.weather_index)
        if self.mode == "fast":
            assert self.fast_encoder is not None
            weather_state = self.single_projection(
                self.fast_encoder(weather_history[:, -self.fast_steps :, :])
            )
        elif self.mode == "slow":
            assert self.slow_encoder is not None
            weather_state = self.single_projection(
                self.slow_encoder(self._slow_history(weather_history))
            )
        else:
            assert self.fast_encoder is not None and self.slow_encoder is not None
            fast_state = self.fast_projection(
                self.fast_encoder(weather_history[:, -self.fast_steps :, :])
            )
            slow_state = self.slow_projection(
                self.slow_encoder(self._slow_history(weather_history))
            )
            gate = torch.sigmoid(self.weather_gate(torch.cat([fast_state, slow_state], dim=1)))
            weather_state = self.weather_normalization(fast_state + gate * slow_state)

        self._last_rec_gate = None
        if self.recurrence_mode == "none":
            return weather_state
        recurrence_state = history[:, -1, :].index_select(1, self.recurrence_index)
        recurrence_representation = self.recurrence_projection(recurrence_state)
        if self.recurrence_mode == "direct":
            self._last_rec_gate = torch.ones(
                (history.shape[0], 1), device=history.device, dtype=history.dtype
            )
        else:
            self._last_rec_gate = self.recurrence_gate(
                torch.cat([weather_state, recurrence_state], dim=1)
            )
        return self.recurrence_normalization(
            weather_state + self._last_rec_gate * recurrence_representation
        )

    def regularization_penalty(self) -> torch.Tensor:
        if (
            self.recurrence_mode != "gate"
            or self._last_rec_gate is None
            or self.gate_penalty_weight <= 0
        ):
            return next(self.parameters()).new_zeros(())
        return self.gate_penalty_weight * self._last_rec_gate.mean()

    def diagnostics(self) -> dict[str, torch.Tensor]:
        return (
            {"rec_gate": self._last_rec_gate}
            if self.recurrence_mode == "gate" and self._last_rec_gate is not None
            else {}
        )


class HistoryRevIN(nn.Module):
    """Normalize each observed history window independently, without future data."""

    def __init__(self, feature_count: int, affine: bool = True, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = float(epsilon)
        self.affine = bool(affine)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, 1, feature_count))
            self.bias = nn.Parameter(torch.zeros(1, 1, feature_count))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        mean = history.mean(dim=1, keepdim=True).detach()
        variance = history.var(dim=1, keepdim=True, unbiased=False).detach()
        normalized = (history - mean) / torch.sqrt(variance + self.epsilon)
        if self.affine:
            normalized = normalized * self.weight + self.bias
        return normalized


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, maximum_length: int, embedding_dim: int):
        super().__init__()
        position = torch.arange(maximum_length, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / max(embedding_dim, 1))
        )
        encoding = torch.zeros(maximum_length, embedding_dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * scale)
        odd_width = encoding[:, 1::2].shape[1]
        encoding[:, 1::2] = torch.cos(position * scale[:odd_width])
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[1] > self.encoding.shape[0]:
            raise ValueError("Sequence is longer than the configured positional encoding")
        return values + self.encoding[: values.shape[1]].to(device=values.device, dtype=values.dtype)


def _transformer_encoder(
    embedding_dim: int,
    attention_heads: int,
    layers: int,
    feedforward_dim: int,
    dropout: float,
) -> nn.TransformerEncoder:
    if embedding_dim % attention_heads != 0:
        raise ValueError("embedding_dim must be divisible by attention_heads")
    layer = nn.TransformerEncoderLayer(
        d_model=embedding_dim,
        nhead=attention_heads,
        dim_feedforward=feedforward_dim,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(
        layer,
        num_layers=layers,
        norm=nn.LayerNorm(embedding_dim),
        enable_nested_tensor=False,
    )


def _patchtst_attention_context(values: torch.Tensor):
    """Avoid CUDA fused-SDP launch limits after PatchTST flattens batch and channels."""
    if values.device.type != "cuda":
        return nullcontext()
    attention = getattr(torch.nn, "attention", None)
    if attention is not None and hasattr(attention, "sdpa_kernel"):
        return attention.sdpa_kernel(attention.SDPBackend.MATH)
    return torch.backends.cuda.sdp_kernel(
        enable_flash=False,
        enable_math=True,
        enable_mem_efficient=False,
        enable_cudnn=False,
    )


class PatchTSTEncoder(nn.Module):
    """Channel-independent patch Transformer followed by a shared hazard representation."""

    def __init__(
        self,
        input_size: int,
        history_steps: int,
        output_size: int = 32,
        patch_length: int = 6,
        patch_stride: int = 3,
        embedding_dim: int = 32,
        attention_heads: int = 4,
        layers: int = 2,
        feedforward_dim: int = 64,
        dropout: float = 0.2,
        revin: bool = True,
    ):
        super().__init__()
        if patch_length < 1 or patch_stride < 1 or patch_length > history_steps:
            raise ValueError("PatchTST patch_length/patch_stride are incompatible with history_steps")
        self.input_size = int(input_size)
        self.history_steps = int(history_steps)
        self.patch_length = int(patch_length)
        self.patch_stride = int(patch_stride)
        self.number_patches = 1 + (self.history_steps - self.patch_length) // self.patch_stride
        self.revin = HistoryRevIN(input_size) if revin else nn.Identity()
        self.patch_projection = nn.Linear(self.patch_length, embedding_dim)
        self.position = SinusoidalPositionEncoding(self.number_patches, embedding_dim)
        self.transformer = _transformer_encoder(
            embedding_dim, attention_heads, layers, feedforward_dim, dropout
        )
        self.patch_pool = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.number_patches * embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.channel_pool = nn.Sequential(
            nn.Linear(self.input_size * embedding_dim, output_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_size = int(output_size)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[1:] != (self.history_steps, self.input_size):
            raise ValueError(
                f"history must have shape [batch, {self.history_steps}, {self.input_size}]"
            )
        normalized = self.revin(history)
        patches = normalized.unfold(1, self.patch_length, self.patch_stride)
        patches = patches.permute(0, 2, 1, 3).contiguous()
        batch_size = patches.shape[0]
        tokens = patches.reshape(batch_size * self.input_size, self.number_patches, self.patch_length)
        tokens = self.position(self.patch_projection(tokens))
        with _patchtst_attention_context(tokens):
            transformed = self.transformer(tokens)
        channel_representation = self.patch_pool(transformed)
        channel_representation = channel_representation.reshape(batch_size, self.input_size, -1)
        return self.channel_pool(channel_representation.flatten(start_dim=1))


class Inception2DBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernels: list[int] | tuple[int, ...]):
        super().__init__()
        if not kernels or any(kernel < 1 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError("TimesNet inception kernels must be positive odd integers")
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=(int(kernel), int(kernel)),
                    padding=(int(kernel) // 2, int(kernel) // 2),
                )
                for kernel in kernels
            ]
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.stack([branch(values) for branch in self.branches], dim=-1).mean(dim=-1)


class TimesBlock(nn.Module):
    """FFT-selected period folding and multi-scale 2D variation modeling."""

    def __init__(
        self,
        embedding_dim: int,
        feedforward_dim: int,
        top_k: int,
        inception_kernels: list[int] | tuple[int, ...],
        dropout: float,
    ):
        super().__init__()
        if top_k < 1:
            raise ValueError("TimesNet top_k must be positive")
        self.top_k = int(top_k)
        self.convolution = nn.Sequential(
            Inception2DBlock(embedding_dim, feedforward_dim, inception_kernels),
            nn.GELU(),
            nn.Dropout(dropout),
            Inception2DBlock(feedforward_dim, embedding_dim, inception_kernels),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, length, channels = values.shape
        spectrum = torch.fft.rfft(values, dim=1)
        global_amplitude = spectrum.abs().mean(dim=(0, 2))
        if global_amplitude.numel() <= 1:
            return values
        global_amplitude = global_amplitude.clone()
        global_amplitude[0] = -torch.inf
        count = min(self.top_k, global_amplitude.numel() - 1)
        frequency_indices = torch.topk(global_amplitude, count).indices
        results: list[torch.Tensor] = []
        sample_amplitudes: list[torch.Tensor] = []
        for frequency_tensor in frequency_indices:
            frequency = max(int(frequency_tensor.item()), 1)
            period = max(length // frequency, 1)
            padded_length = int(math.ceil(length / period) * period)
            if padded_length > length:
                padded = torch.nn.functional.pad(values, (0, 0, 0, padded_length - length))
            else:
                padded = values
            folded = padded.reshape(batch_size, padded_length // period, period, channels)
            folded = folded.permute(0, 3, 1, 2).contiguous()
            convolved = self.convolution(folded)
            restored = convolved.permute(0, 2, 3, 1).reshape(batch_size, padded_length, channels)
            results.append(restored[:, :length, :])
            sample_amplitudes.append(spectrum[:, frequency, :].abs().mean(dim=1))
        stacked = torch.stack(results, dim=-1)
        weights = torch.softmax(torch.stack(sample_amplitudes, dim=1), dim=1)
        combined = (stacked * weights[:, None, None, :]).sum(dim=-1)
        return values + combined


class TimesNetEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        history_steps: int,
        output_size: int = 32,
        embedding_dim: int = 32,
        feedforward_dim: int = 64,
        layers: int = 2,
        top_k: int = 3,
        inception_kernels: list[int] | tuple[int, ...] = (1, 3, 5),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.history_steps = int(history_steps)
        self.value_projection = nn.Linear(input_size, embedding_dim)
        self.position = SinusoidalPositionEncoding(history_steps, embedding_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TimesBlock(embedding_dim, feedforward_dim, top_k, inception_kernels, dropout)
                for _ in range(layers)
            ]
        )
        self.normalizations = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(layers)])
        self.pool = nn.Sequential(
            nn.Linear(embedding_dim * 3, output_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_size = int(output_size)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[1:] != (self.history_steps, self.input_size):
            raise ValueError(
                f"history must have shape [batch, {self.history_steps}, {self.input_size}]"
            )
        hidden = self.input_dropout(self.position(self.value_projection(history)))
        for block, normalization in zip(self.blocks, self.normalizations):
            hidden = normalization(block(hidden))
        summary = torch.cat([hidden[:, -1, :], hidden.mean(dim=1), hidden.amax(dim=1)], dim=1)
        return self.pool(summary)


class ITransformerEncoder(nn.Module):
    """Inverted Transformer with one token per station-level input variable."""

    def __init__(
        self,
        input_size: int,
        history_steps: int,
        output_size: int = 32,
        embedding_dim: int = 32,
        attention_heads: int = 4,
        layers: int = 2,
        feedforward_dim: int = 64,
        dropout: float = 0.2,
        revin: bool = True,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.history_steps = int(history_steps)
        self.revin = HistoryRevIN(input_size) if revin else nn.Identity()
        self.temporal_projection = nn.Linear(history_steps, embedding_dim)
        self.variable_embedding = nn.Embedding(input_size, embedding_dim)
        self.transformer = _transformer_encoder(
            embedding_dim, attention_heads, layers, feedforward_dim, dropout
        )
        self.variable_pool = nn.Sequential(
            nn.Linear(input_size * embedding_dim, output_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_size = int(output_size)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[1:] != (self.history_steps, self.input_size):
            raise ValueError(
                f"history must have shape [batch, {self.history_steps}, {self.input_size}]"
            )
        normalized = self.revin(history).transpose(1, 2)
        variable_indices = torch.arange(self.input_size, device=history.device)
        tokens = self.temporal_projection(normalized) + self.variable_embedding(variable_indices).unsqueeze(0)
        encoded = self.transformer(tokens)
        return self.variable_pool(encoded.flatten(start_dim=1))


class MultiStepHazardHead(nn.Module):
    def __init__(
        self,
        representation_size: int,
        horizon_steps: int,
        horizon_embedding_dim: int = 8,
        initial_log_rate: float = -9.0,
    ):
        super().__init__()
        if horizon_steps < 1:
            raise ValueError("horizon_steps must be positive")
        self.horizon_steps = int(horizon_steps)
        self.horizon_embedding = nn.Embedding(self.horizon_steps, horizon_embedding_dim)
        hidden_size = max(representation_size // 2, 16)
        self.network = nn.Sequential(
            nn.Linear(representation_size + horizon_embedding_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, float(initial_log_rate))

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        batch_size = representation.shape[0]
        horizon = torch.arange(self.horizon_steps, device=representation.device)
        embedding = self.horizon_embedding(horizon).unsqueeze(0).expand(batch_size, -1, -1)
        repeated = representation.unsqueeze(1).expand(-1, self.horizon_steps, -1)
        return self.network(torch.cat([repeated, embedding], dim=2)).squeeze(-1)


class DeepHazardModel(nn.Module):
    def __init__(
        self,
        encoder_type: str,
        input_size: int,
        horizon_steps: int,
        hidden_size: int = 32,
        kernel_size: int = 3,
        dilations: list[int] | tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.2,
        horizon_embedding_dim: int = 8,
        initial_log_rate: float = -9.0,
        masked_feature_indices: list[int] | tuple[int, ...] = (),
        number_stations: int = 0,
        number_cities: int = 0,
        history_steps: int = 0,
        encoder_config: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.architecture_config = {
            "encoder_type": encoder_type,
            "input_size": int(input_size),
            "horizon_steps": int(horizon_steps),
            "hidden_size": int(hidden_size),
            "kernel_size": int(kernel_size),
            "dilations": [int(value) for value in dilations],
            "dropout": float(dropout),
            "horizon_embedding_dim": int(horizon_embedding_dim),
            "initial_log_rate": float(initial_log_rate),
            "masked_feature_indices": sorted({int(value) for value in masked_feature_indices}),
            "number_stations": int(number_stations),
            "number_cities": int(number_cities),
            "history_steps": int(history_steps),
            "encoder_config": dict(encoder_config or {}),
        }
        invalid_masks = [value for value in self.architecture_config["masked_feature_indices"] if value < 0 or value >= input_size]
        if invalid_masks:
            raise ValueError(f"masked feature indices are outside the input: {invalid_masks}")
        if (number_stations > 0) != (number_cities > 0):
            raise ValueError("number_stations and number_cities must either both be positive or both be zero")
        if encoder_type == "gru":
            self.encoder = GRUEncoder(input_size, hidden_size, dropout)
        elif encoder_type == "gru_subset":
            settings = self.architecture_config["encoder_config"]
            self.encoder = FeatureSubsetGRUEncoder(
                input_size,
                [int(value) for value in settings["feature_indices"]],
                hidden_size,
                dropout,
            )
        elif encoder_type == "tcn":
            self.encoder = TCNEncoder(input_size, hidden_size, kernel_size, dilations, dropout)
        elif encoder_type == "patchtst":
            settings = self.architecture_config["encoder_config"]
            self.encoder = PatchTSTEncoder(
                input_size=input_size,
                history_steps=history_steps,
                output_size=int(settings.get("output_size", hidden_size)),
                patch_length=int(settings.get("patch_length", 6)),
                patch_stride=int(settings.get("patch_stride", 3)),
                embedding_dim=int(settings.get("embedding_dim", 32)),
                attention_heads=int(settings.get("attention_heads", 4)),
                layers=int(settings.get("layers", 2)),
                feedforward_dim=int(settings.get("feedforward_dim", 64)),
                dropout=float(settings.get("dropout", dropout)),
                revin=bool(settings.get("revin", True)),
            )
        elif encoder_type == "timesnet":
            settings = self.architecture_config["encoder_config"]
            self.encoder = TimesNetEncoder(
                input_size=input_size,
                history_steps=history_steps,
                output_size=int(settings.get("output_size", hidden_size)),
                embedding_dim=int(settings.get("embedding_dim", 32)),
                feedforward_dim=int(settings.get("feedforward_dim", 64)),
                layers=int(settings.get("layers", 2)),
                top_k=int(settings.get("top_k", 3)),
                inception_kernels=[int(value) for value in settings.get("inception_kernels", [1, 3, 5])],
                dropout=float(settings.get("dropout", dropout)),
            )
        elif encoder_type == "itransformer":
            settings = self.architecture_config["encoder_config"]
            self.encoder = ITransformerEncoder(
                input_size=input_size,
                history_steps=history_steps,
                output_size=int(settings.get("output_size", hidden_size)),
                embedding_dim=int(settings.get("embedding_dim", 32)),
                attention_heads=int(settings.get("attention_heads", 4)),
                layers=int(settings.get("layers", 2)),
                feedforward_dim=int(settings.get("feedforward_dim", 64)),
                dropout=float(settings.get("dropout", dropout)),
                revin=bool(settings.get("revin", True)),
            )
        elif encoder_type == "recurrent_dual":
            settings = self.architecture_config["encoder_config"]
            self.encoder = RecurrentDualTimeEncoder(
                input_size=input_size,
                history_steps=history_steps,
                fast_feature_indices=[int(value) for value in settings["fast_feature_indices"]],
                slow_feature_indices=[int(value) for value in settings["slow_feature_indices"]],
                fast_steps=int(settings.get("fast_steps", 36)),
                fast_hidden_size=int(settings.get("fast_hidden_size", 16)),
                slow_hidden_size=int(settings.get("slow_hidden_size", 8)),
                output_size=int(settings.get("output_size", hidden_size)),
                dropout=float(settings.get("dropout", dropout)),
            )
        elif encoder_type in {
            "weather_fast",
            "weather_slow",
            "weather_dual",
            "weather_dual_rec",
            "weather_dual_rec_gate",
        }:
            settings = self.architecture_config["encoder_config"]
            defaults = {
                "weather_fast": ("fast", "none"),
                "weather_slow": ("slow", "none"),
                "weather_dual": ("dual", "none"),
                "weather_dual_rec": ("dual", "direct"),
                "weather_dual_rec_gate": ("dual", "gate"),
            }
            mode, recurrence_mode = defaults[encoder_type]
            self.encoder = DualResolutionWeatherEncoder(
                input_size=input_size,
                history_steps=history_steps,
                weather_feature_indices=[
                    int(value) for value in settings["weather_feature_indices"]
                ],
                recurrence_feature_indices=[
                    int(value)
                    for value in settings.get("recurrence_feature_indices", [])
                ],
                mode=str(settings.get("mode", mode)),
                fast_steps=int(settings.get("fast_steps", 18)),
                slow_bucket_steps=int(settings.get("slow_bucket_steps", 3)),
                fast_hidden_size=int(settings.get("fast_hidden_size", 8)),
                slow_hidden_size=int(settings.get("slow_hidden_size", 8)),
                output_size=int(settings.get("output_size", hidden_size)),
                dropout=float(settings.get("dropout", dropout)),
                recurrence_mode=str(
                    settings.get("recurrence_mode", recurrence_mode)
                ),
                recurrence_hidden_size=int(
                    settings.get("recurrence_hidden_size", 8)
                ),
                gate_penalty_weight=float(
                    settings.get("gate_penalty_weight", 0.0)
                ),
            )
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")
        self.encoder_type = encoder_type
        self.head = MultiStepHazardHead(
            self.encoder.output_size, horizon_steps, horizon_embedding_dim, initial_log_rate
        )
        self.number_stations = int(number_stations)
        self.number_cities = int(number_cities)
        if self.has_hierarchical_barrier:
            self.city_log_susceptibility = nn.Embedding(self.number_cities + 1, 1, padding_idx=self.number_cities)
            self.station_log_susceptibility = nn.Embedding(
                self.number_stations + 1, 1, padding_idx=self.number_stations
            )
            nn.init.zeros_(self.city_log_susceptibility.weight)
            nn.init.zeros_(self.station_log_susceptibility.weight)

    @property
    def has_hierarchical_barrier(self) -> bool:
        return self.number_stations > 0 and self.number_cities > 0

    def _centered_barrier_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.has_hierarchical_barrier:
            empty = torch.empty(0, device=next(self.parameters()).device)
            return empty, empty
        city = self.city_log_susceptibility.weight[: self.number_cities, 0]
        station = self.station_log_susceptibility.weight[: self.number_stations, 0]
        return city - city.mean(), station - station.mean()

    def barrier_penalty(self, city_l2: float, station_l2: float) -> torch.Tensor:
        penalty = torch.zeros((), device=next(self.parameters()).device)
        if self.has_hierarchical_barrier:
            city, station = self._centered_barrier_weights()
            penalty = (
                penalty
                + float(city_l2) * city.square().mean()
                + float(station_l2) * station.square().mean()
            )
        regularization = getattr(self.encoder, "regularization_penalty", None)
        if regularization is not None:
            penalty = penalty + regularization()
        return penalty

    def forward(
        self,
        history: torch.Tensor,
        station_index: torch.Tensor | None = None,
        city_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        masked = self.architecture_config["masked_feature_indices"]
        if masked:
            history = history.clone()
            history[..., masked] = 0.0
        eta = self.head(self.encoder(history))
        if not self.has_hierarchical_barrier:
            return eta
        if station_index is None or city_index is None:
            raise ValueError("station_index and city_index are required by the hierarchical barrier")
        city_weights, station_weights = self._centered_barrier_weights()
        safe_city = torch.where(
            (city_index >= 0) & (city_index < self.number_cities),
            city_index,
            torch.full_like(city_index, self.number_cities),
        )
        safe_station = torch.where(
            (station_index >= 0) & (station_index < self.number_stations),
            station_index,
            torch.full_like(station_index, self.number_stations),
        )
        city_with_unknown = torch.cat([city_weights, city_weights.new_zeros(1)])
        station_with_unknown = torch.cat([station_weights, station_weights.new_zeros(1)])
        barrier = city_with_unknown[safe_city] + station_with_unknown[safe_station]
        return eta + barrier.unsqueeze(1)

    def diagnostics(self) -> dict[str, torch.Tensor]:
        diagnostic_function = getattr(self.encoder, "diagnostics", None)
        return diagnostic_function() if diagnostic_function is not None else {}

    def to_config(self) -> dict[str, Any]:
        return {
            **self.architecture_config,
            "encoder": self.encoder.__class__.__name__,
            "encoder_output_size": int(self.encoder.output_size),
            "receptive_field_steps": getattr(self.encoder, "receptive_field_steps", None),
            "parameters": int(sum(parameter.numel() for parameter in self.parameters())),
        }

    @classmethod
    def from_config(cls, value: dict[str, Any]) -> "DeepHazardModel":
        return cls(
            encoder_type=str(value["encoder_type"]),
            input_size=int(value["input_size"]),
            horizon_steps=int(value["horizon_steps"]),
            hidden_size=int(value["hidden_size"]),
            kernel_size=int(value["kernel_size"]),
            dilations=[int(item) for item in value["dilations"]],
            dropout=float(value["dropout"]),
            horizon_embedding_dim=int(value["horizon_embedding_dim"]),
            initial_log_rate=float(value["initial_log_rate"]),
            masked_feature_indices=[int(item) for item in value.get("masked_feature_indices", [])],
            number_stations=int(value.get("number_stations", 0)),
            number_cities=int(value.get("number_cities", 0)),
            history_steps=int(value.get("history_steps", 0)),
            encoder_config=dict(value.get("encoder_config", {})),
        )


def cloglog_hazard_probability(eta: torch.Tensor) -> torch.Tensor:
    rate = torch.exp(torch.clamp(eta, max=15.0))
    return -torch.expm1(-rate)


def cumulative_incidence(hazard_probability: torch.Tensor) -> torch.Tensor:
    probability = torch.clamp(hazard_probability, min=0.0, max=1.0 - 1e-7)
    return 1.0 - torch.cumprod(1.0 - probability, dim=-1)


def discrete_hazard_nll(
    eta: torch.Tensor,
    target: torch.Tensor,
    risk_mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    hard_negative_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    numerator, denominator = discrete_hazard_nll_components(
        eta, target, risk_mask, sample_weight, hard_negative_weight
    )
    return numerator / torch.clamp(denominator, min=1.0)


def discrete_hazard_nll_components(
    eta: torch.Tensor,
    target: torch.Tensor,
    risk_mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    hard_negative_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if eta.shape != target.shape or eta.shape != risk_mask.shape:
        raise ValueError("eta, target, and risk_mask must have identical shapes")
    bounded_eta = torch.clamp(eta, max=15.0)
    rate = torch.exp(bounded_eta)
    small_rate = bounded_eta < -10.0
    exact_eta = torch.where(small_rate, torch.full_like(bounded_eta, -10.0), bounded_eta)
    exact_rate = torch.exp(exact_eta)
    exact_positive_loss = -torch.log(-torch.expm1(-exact_rate))
    positive_loss = torch.where(small_rate, -bounded_eta, exact_positive_loss)
    losses = torch.where(target > 0.5, positive_loss, rate)
    weights = risk_mask.to(dtype=losses.dtype)
    if sample_weight is not None:
        weights = weights * sample_weight.to(dtype=losses.dtype).reshape(-1, 1)
    if hard_negative_weight is not None:
        weights = weights * hard_negative_weight.to(dtype=losses.dtype).reshape(-1, 1)
    return (losses * weights).sum(), weights.sum()


@dataclass(frozen=True)
class HorizonRisks:
    risk_1h: torch.Tensor
    risk_3h: torch.Tensor
    risk_6h: torch.Tensor


def standard_horizon_risks(cumulative_risk: torch.Tensor, step_minutes: int = 10) -> HorizonRisks:
    indices = [int(hours * 60 / step_minutes) - 1 for hours in (1, 3, 6)]
    if cumulative_risk.shape[-1] <= indices[-1]:
        raise ValueError("cumulative_risk does not contain a full 6-hour horizon")
    return HorizonRisks(*(cumulative_risk[..., index] for index in indices))
