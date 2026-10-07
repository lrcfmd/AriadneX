"""PXRD CLS embedding adapter backed by the supplied ``last.ckpt`` encoder."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

import numpy as np
import torch

from ..runtime import EncoderRunner
from .artifact import PxrdPattern
from .pxrd_encoder import load_pxrd_patch_transformer_checkpoint


@dataclass(frozen=True)
class PatternGrid:
    """Uniform theta grid expected by an embedding checkpoint."""

    theta_min: float = 5.0
    theta_max: float = 90.0
    n_points: int = 4096

    def __post_init__(self) -> None:
        if int(self.n_points) < 2:
            raise ValueError("PatternGrid.n_points must be at least two")
        if float(self.theta_max) <= float(self.theta_min):
            raise ValueError("PatternGrid.theta_max must exceed theta_min")

    @property
    def theta(self) -> np.ndarray:
        return np.linspace(
            float(self.theta_min), float(self.theta_max), int(self.n_points), dtype=np.float32
        )

    @property
    def step(self) -> float:
        return float((float(self.theta_max) - float(self.theta_min)) / (int(self.n_points) - 1))


@dataclass(frozen=True)
class EmbeddingOutput:
    embeddings: np.ndarray
    theta: np.ndarray

    def __post_init__(self) -> None:
        array = np.asarray(self.embeddings, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("Embedding output must be a 2D array")
        object.__setattr__(self, "embeddings", array)
        object.__setattr__(self, "theta", np.asarray(self.theta, dtype=np.float32).reshape(-1))


class EmbeddingAdapter(Protocol):
    """Adapter seam for converting simulated patterns to embeddings."""

    def embed(
        self, patterns: Iterable[PxrdPattern], *, theta: np.ndarray, batch_size: int, device: str
    ) -> Any: ...


class PxrdEmbeddingAdapter:
    """Run checkpoint-backed PXRD embedding in a child process.

    The adapter serialises only the numeric pattern payload, invokes
    ``ariadnex.library.embedding_worker`` through ``EncoderRunner``, and
    reads a compact NPZ result.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        runner: Any | None = None,
        repo_src: Path | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.repo_src = (
            Path(repo_src).expanduser().resolve()
            if repo_src is not None
            else Path(__file__).resolve().parents[2]
        )
        if runner is None:
            runner = EncoderRunner()
        self.runner = runner

    def embed(
        self,
        patterns: Iterable[PxrdPattern],
        *,
        theta: np.ndarray,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> EmbeddingOutput:
        ordered = tuple(patterns)
        if not ordered:
            raise ValueError("No patterns were supplied for PXRD embedding")
        if int(batch_size) < 1:
            raise ValueError("batch_size must be at least one")
        requested_theta = np.asarray(theta, dtype=np.float32).reshape(-1)
        if requested_theta.size < 2 or not np.all(np.isfinite(requested_theta)):
            raise ValueError("Embedding theta must contain at least two finite points")

        ids: list[str] = []
        offsets = [0]
        x_values: list[np.ndarray] = []
        y_values: list[np.ndarray] = []
        for pattern in ordered:
            x = np.asarray(pattern.x, dtype=np.float32).reshape(-1)
            y = np.asarray(pattern.y, dtype=np.float32).reshape(-1)
            if x.size != y.size or x.size < 2:
                raise ValueError(f"Pattern {pattern.source_id!r} has invalid x/y arrays")
            ids.append(str(pattern.source_id))
            x_values.append(x)
            y_values.append(y)
            offsets.append(offsets[-1] + int(x.size))

        with tempfile.TemporaryDirectory(prefix="ariadnex_embedding_") as scratch:
            scratch_dir = Path(scratch)
            input_path = scratch_dir / "patterns.npz"
            output_path = scratch_dir / "embeddings.npz"
            np.savez_compressed(
                input_path,
                source_ids=np.asarray(ids, dtype=str),
                offsets=np.asarray(offsets, dtype=np.int64),
                x_values=np.concatenate(x_values).astype(np.float32, copy=False),
                y_values=np.concatenate(y_values).astype(np.float32, copy=False),
                theta=requested_theta,
            )
            args = [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--checkpoint",
                str(self.checkpoint_path),
                "--batch-size",
                str(int(batch_size)),
                "--device",
                str(device),
            ]
            environment = self._worker_environment()
            try:
                self.runner.run_module(
                    "ariadnex.library.embedding_worker",
                    args,
                    check=True,
                    env=environment,
                )
            except subprocess.CalledProcessError as exc:
                detail = str(exc.stderr or exc.stdout or "").strip()
                suffix = f": {detail}" if detail else ""
                raise RuntimeError(f"PXRD embedding worker failed{suffix}") from exc
            if not output_path.is_file():
                raise RuntimeError("PXRD embedding worker did not produce an output file")
            with np.load(output_path, allow_pickle=False) as payload:
                if "embeddings" not in payload or "theta" not in payload:
                    raise ValueError("PXRD embedding output must contain embeddings and theta")
                embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
                output_theta = np.asarray(payload["theta"], dtype=np.float32).reshape(-1)
                output_ids = payload.get("source_ids")
                if output_ids is not None and [str(value) for value in output_ids.tolist()] != ids:
                    raise ValueError("PXRD embedding output source IDs do not match input order")
            if embeddings.ndim != 2 or embeddings.shape[0] != len(ordered):
                raise ValueError(
                    f"PXRD embedding output shape {embeddings.shape} does not match {len(ordered)} patterns"
                )
            if output_theta.size < 2 or not np.all(np.isfinite(output_theta)):
                raise ValueError("PXRD embedding output theta is invalid")
            if not np.all(np.isfinite(embeddings)):
                raise ValueError("PXRD embedding output contains non-finite values")
            return EmbeddingOutput(embeddings=embeddings, theta=output_theta)

    def _worker_environment(self) -> dict[str, str]:
        return EncoderRunner.worker_environment(repo_src=self.repo_src)


def normalize_pxrd_sample_max(values: np.ndarray) -> np.ndarray:
    """Apply the training-time sample-max (min/max) normalization."""

    clipped = np.clip(np.asarray(values, dtype=np.float32).reshape(-1), 0.0, None)
    if clipped.size == 0:
        return clipped
    minimum = float(clipped.min())
    maximum = float(clipped.max())
    denominator = maximum - minimum
    if denominator <= 1e-12 or not np.isfinite(denominator):
        return np.zeros_like(clipped, dtype=np.float32)
    return np.clip((clipped - minimum) / denominator, 0.0, 1.0).astype(np.float32)


def resample_patterns(
    patterns: Iterable[PxrdPattern], grid: PatternGrid
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Align patterns to a checkpoint grid and return IDs, raw y, and inputs."""

    theta = grid.theta
    ids: list[str] = []
    raw: list[np.ndarray] = []
    inputs: list[np.ndarray] = []
    for pattern in patterns:
        x = np.asarray(pattern.x, dtype=np.float64).reshape(-1)
        y = np.asarray(pattern.y, dtype=np.float64).reshape(-1)
        if x.size != y.size or x.size < 2:
            raise ValueError(f"Pattern {pattern.source_id!r} has invalid x/y arrays")
        order = np.argsort(x, kind="stable")
        x, y = x[order], y[order]
        unique_x, unique_index = np.unique(x, return_index=True)
        x, y = unique_x, y[unique_index]
        if (
            x.size < 2
            or float(x[0]) > float(grid.theta_min) + grid.step * 1e-2
            or float(x[-1]) < float(grid.theta_max) - grid.step * 1e-2
        ):
            raise ValueError(
                f"Pattern {pattern.source_id!r} does not cover checkpoint theta window "
                f"[{grid.theta_min}, {grid.theta_max}]"
            )
        aligned = x.size == theta.size and np.allclose(
            x, theta, rtol=0.0, atol=max(1e-5, grid.step * 1e-3)
        )
        y_target = y if aligned else np.interp(theta.astype(np.float64), x, y)
        ids.append(pattern.source_id)
        raw.append(np.asarray(y_target, dtype=np.float32))
        inputs.append(normalize_pxrd_sample_max(y_target))
    if not ids:
        raise ValueError("No patterns were supplied for embedding")
    return ids, np.stack(raw, axis=0), np.stack(inputs, axis=0)


@dataclass(frozen=True)
class CheckpointConfig:
    encoder_layers: int
    encoder_embed_dim: int
    encoder_ffn_embed_dim: int
    encoder_attention_heads: int
    max_seq_len: int
    signal_length: int
    patch_size: int
    dropout: float
    emb_dropout: float
    attention_dropout: float
    activation_dropout: float
    activation_fn: str
    post_ln: bool
    position_encoding_type: str
    rope_base: float = 10000.0
    rope_pct: float = 1.0
    rope_coord_mode: str = "index"
    pxrd_x_min: float = 5.0
    pxrd_x_max: float = 90.0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "CheckpointConfig":
        required = {
            "encoder_layers",
            "encoder_embed_dim",
            "encoder_ffn_embed_dim",
            "encoder_attention_heads",
            "max_seq_len",
            "signal_length",
            "patch_size",
            "dropout",
            "emb_dropout",
            "attention_dropout",
            "activation_dropout",
            "activation_fn",
            "post_ln",
            "position_encoding_type",
            "pxrd_x_min",
            "pxrd_x_max",
        }
        missing = sorted(required - set(mapping))
        if missing:
            raise KeyError(f"Checkpoint PXRD encoder config missing keys: {missing}")
        return cls(
            encoder_layers=int(mapping["encoder_layers"]),
            encoder_embed_dim=int(mapping["encoder_embed_dim"]),
            encoder_ffn_embed_dim=int(mapping["encoder_ffn_embed_dim"]),
            encoder_attention_heads=int(mapping["encoder_attention_heads"]),
            max_seq_len=int(mapping["max_seq_len"]),
            signal_length=int(mapping["signal_length"]),
            patch_size=int(mapping["patch_size"]),
            dropout=float(mapping["dropout"]),
            emb_dropout=float(mapping["emb_dropout"]),
            attention_dropout=float(mapping["attention_dropout"]),
            activation_dropout=float(mapping["activation_dropout"]),
            activation_fn=str(mapping["activation_fn"]),
            post_ln=bool(mapping["post_ln"]),
            position_encoding_type=str(mapping["position_encoding_type"]),
            rope_base=float(mapping.get("rope_base", 10000.0)),
            rope_pct=float(mapping.get("rope_pct", 1.0)),
            rope_coord_mode=str(mapping.get("rope_coord_mode", "index")),
            pxrd_x_min=float(mapping["pxrd_x_min"]),
            pxrd_x_max=float(mapping["pxrd_x_max"]),
        )

    @property
    def grid(self) -> PatternGrid:
        return PatternGrid(self.pxrd_x_min, self.pxrd_x_max, self.signal_length)


class CheckpointEmbedder:
    """Load the PXRD encoder lazily and extract CLS tokens."""

    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self._model: Any | None = None
        self._config: CheckpointConfig | None = None
        self._hyper_parameters: dict[str, Any] = {}

    @property
    def config(self) -> CheckpointConfig:
        self._ensure_loaded()
        assert self._config is not None
        return self._config

    @property
    def grid(self) -> PatternGrid:
        return self.config.grid

    @property
    def hyper_parameters(self) -> Mapping[str, Any]:
        self._ensure_loaded()
        return self._hyper_parameters

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        model, pxrd_config = load_pxrd_patch_transformer_checkpoint(
            self.checkpoint_path,
            map_location="cpu",
        )
        config = CheckpointConfig.from_mapping(pxrd_config)
        self._model = model
        self._config = config
        self._hyper_parameters = {"pxrd_encoder": dict(pxrd_config)}

    def embed(
        self,
        patterns: Iterable[PxrdPattern],
        *,
        theta: np.ndarray,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> EmbeddingOutput:
        self._ensure_loaded()
        assert self._model is not None
        ids, _raw, inputs = resample_patterns(patterns, self.grid)
        dense_theta = np.broadcast_to(self.grid.theta.reshape(1, -1), inputs.shape).astype(
            np.float32
        )
        if int(batch_size) < 1:
            raise ValueError("batch_size must be at least one")
        self._model.to(device)
        batches: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(ids), int(batch_size)):
                stop = min(start + int(batch_size), len(ids))
                y = torch.from_numpy(inputs[start:stop]).to(device=device, dtype=torch.float32)
                x = torch.from_numpy(dense_theta[start:stop]).to(device=device, dtype=torch.float32)
                output = self._model(y, pxrd_x_dense=x)
                if not isinstance(output, Mapping) or "cls_token" not in output:
                    raise ValueError("PXRD encoder output does not contain cls_token")
                batches.append(output["cls_token"].detach().cpu().numpy().astype(np.float32))
        return EmbeddingOutput(np.concatenate(batches, axis=0), self.grid.theta)


__all__ = [
    "CheckpointConfig",
    "CheckpointEmbedder",
    "EmbeddingAdapter",
    "EmbeddingOutput",
    "PatternGrid",
    "PxrdEmbeddingAdapter",
    "normalize_pxrd_sample_max",
    "resample_patterns",
]
