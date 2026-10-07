"""PXRD diffraction basis library construction."""

from .artifact import CifMetadata, ClusterResult, PxrdPattern
from .build import build_library
from .clustering import (
    build_threshold_grid,
    cluster_embeddings,
    compute_cosine_distance_matrix,
    cut_linkage_at_count,
    cut_linkage_at_threshold,
    fit_average_linkage,
    l2_normalize_embeddings,
    select_cluster_medoids,
)
from .config import (
    ClusteringProfile,
    DedupProfile,
    FilterProfile,
    as_tuple,
    resolve_library_settings,
)
from .embedding import (
    CheckpointConfig,
    CheckpointEmbedder,
    EmbeddingAdapter,
    EmbeddingOutput,
    PatternGrid,
    PxrdEmbeddingAdapter,
    normalize_pxrd_sample_max,
)
from .filtering import deduplicate_records, evaluate_domain_filter
from .preselection import PreselectionProfile, evaluate_smact
from .simulation import GsasIISimulator, SimulationAdapter
from .validation import (
    CifRewriter,
    CifValidator,
    DefaultCifValidator,
    OversizedCifError,
    PymatgenCifRewriter,
    PymatgenCifValidator,
)

__all__ = [
    "CifMetadata",
    "CifRewriter",
    "CheckpointConfig",
    "CheckpointEmbedder",
    "ClusteringProfile",
    "ClusterResult",
    "CifValidator",
    "DedupProfile",
    "DefaultCifValidator",
    "EmbeddingAdapter",
    "EmbeddingOutput",
    "FilterProfile",
    "GsasIISimulator",
    "OversizedCifError",
    "PatternGrid",
    "PreselectionProfile",
    "PxrdEmbeddingAdapter",
    "PxrdPattern",
    "PymatgenCifRewriter",
    "PymatgenCifValidator",
    "SimulationAdapter",
    "as_tuple",
    "build_library",
    "build_threshold_grid",
    "cluster_embeddings",
    "compute_cosine_distance_matrix",
    "cut_linkage_at_count",
    "cut_linkage_at_threshold",
    "deduplicate_records",
    "evaluate_domain_filter",
    "evaluate_smact",
    "fit_average_linkage",
    "l2_normalize_embeddings",
    "normalize_pxrd_sample_max",
    "resolve_library_settings",
    "select_cluster_medoids",
]
