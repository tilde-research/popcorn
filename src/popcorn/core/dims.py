"""The canonical dim vocabulary and each name's default grid pool.

Like tags, this single table is what prevents near-duplicate names: reuse
before inventing, torch's names where torch has them (`normalized_shape`,
`vocab`, `out_channels`), no abbreviations. A size determined by another dim is
an expression in the annotation (`response+1`, `head_dim/2`,
`seq*num_householder`), never a name. `heads` when an op has one head count,
q_heads/kv_heads when query and kv counts differ. Plurals are counts, singulars
are extents.

A `test_shapes` entry overrides the pool only where the kernel demands it;
None means the default size ladder (`bench.grid.GRID`).
"""

from popcorn.core.constraints import Range

DIMS: dict[str, Range | set[int] | None] = {
    # data layout
    "batch": Range(1, 8),
    "seq": Range(2, 128),
    "tokens": None,
    "total": Range(8, 2048),
    "boundaries": Range(2, 9),
    "first": {9},
    "second": {9},
    # attention
    "heads": {4},
    "q_heads": {4},
    "kv_heads": {2},
    "head_dim": {64},
    "key_dim": {64},
    "value_dim": {64},
    "blocks": {2},
    "block_size": {16},
    "slots": {16},
    "levels": {7},
    "gate_dim": {128},
    # rope
    "cos_batch": {1},
    "half": {16},
    # feature widths
    "hidden": None,
    "intermediate": None,
    "out_features": None,
    "embedding_dim": {64},
    "normalized_shape": None,
    "vocab": Range(2, 4096),
    "channels": None,
    "out_channels": {4},
    "width": {16},
    "kernel_size": {5},
    "experts": {8},
    # matmul
    "rows": {17},
    "cols": {19},
    "inner": {16},
    # losses
    "response": {31, 128},
    "target_hidden": None,
    "teacher_hidden": None,
}
