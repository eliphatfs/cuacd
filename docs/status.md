# Current Status

## Working

- D&C hull, mesh volume, warp sort, plane cut (16 tests) — all tests pass.
- `kdop_hull_block` / `batch_kdop_hull_mesh` — 5 tests pass. Produces exact hull via extreme-point prefilter + D&C.
- `hausdorff_block` — 5 tests pass. Sampling-based bidirectional Hausdorff distance with linear BVH acceleration.
- `decompose_components_block` / `la_decompose_components` — 5 tests pass. Connected-components decomposition via GPU union-find. Integrated as optional post-processing pass in lookahead_decompose (decompose_components parameter). Inner shells (negative signed volume) are filtered out to prevent non-manifold output from downstream plane_cut.
- `lookahead_decompose` — cube, L-shape, octocat, convergence, 49160 tests pass. Uses full cost `max(rv, hausdorff)` for stopping criterion, rv-only for tree search. Default depth=2, quick_depth=0, width=60, width2=5.
- `la_find_concave_edges` — concave edge detection kernel. Generates cutting planes from concave mesh edges for first-layer expansion. Exposed via `n_concave_edges` parameter (default 32). Uses `heap_alloc`/`heap_free` from scratch heap for sort+reservoir scratch. 9 tests pass.
- **Merge-hulls postprocess pass** — port of CoACD's greedy hull merging. Optional (`merge_hulls=True` on `lookahead_decompose`, or `--merge-hulls` on the CLI). Six device kernels in `postprocess_merge.cu`: per-pair cost (rv quick-reject → bbox reject → concat + k-DOP hull → rv cost), separate Hausdorff pass on the filtered-concat mesh (coplanar tris dropped along the shared separating plane) against the merged hull, block-level greedy min-cost matching, apply kernel that builds merged mesh + adopts cached hull, and a single-block in-place compaction. All state lives on the GPU. Bench on bunny: overhead ≈0 ms, slight downward shift in part count (parts-mean 43.55–45.50 off → 43.15–44.85 on; time median 310 ms ≈ 309 ms).

## Known Limitations

- **Lookahead residual pool growth (~1 MB/call)**: After fixing two confirmed leaks (final decomp meshes never freed; seed buffer overwritten without refcount decrement before swap), pool usage still grows ~1 MB/call over 100 repeated calls. Cause unconfirmed — may be allocator bin fragmentation (varying chunk sizes → new slabs) or a remaining minor leak. Not yet experimentally distinguished.

## Not Yet Implemented

- `__cuda_array_interface__` support for GPU tensor input
- **Ternary search refinement** for lookahead cut selection: CoACD refines the MCTS-selected cut position via ternary search (`TernaryMCTS`, up to 10 iterations, epsilon=0.0001) to find the optimal cut within ±interval of the grid point. Our implementation uses only the fixed grid (width/3 cuts per axis). Adding refinement would let the algorithm find exact structural corners (e.g. the L-shape junction) instead of relying on the nearest grid point.
- **Centroid-based mesh volume**: `mesh_volume_warp` computes volume via divergence theorem relative to origin (`signed_tet_volume` sums tetrahedra formed with the origin). For non-watertight meshes (boundary edges from plane_cut ear-clipping giving up), this effectively connects open edges to the origin, introducing volume error proportional to the distance from origin to the hole. Computing relative to the mesh centroid instead would reduce this error since the implicit triangles closing the holes would be much smaller. Low priority — the main convergence issue is rv-only tree search, not volume accuracy.
