## 2024-05-15 - [Vectorizing Array Reductions]
**Learning:** In the `qwip-atlas` codebase, where layer activations can be thousands of dimensions (e.g. `d_mlp` around 14336), computing boolean masks and taking array reductions (`mean`, `std`) inside a Python `for` loop over dimensions causes massive slow-downs (from 0.3s up to 10s per call).
**Action:** Always prioritize calculating aggregations and slice means over the entire tensor dimension across all rows prior to looping through individual rows, thereby doing operations once via NumPy's highly-optimized C backend.

## 2026-07-07 - [Vectorize and Batch Slicing]
**Learning:** Slicing tensors down to required subsets (like the last token) *before* applying expensive activation functions or reshapes, and moving tensors to CPU over entire batches instead of iteratively over rows, eliminates massive overhead in extraction pipelines.
**Action:** Always slice tensors to minimal required shape on GPU before applying activations/transformations, and process/transfer to CPU batched over rows instead of single rows within loops.
