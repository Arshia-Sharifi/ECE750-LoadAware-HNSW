# third_party

Add the Ada-ef authors' hnswlib fork as a pinned submodule (IMPLEMENTATION_PLAN.md §6, P0.2):

```bash
git submodule add https://github.com/chaozhang-cs/hnsw-ada-ef cpp/third_party/hnsw-ada-ef
cd cpp/third_party/hnsw-ada-ef && git checkout <commit>   # record the hash here
```

Pinned commit: `ed463f9993868f7ecc7c103920644e7f94abb377` (verified by `test_backend_parity`)

Rules:
- Never edit files in `hnsw-ada-ef/`.
- Do not use its `CMakeLists.txt` (it hard-codes local paths). Only add its root to the include path, so headers are reached as `hnswlib/...`.
- Only `cpp/src/adaef_backend.cpp` may include `hnswlib/*.h`, plus `cpp/tests/test_backend_parity.cpp`, which includes `adaef_backend.cpp` as a unity TU (plan §0, rule 3).
