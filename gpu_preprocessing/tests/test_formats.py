"""Ada-ef binary format round-trips + fixtures for cpp/tests/test_formats.cpp (IMPLEMENTATION_PLAN.md §13.2)."""
import struct
from pathlib import Path

import numpy as np
import pytest

from prep.adaef_formats import read_estimator, read_samplings, write_estimator, write_samplings

FIXTURES = Path(__file__).resolve().parents[2] / "cpp" / "tests" / "fixtures"


def test_samplings_roundtrip_and_layout(tmp_path):
    q = np.arange(12, dtype=np.float32).reshape(3, 4)
    gt = np.array([[5, 6], [7, 8], [9, 10]], dtype=np.int32)
    p = tmp_path / "s.bin"
    write_samplings(p, q, gt)
    raw = p.read_bytes()
    assert struct.unpack("<ii", raw[:8]) == (3, 4)
    assert struct.unpack("<f", raw[8 + 4 * 4 : 8 + 5 * 4])[0] == 4.0  # row-major: row 1 starts at q[1,0]=4
    off = 8 + 12 * 4
    assert struct.unpack("<ii", raw[off : off + 8]) == (3, 2)
    assert len(raw) == 8 + 48 + 8 + 24
    q2, gt2 = read_samplings(p)
    np.testing.assert_array_equal(q, q2)
    np.testing.assert_array_equal(gt, gt2)


@pytest.mark.parametrize("metric,name", [("cd", "CosineDistanceEstimator"), ("ipd", "InnerProductEstimator")])
def test_estimator_roundtrip(tmp_path, metric, name):
    rng = np.random.default_rng(1)
    a = rng.standard_normal((50, 6))
    cov = np.cov(a, rowvar=False).astype(np.float32)
    mean = a.mean(0).astype(np.float32)
    p = tmp_path / "e.bin"
    write_estimator(p, metric, mean, cov)
    raw = p.read_bytes()
    assert struct.unpack("<Q", raw[:8])[0] == len(name) and raw[8 : 8 + len(name)].decode() == name
    assert len(raw) == 8 + len(name) + 8 + 36 * 4 + 4 + 6 * 4 + 4 + 6 * 4
    t, m2, c2, v2 = read_estimator(p)
    assert t == name
    np.testing.assert_array_equal(m2, mean)
    np.testing.assert_array_equal(c2, cov)
    np.testing.assert_array_equal(v2, np.diag(cov))


def test_rejects_l2(tmp_path):
    with pytest.raises(ValueError):
        write_estimator(tmp_path / "x.bin", "l2", np.zeros(2, np.float32), np.eye(2, dtype=np.float32))


def test_write_cpp_fixtures():
    """Deterministic fixtures consumed by cpp/tests/test_formats.cpp via the AUTHORS' deserializers."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    q = (np.arange(3 * 4, dtype=np.float32).reshape(3, 4) * 0.5)
    gt = (np.arange(3 * 2, dtype=np.int32).reshape(3, 2) + 100)
    write_samplings(FIXTURES / "samplings_3x4_k2.bin", q, gt)
    mean = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    cov = np.array([[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]], dtype=np.float32)
    write_estimator(FIXTURES / "estimator_cd_d3.bin", "cd", mean, cov)
    assert (FIXTURES / "samplings_3x4_k2.bin").exists() and (FIXTURES / "estimator_cd_d3.bin").exists()


def test_registry_has_new_datasets():
    from prep.registry import load_registry
    reg = load_registry()
    for ds in ["quora-384-angular", "bge-wiki-768-angular", "crawl-300-angular"]:
        assert ds in reg
        spec = reg[ds]
        assert spec.metric == "cd"
        assert spec.dim in (384, 768, 300)


def test_fasttext_vec_source_roundtrip(tmp_path):
    from prep.sources import FastTextVecSource
    import zipfile

    vec_path = tmp_path / "test.vec"
    lines = [
        "3 4\n",
        "word1 0.1 0.2 0.3 0.4\n",
        "word2 1.1 1.2 1.3 1.4\n",
        "word3 2.1 2.2 2.3 2.4\n",
    ]
    vec_path.write_text("".join(lines), encoding="utf-8")

    src = FastTextVecSource(vec_path)
    assert src.n == 3 and src.d == 4
    chunks = list(src.iter_chunks(2))
    assert len(chunks) == 2
    assert chunks[0][1].shape == (2, 4)
    assert chunks[1][1].shape == (1, 4)
    np.testing.assert_allclose(chunks[0][1][0], [0.1, 0.2, 0.3, 0.4], atol=1e-5)

    # Test zip support
    zip_path = tmp_path / "test.vec.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(vec_path, arcname="test.vec")

    src_zip = FastTextVecSource(zip_path)
    assert src_zip.n == 3 and src_zip.d == 4
    sampled = src_zip.take_by_index([0, 2])
    assert sampled.shape == (2, 4)
    np.testing.assert_allclose(sampled[1], [2.1, 2.2, 2.3, 2.4], atol=1e-5)
