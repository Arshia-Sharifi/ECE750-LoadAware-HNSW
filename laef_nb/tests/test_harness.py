import itertools
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import laef_nb as nb  # noqa: E402

DATASETS = ("glove-100-angular", "lastfm-64-dot", "deep-image-96-angular")
THREADS = (2, 4, 8, 16, 24)
RHOS = (0.6, 0.8, 1.0, 1.2, 1.3)
FRACTIONS = (0.95, 0.9, 0.8, 0.75)


def frame(datasets=DATASETS, threads=THREADS, rhos=RHOS, fractions=FRACTIONS, repeats=3,
          policies=("laef_uniform", "laef_wf"), profile="constant", with_baseline=True):
    rng = np.random.default_rng(0)
    rows = []
    for ds, t, r, rep in itertools.product(datasets, threads, rhos, range(repeats)):
        common = dict(dataset=ds, threads=t, rho=r, repeat=rep, profile=profile,
                      achieved_qps=100 * t, offered_qps=100 * t, goodput_qps=90 * t,
                      mean_ndis=1000, generator_lag_p99_us=50 + t, frac_full_probe=1.0)
        if with_baseline:
            rows.append(dict(common, policy="ada_ef", slo_fraction=1.0, slo_ms=1.0,
                             run_id=f"{ds}_ada_c{t}_{r}_{rep}_{profile}", p95_ms=1 + r,
                             mean_recall=0.99, p5_recall=0.9, p1_recall=0.8,
                             slo_attainment=max(0.0, 1.4 - r), mean_ef_ratio=1.0,
                             forced_floor_rate=0.0, floor_mode_query_rate=0.0))
        for pol, f in itertools.product(policies, fractions):
            rows.append(dict(common, policy=pol, slo_fraction=f, slo_ms=f,
                             run_id=f"{ds}_{pol}_c{t}_{r}_{f}_{rep}_{profile}",
                             p95_ms=f * (0.8 + 0.5 * r) + rng.normal(0, 0.01),
                             mean_recall=0.95 - 0.05 * r + (0.01 if pol == "laef_wf" else 0.0),
                             p5_recall=0.8, p1_recall=0.6, slo_attainment=min(1.0, 1.5 - r),
                             mean_ef_ratio=0.8, forced_floor_rate=0.1, floor_mode_query_rate=0.05))
    df = pd.DataFrame(rows)
    df["label"] = df.policy.map(nb.config.POLICY_LABEL).fillna(df.policy)
    return df


def study(**kw):
    base = dict(name="test", datasets=DATASETS, threads=THREADS, rhos=RHOS,
                slo_fractions=FRACTIONS, repeats=3)
    return nb.Study(**{**base, **kw})


def test_run_count_matches_the_runner():
    s = study()
    assert s.n_runs == len(frame())
    assert len(nb.experiments.plan(s)) == s.n_runs


def test_generated_yaml_matches_the_runner():
    sys.path.insert(0, str(nb.config.SCRIPTS))
    import run_experiments as rx

    s = study()
    spec = nb.experiments.spec(s)
    runs = [r for key, sec in spec.items() for r in rx.build_runs(key, sec)]
    assert len(runs) == s.n_runs
    assert len({(r.experiment, r.run_id) for r in runs}) == s.n_runs
    assert {r.axis for r in runs} == {"base"}


def test_paired_does_not_duplicate_across_arrival_profiles():
    df = pd.concat([frame(profile="constant"), frame(profile="poisson")], ignore_index=True)
    pair = nb.collect.paired(df)
    assert len(pair) == len(df[df.policy != "ada_ef"])
    assert pair.ada_mean_recall.notna().all()
    assert len(nb.collect.allocator_delta(df)) == len(pair) // 2


def test_duplicate_baselines_raise_instead_of_misleading():
    df = pd.concat([frame(), frame()], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        nb.collect.paired(df)


def test_missing_baseline_warns_and_keeps_going(capsys):
    df = frame()
    df = df[~((df.policy == "ada_ef") & (df.threads == 2))]
    pair = nb.collect.paired(df)
    assert "no ada_ef baseline" in capsys.readouterr().out
    assert pair[pair.threads == 2].recall_loss_pts.isna().all()
    assert pair[pair.threads != 2].recall_loss_pts.notna().all()


def test_figures_do_not_average_over_slo_fractions():
    df = frame()
    kept, frac = nb.figures.pick_fraction(df)
    assert frac == max(FRACTIONS)
    la = kept[kept.policy != "ada_ef"]
    assert la.slo_fraction.nunique() == 1


def test_all_figures_covers_every_dataset(tmp_path):
    s = study()
    object.__setattr__(s, "name", "test")
    df = frame()
    paths = nb.figures.all_figures(pd.DataFrame(), df, nb.collect.paired(df),
                                   nb.collect.allocator_delta(df), s)
    names = [p.name for p in paths]
    for ds in DATASETS:
        assert any(ds in n and n.startswith("06_") for n in names)
        assert any(ds in n and n.startswith("07_") for n in names)


@pytest.mark.parametrize("kw", [
    dict(datasets=("glove-100-angular",), threads=(4,), rhos=(1.0,), fractions=(0.9,), repeats=1),
    dict(policies=("laef_wf",)),
    dict(with_baseline=False),
])
def test_figures_survive_degenerate_grids(kw):
    s = study(name="degenerate")
    df = frame(**kw)
    pair = nb.collect.paired(df) if kw.get("with_baseline", True) else pd.DataFrame()
    delta = nb.collect.allocator_delta(df)
    nb.figures.all_figures(pd.DataFrame(), df, pair, delta, s)


def test_figures_survive_an_all_nan_metric():
    s = study(name="degenerate")
    df = frame(datasets=("glove-100-angular",), threads=(4,), rhos=(1.0,), repeats=1)
    df["slo_attainment"] = np.nan
    nb.figures.slo_heatmap(df, s, dataset="glove-100-angular")
