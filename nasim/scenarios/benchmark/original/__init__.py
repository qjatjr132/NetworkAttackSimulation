import os.path as osp

from nasim.scenarios.benchmark.new.generated import AVAIL_GEN_BENCHMARKS

BENCHMARK_DIR = osp.dirname(osp.abspath(__file__))

AVAIL_STATIC_BENCHMARKS = {
    "tiny": {
        "file": osp.join(BENCHMARK_DIR, "tiny.yaml"),
        "name": "tiny",
        "step_limit": 1000,
        "max_score": 195
    },
    "tiny-hard": {
        "file": osp.join(BENCHMARK_DIR, "tiny-hard.yaml"),
        "name": "tiny-hard",
        "step_limit": 1000,
        "max_score": 192
    },
    "tiny-small": {
        "file": osp.join(BENCHMARK_DIR, "tiny-small.yaml"),
        "name": "tiny-small",
        "step_limit": 1000,
        "max_score": 189
    },
    "tiny_ct": {
        "file": osp.join(BENCHMARK_DIR, "tiny_ct.yaml"),
        "name": "tiny",
        "step_limit": 1000,
        "max_score": 195
    },
    "tiny_kbs": {
        "file": osp.join(BENCHMARK_DIR, "tiny_kbs.yaml"),
        "name": "tiny",
        "step_limit": 1000,
        "max_score": 195
    },
    "small": {
        "file": osp.join(BENCHMARK_DIR, "small.yaml"),
        "name": "small",
        "step_limit": 1000,
        "max_score": 186
    },
    "small-honeypot": {
        "file": osp.join(BENCHMARK_DIR, "small-honeypot.yaml"),
        "name": "small-honeypot",
        "step_limit": 1000,
        "max_score": 186
    },
    "small-linear": {
        "file": osp.join(BENCHMARK_DIR, "small-linear.yaml"),
        "name": "small-linear",
        "step_limit": 1000,
        "max_score": 187
    },
    "medium": {
        "file": osp.join(BENCHMARK_DIR, "medium.yaml"),
        "name": "medium",
        "step_limit": 2000,
        "max_score": 190
    },
    "medium-single-site": {
        "file": osp.join(BENCHMARK_DIR, "medium-single-site.yaml"),
        "name": "medium-single-site",
        "step_limit": 2000,
        "max_score": 195
    },
    "medium-multi-site": {
        "file": osp.join(BENCHMARK_DIR, "medium-multi-site.yaml"),
        "name": "medium-multi-site",
        "step_limit": 2000,
        "max_score": 190
    },
    "smai_gg": {
        "file": osp.join(BENCHMARK_DIR, "smai_gg.yaml"),
        "name": "smai",
        "step_limit": 1000,
        "max_score": 196
    },
    "smai": {
        "file": osp.join(BENCHMARK_DIR, "smai.yaml"),
        "name": "smai",
        "step_limit": 1000,
        "max_score": 196
    },
    "smai_final": {
        "file": osp.join(BENCHMARK_DIR, "smai_final.yaml"),
        "name": "smai_final",
        "step_limit": 1000,
        "max_score": 196
    },
    "smai_tester": {
        "file": osp.join(BENCHMARK_DIR, "smai_pentesterchange.yaml"),
        "name": "smai_tester",
        "step_limit": 1000,
        "max_score": 196
    },
    "kjhbs": {
        "file": osp.join(BENCHMARK_DIR, "kjhbs.yaml"),
        "name": "kjhbs",
        "step_limit": 2000,
        "max_score": 190
    },
    "tiny_alpha": {
        "file": osp.join(BENCHMARK_DIR, "tiny_alpha.yaml"),
        "name": "tiny_alpha",
        "step_limit": 1000,
        "max_score": 195
    },
    "tiny_betta": {
        "file": osp.join(BENCHMARK_DIR, "tiny_betta.yaml"),
        "name": "tiny_betta",
        "step_limit": 1000,
        "max_score": 195
    },
    "tiny_gamma": {
        "file": osp.join(BENCHMARK_DIR, "tiny_gamma.yaml"),
        "name": "tiny_gamma",
        "step_limit": 1000,
        "max_score": 195
    },
    "tiny_final": {
        "file": osp.join(BENCHMARK_DIR, "tiny_final.yaml"),
        "name": "tiny_final",
        "step_limit": 1000,
        "max_score": 195
    },
}

AVAIL_BENCHMARKS = list(AVAIL_STATIC_BENCHMARKS.keys()) \
                    + list(AVAIL_GEN_BENCHMARKS.keys())
