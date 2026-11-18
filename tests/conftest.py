# -*- coding: utf-8 -*-
"""pytest 配置：注册 --run-loocv，默认跳过较慢的留一评测。"""
from __future__ import annotations


def pytest_addoption(parser):
    parser.addoption(
        "--run-loocv",
        action="store_true",
        default=False,
        help="运行较慢的留一交叉验证泛化评测（tests/test_loocv_metrics.py）",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-loocv"):
        return
    skip_loocv = __import__("pytest").mark.skip(reason="需要 --run-loocv 才运行（留一评测较慢）")
    for item in items:
        if "loocv" in item.keywords:
            item.add_marker(skip_loocv)
