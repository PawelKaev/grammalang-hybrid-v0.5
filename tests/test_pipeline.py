"""
test_pipeline.py
Юнит-тесты для всего пайплайна.
"""

import json
from pathlib import Path
import pytest
from src.fast_parser import fast_parse
from src.deep_interpreter import deep_interpret
from src.fusion import apply_fusion_layer

TEST_DATA = json.loads(Path("tests/expected_results.json").read_text(encoding="utf-8"))

@pytest.mark.parametrize("test_case", TEST_DATA)
def test_full_pipeline(test_case):
    text = test_case["text"]
    expected = test_case["expected"]
    syntax = fast_parse(text, use_cache=False)
    assert syntax["syntax_type"] == expected["syntax_type"]
    heidegger = deep_interpret(text, syntax, use_cache=False)
    if "hold_break_risk" in expected:
        assert heidegger.hold_break_risk == expected["hold_break_risk"]
    if "is_ironic" in expected:
        assert heidegger.is_ironic == expected["is_ironic"]
    if "reference" in expected:
        assert heidegger.reference == expected["reference"]
    fusion = apply_fusion_layer(syntax, heidegger)
    final_index = fusion["final_index"]
    assert expected["final_index_min"] <= final_index <= expected["final_index_max"]
    assert fusion["health_status"] == expected["health_status"]
    print(f"PASSED: {text[:40]}... -> final_index={final_index}, health={fusion['health_status']}")

def test_empty_text():
    syntax = fast_parse("", use_cache=False)
    assert syntax["syntax_type"] == "assertion"
    heidegger = deep_interpret("", syntax, use_cache=False)
    assert heidegger.dasein_mode == "unknown"
    fusion = apply_fusion_layer(syntax, heidegger)
    assert fusion["final_index"] == 0.0
