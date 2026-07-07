"""
test_cardiogram.py
Тесты для темпоральной кардиограммы v0.5.0.
"""

import pytest
import sys
from pathlib import Path

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fusion import build_temporal_cardiogram, build_ascii_cardiogram


class TestTemporalCardiogram:
    """Тесты build_temporal_cardiogram."""
    
    def test_empty_results(self):
        """Пустой список результатов."""
        result = build_temporal_cardiogram([])
        assert "error" in result
        assert result["total_points"] == 0
    
    def test_single_point(self):
        """Одна точка данных."""
        results = [{"final_index": 0.5, "health_status": "stable", "syntax": {"type": "assertion"}}]
        result = build_temporal_cardiogram(results)
        
        assert result["total_points"] == 1
        assert result["mean"] == 0.5
        assert result["peak"] == 0.5
        assert result["valley"] == 0.5
        assert result["volatility"] == 0.0
        assert result["trend"] == "stable"
    
    def test_multiple_points(self):
        """Несколько точек данных."""
        results = [
            {"final_index": 0.2, "health_status": "stable", "syntax": {"type": "assertion"}},
            {"final_index": 0.5, "health_status": "high_stability", "syntax": {"type": "definition"}},
            {"final_index": -0.3, "health_status": "stable", "syntax": {"type": "rhetorical_question"}},
            {"final_index": -0.6, "health_status": "critical", "syntax": {"type": "counterfactual"}},
            {"final_index": 0.1, "health_status": "stable", "syntax": {"type": "assertion"}}
        ]
        result = build_temporal_cardiogram(results)
        
        assert result["total_points"] == 5
        assert len(result["timeline"]) == 5
        assert result["peak"] == 0.5
        assert result["valley"] == -0.6
        assert result["volatility"] == 1.1
    
    def test_critical_points_detection(self):
        """Обнаружение критических точек."""
        results = [
            {"final_index": -0.4, "health_status": "critical", "syntax": {"type": "rhetorical_question"}},
            {"final_index": 0.2, "health_status": "stable", "syntax": {"type": "assertion"}},
            {"final_index": -0.7, "health_status": "critical", "syntax": {"type": "counterfactual"}},
        ]
        result = build_temporal_cardiogram(results)
        
        assert 0 in result["critical_points"]
        assert 2 in result["critical_points"]
        assert 1 not in result["critical_points"]
        assert result["stability_break_points"] == [2]  # только -0.7 < -0.5
    
    def test_rising_trend(self):
        """Восходящий тренд."""
        results = [
            {"final_index": -0.5, "health_status": "critical", "syntax": {"type": "rhetorical_question"}},
            {"final_index": 0.0, "health_status": "stable", "syntax": {"type": "assertion"}},
            {"final_index": 0.5, "health_status": "high_stability", "syntax": {"type": "imperative"}},
        ]
        result = build_temporal_cardiogram(results)
        assert result["trend"] == "rising"
        assert result["slope"] > 0
    
    def test_falling_trend(self):
        """Нисходящий тренд."""
        results = [
            {"final_index": 0.5, "health_status": "high_stability", "syntax": {"type": "definition"}},
            {"final_index": 0.0, "health_status": "stable", "syntax": {"type": "assertion"}},
            {"final_index": -0.5, "health_status": "critical", "syntax": {"type": "rhetorical_question"}},
        ]
        result = build_temporal_cardiogram(results)
        assert result["trend"] == "falling"
        assert result["slope"] < 0
    
    def test_stable_trend(self):
        """Стабильный тренд."""
        results = [
            {"final_index": 0.1, "health_status": "stable", "syntax": {"type": "assertion"}},
            {"final_index": 0.15, "health_status": "stable", "syntax": {"type": "assertion"}},
            {"final_index": 0.12, "health_status": "stable", "syntax": {"type": "assertion"}},
        ]
        result = build_temporal_cardiogram(results)
        assert result["trend"] == "stable"
    
    def test_sharp_rise_pattern(self):
        """Обнаружение резкого подъёма."""
        results = [
            {"final_index": -0.5, "health_status": "critical", "syntax": {"type": "rhetorical_question"}},
            {"final_index": 0.5, "health_status": "high_stability", "syntax": {"type": "imperative"}},
        ]
        result = build_temporal_cardiogram(results)
        
        assert len(result["patterns"]) == 1
        assert result["patterns"][0]["type"] == "sharp_rise"
        assert result["patterns"][0]["delta"] == 1.0
    
    def test_sharp_drop_pattern(self):
        """Обнаружение резкого падения."""
        results = [
            {"final_index": 0.5, "health_status": "high_stability", "syntax": {"type": "imperative"}},
            {"final_index": -0.5, "health_status": "critical", "syntax": {"type": "rhetorical_question"}},
        ]
        result = build_temporal_cardiogram(results)
        
        assert len(result["patterns"]) == 1
        assert result["patterns"][0]["type"] == "sharp_drop"
    
    def test_transitions(self):
        """Обнаружение смен статусов."""
        results = [
            {"final_index": 0.5, "health_status": "high_stability", "syntax": {"type": "definition"}},
            {"final_index": 0.2, "health_status": "stable", "syntax": {"type": "assertion"}},
            {"final_index": -0.6, "health_status": "critical", "syntax": {"type": "rhetorical_question"}},
        ]
        result = build_temporal_cardiogram(results)
        
        assert len(result["transitions"]) == 2
        assert result["transitions"][0]["from_status"] == "high_stability"
        assert result["transitions"][0]["to_status"] == "stable"
        assert result["transitions"][1]["to_status"] == "critical"
    
    def test_missing_fields(self):
        """Отсутствие некоторых полей в результатах."""
        results = [
            {"final_index": 0.3},
            {"final_index": -0.2, "syntax": {}},
            {},
        ]
        result = build_temporal_cardiogram(results)
        
        assert result["total_points"] == 3
        assert result["timeline"] == [0.3, -0.2, 0.0]
        assert result["health_sequence"] == ["unknown", "unknown", "unknown"]


class TestAsciiCardiogram:
    """Тесты build_ascii_cardiogram."""
    
    def test_empty_data(self):
        """Пустые данные."""
        result = build_ascii_cardiogram({"timeline": [], "total_points": 0})
        assert "Нет данных" in result
    
    def test_single_point_ascii(self):
        """Одна точка в ASCII."""
        data = build_temporal_cardiogram([
            {"final_index": 0.5, "health_status": "stable", "syntax": {"type": "assertion"}}
        ])
        ascii_art = build_ascii_cardiogram(data)
        
        assert "ТЕМПОРАЛЬНАЯ КАРДИОГРАММА" in ascii_art
        assert "0.500" in ascii_art
        assert "stable" in ascii_art.lower() or "stable" in ascii_art
    
    def test_multiple_points_ascii(self):
        """Несколько точек в ASCII."""
        results = [
            {"final_index": 0.8, "health_status": "high_stability", "syntax": {"type": "imperative"}},
            {"final_index": -0.7, "health_status": "critical", "syntax": {"type": "rhetorical_question"}},
        ]
        data = build_temporal_cardiogram(results)
        ascii_art = build_ascii_cardiogram(data)
        
        assert "🟢" in ascii_art
        assert "🔴" in ascii_art
        assert "Легенда" in ascii_art


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
