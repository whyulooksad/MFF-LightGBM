import pandas as pd

from developer.evaluation.evaluate import evaluate


def test_evaluate_uses_matching_numeric_label_columns(tmp_path):
    predictions = tmp_path / "predictions.csv"
    output = tmp_path / "metrics.json"
    pd.DataFrame({
        "true_label": [0, 1],
        "true_label_name": ["benign", "adware"],
        "pred_label": [0, 1],
        "pred_label_name": ["benign", "adware"],
    }).to_csv(predictions, index=False)
    metrics = evaluate(predictions, output)
    assert metrics["accuracy"] == 1.0
    assert output.exists()
