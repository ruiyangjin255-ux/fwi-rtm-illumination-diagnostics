from __future__ import annotations


def test_visual_score_ranks_lower_errors_and_higher_ssim_better():
    from fwi_visionfm.evaluation.visual_score import add_visual_scores

    rows = [
        {"bridge_name": "bad", "MAE": 10, "RMSE": 20, "SSIM": 0.1, "gradient_error": 5, "edge_MAE": 6},
        {"bridge_name": "good", "MAE": 5, "RMSE": 10, "SSIM": 0.9, "gradient_error": 2, "edge_MAE": 3},
    ]

    scored = add_visual_scores(rows)
    by_name = {row["bridge_name"]: row for row in scored}

    assert by_name["good"]["visual_score"] > by_name["bad"]["visual_score"]
    assert by_name["good"]["visual_score"] == 1.0
    assert by_name["bad"]["visual_score"] == 0.0
