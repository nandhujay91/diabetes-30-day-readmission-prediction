from pathlib import Path

import pandas as pd


def main():

    project_path = Path(__file__).resolve().parents[2]

    report_folder = project_path / "reports" / "model_evaluation"

    result_files = [
        "logistic_regression_comparison.csv",
        "random_forest_results.csv",
        "xgboost_results.csv",
        "lightgbm_results.csv",
    ]

    results = []

    for file in result_files:
        file_path = report_folder / file

        if file_path.exists():
            result = pd.read_csv(file_path)
            results.append(result)
            print(f"Loaded: {file}")

        else:
            print(f"File not found: {file}")

    if not results:
        print("\nNo model results found.")
        return

    comparison = pd.concat(results, ignore_index=True)

    comparison = comparison[["model", "pr_auc", "roc_auc", "brier_score"]]

    comparison = comparison.sort_values(
        by="pr_auc",
        ascending=False,
    )

    comparison = comparison.reset_index(drop=True)

    comparison.insert(
        0,
        "rank",
        range(1, len(comparison) + 1),
    )

    comparison.to_csv(
        report_folder / "model_comparison.csv",
        index=False,
    )

    print("\nMODEL COMPARISON")
    print(comparison.round(4).to_string(index=False))

    print("\nComparison saved in:", report_folder)


if __name__ == "__main__":
    main()
