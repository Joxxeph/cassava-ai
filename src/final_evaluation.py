import numpy as np
import pandas as pd


def validate_final_results(
    dataframe,
    architectures,
    test_images=3210
):

    expected_models = set(architectures) | {
        "heterogeneous_ensemble"
    }

    assert len(dataframe) == 12

    assert not dataframe.duplicated(
        subset=["model", "approach"]
    ).any()

    assert (
        dataframe["num_samples"] == test_images
    ).all()

    for approach in [
        "centralized",
        "federated"
    ]:

        subset = dataframe[
            dataframe["approach"] == approach
        ]

        assert set(
            subset["model"]
        ) == expected_models

    metrics = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "balanced_accuracy"
    ]

    for metric in metrics:

        assert np.isfinite(
            dataframe[metric]
        ).all()

        assert dataframe[metric].between(
            0, 1
        ).all()

    return True


def compare_approaches(dataframe):

    metrics = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted"
    ]

    centralized = dataframe[
        dataframe["approach"] == "centralized"
    ]

    federated = dataframe[
        dataframe["approach"] == "federated"
    ]

    comparison = centralized[
        ["model", *metrics]
    ].merge(

        federated[["model", *metrics]],

        on="model",

        suffixes=(
            "_centralized",
            "_federated"
        ),

        validate="one_to_one"

    )

    for metric in metrics:

        comparison[f"{metric}_difference_pp"] = (

            comparison[f"{metric}_federated"]

            - comparison[f"{metric}_centralized"]

        ) * 100

    return comparison


def summarize_approaches(dataframe):

    results = []

    for approach in [
        "centralized",
        "federated"
    ]:

        subset = dataframe[
            dataframe["approach"] == approach
        ]

        individual = subset[
            subset["model"]
            != "heterogeneous_ensemble"
        ]

        ensemble = subset[
            subset["model"]
            == "heterogeneous_ensemble"
        ].iloc[0]

        best = individual.loc[
            individual["f1_macro"].idxmax()
        ]

        results.append({

            "approach": approach,

            "best_individual_model":
                best["model"],

            "best_individual_f1":
                float(best["f1_macro"]),

            "mean_individual_f1":
                float(
                    individual["f1_macro"].mean()
                ),

            "ensemble_f1":
                float(ensemble["f1_macro"]),

            "ensemble_accuracy":
                float(ensemble["accuracy"]),

            "ensemble_gain_pp":
                float(
                    (
                        ensemble["f1_macro"]
                        - best["f1_macro"]
                    ) * 100
                )

        })

    return pd.DataFrame(results)
