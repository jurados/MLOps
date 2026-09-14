from kfp.dsl import Dataset, Input, Metrics, Model, Output, component


@component(
    # Esta imagen que queremos usasr
    base_image="gcr.io/deeplearning-platform-release/tf2-cpu.2-6:latest",
    packages_to_install=[
        # Paquetes que queremos usar/instalar
        "pandas==1.3.5",
        "joblib==1.1.0",
    ],
)
def choose_best_model(
    test_dataset: Input[Dataset],
    decision_tree_model: Input[Model],
    random_forest_model: Input[Model],
    metrics: Output[Metrics],
    best_model: Output[Model],
):
    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score

    test_data = pd.read_csv(test_dataset.path)

    dt = joblib.load(decision_tree_model.path)
    rf = joblib.load(random_forest_model.path)

    dt_pred = dt.predict(test_data.drop("Species", axis=1))
    rf_pred = rf.predict(test_data.drop("Species", axis=1))

    dt_accuracy = accuracy_score(test_data["Species"], dt_pred)
    rf_accuracy = accuracy_score(test_data["Species"], rf_pred)

    metrics.log_metrics("Decision Tree (Accuracy)", (dt_accuracy))
    metrics.log_metrics("Random FOrest (Accuracy)", (rf_accuracy))

    joblib.dump(dt, best_model.path) if dt_accuracy > rf_accuracy else joblib.dump(
        rf, best_model.path
    )
