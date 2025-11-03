import os.path
import joblib
import wandb
import numpy as np
import pandas as pd
import optuna
import json
from cosinecows.config import RUNNING_MODE, configs, RunMode, Imputer, OutlierDetector, Regressor
from cosinecows.dataset import load_train_data, load_test_data, MODELS_DIR, REGRESSORS_DIR
from cosinecows.dataset import RAW_DATA_DIR, PROCESSED_DATA_DIR
from cosinecows.io import load_best_params, save_results_locally
from cosinecows.modeling.train import train_model, run_cv_experiment
from cosinecows.utils_optuna import objective_stacker, objective
from cosinecows.utils_wandb import log_results_to_wandb, sweep_train

print('Loading training data (optuna global access)...')
x_train, y_train = load_train_data()


def run_final_evaluation():
    # Generates submission.csv using the single configuration defined in the global 'configs' dict
    print(f"🚀 Running final evaluation pipeline with:")
    print(f"Loading configuration from: {configs['optuna']['load_file']}...")
    load_best_params(json_file=configs['optuna']['load_file'])

    print(f"   Final Model: {configs['regression_method'].name}")
    print(f"   Final Outlier Detector: {configs['outlier_detector']['method'].name}")
    print(f"   Final Selection Percentile: {configs['selection'].get('percentile', 'N/A')}")
    # --- END NEW ---

    x_test = load_test_data()
    final_pipeline_path = REGRESSORS_DIR / 'impKnn-outlierPcaIsofor-featSelect-stacking.pkl'
    if os.path.isfile(final_pipeline_path):
        print(f"\nLoading pre-trained pipeline from {final_pipeline_path}...")
        pipeline_components = joblib.load(final_pipeline_path)
        imputer = pipeline_components['imputer']
        selection = pipeline_components['selection']
        model = pipeline_components['model']
        print("✅ Pipeline loaded from disk.")

    else:
        print(f"\nNo pre-trained pipeline found. Training model from scratch...")
        imputer, detector, selection, model, _, _ = train_model(x_train, y_train, i=None)
        print(f"\nSaving trained pipeline to {final_pipeline_path}...")
        pipeline_components = {
            'imputer': imputer,
            'selection': selection,
            'model': model
        }
        joblib.dump(pipeline_components, final_pipeline_path)
        print("✅ Pipeline trained and saved.")

    print("\nGenerating predictions on test data...")
    x_test_imputed = imputer.transform(x_test)
    x_test_selected = selection.transform(x_test_imputed)  # apply feature selection, NO outlier removal
    y_test_pred = model.predict(x_test_selected)

    # Save predictions to submission file
    table = pd.DataFrame({"id": np.arange(0, y_test_pred.shape[0]), "y": y_test_pred.flatten()})
    table.to_csv(PROCESSED_DATA_DIR / "submission.csv", index=False)
    print("\n✅ Successfully generated submission.csv")


def run_wandb():
    # Runs a single CV experiment (using 'configs') and logs to W&B.
    print(f"🚀 Starting W&B run for: {configs['regression_method']} + {configs['outlier_detector']['method']}")
    with wandb.init(
            project="AML_task1",
            config=configs,
            tags=["regression", configs["regression_method"], configs["outlier_detector"]['method']],
            name=f"regressor {configs['regression_method']}_{configs['outlier_detector']['method']}",
            notes=f''
    ) as run:
        cv_df = run_cv_experiment(x_train, y_train)
        log_results_to_wandb(cv_df, run)

def run_wandb_sweep():
    sweep_config = {
        'method': 'random',  # Options: 'random', 'grid', 'bayes'
        'metric': {
            'name': 'val/r2',
            'goal': 'maximize'
        },
        'parameters': {
            'n_estimators': {
                'distribution': 'int_uniform',
                'min': 500,
                'max': 2000
            },
            'learning_rate': {
                'distribution': 'uniform',
                'min': 0.01,
                'max': 0.3
            },
            'max_depth': {
                'distribution': 'int_uniform',
                'min': 3,
                'max': 6
            },
            'min_samples_split': {
                'distribution': 'int_uniform',
                'min': 2,
                'max': 20
            }
            # Add more parameters as needed, e.g., 'subsample': {'min': 0.8, 'max': 1.0}
        },
        'project': 'AML_task1',
        'tags': ['sweep', 'GradientBoostingRegressor']
    }

    # Create the sweep
    sweep_id = wandb.sweep(sweep_config)

    # Run the agent (adjust 'count' to the number of runs you want)
    print("🚀 Starting WandB sweep agent...")
    wandb.agent(sweep_id, function=sweep_train, count=50)  # Run 50 trials; adjust as needed



def run_current_config():
    print(
        f"🚀 Starting single local CV run for: {configs['regression_method'].name} + {configs['outlier_detector']['method'].name}")
    cv_df = run_cv_experiment(x_train, y_train)
    save_results_locally(cv_df, is_grouped_run=False)  # Use the helper


def run_grid():
    # Runs all combinations of models and outlier detectors locally. Saves one CSV and one plot with all results.
    print("🚀 Starting local 'Run All' comparison...")
    all_results_dfs = []

    for model_name in Regressor:
        configs["regression_method"] = model_name  # !update global config

        # for outlier_method in OutlierDetector:
        # configs["outlier_detection"] = outlier_method  # !update global config
        cv_df = run_cv_experiment(x_train, y_train)
        all_results_dfs.append(cv_df)

    results_df = pd.concat(all_results_dfs)
    save_results_locally(results_df, is_grouped_run=True)


def run_optuna_search():
    print("🚀 Starting Optuna hyperparameter search...")
    storage_name = 'sqlite:///optuna_study.db'

    if configs['optuna']['objective_to_run'] == 'stacker':
        study_name = 'aml-stacker-pipeline'
        objective_func = objective_stacker
        print(f"Tuning STACKER PIPELINE (study: {study_name})")
    else:
        study_name = 'aml-xgb-single'
        objective_func = objective
        print(f"Tuning SINGLE XGB (study: {study_name})")

    study = optuna.create_study(storage=storage_name, study_name=study_name, direction='maximize', load_if_exists=True)
    # run 50 different trials. may take long
    study.optimize(
        lambda trial: objective_func(trial, x_train, y_train),
        n_trials=50
    )

    print("\n\n--- 🏆 Optuna Search Complete ---")
    print(f'   study: {study.study_name}')
    print(f"Best Validation R²: {study.best_value:.4f}")
    print("Best Parameters:")
    print(study.best_params)

    # Save best params to a file so you can use them later
    best_params_file = f"best_params_{configs['optuna']['objective_to_run']}.json"
    with open(best_params_file, "w") as f:
        json.dump(study.best_params, f, indent=4)
    print(f"\n✅ Best parameters saved to {best_params_file}")


def run_optuna_config():
    load_best_params(json_file=configs['optuna']['load_file'])
    print(
        f"🚀 Starting single local CV run for: {configs['regression_method'].name} + {configs['outlier_detector']['method'].name}")
    cv_df = run_cv_experiment(x_train, y_train)
    save_results_locally(cv_df, is_grouped_run=False)


def run(mode: RunMode):
    match mode:
        case RunMode.final_evaluation:
            run_final_evaluation()
        case RunMode.wandb:
            run_wandb()
        case RunMode.current_config:
            run_current_config()
        case RunMode.grid:
            run_grid()
        case RunMode.optuna_search:
            run_optuna_search()
        case RunMode.optuna_config:
            run_optuna_config()
