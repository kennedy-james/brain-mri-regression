import json

from matplotlib import pyplot as plt
import seaborn as sns

from cosinecows.config import configs, Regressor, OutlierDetector, Imputer
from cosinecows.dataset import REPORTS_DIR


def load_best_params(json_file="best_params.json"):
    """Loads the best parameters from the Optuna JSON file.

    Parameters:
    ----------
    json_file: Path to the JSON file containing best parameters.

    Returns:
    ----------
    None
    """
    try:
        with open(json_file, "r") as f:
            best_params = json.load(f)
            print(f"✅ Successfully loaded best parameters from {json_file}")

        # imputer and outlier params
        if 'impute_method' in best_params:
            configs['impute_method'] = Imputer[best_params['impute_method']]

        if 'outlier_method_name' in best_params:
            outlier_method = OutlierDetector[best_params['outlier_method_name']]
            configs['outlier_method'] = outlier_method
            if outlier_method == OutlierDetector.isoforest and 'isoforest_contamination' in best_params:
                configs['isoforest_contamination'] = best_params['isoforest_contamination']
            elif outlier_method == OutlierDetector.zscore and 'zscore_std' in best_params:
                configs['zscore_std'] = best_params['zscore_std']

        # stacked ensemble params
        if 'selection_percentile' in best_params:
            configs['selection_percentile'] = best_params['selection_percentile']
            print(f"   Loaded selection_percentile: {best_params['selection_percentile']}")

        if 'pca_isoforest_contamination' in best_params:
            configs['pca_isoforest_contamination'] = best_params['pca_isoforest_contamination']
            print(f"   Loaded pca_isoforest_contamination: {best_params['pca_isoforest_contamination']}")

        if 'pca_n_components' in best_params:
            configs['pca_n_components'] = best_params['pca_n_components']
            print(f"   Loaded pca_n_components: {best_params['pca_n_components']}")

        # xgb specific params if present
        if 'n_estimators' in best_params and 'max_depth' in best_params:
            print("   Loading tuned XGBoost parameters...")
            configs['regression_method'] = Regressor.xgb  # It was tuned for XGB
            configs['regression_params'] = {
                'random_state': configs["random_state"],
                'n_estimators': best_params['n_estimators'],
                'max_depth': best_params['max_depth'],
                'min_child_weight': best_params['min_child_weight'],
                'gamma': best_params['gamma'],
                'subsample': best_params['subsample'],
                'colsample_bytree': best_params['colsample_bytree'],
                'reg_alpha': best_params['reg_alpha'],
                'reg_lambda': best_params['reg_lambda'],
                'learning_rate': best_params['learning_rate'],
                'verbosity': 0
            }

        # if stacker params are loaded, set regressor to stacking ensemble
        elif 'selection_percentile' in best_params:
            configs['regression_method'] = Regressor.stacking
            print(f"   Set regression_method to: {configs['regression_method'].name}")


    except FileNotFoundError:
        print(f"⚠️ WARNING: {json_file} not found. Using default configs.")
    except Exception as e:
        print(f"⚠️ ERROR loading {json_file}: {e}. Using default configs.")



def save_results_locally(results_df, is_grouped_run):
    """Saves a results DataFrame locally to CSV and creates a boxplot.

    Parameters:
    ----------
    results_df: DataFrame with CV results
    is_grouped_run: Boolean indicating if multiple model/outlier combinations are included.

    Returns:
    -----------
    None
    """
    print("\n\n--- 📊 Final Performance Summary ---")

    # save csv
    csv_filename = REPORTS_DIR / "cv_run_results.csv"
    if is_grouped_run:
        csv_filename = REPORTS_DIR / "cv_run_results_all.csv"

    results_df.to_csv(csv_filename, index=False)
    print(f"\n✅ All results saved to '{csv_filename}'")

    print("\n--- Validation R² Summary ---")
    if is_grouped_run:
        summary_stats = results_df.groupby(['model', 'outlier_method'])['validation_score'].describe()
    else:
        summary_stats = results_df['validation_score'].describe()
    print(summary_stats)

    # generate boxplot
    fig, ax = plt.subplots(figsize=(14, 8))
    plot_filename = "cv_run_boxplot.png"

    if is_grouped_run:
        sns.boxplot(data=results_df, x='outlier_method', y='validation_score', hue='model', ax=ax)
        ax.set_title("Model Comparison by Outlier Method (Validation R²)")
        ax.set_xlabel("Outlier Detection Method")
        ax.legend(title="Model")
        plot_filename = "cv_run_boxplot_all.png"
    else:
        sns.boxplot(data=results_df[["train_score", "validation_score"]], ax=ax)
        ax.set_title(f"CV Results: {configs['regression_method'].name} + {configs['outlier_method'].name} Detector + {'Feature Selection' if configs['selection_is_enabled'] else 'No Feature Selection'} + {configs['impute_method'].name} Imputation")
        ax.set_xlabel("Score Type")

    ax.set_ylabel("R² Score")
    plt.savefig(plot_filename)
    print(f"\n✅ Saved boxplot to '{plot_filename}'")
    print("\n✅ Local run complete.")
