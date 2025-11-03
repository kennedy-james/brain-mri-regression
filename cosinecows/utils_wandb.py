import wandb
from matplotlib import pyplot as plt

from cosinecows.config import configs
from cosinecows.modeling.train import run_cv_experiment
from cosinecows.plots import generate_plot


def log_results_to_wandb(cv_df, run):
    """Logs a CV result DataFrame to a wandb run.

    Parameters:
    ----------
    cv_df: DataFrame with CV results
    run: W&B run object

    Returns:
    -----------
    None
    """
    print("\nLogging results to W&B...")
    # Generate and log boxplot
    fig = generate_plot(cv_df)
    run.log({"CV_Boxplot": wandb.Image(fig)})
    plt.close(fig)
    # Store raw CV results in table
    cv_table = wandb.Table(dataframe=cv_df)
    run.log({"CV Results": cv_table})
    # Log summary statistics
    run.summary["mean_train_score"] = cv_df["train_score"].mean()
    run.summary["mean_validation_score"] = cv_df["validation_score"].mean()
    run.summary["std_train_score"] = cv_df["train_score"].std()
    run.summary["std_validation_score"] = cv_df["validation_score"].std()
    print("✅ W&B run complete.")

def sweep_train():
    """Train function for WandB sweep. Loads data, runs CV, logs metrics."""
    # Load data inside the sweep function to ensure isolation per run
    x_training_data = pd.read_csv("./data/X_train.csv", skiprows=1, header=None).values[:, 1:]
    y_training_data = (pd.read_csv("./data/y_train.csv", skiprows=1, header=None).values[:, 1:].ravel())

    run = wandb.init()
    config = wandb.config

    # Override GradientBoostingRegressor hyperparameters from sweep config
    configs['gb_n_estimators'] = config.n_estimators
    configs['gb_learning_rate'] = config.learning_rate
    configs['gb_max_depth'] = config.max_depth
    configs['gb_min_samples_split'] = config.min_samples_split

    # Optionally log the updated configs for inspection
    run.config.update(configs)

    print(f"🚀 Starting sweep run for GradientBoostingRegressor with params: {config}")

    cv_df = run_cv_experiment(x_training_data, y_training_data)

    # Compute and log the primary metric for sweep optimization
    mean_val_score = cv_df["validation_score"].mean()
    wandb.log({"val/r2": mean_val_score})

    # Log additional metrics
    wandb.log({
        "mean_train_r2": cv_df["train_score"].mean(),
        "std_val_r2": cv_df["validation_score"].std(),
        "std_train_r2": cv_df["train_score"].std(),
    })

    # Log detailed results
    log_results_to_wandb(cv_df, run)

    wandb.finish()
