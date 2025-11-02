import wandb
from matplotlib import pyplot as plt

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

