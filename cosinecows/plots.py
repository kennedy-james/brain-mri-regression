import matplotlib.pyplot as plt
import seaborn as sns

from cosinecows.config import configs


def generate_plot(cv_df):
    """Generates a boxplot comparing training and validation scores from CV results.

    Parameters:
    ----------
    cv_df: DataFrame with CV results containing 'train_score' and 'validation_score' columns.

    Returns:
    -----------
    fig: Matplotlib figure object containing the boxplot.
    """
    fig, ax = plt.subplots(figsize=(11, 13))
    sns.boxplot(data=cv_df[["train_score", "validation_score"]], ax=ax)
    ax.set_title(
        f"CV Results: {configs['regression_method'].name} + {configs['outlier_detector']['method'].name} Detector + {'Feature Selection' if configs['selection']['is_enabled'] else 'No Feature Selection'} + {configs['impute_method'].name} Imputation")
    ax.set_ylabel("R² Score")
    ax.set_xlabel("Score Type")
    return fig
