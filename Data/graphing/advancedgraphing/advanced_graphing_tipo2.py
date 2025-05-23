import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import logging
import scipy.stats as stats
from dataclasses import dataclass
from typing import Callable, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PlotConfig:
    style: str = 'default'
    figsize: tuple = (10, 6)
    color_theme: str = 'viridis'

# Global configuration instance
plot_cfg = PlotConfig()

def set_plot_config(style: str = 'default', figsize: tuple = (10, 6), color_theme: str = 'viridis') -> None:
    """Set global configuration for plotting."""
    global plot_cfg
    if style not in plt.style.available:
        logger.warning(f"Invalid style '{style}' provided. Defaulting to 'default'.")
        style = 'default'
        
    plot_cfg.style = style
    plot_cfg.figsize = figsize
    plot_cfg.color_theme = color_theme
    plt.style.use(style)
    
    # Validate color palette
    if color_theme in sns.palettes.SEABORN_PALETTES.keys():
        sns.set_palette(color_theme)
    else:
        logger.warning(f"Invalid color palette '{color_theme}' provided. Defaulting to 'viridis'.")
        sns.set_palette('viridis')

def get_available_plot_styles() -> list:
    """Returns a list of available matplotlib styles."""
    return plt.style.available

def apply_plot_config() -> None:
    """Applies the current configuration settings."""
    try:
        plt.style.use(plot_cfg.style if plot_cfg.style != 'default' else 'default')
        plt.rcParams['figure.figsize'] = plot_cfg.figsize
        sns.set_palette(plot_cfg.color_theme)
    except Exception as e:
        logger.error(f"Error applying configuration: {str(e)}")
        logger.info("Reverting to default matplotlib style")
        plt.style.use('default')

def secure_plot(plot_fn: Callable[..., None], *args: Any, **kwargs: Any) -> None:
    """Safely executes a plotting function with error handling."""
    try:
        plot_fn(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error in plotting: {str(e)}")
        raise

def draw_line(x_vals: np.ndarray, y_vals: np.ndarray, title: str = "Line Plot", x_label: str = "X-axis", y_label: str = "Y-axis", interactive: bool = False) -> None:
    """Creates a line plot."""
    apply_plot_config()
    if interactive:
        fig = go.Figure(data=go.Scatter(x=x_vals, y=y_vals, mode='lines'))
        fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
        fig.show()
    else:
        plt.figure()
        plt.plot(x_vals, y_vals)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)
        plt.show()

def draw_bar_chart(categories: list, values: list, title: str = "Bar Chart", x_label: str = "Categories", y_label: str = "Values") -> None:
    """Creates a bar chart."""
    apply_plot_config()
    plt.figure()
    plt.bar(categories, values)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45)
    plt.show()

def draw_scatter(x_vals: np.ndarray, y_vals: np.ndarray, title: str = "Scatter Plot", x_label: str = "X-axis", y_label: str = "Y-axis") -> None:
    """Creates a scatter plot."""
    apply_plot_config()
    plt.figure()
    plt.scatter(x_vals, y_vals)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()

def draw_pie_chart(labels: list, sizes: list, title: str = "Pie Chart") -> None:
    """Creates a pie chart."""
    apply_plot_config()
    plt.figure()
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title(title)
    plt.show()

def draw_subplots(plot_fns: list, nrows: int, ncols: int, titles: list) -> None:
    """Creates subplots from a list of plotting functions."""
    apply_plot_config()
    fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    axs = axs.flatten()
    for ax, plot_fn, title in zip(axs, plot_fns, titles):
        plt.sca(ax)
        plot_fn()
        plt.title(title)
    plt.tight_layout()
    plt.show()

def show_heatmap(data_vals: np.ndarray, title: str = "Heatmap") -> None:
    """Creates a heatmap."""
    apply_plot_config()
    plt.figure()
    sns.heatmap(data_vals, annot=True, cmap="YlGnBu")
    plt.title(title)
    plt.show()

def normalize_values(data_vals: np.ndarray) -> np.ndarray:
    """Normalizes the data to a range of [0, 1]."""
    return (data_vals - np.min(data_vals)) / (np.max(data_vals) - np.min(data_vals))

def calculate_moving_avg(data_vals: np.ndarray, window_size: int) -> np.ndarray:
    """Calculates the moving average of the data."""
    return np.convolve(data_vals, np.ones(window_size), 'valid') / window_size

def set_color_theme(theme: str) -> None:
    """Sets the color theme for plots."""
    sns.set_palette(theme)

def annotate_data_point(x_val: float, y_val: float, text: str) -> None:
    """Annotates a point on the plot."""
    plt.annotate(text, (x_val, y_val), xytext=(5, 5), textcoords='offset points')

def qq_probability_plot(data_vals: np.ndarray, title: str = "Q-Q Plot") -> None:
    """Creates a Q-Q plot."""
    apply_plot_config()
    plt.figure()
    stats.probplot(data_vals, dist="norm", plot=plt)
    plt.title(title)
    plt.show()
