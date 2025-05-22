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
class GraphConfig:
    style: str = 'default'
    figsize: tuple = (10, 6)
    color_palette: str = 'viridis'

# Global configuration instance
config = GraphConfig()

def configurate(style: str = 'default', figsize: tuple = (10, 6), color_palette: str = 'viridis') -> None:
    """Set global configuration for plotting."""
    global config
    if style not in plt.style.available:
        logger.warning(f"Invalid style '{style}' provided. Defaulting to 'default'.")
        style = 'default'
        
    config.style = style
    config.figsize = figsize
    config.color_palette = color_palette
    plt.style.use(style)
    
    # Validate color palette
    if color_palette in sns.palettes.SEABORN_PALETTES.keys():
        sns.set_palette(color_palette)
    else:
        logger.warning(f"Invalid color palette '{color_palette}' provided. Defaulting to 'viridis'.")
        sns.set_palette('viridis')

def get_styles() -> list:
    """Returns a list of available matplotlib styles."""
    return plt.style.available

def apply_settings() -> None:
    """Applies the current configuration settings."""
    try:
        plt.style.use(config.style if config.style != 'default' else 'default')
        plt.rcParams['figure.figsize'] = config.figsize
        sns.set_palette(config.color_palette)
    except Exception as e:
        logger.error(f"Error applying configuration: {str(e)}")
        logger.info("Reverting to default matplotlib style")
        plt.style.use('default')

def safe_plotting(plotter: Callable[..., None], *args: Any, **kwargs: Any) -> None:
    """Safely executes a plotting function with error handling."""
    try:
        plotter(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error in plotting: {str(e)}")
        raise

def line_chart(x_data: np.ndarray, y_data: np.ndarray, title: str = "Line Plot", x_label: str = "X-axis", y_label: str = "Y-axis", interactive: bool = False) -> None:
    """Creates a line plot."""
    apply_settings()
    if interactive:
        fig = go.Figure(data=go.Scatter(x=x_data, y=y_data, mode='lines'))
        fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
        fig.show()
    else:
        plt.figure()
        plt.plot(x_data, y_data)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)
        plt.show()

def bar_graph(categories_data: list, values_data: list, title: str = "Bar Chart", x_label: str = "Categories", y_label: str = "Values") -> None:
    """Creates a bar chart."""
    apply_settings()
    plt.figure()
    plt.bar(categories_data, values_data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45)
    plt.show()

def scatter_chart(x_data: np.ndarray, y_data: np.ndarray, title: str = "Scatter Plot", x_label: str = "X-axis", y_label: str = "Y-axis") -> None:
    """Creates a scatter plot."""
    apply_settings()
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()

def pie_chart_plot(labels_data: list, sizes_data: list, title: str = "Pie Chart") -> None:
    """Creates a pie chart."""
    apply_settings()
    plt.figure()
    plt.pie(sizes_data, labels=labels_data, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title(title)
    plt.show()

def multiple_plots(plot_functions: list, rows: int, cols: int, titles_list: list) -> None:
    """Creates subplots from a list of plotting functions."""
    apply_settings()
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    axes = axes.flatten()
    for ax, plot_function, title in zip(axes, plot_functions, titles_list):
        plt.sca(ax)
        plot_function()
        plt.title(title)
    plt.tight_layout()
    plt.show()

def heat_map(data_array: np.ndarray, title: str = "Heatmap") -> None:
    """Creates a heatmap."""
    apply_settings()
    plt.figure()
    sns.heatmap(data_array, annot=True, cmap="YlGnBu")
    plt.title(title)
    plt.show()

def normalize_values(data_array: np.ndarray) -> np.ndarray:
    """Normalizes the data to a range of [0, 1]."""
    return (data_array - np.min(data_array)) / (np.max(data_array) - np.min(data_array))

def moving_avg(data_array: np.ndarray, window_size: int) -> np.ndarray:
    """Calculates the moving average of the data."""
    return np.convolve(data_array, np.ones(window_size), 'valid') / window_size

def set_palette_colors(palette_name: str) -> None:
    """Sets the color palette for plots."""
    sns.set_palette(palette_name)

def annotate_data_point(x_coord: float, y_coord: float, text_label: str) -> None:
    """Annotates a point on the plot."""
    plt.annotate(text_label, (x_coord, y_coord), xytext=(5, 5), textcoords='offset points')

def qq_plotting(data_array: np.ndarray, title: str = "Q-Q Plot") -> None:
    """Creates a Q-Q plot."""
    apply_settings()
    plt.figure()
    stats.probplot(data_array, dist="norm", plot=plt)
    plt.title(title)
    plt.show()