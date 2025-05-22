import logging 

import numpy as np
from advanced_graphing import (
    set_config, line_plot, bar_chart, scatter_plot, pie_chart, 
    subplot, heatmap, normalize_data, moving_average, 
    set_color_palette, annotate_point, qq_plot, safe_plot,
    get_available_styles
)

# Print available styles
av_styles = get_available_styles()
logging.info("Available styles: %s", av_styles)

# Set global configuration
pref_style = 'seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in av_styles else 'ggplot'
set_config(style=pref_style, figsize=(12, 8), color_palette='Set2')

# Generate example data
x_data = np.linspace(0, 10, 100)
y_data = np.sin(x_data)

# Line plot
safe_plot(line_plot, x_data, y_data, title="Sine Wave", xlabel="X", ylabel="sin(x)")

# Bar chart
cats = ['A', 'B', 'C', 'D', 'E']
vals = np.random.rand(5) * 10
safe_plot(bar_chart, cats, vals, title="Random Bar Chart")

# Scatter plot
x_sc = np.random.rand(50)
y_sc = x_sc + np.random.normal(0, 0.1, 50)
safe_plot(scatter_plot, x_sc, y_sc, title="Scatter Plot with Noise")

# Pie chart
sizes_pie = [35, 30, 20, 15]
labels_pie = ['A', 'B', 'C', 'D']
safe_plot(pie_chart, labels_pie, sizes_pie, title="Sample Pie Chart")

# Heatmap
data_hm = np.random.rand(10, 10)
safe_plot(heatmap, data_hm, title="Random Heatmap")

# Subplots
def plot_sin_cos():
    line_plot(x_data, np.sin(x_data), title="Sin(x)")

def plot_cos_sin():
    line_plot(x_data, np.cos(x_data), title="Cos(x)")

safe_plot(subplot, [plot_sin_cos, plot_cos_sin], 1, 2, ["Sin(x)", "Cos(x)"])

# Data preprocessing
norm_data = normalize_data(y_data)
safe_plot(line_plot, x_data, norm_data, title="Normalized Sine Wave")

mov_avg = moving_average(y_data, 10)
safe_plot(line_plot, x_data[9:], mov_avg, title="Moving Average of Sine Wave")

# Color palette
set_color_palette("Set3")
safe_plot(scatter_plot, x_data, y_data, title="Sine Wave with New Color Palette")

# Annotation
def annotated_plot_func():
    line_plot(x_data, y_data, title="Annotated Sine Wave")
    annotate_point(np.pi, 0, "y = 0 at π")

safe_plot(annotated_plot_func)

# Statistical plot
data_stat = np.random.normal(0, 1, 1000)
safe_plot(qq_plot, data_stat, title="Q-Q Plot of Normal Distribution")

logging.info("All plots have been generated and displayed.")