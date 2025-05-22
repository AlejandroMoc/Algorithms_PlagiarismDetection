import logging 

import numpy as np
from advanced_graphing import (
    set_config, line_plot, bar_chart, scatter_plot, pie_chart, 
    subplot, heatmap, normalize_data, moving_average, 
    set_color_palette, annotate_point, qq_plot, safe_plot,
    get_available_styles
)

def get_styles():
    return get_available_styles()

def plot_sine_wave():
    x_data = np.linspace(0, 10, 100)
    y_data = np.sin(x_data)
    safe_plot(line_plot, x_data, y_data, title="Sine Wave", xlabel="X", ylabel="sin(x)")

def plot_random_bar_chart():
    categories = ['A', 'B', 'C', 'D', 'E']
    values = np.random.rand(5) * 10
    safe_plot(bar_chart, categories, values, title="Random Bar Chart")

def plot_scatter_noise():
    x_scatter = np.random.rand(50)
    y_scatter = x_scatter + np.random.normal(0, 0.1, 50)
    safe_plot(scatter_plot, x_scatter, y_scatter, title="Scatter Plot with Noise")

def plot_sample_pie_chart():
    sizes = [35, 30, 20, 15]
    labels = ['A', 'B', 'C', 'D']
    safe_plot(pie_chart, labels, sizes, title="Sample Pie Chart")

def plot_random_heatmap():
    data = np.random.rand(10, 10)
    safe_plot(heatmap, data, title="Random Heatmap")

def plot_sin_cos_subplot():
    def plot_sin():
        line_plot(x_data, np.sin(x_data), title="Sin(x)")

    def plot_cos():
        line_plot(x_data, np.cos(x_data), title="Cos(x)")

    safe_plot(subplot, [plot_sin, plot_cos], 1, 2, ["Sin(x)", "Cos(x)"])

def plot_normalized_sine_wave():
    normalized_data = normalize_data(np.sin(x_data))
    safe_plot(line_plot, x_data, normalized_data, title="Normalized Sine Wave")

def plot_moving_avg_sine_wave():
    moving_avg = moving_average(np.sin(x_data), 10)
    safe_plot(line_plot, x_data[9:], moving_avg, title="Moving Average of Sine Wave")

def plot_sine_wave_color_palette():
    set_color_palette("Set3")
    safe_plot(scatter_plot, x_data, np.sin(x_data), title="Sine Wave with New Color Palette")

def plot_annotated_sine_wave():
    def annotated_plot():
        line_plot(x_data, np.sin(x_data), title="Annotated Sine Wave")
        annotate_point(np.pi, 0, "y = 0 at π")

    safe_plot(annotated_plot)

def plot_qq_plot_normal_distribution():
    data = np.random.normal(0, 1, 1000)
    safe_plot(qq_plot, data, title="Q-Q Plot of Normal Distribution")

available_styles = get_styles()
logging.info("Available styles: %s", available_styles)

preferred_style = 'seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in available_styles else 'ggplot'
set_config(style=preferred_style, figsize=(12, 8), color_palette='Set2')

x_data = np.linspace(0, 10, 100)

plot_sine_wave()
plot_random_bar_chart()
plot_scatter_noise()
plot_sample_pie_chart()
plot_random_heatmap()
plot_sin_cos_subplot()
plot_normalized_sine_wave()
plot_moving_avg_sine_wave()
plot_sine_wave_color_palette()
plot_annotated_sine_wave()
plot_qq_plot_normal_distribution()

logging.info("All plots have been generated and displayed.")