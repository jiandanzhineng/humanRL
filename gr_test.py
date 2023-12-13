import math
import gradio as gr
import plotly.express as px
import numpy as np


plot_end = 2 * math.pi


def get_plot(period=1):
    return str(period)


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("Change the value of the slider to automatically update the plot")
            period = gr.Slider(label="Period of plot", value=1, minimum=0, maximum=10, step=1)
            gr.Markdown("发射点发射点发射点发发射点发射点发射点发发射点发射点发射点发发射点发射点发射点发")
            plot = gr.Markdown('   ASDF')
            gr.Markdown("发射点发射点发射点发发射点发射点发射点发发射点发射点发射点发发射点发射点发射点发")

    dep = demo.load(get_plot, period, plot, every=1)
    period.change(get_plot, period, plot, every=1, cancels=[dep])


if __name__ == "__main__":
    demo.queue().launch()