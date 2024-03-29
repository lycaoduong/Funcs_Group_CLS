import numpy as np
import gradio as gr
import io
from FcgEngine.engine import PredictorCls
from FcgEngine.utils import plot_result, plot_spectra, plot_self_attention_map
import os
from huggingface_hub import login


model_path = os.getenv("API_MODEL")
auth_token = os.getenv("API_TOKEN")
login(token=auth_token)
engine = PredictorCls(model_path=model_path)


def process(array_binary, th, option):
    spectra = np.load(io.BytesIO(array_binary[0]))
    outputs = engine(spectra)
    if option == 'Positive only':
        fcn_groups, probabilities = engine.get_result(outputs, th=th, pos_only=True)
    else:
        fcn_groups, probabilities = engine.get_result(outputs, th=th)
    prediction_fg = plot_result(fcn_groups, probabilities)

    spectra_fg = plot_spectra(spectra)

    att = engine.model.get_self_attention(layer_value=1)
    att = np.sum(att[0], axis=0)
    attention_fg = plot_self_attention_map(spectra, att)

    return spectra_fg, prediction_fg, attention_fg


def clear():
    return None, None, None


examples_signal = [
    ['examples/2373797.npy'],
    ['examples/2459054.npy'],
    ['examples/17451182.npy'],
    ['examples/19013106.npy'],
    ['examples/168833805.npy'],
]


block = gr.Blocks(title="FcgFormer APP - Ohlabs")
with block:
    with gr.Row():
        gr.Markdown("## Transformer based for Spectra Characterization")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                spectra_input = gr.File(file_count="multiple",
                                        label="Select File (*.npy)-signal with shape [1, n], n: signal length",
                                        type="binary", file_types=['.npy'])
            with gr.Row():
                with gr.Accordion("Advanced options", open=False):
                    threshold = gr.Slider(label="Control Confidence", minimum=0.05, maximum=0.95, value=0.5, step=0.05)
                    draw_option = gr.Radio(["Positive only", "All classes"], value="Positive only", label="Draw Prediction option")
            with gr.Row():
                with gr.Column():
                    run_button = gr.Button(value="Run", variant="primary")
                with gr.Column():
                    clear_button = gr.Button(value="Clear")

            with gr.Row():
                download_example = gr.File(file_count="multiple",
                                           label="Please select an example file below and then click on the file size, "
                                                 "which is typically highlighted in blue, to download the example signal "
                                                 "for testing purposes.",
                                           type="binary", file_types=['.npy'])
            with gr.Row():
                gr.Examples(examples=examples_signal,
                            inputs=download_example, label="Download example file")

        with gr.Column():
            with gr.Tab("Prediction"):
                predicted_plot = gr.Plot(label="Result")
            with gr.Tab("Attention map"):
                attention_plot = gr.Plot(label="Attention Map")
            with gr.Tab("Input Spectra"):
                signal_plot = gr.Plot(label="Spectra Signal")

    ips = [spectra_input, threshold, draw_option]
    ops = [signal_plot, predicted_plot, attention_plot]
    run_button.click(fn=process, inputs=ips, outputs=ops)
    clear_button.click(fn=clear, inputs=[], outputs=ops)

block.launch(server_name='0.0.0.0', share=False)

