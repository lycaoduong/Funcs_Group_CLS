import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import io
from OhlabsFcg.model import PredictorCls
import time
import os
from huggingface_hub import login


model_path = os.getenv("API_MODEL")
auth_token = os.getenv("API_TOKEN")
login(token=auth_token)
model = PredictorCls(model_path=model_path)


def process(array_binary, th):
    try:
        spectral = np.load(io.BytesIO(array_binary[0]))
        outputs = model(spectral)
        result = model.decode(outputs, th)
        # plt.figure()
        # plt.title("Spectra Signal")
        # plt.plot(spectral)
        stream_result = ''
        plt.figure(1)
        plt.title("Spectra Signal")
        for r, text in enumerate(result):
            plt.clf()
            plt.plot(spectral)
            if r != len(result)-1:
                stream_result += text + ' '
            else:
                stream_result += '. '
            time.sleep(0.01)
            yield plt, stream_result
    except:
        stream_result = 'Invalid File'
        plt.figure(1)
    return plt, stream_result

def clear():
    return None, None


block = gr.Blocks(title="Ohlabs-Fcg API")
with block:
    with gr.Row():
        gr.Markdown("## Transformer based for Spectra Characterization - Ohlabs")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                spectra_input = gr.File(file_count="multiple",
                                        label="Select File (*.npy)-signal with shape [1, n], n: signal length",
                                        type="binary", file_types=['.npy'])
            with gr.Row():
                with gr.Accordion("Advanced options", open=False):
                    threshold = gr.Slider(label="Control Confidence", minimum=0.05, maximum=0.95, value=0.5, step=0.05)
            with gr.Row():
                with gr.Column():
                    run_button = gr.Button(value="Run", variant="primary")
                with gr.Column():
                    clear_button = gr.Button(value="Clear")
        with gr.Column():
            plot = gr.Plot(label="Spectra Signal")
            results = gr.Textbox(label="Prediction Result")
    ips = [spectra_input, threshold]
    ops = [plot, results]
    run_button.click(fn=process, inputs=ips, outputs=ops)
    clear_button.click(fn=clear, inputs=[], outputs=ops)
block.launch(server_name='0.0.0.0', share=False)
