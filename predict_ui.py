import numpy as np
from src.predictor import PredictorCls
import torch
import gradio as gr
import matplotlib.pyplot as plt
import os


ckpt = '../runs/train/FCG_Classification/Fcg-B_ohlabsFcg/single_run/2023.12.28_21.12.53/last.pt'
model = PredictorCls(model_name='Fcg-B', ckpt=ckpt, device='cuda')

def process(input_file):
    if os.path.isfile(input_file):
        spectral = np.load(input_file)
        plt.figure(1)
        plt.title("Spectra Signal")
        plt.plot(spectral)
        with torch.no_grad():
            outputs = model(spectral, th=0.5)
        rs = ''
        for o in outputs:
            rs += o + ', '
    else:
        rs = 'Invalid File'
        plt.figure(1)
    return rs, plt


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Spectra Characterization API")
    with gr.Row():
        with gr.Column():
            input_files = gr.File(file_count="single", label="Input Spectra File", type="filepath")
            results = gr.Textbox(label="Prediction Result")
            run_button = gr.Button(value="Run")
        with gr.Column():
            plot = gr.Plot(label="Plot")
    ips = [input_files]
    ops = [results, plot]
    run_button.click(fn=process, inputs=ips, outputs=ops)

block.launch(server_name='localhost', share=False)