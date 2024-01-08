import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import io
import onnxruntime
import cv2


class PredictorClsONNX(object):
    def __init__(self, onnx_dir, signal_size=1024, device='cpu'):
        if device == 'cuda':
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 0.5 * 1024 * 1024 * 1024,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                })
            ]
            session = onnxruntime.InferenceSession(onnx_dir, None, providers=providers)
        else:
            session = onnxruntime.InferenceSession(onnx_dir, providers=['CPUExecutionProvider'])
        session.get_modelmeta()
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name
        self.model = session
        self.device = device
        self.signal_size = signal_size
        self.cls_dic = {0: "alkane", 1: "methyl", 2: "alkene", 3: "alkyne", 4: "alcohols", 5: "amines", 6: "nitriles",
                        7: "aromatics", 8: "alkyl halides", 9: "esters", 10: "ketones", 11: "aldehydes",
                        12: "carboxylic acids", 13: "ether", 14: "acyl halides", 15: "amides", 16: "nitro"}

    def __call__(self, signal, th=0.3):
        signal = np.expand_dims(signal, axis=0)
        max = np.max(signal, axis=1)
        min = np.min(signal, axis=1)
        signal = ((signal.astype(np.float32) - min) / (max - min))
        signal = cv2.resize(signal, (self.signal_size, 1), interpolation=cv2.INTER_CUBIC)
        input_blob = np.expand_dims(signal, axis=0).astype(np.float32)
        outputs = self.model.run([self.output_name], {self.input_name: input_blob})[0][0]
        clses = list(np.where(outputs >= th))[0]
        results = []
        for cls in clses:
            cls_name = self.cls_dic[cls]
            results.append(cls_name)
        return results


ckpt = 'cls.onnx'
model = PredictorClsONNX(onnx_dir=ckpt)

def process(inputdata):
    try:
        spectral = np.load(io.BytesIO(inputdata))
        outputs = model(spectral, th=0.5)
        rs = ''
        for o in outputs:
            rs += o + ', '
        fig = plt.figure()
        plt.title("Spectra Signal")
        # plt.ion()
        # for i in range(len(spectral)):
        #     # fig = plt.figure()
        #     plt.title("Spectra Signal")
        #     plt.plot(spectral[:i])
        #     yield rs, plt
        plt.plot(spectral)
    except:
        rs = 'Invalid File'
        plt.figure(1)
    return rs, plt


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Spectra Characterization API")
    with gr.Row():
        with gr.Column():
            input_data = gr.File(file_count="single", label="Input Spectra File", type="binary")
            # input_data = gr.Image(type="numpy", label="Input Spectra npy File", image_mode= "1")
            results = gr.Textbox(label="Prediction Result")
            run_button = gr.Button(value="Run")
        with gr.Column():
            plot = gr.Plot(label="Plot")
    ips = [input_data]
    ops = [results, plot]
    run_button.click(fn=process, inputs=ips, outputs=ops)

block.launch(server_name='localhost', share=True)