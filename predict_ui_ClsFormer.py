import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import io
import onnxruntime
import cv2


funcs_name = ["alkane", "methyl", "alkene", "alkyne", "alcohols", "amines", "nitriles", "aromatics",
              "alkyl halides", "esters", "ketones", "aldehydes", "carboxylic acids",
              "ether", "acyl halides", "amides", "nitro"]


class PredictorClsFormerONNX(object):
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
        self.result_name = session.get_outputs()[0].name
        self.att_name = session.get_outputs()[1].name
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
        outputs, att_map = self.model.run([self.result_name, self.att_name], {self.input_name: input_blob})
        outputs = outputs[0]
        clses = list(np.where(outputs >= th))[0]
        results = []
        for cls in clses:
            cls_name = self.cls_dic[cls]
            results.append(cls_name)
        return results, np.mean(att_map, axis=1)[0]


ckpt = 'cls_former.onnx'
model = PredictorClsFormerONNX(onnx_dir=ckpt)

def process(inputdata):
    try:
        spectra = np.load(io.BytesIO(inputdata))
        outputs, att_map = model(spectra, th=0.5)
        rs = ''
        for o in outputs:
            rs += o + ', '

        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = plt.twinx().twiny()
        ax2.set_xlim(0, len(spectra))
        ax1.set_yticks(np.arange(len(funcs_name)), labels=funcs_name)
        # ax1.set_xlim(0, 64)

        tem = np.zeros_like(att_map)
        for i, r in enumerate(funcs_name):
            if r in outputs:
                tem[i, :] = att_map[i, :]

        ax1.imshow(tem[:, :], cmap='viridis', interpolation='nearest', aspect="auto")
        ax2.plot(spectra)

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
                att_map = gr.Plot(label="Attention Map")
    ips = [input_data]
    ops = [results, att_map]
    run_button.click(fn=process, inputs=ips, outputs=ops)

block.launch(server_name='localhost', share=False)