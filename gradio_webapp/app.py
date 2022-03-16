import gradio as gr
import model
from examples import example

input_1 = gr.inputs.Textbox(lines=1, placeholder='Feature Text', default="", label=None, optional=False)
input_2 = gr.inputs.Textbox(lines=5, placeholder='Patient History', default="", label=None, optional=False)

output_1 = gr.outputs.Textbox(type="auto", label=None)

iface = gr.Interface(
    model.get_predictions, 
    inputs=[input_1, input_2], 
    outputs=[output_1],
    examples=example,
    title='Identify Key Phrases in Patient Notes from Medical Licensing Exams',
    theme='dark', # 'dark'
)
iface.launch()