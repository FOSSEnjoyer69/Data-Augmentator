import gradio as gr
from image_editing import augment_dataset

with gr.Blocks(title="Data Augmentator") as app:
    input_folder = gr.Text(label="Input Folder")
    folder_type = gr.Radio(label="Type", choices=["Single Folder", "Parent Of Multiple"])
    with gr.Row():
        iterations_count = gr.Slider(label="Iterations", minimum=1, value=1, maximum=100, step=1)
        create_varitations_btn = gr.Button("Augment Dataset")

    create_varitations_btn.click(augment_dataset, inputs=[input_folder, folder_type, iterations_count])

app.launch()
