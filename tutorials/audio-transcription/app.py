######IMPORTANT READ THIS PLS!!!!!!!!!!
######
##for development mode in Gradio... run `gradio app.py` in your terminal - trust me...
##make sure you have all dependencies and actually ran setup.ipynb
########
import gradio as gr
import torch

import nemo.collections.asr as nemo_asr

TITLE = "NeMo ASR"
DESCRIPTION = "Demo of Various Models in NeMo ASR"
DEFAULT_EN_MODEL = "stt_en_quartznet15x5"

MARKDOWN = f"""
# {TITLE}
## {DESCRIPTION}
"""

CSS = """
p.big {
  font-size: 20px;
}
"""

ARTICLE = """
<br><br>
<p class='big' style='text-align: center'>
    <a href='https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/intro.html' target='_blank'>NeMo ASR</a> 
    | 
    <a href='https://github.com/NVIDIA/NeMo#nvidia-nemo' target='_blank'>Github Repo</a>
</p>
"""

SUPPORTED_MODEL_NAMES = set([])
available_models = nemo_asr.models.ASRModel.list_available_models()

for mdl in available_models:
    mdlnameswewant = mdl.pretrained_model_name.startswith("stt_en") ## seemed good to start with english
    if (mdlnameswewant):
        SUPPORTED_MODEL_NAMES.add(mdl.pretrained_model_name)
    continue

SUPPORTED_MODEL_NAMES = sorted(list(SUPPORTED_MODEL_NAMES))

def transcribe(microphone, audio_file, model_name):

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    warn_output = ""
    if (microphone is not None) and (audio_file is not None):
        warn_output = (
            "WARNING: You've uploaded an audio file and used the microphone. "
            "The recorded file from the microphone will be used and the uploaded audio will be discarded.\n"
        )
        audio_data = microphone

    elif (microphone is None) and (audio_file is None):
        return "ERROR: You have to either use the microphone or upload an audio file"

    elif microphone is not None:
        audio_data = microphone
    else:
        audio_data = audio_file

    try:
        print(audio_data)
        # Use HF API for transcription
        transcriptions = model.transcribe(paths2audio_files=[audio_data])

    except Exception as e:
        transcriptions = ""
        warn_output = warn_output + "\n\n"
        warn_output += (
            f"Error: {e}"
        )

    return f'{warn_output} {transcriptions[0]}'



demo = gr.Blocks(title=TITLE, css=CSS)
with demo:
    header = gr.Markdown(MARKDOWN)

    with gr.Row() as row:
        file_upload = gr.components.Audio(source="upload", type='filepath', label='Upload File')
        microphone = gr.components.Audio(source="microphone", type='filepath', label='Microphone')

    models = gr.components.Dropdown(
        choices=sorted(list(SUPPORTED_MODEL_NAMES)),
        value=DEFAULT_EN_MODEL,
        label="Models",
        interactive=True,
    )
    transcript = gr.components.Label(label='Transcript')

    run = gr.components.Button('Transcribe')
    run.click(transcribe, inputs=[microphone, file_upload, models], outputs=[transcript])

    gr.components.HTML(ARTICLE)

demo.queue(concurrency_count=1)
demo.launch()