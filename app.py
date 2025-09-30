import gradio as gr
import torch
from TTS.api import TTS
import os
import subprocess

# Define paths
check_point_folder = "./ckpts"
model_path = f"{check_point_folder}/best_model.pth"
config_path = f"{check_point_folder}/config.json"

# Check if model and config files exist, download if they don't
if not (os.path.exists(model_path) and os.path.exists(config_path)):
    print("Model or config file not found. Running download script...")
    try:
        subprocess.run(["python", "scripts/download_ckpts.py"], check=True)
        print("Download completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running download script: {e}")
        exit(1)

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize TTS model
tts = TTS(model_path=model_path, config_path=config_path, progress_bar=True)

# Move model to the specified device
tts.to(device)

def generate_audio(text):
    # Define output path
    out_path = "tts_output.wav"
    
    # Perform inference and save to file
    tts.tts_to_file(text=text, file_path=out_path, speaker=None, language=None, split_sentences=False)
    
    # Return the path to the generated audio file
    return out_path

# Create Gradio interface
iface = gr.Interface(
    fn=generate_audio,
    inputs=gr.Textbox(
        label="Input Text",
        placeholder="Enter text to convert to speech...",
        lines=5,
        value=("Trong khi đó, tại bến tàu du lịch Nha Trang, hàng ngàn du khách chen nhau "
        "để đi đến các đảo trên vịnh Nha Trang, lực lượng cảnh sát đường thủy đã "
        "tăng cường quân số để quản lý, đảm bảo an toàn cho du khách.")
    ),
    outputs=gr.Audio(label="Generated Audio"),
    title="Viet GlowTTS finetuning",
    description="Enter text to generate speech using the Viet-Glow-TTS-finetuning model.",
    allow_flagging="never"
)

# Launch the app
if __name__ == "__main__":
    iface.launch()