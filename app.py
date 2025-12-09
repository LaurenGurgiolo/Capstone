import gradio as gr
import pandas as pd
import pickle
import random
import requests
try:
    import sclib
except ImportError:
    print("sclib not found, attempting installation...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sclib"])
    print("sclib installed.")
import cv2
import numpy as np
import torch
from PIL import Image
import soundfile as sf
import os
import tempfile
from sclib import SoundcloudAPI
from playsound3 import playsound
import sys
import subprocess
from transformers import AutoImageProcessor, AutoModelForImageClassification, ViTImageProcessor

# Import your specific emotion recognition model library (e.g., from facenet_pytorch or transformers)

# --- 1. Load the emotion recognition model ---
# This part depends on your specific model. The example below is conceptual.
# You will need to replace this with your actual model loading code.
# Example: Using a placeholder function for demonstration.
def load_emotion_model():
    model_name = "LaurenGurgiolo/vit-micro-facial-expressions"
# Load model directly


    processor = ViTImageProcessor.from_pretrained(model_name)
    # Load model here (e.g., using fastai load_learner, or a transformers pipeline)
    # Placeholder for model loading
    print("Loading emotion recognition model...")
    # Example: model = load_learner('emotion_model.pkl') 
    model = AutoModelForImageClassification.from_pretrained(model_name)# Replace with actual model
    return model

# --- 2. Load the pickle file with songs ---
def load_song_urls(filepath='emotion_songs.pkl'):
    try:
        with open(filepath, 'rb') as f:
            songs_data = pickle.load(f)
        return songs_data
    except FileNotFoundError:
        return {} # Return empty dict if file not found
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return {}

# --- 3. Function to get a direct streamable URL from SoundCloud URL (complex and potentially unstable) ---
# Direct streaming from SoundCloud URLs requires complex API interaction or third-party tools (like yt-dlp).
# For simplicity and stability on HF Spaces, this example uses a placeholder or assumes direct MP3 URLs are in the pickle file.
# If you have SoundCloud page URLs, you might need a service to resolve them to direct audio files which can expire.
def get_stream_url(url):
   
    # Create a SoundcloudAPI object
    api = SoundcloudAPI()

    # Resolve the track from the SoundCloud URL
    track = api.resolve(url)

    if not track:
        print("Could not resolve track from URL.")
        return

    print(f"Streaming: {track.artist} - {track.title}")

    # Create a temporary file to store the MP3 data
    # Use a context manager to ensure the temporary file is handled properly
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        temp_filename = temp_file.name
        # Write the track's MP3 data to the temporary file
        track.write_mp3_to(temp_file)
        print(f"Track saved to temporary file: {temp_filename}")

    # Play the audio stream asynchronously (in the background) from the local file
    audio_stream = playsound(temp_filename, block=False)

    return temp_filename, track.artist, track.title

    # Clean up the temporary file
    
    print("Temporary file removed.")

    

# --- 4. Main prediction function ---
def analyze_emotion_and_get_songs(image_input):
    # Convert PIL image to format required by your model (e.g., numpy array)
    image = image_input.convert("RGB") 
    image = np.array(image)
    model_name = "LaurenGurgiolo/vit-micro-facial-expressions"
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    
    
    
    
    
    inputs = feature_extractor(images=image, return_tensors="pt")
    id_to_emotion = {
    0: "Angry",
    1: "Contempt",
    2: "Disgust",
    3: "Fear",
    4: "Happy",
    5: "Neutral",
    6: 'Sad',
    7: "Tired",
    8: "Surprised",
}
    model = load_emotion_model()
    # image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) # if needed

    # Run emotion model (placeholder)
   
    # Get predictions from the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label_id = logits.argmax(-1).item()

    # Get the true and predicted labels as strings
    
    predicted_emotion = id_to_emotion[predicted_label_id ]
    

    # Load songs data and select 20 random URLs
    songs_db = load_song_urls()
    if predicted_emotion in songs_db:
        emotion_urls = songs_db[predicted_emotion]
        selected_urls = random.sample(emotion_urls, min(20, len(emotion_urls)))
    else:
        selected_urls = []
    
    # Generate HTML playlist (Gradio doesn't have a built-in playlist component, use HTML for embedding)
    if selected_urls:
        data = {}
        title = []
        path = []
        data['Title'] = []
        data['Path'] = []
        playlist_html = "<h3 style='text-align: center;'>Selected Songs for " + predicted_emotion.capitalize() + "</h3>"
        for url in selected_urls:
            x, artist, title = get_stream_url(url)
        
            data['Title'].append("Artist: " + artist + " Title: " + title)

            data['Path'].append(x)

    
    

        df = pd.DataFrame(data)
        return df, None
            # Assuming the URL is a direct MP3 link, you can use a simple HTML audio tag
        
        # Ensure proper quoting for HTML attributes
            

def select_track_for_playback(evt: gr.SelectData, playlist_df):
    """
    Called when a user selects a row in the DataFrame.
    evt.index[0] gives the row index of the selection.
    """
    if evt.index:
        selected_index = evt.index[0]
        # Get the actual file path from the hidden data source (or from the displayed DataFrame)
        selected_path = playlist_df.iloc[selected_index]['Path']
        return selected_path
    return None

# --- 5. Gradio Interface ---
# Load your model globally or within the function if preferred
# model_instance = load_emotion_model() 
# 3. Define the Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## Austin-Based SoundCloud Playlist Generator")
    
    # Input component (e.g., webcam for facial recognition)
    user_input = gr.Image(label="UUpload an image of a face to detect the emotion and get a 20-song playlist from SoundCloud.", sources=["upload", "webcam"], type="pil")

    # Hidden State to store the full playlist DataFrame after generation
    # This is important for the select_track_for_playback function to work
    playlist_state = gr.State(value=pd.DataFrame())

    generate_btn = gr.Button("Analyze Emotion and Create Playlist")

    # Playlist display using a DataFrame (user interactive)
    # We hide the 'Path' column from the user view but keep it in the data
    playlist_df = gr.DataFrame(headers=["Title", "Path"], visible=True) 

    # Single Audio player component (output-only)
    audio_player = gr.Audio(label="Now Playing")
    
    # Event Handlers:
    
    # When the button is clicked, generate the playlist and update the UI components
    generate_btn.click(
        fn=analyze_emotion_and_get_songs, 
        inputs=[user_input], 
        outputs=[playlist_df, audio_player]
    ).then(
        # After updating the playlist_df visual, store the *full* data in the state
        lambda df: df, 
        inputs=[playlist_df], 
        outputs=[playlist_state]
    )
    
    # When a row in the DataFrame is selected, update the single audio player
    playlist_df.select(
        fn=select_track_for_playback,
        inputs=[playlist_state], # Pass the hidden state DataFrame to access the paths
        outputs=[audio_player]
    )



if __name__ == "__main__":
    demo.launch()
