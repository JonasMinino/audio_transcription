from boto3 import Session, client
from collections import Counter
from gtts import gTTS
from gtts.lang import tts_langs
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from openai import OpenAI
from pydub import AudioSegment, effects
from pydub.silence import split_on_silence
import asyncio
import boto3
import gradio as gr
import json
import nltk
import numpy as np
import os
import threading
import torch
import whisper
import sounddevice as sd
import soundfile as sf
import re


#Variables 
last_message = ''
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('punkt')

def speak_text(text):
    def speak():
        try:
            # Create speech using gTTS
            tts = gTTS(text, lang='en', slow=False)
            
            # Save the audio to a temporary file
            temp_file = 'temp_audio.wav'
            tts.save(temp_file)
            
            # Load the audio file
            data, fs = sf.read(temp_file, dtype='float32')
            
            # Play the audio file using sounddevice
            sd.play(data, fs)
            sd.wait()
            
            # Remove the temporary file
            os.remove(temp_file)
        
        except Exception as e:
            print(f"Error: {e}")

    # Create a thread to run the speak function
    threading.Thread(target=speak).start()

# Gets the secret from AWS
def get_secret():

    secret_name = "openai-secret"
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    get_secret_value_response = client.get_secret_value(
        SecretId=secret_name
    )

    secret = get_secret_value_response['SecretString']
    return secret

# Starts Openai
gpt_key = json.loads(get_secret())['OPENAI_API_KEY']
openai = OpenAI(api_key=gpt_key)

#cleans the message from stop words
def clean_message(message):
    # Remove '*' and '#'
    message = word_tokenize(message)
    stop_words = set(stopwords.words('english'))
    message = [word for word in message if word.lower() not in stop_words] #clean stop words
    message = ' '.join(message)
    return message

#Chunking the audio file into audio segments
def preprocess_audio(audio):
    '''Convert audio file to proper format before passing to Whisper'''
    audio = AudioSegment.from_file(audio)

    # Ensure the audio is mono (single channel) and set the proper sampling rate (e.g., 16kHz or 44.1kHz)
    audio = audio.set_channels(1).set_frame_rate(16000)

    # Now split into chunks
    chunk_duration = 10000  # 10 seconds for example
    chunks = []

    for start_ms in range(0, len(audio), chunk_duration):
        end_ms = min(start_ms + chunk_duration, len(audio))
        chunk = audio[start_ms:end_ms]
        chunks.append(chunk)

    return chunks
    
#Converts the audio into a numpy array
def audio_segment_to_numpy(chunk):
    '''Convert an AudioSegment chunk to a NumPy array'''    
    if not isinstance(chunk, AudioSegment):
        raise ValueError(f"Expected AudioSegment, got {type(chunk)}")

    # Convert AudioSegment to raw data (16-bit PCM)
    raw_data = chunk.raw_data

    # Ensure that the audio array is in the correct shape and dtype to avoid memory overflow
    audio_array = np.frombuffer(raw_data, dtype=np.int16)

    # Normalize and convert to float32
    audio_array = audio_array.astype(np.float32) / 32768.0

    # Reshape to match the number of channels (if stereo, reshape accordingly)
    if chunk.channels > 1:
        audio_array = audio_array.reshape((-1, chunk.channels))

    return audio_array

#Detecting the language from the first 10 segements of a list
def language_detection(results):

    main_language =''
    languages=[]

    for result in results:
        languages.append(result['language'])
            
    if languages:
        language_counter = Counter(languages)
        main_language = language_counter.most_common(1)[0][0]
        
    return main_language

# Transcribe and detects the language
async def transcribe_async(audio):
    '''Asynchronous transcription of audio file using Whisper'''
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("small").to(device)
    transcription = ''
    main_language = ''
    results = []
    try:
        # Use Whisper to transcribe the audio file
        audio_chunks = preprocess_audio(audio)

        #Detecting the language
        for i in range(min(len(audio_chunks), 10)):           
            # Convert each chunk to NumPy array
            chunk = audio_chunks[i]
            audio_data = audio_segment_to_numpy(chunk)
            audio_data = whisper.pad_or_trim(audio_data)
            audio_tensor = torch.from_numpy(audio_data).to(device)
            
            # Transcribe the tensor
            res = await asyncio.to_thread(model.transcribe, audio_tensor, language=None, task='transcribe')
            results.append(res)

        main_language = language_detection(results)
        print(f'Language: {main_language}')

        #Iterating the entire list of audio segments
        for i, chunk in enumerate(audio_chunks):            
            # Convert each chunk to NumPy array
            audio_data = audio_segment_to_numpy(chunk)
            audio_data = whisper.pad_or_trim(audio_data)
            audio_tensor = torch.from_numpy(audio_data).to(device)
            
            # Transcribe the tensor
            if main_language == 'en':
                result = await asyncio.to_thread(model.transcribe, audio_tensor, language=None, task='transcribe')
                print(f"Transcription result for chunk {i+1}: {result}")
            else:
                result = await asyncio.to_thread(model.transcribe, audio_tensor, language=main_language, task='translate')
                print(f"Transcription result for chunk {i+1}: {result}")
            
            # Check if the transcription has meaningful text
            if result['text'].strip():  # If text is not empty or too short
                transcription += result['text']
            else:
                print(f"Skipping chunk {i+1} due to lack of meaningful transcription.")
            
            print(f"Processing chunk {i + 1}/{len(audio_chunks)}...")

        clean_message(transcription)
        transcription += f"Original Audio Language: {main_language}"
        return write_minutes(transcription, 'en')
    
    except Exception as e:
        print(f"Error during transcription: {e}")
        return f"Error during transcription {e}"

#Cleans a text from markdown symbols
def clean_markdown(last_message):
    '''Cleans a message from Markdown symbols'''
    # Remove bold (**), italic (*), and code block symbols (```)
    last_message = re.sub(r'(\*\*|\*|__|_|\`{1,2})', '', last_message)
    
    # Remove headers (# and ## and more)
    last_message = re.sub(r'(^|\s)(#{1,6})\s', '\n', last_message)
    
    # Remove lists (ordered and unordered)
    last_message = re.sub(r'^\s*(\*|\+|\-|\d+\.)\s+', '', last_message, flags=re.MULTILINE)
    
    # Remove links in the form [text](url)
    last_message = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', last_message)
    
    # Remove images in the form ![alt](url)
    last_message = re.sub(r'!\[([^\]]+)\]\([^\)]+\)', r'\1', last_message)
    
    # Remove blockquotes (lines starting with >)
    last_message = re.sub(r'^\s*>', '', last_message, flags=re.MULTILINE)
    
    # Remove horizontal rules (---, ***, ___)
    last_message = re.sub(r'(\*\*\*|---|___)\s*', '', last_message)
    
    return last_message.strip()

# Write meeting minutes
def write_minutes(text, language) -> str:
    '''Uses openai to write minutes from a text'''
    global last_message

    system_message = system_message = '''
        You are an assistant that writes minutes from audio or text. 
        Your task is to summarize the most important information from the meeting such as:
        - Date and time of the meeting
        - Location of the meeting
        - Orignial Audio Language
        - Attendees
        - Agenda items
        - Action items with their assignees and deadlines
        - A concise summary at the end of the meeting

        The summary should be brief and focused on key takeaways and decisions made during the meeting.
        Translate the following ISO 639-1 language code into its corresponding language, for example en should be Orignial Audio Language: English.
        The meeting minutes should be returned in markdown format.
        '''

    user_message = f'Language: {language} ' + f'Meeting text: {text}'

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}],
            max_tokens=1000
    )
    last_message = response.choices[0].message.content
    last_message = clean_markdown(last_message)
    return last_message

# Openai chatbot
def openai_chatbot(message):
    system_message = "You are an assistant that helps automate office tasks. Always respond in a short way."
    system_message += "You can assist with text summarization, audio transcription, and writing meeting minutes."
    system_message += "The use can upload an audio file or give you a text file for transcription."

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": message}],
        max_tokens=150
    )
         
    return response.choices[0].message.content

# Gradio Interface Components
async def chat_interface(audio=None, message=None, conversation_history=None):
    '''Handles chat input and returns response, transcribes the audio file, and outputs text'''
    global last_message
    if conversation_history is None:
        conversation_history = []

    try:
        if audio:
            transcription = await transcribe_async(audio)
            conversation_history.append({"role": "assistant", "content": f'Assistant: {transcription}'})
            response = transcription
            last_message = response
        elif message:
            # Process the text message and get a response
            conversation_history.append({"role": "user", "content": f'User: {message}'})
            assistant_response = openai_chatbot(message)
            conversation_history.append({"role": "assistant", "content": f'Assistant: {assistant_response}'})
            response = assistant_response
            last_message = response
        else:
            response = initial_message
    except Exception as e:
        print(f'Error during transcription {e}')

    return gr.update(value="\n".join([msg['content'] for msg in conversation_history])), conversation_history

initial_message = 'How can I assist you today?'

# Gradio Interface setup
with gr.Blocks() as demo:
    gr.Markdown("# Meeting Minutes - Audio and Chat Interface")
      
    conversation_history = gr.State([])
    
    with gr.Row():
        with gr.Column():
            chat_output = gr.Textbox(label="Chat and Transcription Output", value="How can I assist you today?", lines=10, interactive=False)
            message_input = gr.Textbox(placeholder="Type your message here...", label="Type your message", lines=1)
        
        with gr.Column():
            audio_input = gr.File(label="Upload an Audio File", file_types=["audio"])
            submit_button = gr.Button("Submit Audio")

    submit_button.click(
        chat_interface, 
        inputs=[audio_input, gr.State(None), conversation_history], 
        outputs=[chat_output, conversation_history]
    )

    message_input.submit(
        chat_interface, 
        inputs=[gr.State(None), message_input, conversation_history], 
        outputs=[chat_output, conversation_history]
    )

    message_input.submit(lambda x: gr.update(value=''), [], [message_input], queue=False)


    #Speak the last message and clear the audio input on output change
    chat_output.change(lambda x: speak_text(last_message), None, None)

# Launch the app
speak_text(initial_message)   
demo.launch(debug=True, inbrowser=True)