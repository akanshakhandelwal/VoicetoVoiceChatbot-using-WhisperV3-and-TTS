from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from pydantic import BaseModel, Field
import json
import os
import re
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from transformers import pipeline
from langchain.text_splitter import SpacyTextSplitter
import transformers
import torch
import warnings
warnings.filterwarnings("ignore")
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from pathlib import Path
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
from openai import OpenAI
from pydub import AudioSegment  
from pydub.playback import play
from playsound import playsound
import pygame

def load_document(file):
    print(file)
    name, extension = os.path.splitext(file)

    
    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension=='.xlsx':
        from langchain.document_loaders import UnstructuredExcelLoader
        print(f'Loading {file}')
        loader=UnstructuredExcelLoader(file,mode="elements")
    elif extension =='.csv':
        from langchain.document_loaders.csv_loader import CSVLoader
        print(f'Loading {file}')
        loader=CSVLoader(file_path=file,csv_args={
        "delimiter": ",",
        "quotechar": '"',
        "fieldnames": ["Cases"],
    })
    else:
        print('Document format is not supported!')
        return None
    #data = loader.load()
    return loader


def chunk_data(data, chunk_size=150):
 
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    return chunks
 

def create_faiss_db(docs):
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db

def play_audio(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

file_path ="/Users/akansha/Downloads/annual report.pdf"
loader = load_document(file_path)
pages = loader.load()
chunks = chunk_data(pages)

vectorstore = create_faiss_db(chunks)

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
prompt_template = """  You have been provided annual report of L&T Finance Holdings. Please respond to user's questions based on information provided in context.
    {context}

    Question: {question}
    Answer:"""

PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
chain_type_kwargs = {"prompt": PROMPT}
  
llm = ChatOpenAI(model='gpt-4-1106-preview', temperature=0.2)

qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0), vectorstore.as_retriever(),memory=memory)


device = "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True,resume_download=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
    
)


def transcribe(audio):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    return pipe({"sampling_rate": sr, "raw": y})["text"]


# Front end web app
import gradio as gr
with gr.Blocks() as demo:
  
   
    
    chat_history = []

    

    def process_audio(original_text,audio,state):
        global chat_history
        conversation = state if state else ""
        user_message = ""
        global msg, chat_history
        if audio:
            user_message =  transcribe(audio)
        elif original_text:
            user_message = original_text
        
        
        response = qa({"question": user_message, "chat_history": chat_history})
       

        chat_history.append(response["answer"])
        speech_file_path = Path(__file__).parent / "speech.mp3"
        client = OpenAI()
        audioresponse = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=response["answer"]
        )

        audioresponse.stream_to_file(speech_file_path)

        play_audio(speech_file_path)
        updated_conversation = f"{conversation}\nUser: {user_message}\nBot: {response['answer']}"
        return updated_conversation,updated_conversation


    gr.Interface(
    fn= process_audio,
    inputs=[
        gr.Textbox(), gr.components.Audio(sources=["microphone"], type="numpy", label="Record Audio"),gr.State()
       
       
    ],
    outputs = [gr.components.Textbox(label="Conversation History", interactive=False),
    gr.State()],

    title="Voice Assistant",
    
)


demo.launch(debug=True)