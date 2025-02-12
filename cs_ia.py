import cv2
import pandas as pd
import openai
import streamlit as st
import yt_dlp
from ultralytics import YOLO

# Configuração da API OpenAI
openai.api_key = "API_KEY_OPENAI"

def download_video(youtube_url, output_path="video.mp4"):
    ydl_opts = {
        "format": "mp4",
        "outtmpl": output_path,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return output_path

def process_video(video_path):
    model = YOLO("yolov8n.pt")  # Modelo YOLO pré-treinado
    cap = cv2.VideoCapture(video_path)
    stats = {"kills": 0, "movements": 0, "accuracy": 0}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        for result in results:
            for box in result.boxes:
                label = result.names[int(box.cls[0])]
                if label in ["person", "weapon"]:
                    stats["kills"] += 1 if label == "weapon" else 0
                    stats["movements"] += 1 if label == "person" else 0
    
    cap.release()
    return stats

# Função para gerar insights
def generate_insights(stats):
    prompt = f"""
    Baseado nesses dados de jogabilidade:
    - Kills: {stats['kills']}
    - Movimentação: {stats['movements']}
    - Precisão: {stats['accuracy']}
    Sugira como o jogador pode melhorar.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "Você é um coach de CS profissional."},
                  {"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

# Interface no Streamlit
st.title("CS Tactics AI - Analisador de Partidas")

# Input do vídeo
video_url = st.text_input("Cole o link do vídeo do YouTube")

if st.button("Baixar e Analisar Vídeo"):
    if video_url:
        with st.spinner('Baixando e processando o vídeo...'):
            video_path = download_video(video_url, "youtube_video.mp4")
            stats = process_video(video_path)
            insights = generate_insights(stats)

            st.write("### 📊 Estatísticas de Jogo")
            st.json(stats)
            
            st.write("### 🎯 Sugestões de Melhorias")
            st.write(insights)

# Enviar vídeo próprio
uploaded_file = st.file_uploader("Ou envie seu próprio vídeo de gameplay", type=["mp4", "avi", "mov"])

if uploaded_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner('Processando seu vídeo...'):
        stats = process_video("temp_video.mp4")
        insights = generate_insights(stats)
        
        st.write("### 📊 Estatísticas de Jogo")
        st.json(stats)
        
        st.write("### 🎯 Sugestões de Melhorias")
        st.write(insights)
