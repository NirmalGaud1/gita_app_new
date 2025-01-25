#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Configure models
genai.configure(api_key="AIzaSyA-9-lTQTWdNM43YdOXMQwGKDy0SrMwo6c")
gemini = genai.GenerativeModel('gemini-pro')
embedder = SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    df = pd.read_csv('bhagwatgeeta.csv')
    
    df['context'] = df.apply(
        lambda row: (f"Chapter {row['Chapter']}, Verse {row['Verse']}\n"
                    f"Teaching: {row['EngMeaning']}\n"
                    f"Key Terms: {row['WordMeaning']}"), 
        axis=1
    )
    
    embeddings = embedder.encode(df['context'].tolist(), show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    
    return df, index

df, faiss_index = load_data()

st.title("Bhagavad Gita Chatbot")
st.subheader("Get philosophical insights + practical life solutions")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def get_life_advice(query):
    query_embedding = embedder.encode([query])
    D, I = faiss_index.search(query_embedding.astype('float32'), k=5)
    
    contexts = [df.iloc[i]['context'] for i in I[0]]
    
    prompt = f"""
    You are a wise Gita counselor. Structure your answer as:
    
    [Analysis]
    1. Key Teachings: 3-5 main philosophical points from contexts
    2. Sanskrit Terms: Important words with meanings
    3. Chapter References: Relevant verses
    
    [Personal Guidance]
    Write 1-2 empathetic paragraphs connecting teachings to the asker's situation. Use:
    - "In modern life..." 
    - "When facing..."
    - "Practical steps..."
    - "You might consider..."
    - "Remember that..."
    
    Contexts:
    {contexts}
    
    Question: {query}
    """
    
    response = gemini.generate_content(prompt)
    return response.text

if prompt := st.chat_input("What life challenge are you facing?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Finding wisdom for your situation..."):
        try:
            response = get_life_advice(prompt)
            # Formatting enhancements
            response = response.replace("[Analysis]", "## Philosophical Insights")
            response = response.replace("[Personal Guidance]", "## Life Application")
            response = response.replace("1. ", "◆ ")
        except:
            response = "Please ask your question in different words"
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

