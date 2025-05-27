import streamlit as st
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from transformers import pipeline
import plotly.express as px
import joblib
import os

st.set_page_config(layout="wide", page_title="jp_company")

st.title("Análisis de Opiniones de Clientes")
st.markdown("---")

MODEL_PATH = "sentiment_model.pkl"

@st.cache_resource
def load_sentiment_model():
    if os.path.exists(MODEL_PATH):
        st.write(f"Cargando el modelo de sentimientos desde {MODEL_PATH}...")
        return joblib.load(MODEL_PATH)
    else:
        st.write("Descargando y guardando el modelo de sentimientos (primera vez)...")
        classifier = pipeline("sentiment-analysis", model="finiteautomata/beto-sentiment-analysis")
        joblib.dump(classifier, MODEL_PATH)
        st.success(f"Modelo guardado en {MODEL_PATH}")
        return classifier

classifier = load_sentiment_model()

def clean_text(text):
    text = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ\s]', '', text)
    text = text.lower()
    return text

def remove_stopwords(text, stopwords_list):
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords_list]
    return " ".join(filtered_words)

spanish_stopwords = [
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para", "con", "no", "una", "su", "al", "es", "o", "lo", "como", "más", "pero", "sus", "le", "ha", "me", "si", "sin", "sobre", "este", "mi", "ya", "muy", "también", "esto", "ir", "haber", "ser", "tener", "estar", "hacer", "poder", "decir", "ver", "dar", "saber", "querer", "llegar", "pasar", "deber", "quedar", "parecer", "creer", "hablar", "llevar", "dejar", "seguir", "encontrar", "llamar", "venir", "pensar", "tomar", "conocer", "salir", "quedar", "entender", "recibir", "recordar", "cumplir", "esperar", "buscar", "resultar", "volver", "cambiar", "sentir", "enviar", "comenzar", "ayudar", "entrar", "presentar", "demostrar", "obtener", "incluir", "considerar", "desarrollar", "permanecer", "establecer", "construir", "organizar", "producir", "ofrecer", "necesitar", "realizar", "aceptar", "compartir", "dirigir", "participar", "evitar", "obtener", "comprar", "vender", "usar", "utilizar", "mejor", "peor", "bueno", "malo", "gran", "poco", "mucho", "todo", "nada", "cada", "mismo", "otro", "varios", "algunos", "primer", "segundo", "último", "nuevo", "viejo", "grande", "pequeño", "alto", "bajo", "cerca", "lejos", "aquí", "allí", "siempre", "nunca", "a veces", "antes", "después", "durante", "entonces", "luego", "pronto", "tarde", "hoy", "mañana", "ayer", "ahora", "donde", "cuando", "como", "cuanto", "quien", "que"
]

st.header("1. Subir Opiniones de Clientes (Archivo .csv)")
uploaded_file = st.file_uploader("Sube tu archivo .csv (una opinión por fila/columna). Asegúrate de que las opiniones estén en una columna llamada 'opinion' o 'texto'.", type="csv")

opiniones_df = pd.DataFrame()

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        if 'opinion' in data.columns:
            opiniones_df = data[['opinion']].copy()
        elif 'texto' in data.columns:
            opiniones_df = data[['texto']].copy()
            opiniones_df.rename(columns={'texto': 'opinion'}, inplace=True)
        elif len(data.columns) > 0: 
            opiniones_df = data.iloc[:, 0:1].copy()
            opiniones_df.columns = ['opinion']
        else:
            st.error("No se encontraron columnas de opiniones en el archivo. Por favor, asegúrate de que contenga una columna llamada 'opinion' o 'texto', o que las opiniones sean la primera columna.")
            opiniones_df = pd.DataFrame()

        if not opiniones_df.empty:
            if len(opiniones_df) > 20:
                st.warning(f"Se detectaron {len(opiniones_df)} opiniones. Se tomarán las primeras 20 para el análisis.")
                opiniones_df = opiniones_df.head(20)

            st.success(f"Se han cargado {len(opiniones_df)} opiniones correctamente.")
            st.dataframe(opiniones_df)

            # --- Análisis de texto (Nube de palabras y palabras más frecuentes) ---
            st.markdown("---")
            st.header("2. Análisis de Texto: Nube de Palabras y Frecuencia")
            if not opiniones_df.empty:
                all_words = " ".join(opiniones_df['opinion'].astype(str).apply(clean_text))
                all_words_filtered = remove_stopwords(all_words, spanish_stopwords)

                if all_words_filtered:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Nube de Palabras")
                        wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(all_words_filtered)
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)

                    with col2:
                        st.subheader("10 Palabras Más Repetidas")
                        words = all_words_filtered.split()
                        word_counts = Counter(words)
                        top_10_words = word_counts.most_common(10)
                        
                        if top_10_words:
                            df_top_words = pd.DataFrame(top_10_words, columns=['Palabra', 'Frecuencia'])
                            fig_bar = px.bar(df_top_words, x='Palabra', y='Frecuencia', 
                                             title='Top 10 Palabras Más Frecuentes',
                                             labels={'Palabra': 'Palabra', 'Frecuencia': 'Frecuencia'},
                                             color='Frecuencia', color_continuous_scale=px.colors.sequential.Plasma)
                            st.plotly_chart(fig_bar, use_container_width=True)
                        else:
                            st.info("No hay palabras para mostrar después de eliminar stopwords.")
                else:
                    st.info("No hay texto válido para generar la nube de palabras y el gráfico de frecuencia después de la limpieza y eliminación de stopwords.")

            # --- Clasificación de Sentimientos ---
            st.markdown("---")
            st.header("3. Clasificación de Sentimientos de las Opiniones")
            if not opiniones_df.empty:
                @st.cache_data
                def classify_sentiments(opinions):
                    results = classifier(opinions.tolist())
                    sentiments = [res['label'] for res in results]
                    scores = [res['score'] for res in results]
                    return sentiments, scores

                with st.spinner("Clasificando sentimientos... esto puede tomar un momento."):
                    sentiments, scores = classify_sentiments(opiniones_df['opinion'])
                    opiniones_df['Sentimiento'] = sentiments
                    opiniones_df['Confianza'] = scores

                st.subheader("Resultados de la Clasificación por Opinión")
                st.dataframe(opiniones_df[['opinion', 'Sentimiento', 'Confianza']], use_container_width=True)

                st.subheader("Porcentaje de Opiniones por Clase de Sentimiento")
                sentiment_counts = opiniones_df['Sentimiento'].value_counts(normalize=True) * 100
                df_sentiment_counts = sentiment_counts.reset_index()
                df_sentiment_counts.columns = ['Sentimiento', 'Porcentaje']

                fig_pie = px.pie(df_sentiment_counts, values='Porcentaje', names='Sentimiento', 
                                 title='Distribución de Sentimientos',
                                 color_discrete_map={'POS': 'green', 'NEG': 'red', 'NEU': 'blue'})
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Sube opiniones para ver la clasificación de sentimientos.")

    except Exception as e:
        st.error(f"Error al leer el archivo CSV o procesar las opiniones: {e}")
        st.info("Asegúrate de que el archivo CSV esté bien formado y que las opiniones estén en la primera columna o en una columna llamada 'opinion' o 'texto'.")

# --- Sección de Comentarios Nuevos ---
st.markdown("---")
st.header("4. Analizar un Comentario Nuevo")
new_comment = st.text_area("Ingresa un nuevo comentario para analizar:")

if st.button("Clasificar Comentario"):
    if new_comment:
        with st.spinner("Analizando comentario..."):
            result = classifier(new_comment)[0]
            sentiment_label = result['label']
            sentiment_score = result['score']

            st.subheader("Resultado del Análisis")
            st.write(f"**Sentimiento:** {sentiment_label} (Confianza: {sentiment_score:.2f})")

            # Mini resumen (simple, se podría mejorar con modelos de resumen)
            if sentiment_label == 'POS':
                st.info("Este comentario parece ser **positivo**.")
            elif sentiment_label == 'NEG':
                st.info("Este comentario parece ser **negativo**.")
            else:
                st.info("Este comentario parece ser **neutro**.")
            
            st.markdown(f"**Resumen/Observación:** La herramienta clasificó este comentario como **{sentiment_label}** con una confianza del **{sentiment_score:.2%}**.")
            st.markdown("Ten en cuenta que este es un resumen básico; un modelo de resumen más avanzado podría dar un resumen más contextual.")

    else:
        st.warning("Por favor, ingresa un comentario para analizar.")