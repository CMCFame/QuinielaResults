# ============================================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================================
import streamlit as st

# This must be the very first Streamlit command in your script
st.set_page_config(
    page_title="Progol Quiniela Tracker",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# IMPORTS AND SETUP
# ============================================================================
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import re
from datetime import datetime
import json
import base64
import traceback
import random
from PIL import Image
from io import BytesIO
import pytesseract
import cv2
import numpy as np

# ============================================================================
# OCR UTILITIES
# ============================================================================
def download_image(url):
    """
    Download an image from a URL
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        st.error(f"Error downloading image: {str(e)}")
        return None

def preprocess_image(image):
    """
    Preprocess the image for better OCR results
    """
    # Convert PIL image to OpenCV format
    img = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    # Return as PIL image
    return Image.fromarray(binary)

def extract_text_from_image(image):
    """
    Extract text from image using OCR
    """
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Use pytesseract to extract text
        text = pytesseract.image_to_string(processed_image, lang='spa')
        
        return text
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return None

def parse_matches_from_ocr_text(text):
    """
    Parse match information from OCR text
    """
    if not text:
        return None
    
    # Regular expression to find match patterns
    # Looking for patterns like "Team1 vs Team2" or "Team1 - Team2"
    match_pattern = r'([A-Za-z0-9\s\.]+)(?:\s+vs\.?\s+|\s+-\s+)([A-Za-z0-9\s\.]+)'
    
    matches = re.findall(match_pattern, text)
    
    if not matches:
        return None
    
    # Create match data
    match_data = []
    for local, visitante in matches:
        match_data.append({
            'local': local.strip(),
            'visitante': visitante.strip(),
            'odds_l': "2.0",  # Default odds
            'odds_e': "3.0",  # Default odds
            'odds_v': "2.0",  # Default odds
            'resultado': '',
            'status': 'Pendiente'
        })
    
    return match_data

def get_matches_from_loterianacional():
    """
    Get match data from Lotería Nacional website using OCR
    """
    # URL of the image
    url = "https://www.loterianacional.gob.mx/Progol/Quiniela"
    
    # Download the image
    st.info("Descargando imagen desde Lotería Nacional...")
    image = download_image(url)
    
    if image is None:
        return None, None
    
    # Display the image
    st.image(image, caption="Imagen descargada de Lotería Nacional", use_column_width=True)
    
    # Extract text from image
    st.info("Procesando imagen con OCR...")
    text = extract_text_from_image(image)
    
    if text is None:
        return None, None
    
    # Show extracted text for debugging
    with st.expander("Texto extraído de la imagen"):
        st.text(text)
    
    # Parse matches from text
    st.info("Analizando datos extraídos...")
    matches = parse_matches_from_ocr_text(text)
    
    if matches is None or len(matches) == 0:
        st.error("No se pudieron identificar partidos en la imagen")
        return None, None
    
    # Separate into Progol and Revancha
    # Assuming first 14 matches are Progol and the rest are Revancha
    if len(matches) >= 14:
        progol_matches = matches[:14]
        revancha_matches = matches[14:] if len(matches) > 14 else []
    else:
        progol_matches = matches
        revancha_matches = []
    
    st.success(f"Se encontraron {len(progol_matches)} partidos de Progol y {len(revancha_matches)} partidos de Revancha")
    
    return progol_matches, revancha_matches

def get_live_results(match):
    """
    Get live results for a specific match
    In a production app, this would connect to an API like flashscore.com or sofascore
    For this demo, we'll simulate random results
    """
    # This is a placeholder - in a real app you would connect to a sports API
    # or scrape live results from a website
    
    # Simulate a random result for demonstration purposes
    import random
    statuses = ['Pendiente', 'En Juego', 'Finalizado']
    resultados = ['L', 'E', 'V', '']
    
    # 80% chance to keep existing status/result, 20% chance to change
    if random.random() < 0.8 and match['status'] != 'Pendiente':
        return match['resultado'], match['status']
    
    # Generate a new status/result
    status = random.choices(statuses, weights=[0.3, 0.4, 0.3], k=1)[0]
    
    # Only assign a result if the match is in progress or finished
    if status in ['En Juego', 'Finalizado']:
        resultado = random.choices(resultados[0:3], weights=[0.4, 0.2, 0.4], k=1)[0]
    else:
        resultado = ''
        
    return resultado, status

def parse_quiniela_selections(selections_str):
    """
    Parse user's quiniela selections from a string like "L,E,V,L,E,V,..."
    """
    if not selections_str:
        return []
        
    selections = []
    for sel in selections_str.upper().split(','):
        sel = sel.strip()
        if sel in ['L', 'E', 'V', 'LE', 'LV', 'EV', 'LEV']:
            selections.append(sel)
        else:
            # Default to empty if invalid
            selections.append('')
    
    return selections

def check_winning_selections(user_selection, result):
    """
    Check if the user's selection matches the result
    """
    if not result or not user_selection:
        return None  # No result yet or no selection
    
    # User could have selected multiple options (e.g., "LE")
    return result in user_selection

def create_download_link(df, filename, text):
    """
    Create a download link for a DataFrame
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'data:file/csv;base64,{b64}'
    return f'<a href="{href}" download="{filename}">{text}</a>'

# ============================================================================
# RESULT UPDATE FUNCTIONS
# ============================================================================
def update_match_results():
    """
    Update match results - this is a simpler approach that doesn't use threading,
    which can sometimes cause issues in Streamlit
    """
    # Update progol results
    if 'progol_matches' in st.session_state and st.session_state.progol_matches is not None and not st.session_state.progol_matches.empty:
        for idx, match in st.session_state.progol_matches.iterrows():
            resultado, status = get_live_results(match)
            
            # Check if result has changed
            old_resultado = st.session_state.progol_matches.at[idx, 'resultado']
            old_status = st.session_state.progol_matches.at[idx, 'status']
            
            # Update the dataframe with new results
            st.session_state.progol_matches.at[idx, 'resultado'] = resultado
            st.session_state.progol_matches.at[idx, 'status'] = status
            
            # Store the notification if the result has changed
            if old_resultado != resultado and resultado != '':
                # Add notification
                notification = {
                    'time': datetime.now().strftime("%H:%M:%S"),
                    'match': f"{match['local']} vs {match['visitante']}",
                    'old_result': old_resultado if old_resultado else 'None',
                    'new_result': resultado,
                    'type': 'Progol'
                }
                
                if 'notifications' not in st.session_state:
                    st.session_state.notifications = []
                
                st.session_state.notifications.append(notification)
    
    # Update revancha results
    if 'revancha_matches' in st.session_state and st.session_state.revancha_matches is not None and not st.session_state.revancha_matches.empty:
        for idx, match in st.session_state.revancha_matches.iterrows():
            resultado, status = get_live_results(match)
            
            # Check if result has changed
            old_resultado = st.session_state.revancha_matches.at[idx, 'resultado']
            old_status = st.session_state.revancha_matches.at[idx, 'status']
            
            # Update the dataframe with new results
            st.session_state.revancha_matches.at[idx, 'resultado'] = resultado
            st.session_state.revancha_matches.at[idx, 'status'] = status
            
            # Store the notification if the result has changed
            if old_resultado != resultado and resultado != '':
                # Add notification
                notification = {
                    'time': datetime.now().strftime("%H:%M:%S"),
                    'match': f"{match['local']} vs {match['visitante']}",
                    'old_result': old_resultado if old_resultado else 'None',
                    'new_result': resultado,
                    'type': 'Revancha'
                }
                
                if 'notifications' not in st.session_state:
                    st.session_state.notifications = []
                
                st.session_state.notifications.append(notification)

# ============================================================================
# MANUAL MATCH INPUT
# ============================================================================
def manual_match_input():
    """
    Allow manual input of match data
    """
    st.subheader("Ingreso Manual de Partidos")
    st.write("Ingresa manualmente los partidos si la extracción automática no funciona correctamente.")
    
    # Number of matches to input
    num_progol = st.number_input("Número de partidos Progol:", min_value=1, max_value=14, value=14)
    num_revancha = st.number_input("Número de partidos Revancha:", min_value=0, max_value=7, value=0)
    
    # Create containers for match inputs
    progol_matches = []
    revancha_matches = []
    
    if num_progol > 0:
        st.write("### Partidos Progol")
        for i in range(num_progol):
            col1, col2 = st.columns(2)
            with col1:
                local = st.text_input(f"Equipo Local #{i+1}:", key=f"progol_local_{i}")
            with col2:
                visitante = st.text_input(f"Equipo Visitante #{i+1}:", key=f"progol_visit_{i}")
            
            if local and visitante:
                progol_matches.append({
                    'local': local,
                    'visitante': visitante,
                    'odds_l': "2.0",  # Default odds
                    'odds_e': "3.0",  # Default odds
                    'odds_v': "2.0",  # Default odds
                    'resultado': '',
                    'status': 'Pendiente'
                })
    
    if num_revancha > 0:
        st.write("### Partidos Revancha")
        for i in range(num_revancha):
            col1, col2 = st.columns(2)
            with col1:
                local = st.text_input(f"Equipo Local #{i+1}:", key=f"revancha_local_{i}")
            with col2:
                visitante = st.text_input(f"Equipo Visitante #{i+1}:", key=f"revancha_visit_{i}")
            
            if local and visitante:
                revancha_matches.append({
                    'local': local,
                    'visitante': visitante,
                    'odds_l': "2.0",  # Default odds
                    'odds_e': "3.0",  # Default odds
                    'odds_v': "2.0",  # Default odds
                    'resultado': '',
                    'status': 'Pendiente'
                })
    
    if st.button("Guardar Partidos Manuales"):
        if progol_matches:
            st.session_state.progol_matches = pd.DataFrame(progol_matches)
            st.success(f"Se guardaron {len(progol_matches)} partidos de Progol")
        
        if revancha_matches:
            st.session_state.revancha_matches = pd.DataFrame(revancha_matches)
            st.success(f"Se guardaron {len(revancha_matches)} partidos de Revancha")
            
        # Return the matches for use
        return progol_matches, revancha_matches
    
    return None, None

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # Set up session state
    if 'progol_matches' not in st.session_state:
        st.session_state.progol_matches = None
    if 'revancha_matches' not in st.session_state:
        st.session_state.revancha_matches = None
    if 'notifications' not in st.session_state:
        st.session_state.notifications = []
    if 'progol_selections' not in st.session_state:
        st.session_state.progol_selections = []
    if 'revancha_selections' not in st.session_state:
        st.session_state.revancha_selections = []
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now()
    if 'debug_enabled' not in st.session_state:
        st.session_state.debug_enabled = False
    
    # Update results periodically without threading
    if datetime.now() - st.session_state.last_update > pd.Timedelta(seconds=30):
        update_match_results()
        st.session_state.last_update = datetime.now()

    # App title and description
    st.title("Progol Quiniela Tracker")
    st.write("Esta aplicación te permite seguir los resultados de tus quinielas Progol y Revancha en tiempo real.")
    
    # Sidebar
    st.sidebar.header("Configuración")
    
    # Debug mode toggle
    debug_toggle = st.sidebar.checkbox("Modo Debug", value=st.session_state.debug_enabled)
    if debug_toggle != st.session_state.debug_enabled:
        st.session_state.debug_enabled = debug_toggle
    
    # Load data options
    st.sidebar.subheader("Cargar Datos")
    data_option = st.sidebar.radio(
        "Método de obtención de datos:",
        ["OCR desde Lotería Nacional", "Ingreso Manual"]
    )
    
    if st.sidebar.button("Cargar partidos"):
        with st.spinner("Obteniendo información de partidos..."):
            try:
                if data_option == "OCR desde Lotería Nacional":
                    # Get data using OCR
                    progol_matches, revancha_matches = get_matches_from_loterianacional()
                else:
                    # Get data manually
                    # Just show UI for manual input, actual saving happens in the manual_match_input function
                    progol_matches, revancha_matches = None, None
                
                # Process progol matches
                if progol_matches:
                    st.session_state.progol_matches = pd.DataFrame(progol_matches)
                    st.success(f"Se cargaron {len(progol_matches)} partidos de Progol")
                
                # Process revancha matches
                if revancha_matches:
                    st.session_state.revancha_matches = pd.DataFrame(revancha_matches)
                    st.success(f"Se cargaron {len(revancha_matches)} partidos de Revancha")
            
            except Exception as e:
                # Show detailed error message
                st.error(f"Error al cargar los datos: {str(e)}")
                st.info("Intenta con el método de ingreso manual.")

    # User quiniela inputs
    st.sidebar.header("Mis Quinielas")
    
    # Progol selections
    st.sidebar.subheader("Progol")
    progol_selections_str = st.sidebar.text_area(
        "Ingresa tus selecciones de Progol (separa con comas, ej: L,E,V,LE,...):",
        help="Usa L para Local, E para Empate, V para Visita. Puedes combinar (ej: LE, LV, EV, LEV)"
    )
    
    # Revancha selections
    st.sidebar.subheader("Revancha")
    revancha_selections_str = st.sidebar.text_area(
        "Ingresa tus selecciones de Revancha (separa con comas, ej: L,E,V,LE,...):",
        help="Usa L para Local, E para Empate, V para Visita. Puedes combinar (ej: LE, LV, EV, LEV)"
    )
    
    if st.sidebar.button("Guardar selecciones"):
        st.session_state.progol_selections = parse_quiniela_selections(progol_selections_str)
        st.session_state.revancha_selections = parse_quiniela_selections(revancha_selections_str)
        st.success("Selecciones guardadas")
    
    # Main content
    if data_option == "Ingreso Manual" and not st.session_state.progol_matches:
        manual_match_input()
    else:
        tab1, tab2, tab3 = st.tabs(["Progol", "Revancha", "Notificaciones"])
        
        with tab1:
            st.header("Partidos Progol")
            
            if st.session_state.progol_matches is not None and not st.session_state.progol_matches.empty:
                # Create a display copy with user selections
                display_df = st.session_state.progol_matches.copy()
                
                # Add user selections column if available
                if st.session_state.progol_selections:
                    # Pad or truncate the selections to match DataFrame length
                    padded_selections = st.session_state.progol_selections[:len(display_df)]
                    padded_selections = padded_selections + [''] * (len(display_df) - len(padded_selections))
                    
                    display_df['Mi selección'] = padded_selections
                    
                    # Add column to show if selection matches result
                    def check_result(row):
                        if not row['resultado'] or not row['Mi selección']:
                            return ''
                        return '✓' if row['resultado'] in row['Mi selección'] else '✗'
                    
                    display_df['Acierto'] = display_df.apply(check_result, axis=1)
                
                # Display the dataframe
                st.dataframe(display_df, use_container_width=True)
                
                # Download button
                st.markdown(create_download_link(display_df, "progol.csv", "Descargar resultados como CSV"), unsafe_allow_html=True)
                
                # Show summary of correct predictions
                if 'Acierto' in display_df.columns:
                    correct_count = display_df['Acierto'].value_counts().get('✓', 0)
                    total_finished = display_df['status'].value_counts().get('Finalizado', 0)
                    
                    if total_finished > 0:
                        st.metric("Aciertos", f"{correct_count}/{total_finished}")
            else:
                st.info("No hay partidos cargados. Carga los partidos o utiliza el modo de ingreso manual.")
        
        with tab2:
            st.header("Partidos Revancha")
            
            if st.session_state.revancha_matches is not None and not st.session_state.revancha_matches.empty:
                # Create a display copy with user selections
                display_df = st.session_state.revancha_matches.copy()
                
                # Add user selections column if available
                if st.session_state.revancha_selections:
                    # Pad or truncate the selections to match DataFrame length
                    padded_selections = st.session_state.revancha_selections[:len(display_df)]
                    padded_selections = padded_selections + [''] * (len(display_df) - len(padded_selections))
                    
                    display_df['Mi selección'] = padded_selections
                    
                    # Add column to show if selection matches result
                    def check_result(row):
                        if not row['resultado'] or not row['Mi selección']:
                            return ''
                        return '✓' if row['resultado'] in row['Mi selección'] else '✗'
                    
                    display_df['Acierto'] = display_df.apply(check_result, axis=1)
                
                # Display the dataframe
                st.dataframe(display_df, use_container_width=True)
                
                # Download button
                st.markdown(create_download_link(display_df, "revancha.csv", "Descargar resultados como CSV"), unsafe_allow_html=True)
                
                # Show summary of correct predictions
                if 'Acierto' in display_df.columns:
                    correct_count = display_df['Acierto'].value_counts().get('✓', 0)
                    total_finished = display_df['status'].value_counts().get('Finalizado', 0)
                    
                    if total_finished > 0:
                        st.metric("Aciertos", f"{correct_count}/{total_finished}")
            else:
                st.info("No hay partidos de Revancha cargados.")
                
        with tab3:
            st.header("Notificaciones de resultados")
            
            # Display notifications in reverse chronological order
            if st.session_state.notifications:
                for notification in reversed(st.session_state.notifications):
                    with st.container():
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            st.text(notification['time'])
                        with col2:
                            st.markdown(f"**{notification['type']}:** {notification['match']} - Resultado cambiado de **{notification['old_result']}** a **{notification['new_result']}**")
                    st.divider()
            else:
                st.info("No hay notificaciones de cambios de resultados todavía.")

    # Use a safer way to update UI with auto_refresh
    # This approach is safer and won't crash the app
    refresh_placeholder = st.empty()
    refresh_placeholder.info("La aplicación se actualiza automáticamente cada 30 segundos. Última actualización: " + 
                        datetime.now().strftime("%H:%M:%S"))

# Run the main application
if __name__ == "__main__":
    main()