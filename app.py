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
import threading
import re
from datetime import datetime
import json
import base64

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_matches_from_oddschecker(progol_number):
    """
    Scrape match data from oddschecker.com for the given Progol contest number
    """
    url = f"https://www.oddschecker.com/es/pronosticos/futbol/predicciones-progol-revancha-quiniela-esta-semana-{progol_number}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the tables for Progol and Revancha
        tables = soup.find_all('table')
        
        if len(tables) < 2:
            return None, None
        
        # Process Progol table
        progol_matches = []
        progol_rows = tables[0].find_all('tr')[1:]  # Skip header row
        
        for row in progol_rows:
            cols = row.find_all('td')
            if len(cols) >= 5:
                local = cols[0].text.strip()
                visitante = cols[1].text.strip()
                odds_l = cols[2].text.strip()
                odds_e = cols[3].text.strip()
                odds_v = cols[4].text.strip()
                
                progol_matches.append({
                    'local': local,
                    'visitante': visitante,
                    'odds_l': odds_l,
                    'odds_e': odds_e,
                    'odds_v': odds_v,
                    'resultado': '',
                    'status': 'Pendiente'
                })
        
        # Process Revancha table if exists (second table)
        revancha_matches = []
        if len(tables) > 1:
            revancha_rows = tables[1].find_all('tr')[1:]  # Skip header row
            
            for row in revancha_rows:
                cols = row.find_all('td')
                if len(cols) >= 5:
                    local = cols[0].text.strip()
                    visitante = cols[1].text.strip()
                    odds_l = cols[2].text.strip()
                    odds_e = cols[3].text.strip()
                    odds_v = cols[4].text.strip()
                    
                    revancha_matches.append({
                        'local': local,
                        'visitante': visitante,
                        'odds_l': odds_l,
                        'odds_e': odds_e,
                        'odds_v': odds_v,
                        'resultado': '',
                        'status': 'Pendiente'
                    })
        
        return progol_matches, revancha_matches
    
    except Exception as e:
        st.error(f"Error fetching data from oddschecker: {str(e)}")
        return None, None

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
# BACKGROUND RESULT FETCHING
# ============================================================================
def update_results_background(progol_df, revancha_df, stop_event):
    """
    Background thread to update match results periodically
    """
    while not stop_event.is_set():
        # Update progol results
        if progol_df is not None and not progol_df.empty:
            for idx, match in progol_df.iterrows():
                resultado, status = get_live_results(match)
                
                # Check if result has changed
                old_resultado = progol_df.at[idx, 'resultado']
                old_status = progol_df.at[idx, 'status']
                
                # Update the dataframe with new results
                progol_df.at[idx, 'resultado'] = resultado
                progol_df.at[idx, 'status'] = status
                
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
        if revancha_df is not None and not revancha_df.empty:
            for idx, match in revancha_df.iterrows():
                resultado, status = get_live_results(match)
                
                # Check if result has changed
                old_resultado = revancha_df.at[idx, 'resultado']
                old_status = revancha_df.at[idx, 'status']
                
                # Update the dataframe with new results
                revancha_df.at[idx, 'resultado'] = resultado
                revancha_df.at[idx, 'status'] = status
                
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
        
        # Sleep for a while before checking again
        time.sleep(10)  # Update every 10 seconds

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # Set up session state
    if 'progol_matches' not in st.session_state:
        st.session_state.progol_matches = None
    if 'revancha_matches' not in st.session_state:
        st.session_state.revancha_matches = None
    if 'stop_thread' not in st.session_state:
        st.session_state.stop_thread = threading.Event()
    if 'update_thread' not in st.session_state:
        st.session_state.update_thread = None
    if 'notifications' not in st.session_state:
        st.session_state.notifications = []
    if 'progol_selections' not in st.session_state:
        st.session_state.progol_selections = []
    if 'revancha_selections' not in st.session_state:
        st.session_state.revancha_selections = []

    # App title and description
    st.title("Progol Quiniela Tracker")
    st.write("Esta aplicación te permite seguir los resultados de tus quinielas Progol y Revancha en tiempo real.")
    
    # Sidebar
    st.sidebar.header("Configuración")
    
    # Input for Progol contest number
    progol_number = st.sidebar.text_input("Número de concurso Progol:", value="2272")
    
    if st.sidebar.button("Cargar partidos"):
        with st.spinner("Obteniendo información de partidos..."):
            progol_matches, revancha_matches = get_matches_from_oddschecker(progol_number)
            
            if progol_matches:
                st.session_state.progol_matches = pd.DataFrame(progol_matches)
                st.success(f"Se cargaron {len(progol_matches)} partidos de Progol")
            else:
                st.error("No se pudieron cargar los partidos de Progol")
            
            if revancha_matches:
                st.session_state.revancha_matches = pd.DataFrame(revancha_matches)
                st.success(f"Se cargaron {len(revancha_matches)} partidos de Revancha")
            else:
                st.warning("No se pudieron cargar los partidos de Revancha")
            
            # Stop existing thread if it's running
            if st.session_state.update_thread and st.session_state.update_thread.is_alive():
                st.session_state.stop_thread.set()
                st.session_state.update_thread.join()
                st.session_state.stop_thread.clear()
            
            # Start new background thread for updates
            if progol_matches or revancha_matches:
                st.session_state.update_thread = threading.Thread(
                    target=update_results_background,
                    args=(st.session_state.progol_matches, st.session_state.revancha_matches, st.session_state.stop_thread)
                )
                st.session_state.update_thread.daemon = True
                st.session_state.update_thread.start()

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
    
    # Main content tabs
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
            st.markdown(create_download_link(display_df, f"progol_{progol_number}.csv", "Descargar resultados como CSV"), unsafe_allow_html=True)
            
            # Show summary of correct predictions
            if 'Acierto' in display_df.columns:
                correct_count = display_df['Acierto'].value_counts().get('✓', 0)
                total_finished = display_df['status'].value_counts().get('Finalizado', 0)
                
                if total_finished > 0:
                    st.metric("Aciertos", f"{correct_count}/{total_finished}")
        else:
            st.info("No hay partidos cargados. Ingresa un número de concurso y haz clic en 'Cargar partidos'.")
    
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
            st.markdown(create_download_link(display_df, f"revancha_{progol_number}.csv", "Descargar resultados como CSV"), unsafe_allow_html=True)
            
            # Show summary of correct predictions
            if 'Acierto' in display_df.columns:
                correct_count = display_df['Acierto'].value_counts().get('✓', 0)
                total_finished = display_df['status'].value_counts().get('Finalizado', 0)
                
                if total_finished > 0:
                    st.metric("Aciertos", f"{correct_count}/{total_finished}")
        else:
            st.info("No hay partidos cargados. Ingresa un número de concurso y haz clic en 'Cargar partidos'.")
            
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

    # Add auto-refresh to update UI
    st.empty()
    time.sleep(1)
    st.experimental_rerun()

# Run the main application
if __name__ == "__main__":
    main()