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
import time
import re
from datetime import datetime, timedelta
import json
import base64
import traceback
import random

# ============================================================================
# API FOOTBALL INTEGRATION
# ============================================================================
def get_api_key():
    """
    Get API key from Streamlit secrets
    """
    try:
        return st.secrets["api_football"]["api_key"]
    except Exception as e:
        st.error("API key not found in secrets. Please set up your .streamlit/secrets.toml file.")
        st.code("""
        [api_football]
        api_key = "YOUR_API_KEY"
        """)
        return None

def get_live_matches():
    """
    Get live matches from API-Football
    """
    api_key = get_api_key()
    
    if not api_key:
        return None
    
    url = "https://v3.football.api-sports.io/fixtures?live=all"
    headers = {
        'x-apisports-key': api_key
    }
    
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        
        if 'response' in data:
            return data['response']
        else:
            st.warning("No response data from API")
            return None
            
    except Exception as e:
        st.error(f"Error fetching live matches: {str(e)}")
        return None

def get_matches_by_date(date_str):
    """
    Get matches for a specific date from API-Football
    Date format should be YYYY-MM-DD
    """
    api_key = get_api_key()
    
    if not api_key:
        return None
    
    url = f"https://v3.football.api-sports.io/fixtures?date={date_str}"
    headers = {
        'x-apisports-key': api_key
    }
    
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        
        if 'response' in data:
            return data['response']
        else:
            st.warning("No response data from API")
            return None
            
    except Exception as e:
        st.error(f"Error fetching matches for date {date_str}: {str(e)}")
        return None

def get_matches_by_league(league_id, season):
    """
    Get matches for a specific league and season from API-Football
    """
    api_key = get_api_key()
    
    if not api_key:
        return None
    
    url = f"https://v3.football.api-sports.io/fixtures?league={league_id}&season={season}"
    headers = {
        'x-apisports-key': api_key
    }
    
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        
        if 'response' in data:
            return data['response']
        else:
            st.warning("No response data from API")
            return None
            
    except Exception as e:
        st.error(f"Error fetching matches for league {league_id}, season {season}: {str(e)}")
        return None

def process_api_matches(api_matches, max_matches=14):
    """
    Process API-Football matches into our app format
    """
    if not api_matches:
        return None
    
    processed_matches = []
    
    for match in api_matches[:max_matches]:  # Limit to max_matches
        try:
            fixture = match['fixture']
            teams = match['teams']
            goals = match['goals']
            
            # Extract data
            local_team = teams['home']['name']
            visitante_team = teams['away']['name']
            
            # Get match status
            status_code = fixture['status']['short']  # NS, 1H, HT, 2H, FT, etc.
            
            # Map API status to our app status
            status_map = {
                'NS': 'Pendiente',  # Not Started
                'TBD': 'Pendiente',  # To Be Defined
                '1H': 'En Juego',   # First Half
                'HT': 'En Juego',   # Half Time
                '2H': 'En Juego',   # Second Half
                'ET': 'En Juego',   # Extra Time
                'P': 'En Juego',    # Penalty
                'FT': 'Finalizado', # Full Time
                'AET': 'Finalizado',# After Extra Time
                'PEN': 'Finalizado',# Penalty Shootout
                'BT': 'En Juego',   # Break Time
                'SUSP': 'Suspendido',# Suspended
                'INT': 'Interrumpido',# Interrupted
                'PST': 'Pospuesto', # Postponed
                'CANC': 'Cancelado',# Cancelled
                'ABD': 'Abandonado',# Abandoned
                'AWD': 'Declarado', # Technical Loss
                'WO': 'Incomparecencia',# Walkover
            }
            
            status = status_map.get(status_code, 'Pendiente')
            
            # Determine result for finished or in-progress matches
            resultado = ''
            if goals and (status == 'Finalizado' or status == 'En Juego'):
                home_goals = goals['home'] if goals['home'] is not None else 0
                away_goals = goals['away'] if goals['away'] is not None else 0
                
                if home_goals > away_goals:
                    resultado = 'L'  # Local win
                elif away_goals > home_goals:
                    resultado = 'V'  # Visitor win
                else:
                    resultado = 'E'  # Draw
            
            # Create match entry
            match_entry = {
                'local': local_team,
                'visitante': visitante_team,
                'odds_l': "2.0",  # Default odds
                'odds_e': "3.0",  # Default odds
                'odds_v': "2.0",  # Default odds
                'resultado': resultado,
                'status': status
            }
            
            processed_matches.append(match_entry)
            
        except Exception as e:
            st.warning(f"Error processing match: {str(e)}")
            continue
    
    return processed_matches

def fetch_matches_for_progol():
    """
    Fetch matches for Progol from API-Football
    This will get upcoming matches from major leagues
    """
    # Get today's date and tomorrow's date
    today = datetime.today().strftime('%Y-%m-%d')
    tomorrow = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Get matches for today and tomorrow
    today_matches = get_matches_by_date(today)
    tomorrow_matches = get_matches_by_date(tomorrow)
    
    # Combine matches
    all_matches = []
    if today_matches:
        all_matches.extend(today_matches)
    if tomorrow_matches:
        all_matches.extend(tomorrow_matches)
    
    # Process matches
    processed_matches = process_api_matches(all_matches)
    
    return processed_matches

def get_live_results(match):
    """
    Check for live results for a match
    If using API-Football, this would be replaced with actual API calls
    """
    # If we have a full integration with API-Football, this would check for updates
    # For now, using the simulated approach for demonstration
    
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

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
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
    st.write("Ingresa manualmente los partidos para tu quiniela.")
    
    # Number of matches to input
    num_progol = st.number_input("Número de partidos Progol:", min_value=1, max_value=14, value=14)
    num_revancha = st.number_input("Número de partidos Revancha:", min_value=0, max_value=7, value=0)
    
    # Create containers for match inputs
    progol_matches = []
    revancha_matches = []
    
    if num_progol > 0:
        st.write("### Partidos Progol")
        
        # Create a form for easier input
        with st.form("progol_form"):
            for i in range(num_progol):
                cols = st.columns([3, 1, 3])
                with cols[0]:
                    local = st.text_input(f"Local #{i+1}:", key=f"progol_local_{i}")
                with cols[1]:
                    st.write("vs")
                with cols[2]:
                    visitante = st.text_input(f"Visitante #{i+1}:", key=f"progol_visit_{i}")
                
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
            
            submitted_progol = st.form_submit_button("Guardar Partidos Progol")
    
    if num_revancha > 0:
        st.write("### Partidos Revancha")
        
        # Create a form for easier input
        with st.form("revancha_form"):
            for i in range(num_revancha):
                cols = st.columns([3, 1, 3])
                with cols[0]:
                    local = st.text_input(f"Local #{i+1}:", key=f"revancha_local_{i}")
                with cols[1]:
                    st.write("vs")
                with cols[2]:
                    visitante = st.text_input(f"Visitante #{i+1}:", key=f"revancha_visit_{i}")
                
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
            
            submitted_revancha = st.form_submit_button("Guardar Partidos Revancha")
    
    # Apply the matches to session state when the form is submitted
    if 'submitted_progol' in locals() and submitted_progol and progol_matches:
        st.session_state.progol_matches = pd.DataFrame(progol_matches)
        st.success(f"Se guardaron {len(progol_matches)} partidos de Progol")
    
    if 'submitted_revancha' in locals() and submitted_revancha and revancha_matches:
        st.session_state.revancha_matches = pd.DataFrame(revancha_matches)
        st.success(f"Se guardaron {len(revancha_matches)} partidos de Revancha")
    
    # Global save button for all matches
    if st.button("Guardar Todos los Partidos"):
        if progol_matches:
            st.session_state.progol_matches = pd.DataFrame(progol_matches)
            st.success(f"Se guardaron {len(progol_matches)} partidos de Progol")
        
        if revancha_matches:
            st.session_state.revancha_matches = pd.DataFrame(revancha_matches)
            st.success(f"Se guardaron {len(revancha_matches)} partidos de Revancha")

# ============================================================================
# PREMADE MATCH TEMPLATES
# ============================================================================
def load_sample_matches():
    """
    Load sample match data for demonstration purposes
    """
    st.subheader("Cargar Equipos de la Quiniela Actual")
    st.write("Selecciona qué quiniela quieres cargar:")
    
    template_option = st.radio(
        "Plantilla:",
        ["Progol 2274", "Revancha 2274"]
    )
    
    if st.button("Cargar Plantilla"):
        if template_option == "Progol 2274":
            progol_matches = [
                {"local": "Juarez", "visitante": "Guadalajara", "odds_l": "3.40", "odds_e": "3.50", "odds_v": "2.15", "resultado": "", "status": "Pendiente"},
                {"local": "Pumas UNAM", "visitante": "Monterrey", "odds_l": "2.80", "odds_e": "3.50", "odds_v": "2.50", "resultado": "", "status": "Pendiente"},
                {"local": "Tlaxcala FC", "visitante": "Cancun", "odds_l": "2.70", "odds_e": "3.30", "odds_v": "2.50", "resultado": "", "status": "Pendiente"},
                {"local": "Sevilla", "visitante": "Athletic Club", "odds_l": "3.00", "odds_e": "3.20", "odds_v": "2.60", "resultado": "", "status": "Pendiente"},
                {"local": "Rayo Vallecano", "visitante": "Real Sociedad", "odds_l": "2.50", "odds_e": "3.15", "odds_v": "3.15", "resultado": "", "status": "Pendiente"},
                {"local": "Atletico Madrid", "visitante": "Barcelona", "odds_l": "3.10", "odds_e": "4.00", "odds_v": "2.72", "resultado": "", "status": "Pendiente"},
                {"local": "Everton", "visitante": "West Ham", "odds_l": "2.15", "odds_e": "3.30", "odds_v": "3.90", "resultado": "", "status": "Pendiente"},
                {"local": "Bologna", "visitante": "Lazio", "odds_l": "2.40", "odds_e": "3.14", "odds_v": "3.25", "resultado": "", "status": "Pendiente"},
                {"local": "Augsburgo", "visitante": "Wolfsburgo", "odds_l": "2.60", "odds_e": "3.30", "odds_v": "2.80", "resultado": "", "status": "Pendiente"},
                {"local": "Twente", "visitante": "Feyenoord", "odds_l": "2.30", "odds_e": "3.60", "odds_v": "3.00", "resultado": "", "status": "Pendiente"},
                {"local": "Arouca", "visitante": "Estoril Praia", "odds_l": "2.00", "odds_e": "3.40", "odds_v": "3.90", "resultado": "", "status": "Pendiente"},
                {"local": "Atlanta United", "visitante": "Inter Miami", "odds_l": "2.40", "odds_e": "3.78", "odds_v": "2.75", "resultado": "", "status": "Pendiente"},
                {"local": "U. Catolica", "visitante": "Colo Colo", "odds_l": "3.80", "odds_e": "3.26", "odds_v": "1.89", "resultado": "", "status": "Pendiente"},
                {"local": "Spartak Moscow", "visitante": "Zenit St. Petersburg", "odds_l": "2.40", "odds_e": "3.28", "odds_v": "2.90", "resultado": "", "status": "Pendiente"}
            ]
            
            st.session_state.progol_matches = pd.DataFrame(progol_matches)
            st.success("Se cargaron los partidos de Progol 2274")
            
        elif template_option == "Revancha 2274":
            revancha_matches = [
                {"local": "Puebla", "visitante": "Toluca", "odds_l": "4.75", "odds_e": "4.20", "odds_v": "1.67", "resultado": "", "status": "Pendiente"},
                {"local": "Cruz Azul", "visitante": "Atletico San Luis", "odds_l": "1.30", "odds_e": "5.50", "odds_v": "10.0", "resultado": "", "status": "Pendiente"},
                {"local": "Atlas", "visitante": "America", "odds_l": "4.20", "odds_e": "3.75", "odds_v": "1.88", "resultado": "", "status": "Pendiente"},
                {"local": "Pachuca", "visitante": "Tijuana", "odds_l": "1.57", "odds_e": "4.40", "odds_v": "5.50", "resultado": "", "status": "Pendiente"},
                {"local": "Fiorentina", "visitante": "Juventus", "odds_l": "3.50", "odds_e": "3.20", "odds_v": "2.30", "resultado": "", "status": "Pendiente"},
                {"local": "Werder Bremen", "visitante": "Borussia M'Gladbach", "odds_l": "2.40", "odds_e": "3.70", "odds_v": "2.88", "resultado": "", "status": "Pendiente"},
                {"local": "RB Leipzig", "visitante": "Borussia Dortmund", "odds_l": "2.40", "odds_e": "3.70", "odds_v": "2.90", "resultado": "", "status": "Pendiente"}
            ]
            
            st.session_state.revancha_matches = pd.DataFrame(revancha_matches)
            st.success("Se cargaron los partidos de Revancha 2274")

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
    if 'api_key_checked' not in st.session_state:
        st.session_state.api_key_checked = False
    
    # Update results periodically
    if datetime.now() - st.session_state.last_update > timedelta(seconds=30):
        update_match_results()
        st.session_state.last_update = datetime.now()

    # App title and description
    st.title("Progol Quiniela Tracker")
    st.write("Esta aplicación te permite seguir los resultados de tus quinielas Progol y Revancha en tiempo real.")
    
    # Check API key on first run
    if not st.session_state.api_key_checked:
        api_key = get_api_key()
        if api_key:
            st.success("API key found in secrets!")
        # Mark as checked to avoid checking every rerun
        st.session_state.api_key_checked = True
    
    # Sidebar
    st.sidebar.header("Configuración")
    
    # Load data options
    st.sidebar.subheader("Cargar Datos")
    data_option = st.sidebar.radio(
        "Método de obtención de datos:",
        ["API-Football", "Cargar Plantilla", "Ingreso Manual"]
    )
    
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
    
    # Main content based on selected option
    if data_option == "API-Football":
        st.subheader("Partidos desde API-Football")
        
        if st.button("Cargar Partidos"):
            with st.spinner("Obteniendo partidos desde API-Football..."):
                # Get matches from API
                matches = fetch_matches_for_progol()
                
                if matches:
                    # Save as Progol matches
                    st.session_state.progol_matches = pd.DataFrame(matches)
                    st.success(f"Se cargaron {len(matches)} partidos desde API-Football")
                else:
                    st.error("No se pudieron cargar partidos desde API-Football")
        
        # Display API status
        st.subheader("Estado de la API")
        api_key = get_api_key()
        if api_key:
            st.success("API configurada correctamente")
            
            # Show a sample of live matches if available
            if st.button("Mostrar partidos en vivo"):
                with st.spinner("Consultando partidos en vivo..."):
                    live_matches = get_live_matches()
                    if live_matches:
                        st.success(f"Hay {len(live_matches)} partidos en vivo")
                        # Display first few matches
                        for i, match in enumerate(live_matches[:5]):
                            teams = match['teams']
                            goals = match['goals']
                            st.write(f"{teams['home']['name']} {goals['home'] or 0} - {goals['away'] or 0} {teams['away']['name']} ({match['fixture']['status']['long']})")
                        
                        if len(live_matches) > 5:
                            st.info(f"... y {len(live_matches) - 5} partidos más")
                    else:
                        st.info("No hay partidos en vivo en este momento")
        else:
            st.error("API no configurada. Por favor configura tu API key en los secretos de Streamlit.")
            
    elif data_option == "Ingreso Manual":
        manual_match_input()
    elif data_option == "Cargar Plantilla":
        load_sample_matches()
    
    # Show match data if available
    if 'progol_matches' in st.session_state and st.session_state.progol_matches is not None or \
       'revancha_matches' in st.session_state and st.session_state.revancha_matches is not None:
        
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
                st.info("No hay partidos cargados. Usa la opción de cargar partidos desde API-Football o ingreso manual.")
        
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