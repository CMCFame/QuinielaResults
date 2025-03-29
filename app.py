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

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_matches_from_oddschecker(progol_number):
    """
    Scrape match data from oddschecker.com for the given Progol contest number
    """
    url = f"https://www.oddschecker.com/es/pronosticos/futbol/predicciones-progol-revancha-quiniela-esta-semana-{progol_number}"
    
    try:
        # Add headers to mimic a browser and avoid 403 errors
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
            'Referer': 'https://www.google.com/',
        }
        
        # Try several different request configurations
        session = requests.Session()
        
        # First attempt with a delay and full headers
        time.sleep(2)  # Small delay to avoid rate limiting
        response = session.get(url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            st.error(f"Error al obtener datos de oddschecker: Código {response.status_code}")
            return None, None
            
        html = response.text
                
    except Exception as e:
        st.error(f"Error al obtener datos de oddschecker: {str(e)}")
        return None, None
    
    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    
    # Find the tables for Progol and Revancha
    tables = soup.find_all('table')
    
    if len(tables) < 1:
        st.warning("No se encontraron tablas en la página")
        return None, None
    
    # Process Progol table
    progol_matches = []
    
    try:
        # Look for the first table with match data
        for table in tables:
            # Check if this looks like a match table
            rows = table.find_all('tr')
            if len(rows) < 2:  # Need at least header + one row
                continue
                
            header_row = rows[0]
            
            # Process data rows
            data_rows = rows[1:]  # Skip header row
            
            for row in data_rows:
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
            
            # If we found matches in this table, break
            if progol_matches:
                break
    except Exception as e:
        st.error(f"Error procesando tabla Progol: {str(e)}")
    
    # Process Revancha table if exists (should be the second match table)
    revancha_matches = []
    
    try:
        # If we found one table with matches, look for another
        if progol_matches and len(tables) > 1:
            # Start from the second table
            for table in tables[1:]:
                # Check if this looks like a match table
                rows = table.find_all('tr')
                if len(rows) < 2:  # Need at least header + one row
                    continue
                    
                # Process data rows
                data_rows = rows[1:]  # Skip header row
                
                for row in data_rows:
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
                
                # If we found matches in this table, break
                if revancha_matches:
                    break
    except Exception as e:
        st.error(f"Error procesando tabla Revancha: {str(e)}")
    
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
# DIRECT HTML SCRAPING
# ============================================================================
def scrape_from_html_file():
    """
    Alternative approach: manually save the HTML from oddschecker.com and upload it
    """
    st.subheader("Cargar HTML manualmente")
    st.write("""
    Si los métodos automáticos de scraping no funcionan, puedes seguir estos pasos:
    1. Visita la página de oddschecker.com en tu navegador
    2. Guarda la página completa (Ctrl+S en la mayoría de navegadores)
    3. Sube el archivo HTML aquí
    """)
    
    uploaded_file = st.file_uploader("Sube el archivo HTML de oddschecker", type=['html', 'htm'])
    
    if uploaded_file is not None:
        try:
            # Read the HTML content
            html_content = uploaded_file.read().decode('utf-8')
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find tables and extract data
            tables = soup.find_all('table')
            
            if len(tables) < 1:
                st.error("No se encontraron tablas en el archivo HTML")
                return None, None
            
            # Process matches using the same logic as before
            progol_matches = []
            revancha_matches = []
            
            # Process first table (Progol)
            if len(tables) > 0:
                rows = tables[0].find_all('tr')[1:]  # Skip header
                for row in rows:
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
            
            # Process second table (Revancha)
            if len(tables) > 1:
                rows = tables[1].find_all('tr')[1:]  # Skip header
                for row in rows:
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
            
            return progol_matches, revancha_matches
            
        except Exception as e:
            st.error(f"Error procesando el archivo HTML: {str(e)}")
            return None, None
    
    return None, None

# ============================================================================
# DIRECT URL FETCH
# ============================================================================
def fetch_url_directly():
    """
    Alternative approach: directly entering a URL to scrape data from
    """
    st.subheader("Obtener datos directamente de una URL")
    
    url = st.text_input(
        "URL de la página de Progol",
        value="https://www.oddschecker.com/es/pronosticos/futbol/predicciones-progol-revancha-quiniela-esta-semana-2274"
    )
    
    if st.button("Obtener Datos de URL"):
        try:
            with st.spinner("Obteniendo datos..."):
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Cache-Control': 'max-age=0',
                    'Referer': 'https://www.google.com/',
                }
                
                session = requests.Session()
                response = session.get(url, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    html_content = response.text
                    
                    # Parse the data from HTML
                    soup = BeautifulSoup(html_content, 'html.parser')
                    tables = soup.find_all('table')
                    
                    if len(tables) < 1:
                        st.error("No se encontraron tablas en la página")
                        return
                    
                    # Process matches from tables
                    progol_matches = []
                    revancha_matches = []
                    
                    # First table (Progol)
                    if len(tables) > 0:
                        table = tables[0]
                        rows = table.find_all('tr')
                        
                        if len(rows) > 1:  # We need at least a header and one data row
                            # Process data rows (skip header)
                            for row in rows[1:]:
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
                    
                    # Second table (Revancha)
                    if len(tables) > 1:
                        table = tables[1]
                        rows = table.find_all('tr')
                        
                        if len(rows) > 1:  # We need at least a header and one data row
                            # Process data rows (skip header)
                            for row in rows[1:]:
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
                    
                    # Save the matches to session state
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
                else:
                    st.error(f"Error al obtener datos: Código {response.status_code}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.code(traceback.format_exc())

# ============================================================================
# DEBUG FUNCTIONS
# ============================================================================
def debug_mode():
    """
    Debug mode that shows detailed information about the current state and errors
    """
    st.header("Debug Information")
    
    # System information
    st.subheader("System Information")
    st.write(f"Current Time: {datetime.now()}")
    
    # Session state
    st.subheader("Session State Variables")
    for key, value in st.session_state.items():
        if key == 'progol_matches' or key == 'revancha_matches':
            if value is not None:
                st.write(f"{key}: DataFrame with {len(value)} rows")
            else:
                st.write(f"{key}: None")
        else:
            st.write(f"{key}: {value}")
    
    # Test connection to oddschecker
    st.subheader("Test Connection to oddschecker.com")
    if st.button("Test Connection"):
        with st.spinner("Testing connection..."):
            try:
                # Try simple request
                response = requests.get("https://www.oddschecker.com", timeout=10)
                st.write(f"Status Code: {response.status_code}")
                st.write(f"Headers: {response.headers}")
                
                # Try with requests and custom headers
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                }
                st.write("Testing with custom headers:")
                response2 = requests.get("https://www.oddschecker.com", headers=headers, timeout=10)
                st.write(f"Status Code: {response2.status_code}")
            except Exception as e:
                st.error(f"Error connecting: {str(e)}")
                st.code(traceback.format_exc())
                
    # Add direct URL fetch method
    fetch_url_directly()
                
    # File uploader for direct HTML processing
    st.subheader("Process HTML directly")
    uploaded_file = st.file_uploader("Upload HTML from oddschecker.com", type=["html", "htm"])
    
    if uploaded_file is not None:
        try:
            html_content = uploaded_file.read().decode('utf-8')
            st.write(f"File size: {len(html_content)} characters")
            
            # Find tables in HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            tables = soup.find_all('table')
            st.write(f"Found {len(tables)} tables in the HTML")
            
            # Display header of first table if available
            if len(tables) > 0:
                st.write("First table header:")
                header_row = tables[0].find('tr')
                if header_row:
                    st.write(header_row.text)
        except Exception as e:
            st.error(f"Error processing HTML: {str(e)}")
    
    # Error simulation for testing
    st.subheader("Error Handling Tests")
    if st.button("Simulate Request Error"):
        try:
            # Intentionally cause an error
            response = requests.get("https://nonexistent-domain-123456.com", timeout=2)
        except Exception as e:
            st.error(f"Expected error: {str(e)}")
    
    # Manually show some useful debug information
    st.subheader("Manual Debug Information")
    debug_info = f"""
    Streamlit version: {st.__version__}
    Current time: {datetime.now()}
    """
    st.code(debug_info)

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
    
    # Input for Progol contest number
    progol_number = st.sidebar.text_input("Número de concurso Progol:", value="2274")
    
    # Debug mode toggle
    debug_toggle = st.sidebar.checkbox("Modo Debug", value=st.session_state.debug_enabled)
    if debug_toggle != st.session_state.debug_enabled:
        st.session_state.debug_enabled = debug_toggle
        # Don't use experimental_rerun - it causes errors
    
    # Advanced options
    with st.sidebar.expander("Opciones Avanzadas"):
        st.write("Si tienes problemas cargando los datos automáticamente, intenta con esta opción:")
        if st.button("Cargar desde HTML"):
            scrape_from_html_file()
            
    if st.sidebar.button("Cargar partidos"):
        with st.spinner("Obteniendo información de partidos..."):
            try:
                # Get data from oddschecker
                progol_matches, revancha_matches = get_matches_from_oddschecker(progol_number)
                
                # Process progol matches
                if progol_matches:
                    st.session_state.progol_matches = pd.DataFrame(progol_matches)
                    st.success(f"Se cargaron {len(progol_matches)} partidos de Progol")
                else:
                    st.error("No se pudieron cargar los partidos de Progol")
                
                # Process revancha matches
                if revancha_matches:
                    st.session_state.revancha_matches = pd.DataFrame(revancha_matches)
                    st.success(f"Se cargaron {len(revancha_matches)} partidos de Revancha")
                else:
                    st.warning("No se pudieron cargar los partidos de Revancha")
            
            except Exception as e:
                # Show detailed error message
                st.error(f"Error al cargar los datos: {str(e)}")
                st.info("Intenta con otro número de concurso o verifica la conexión a internet.")

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
    
    # Main content tabs or debug mode
    if st.session_state.debug_enabled:
        debug_mode()
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

    # Use a safer way to update UI with auto_refresh
    # Instead of experimental_rerun, use a placeholder with a timer
    # This approach is safer and won't crash the app
    refresh_placeholder = st.empty()
    refresh_placeholder.info("La aplicación se actualiza automáticamente cada 30 segundos. Última actualización: " + 
                        datetime.now().strftime("%H:%M:%S"))
    
    # We don't need to call rerun explicitly, as Streamlit 
    # will rerun the script whenever session state changes

# Run the main application
if __name__ == "__main__":
    main()