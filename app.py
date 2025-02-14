import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# Configuración de la página
st.set_page_config(
    page_title="Resultados de Fútbol",
    page_icon="⚽",
    layout="wide"
)

# Constantes
API_BASE_URL = "https://v3.football.api-sports.io"

# Obtener API key desde los secretos de Streamlit
API_KEY = st.secrets["FOOTBALL_API_KEY"]
HEADERS = {
    'x-rapidapi-host': 'v3.football.api-sports.io',
    'x-rapidapi-key': API_KEY
}

# Diccionario de ligas con sus IDs
LEAGUES = {
    'México - Liga MX': 262,
    'México - Liga Expansión': 263,
    'México - Liga Femenil': 264,
    'España - LaLiga': 140,
    'Inglaterra - Premier League': 39,
    'Países Bajos - Eredivisie': 88,
    'Alemania - Bundesliga': 78,
    'Portugal - Primeira Liga': 94,
    'Francia - Ligue 1': 61,
    'Argentina - Primera División': 128,
    'Brasil - Série A': 71,
    'España - LaLiga2': 141
}

# Cache para las respuestas de la API
@st.cache_data(ttl=3600)  # Cache por 1 hora
def fetch_api_data(endpoint, params=None):
    """
    Función genérica para hacer peticiones a la API con caché
    
    Args:
        endpoint (str): Endpoint de la API
        params (dict): Parámetros de la consulta
    
    Returns:
        dict: Respuesta de la API
    """
    try:
        url = f"{API_BASE_URL}/{endpoint}"
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()  # Lanza una excepción para códigos de error HTTP
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error al conectar con la API: {str(e)}")
        return None

def get_current_season():
    """
    Determina la temporada actual basada en la fecha
    
    Returns:
        int: Año de la temporada actual
    """
    current_date = datetime.now()
    if current_date.month < 7:  # Si estamos antes de julio
        return current_date.year - 1
    return current_date.year

def get_standings(league_id, season=None):
    """
    Obtiene la tabla de posiciones de una liga
    
    Args:
        league_id (int): ID de la liga
        season (int): Año de la temporada (opcional)
    
    Returns:
        list: Lista de equipos con sus posiciones
    """
    if season is None:
        season = get_current_season()
        
    data = fetch_api_data("standings", {
        'league': league_id,
        'season': season
    })
    
    if data and data['response']:
        return data['response'][0]['league']['standings'][0]
    return None

def get_last_matches(team_id, last_n=5):
    """
    Obtiene los últimos N partidos de un equipo
    
    Args:
        team_id (int): ID del equipo
        last_n (int): Número de partidos a obtener
    
    Returns:
        list: Lista de partidos
    """
    data = fetch_api_data("fixtures", {
        'team': team_id,
        'last': last_n
    })
    
    if data and data['response']:
        return data['response']
    return None

def format_match_result(match):
    """
    Formatea el resultado de un partido
    
    Args:
        match (dict): Datos del partido
    
    Returns:
        str: Resultado formateado
    """
    home_team = match['teams']['home']['name']
    away_team = match['teams']['away']['name']
    home_score = match['goals']['home']
    away_score = match['goals']['away']
    match_date = datetime.strptime(match['fixture']['date'], 
                                 '%Y-%m-%dT%H:%M:%S%z').strftime('%d/%m/%Y')
    
    # Determinar el color del resultado
    if match['teams']['home']['winner']:
        home_color = "green"
        away_color = "red"
    elif match['teams']['away']['winner']:
        home_color = "red"
        away_color = "green"
    else:
        home_color = away_color = "orange"
    
    # Formatear el resultado con colores
    return f"{match_date}: {home_team} <span style='color:{home_color}'>{home_score}</span>-<span style='color:{away_color}'>{away_score}</span> {away_team}"

def show_team_form(team_data):
    """
    Muestra la forma reciente del equipo con círculos de colores
    
    Args:
        team_data (dict): Datos del equipo
    """
    form = team_data.get('form', '')
    if form:
        form_html = ''
        for result in form:
            if result == 'W':
                color = 'green'
            elif result == 'L':
                color = 'red'
            else:
                color = 'orange'
            form_html += f'<span style="color:{color}">●</span> '
        st.markdown(f"Forma reciente: {form_html}", unsafe_allow_html=True)

def main():
    st.title("⚽ Resultados de Fútbol")
    
    # Verificar que la API key está configurada
    if "FOOTBALL_API_KEY" not in st.secrets:
        st.error("⚠️ La API key no está configurada. Por favor, configura el secreto 'FOOTBALL_API_KEY' en la configuración de Streamlit.")
        return
    
    # Selector de liga
    selected_league = st.selectbox(
        "Selecciona una liga",
        list(LEAGUES.keys())
    )
    
    league_id = LEAGUES[selected_league]
    
    # Selector de temporada
    current_season = get_current_season()
    season = st.selectbox(
        "Selecciona la temporada",
        range(current_season, current_season-5, -1),
        index=0
    )
    
    # Obtener y mostrar la tabla de posiciones
    with st.spinner("Cargando datos..."):
        standings = get_standings(league_id, season)
        
        if standings:
            st.subheader("Tabla de Posiciones")
            
            # Crear DataFrame para la tabla de posiciones
            standings_data = []
            for team in standings:
                standings_data.append({
                    'Pos': team['rank'],
                    'Equipo': team['team']['name'],
                    'PJ': team['all']['played'],
                    'G': team['all']['win'],
                    'E': team['all']['draw'],
                    'P': team['all']['lose'],
                    'GF': team['all']['goals']['for'],
                    'GC': team['all']['goals']['against'],
                    'DG': team['goalsDiff'],
                    'Pts': team['points']
                })
            
            df_standings = pd.DataFrame(standings_data)
            st.dataframe(
                df_standings,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Pos": st.column_config.NumberColumn(
                        "Pos",
                        help="Posición en la tabla",
                        format="%d"
                    ),
                    "Pts": st.column_config.NumberColumn(
                        "Pts",
                        help="Puntos totales",
                        format="%d"
                    )
                }
            )
            
            # Mostrar últimos 5 partidos de cada equipo
            st.subheader("Últimos 5 partidos por equipo")
            
            # Crear columnas para mostrar los resultados
            cols = st.columns(2)
            col_index = 0
            
            for team in standings:
                team_name = team['team']['name']
                team_id = team['team']['id']
                
                with cols[col_index]:
                    st.markdown(f"### {team_name}")
                    show_team_form(team)
                    matches = get_last_matches(team_id)
                    
                    if matches:
                        for match in matches:
                            st.markdown(format_match_result(match), unsafe_allow_html=True)
                    else:
                        st.write("No hay datos de partidos disponibles")
                    
                    st.markdown("---")
                
                # Alternar entre columnas
                col_index = (col_index + 1) % 2
        else:
            st.error("No se pudieron cargar los datos de la liga seleccionada")

if __name__ == "__main__":
    main()