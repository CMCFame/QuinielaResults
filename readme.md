# README.md
"""
# Progol Optimizer - Metodología Definitiva

Implementación exacta de la **Metodología Definitiva Progol** según el documento técnico de 7 partes.

## 🎯 Características Principales

- **Calibración Bayesiana**: Fórmula exacta `p_final = p_raw * (1 + k1*ΔForma + k2*Lesiones + k3*Contexto) / Z`
- **Taxonomía de Partidos**: Clasificación Ancla/Divisor/TendenciaX/Neutro según umbrales específicos
- **Arquitectura Core + Satélites**: 4 quinielas Core + 26 satélites en 13 pares anticorrelados
- **Optimización GRASP-Annealing**: Maximiza `F = 1 - ∏(1 - Pr[≥11])`
- **Validación Completa**: 6 reglas obligatorias del documento

## 🚀 Instalación Rápida

1. **Clonar/descargar** los archivos del proyecto
2. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Ejecutar la aplicación**:
   ```bash
   streamlit run progol_optimizer/ui/streamlit_app.py
   ```

## 📋 Estructura del Proyecto

```
progol_optimizer/
├── __init__.py
├── config/
│   ├── __init__.py
│   └── constants.py          # Configuración exacta del documento
├── data/
│   ├── __init__.py
│   ├── loader.py             # Carga CSV/datos
│   └── validator.py          # Validación de entrada
├── models/
│   ├── __init__.py
│   ├── classifier.py         # Clasificación Ancla/Divisor/TendenciaX/Neutro
│   ├── calibrator.py         # Calibración Bayesiana EXACTA
│   └── probability.py        # Bivariate-Poisson (opcional avanzado)
├── portfolio/
│   ├── __init__.py
│   ├── core_generator.py     # 4 Quinielas Core
│   ├── satellite_generator.py # Pares Satélites anticorrelados
│   └── optimizer.py          # GRASP-Annealing
├── validation/
│   ├── __init__.py
│   └── portfolio_validator.py # Validación completa
├── export/
│   ├── __init__.py
│   └── exporter.py           # CSV, JSON, PDF
├── ui/
│   ├── __init__.py
│   └── streamlit_app.py      # Interfaz gráfica
└── main.py                   # Orquestador principal
```

## 🎮 Uso de la Interfaz Gráfica

1. **Abrir la aplicación web**: `streamlit run progol_optimizer/ui/streamlit_app.py`
2. **Cargar datos**: 
   - Usar "Datos de Ejemplo" para pruebas rápidas
   - O subir CSV con columnas: `home, away, liga, prob_local, prob_empate, prob_visitante`
3. **Ejecutar optimización**: Botón "🚀 Ejecutar Optimización Completa"
4. **Ver resultados**: Pestañas con visualizaciones, validación y descarga

## 💻 Uso Programático

```python
from progol_optimizer.main import ProgolOptimizer

# Inicializar optimizador
optimizer = ProgolOptimizer()

# Procesar concurso
resultado = optimizer.procesar_concurso(
    archivo_datos="mi_concurso.csv",
    concurso_id="2283"
)

# Acceder a resultados
portafolio = resultado["portafolio"]  # 30 quinielas optimizadas
validacion = resultado["validacion"]  # Resultado de validación
archivos = resultado["archivos_exportados"]  # Archivos generados
```

## 📊 Formato de Datos de Entrada

### CSV Mínimo Requerido:
```csv
home,away,liga,prob_local,prob_empate,prob_visitante
Real Madrid,Barcelona,La Liga,0.45,0.30,0.25
Manchester City,Arsenal,Premier League,0.55,0.25,0.20
PSG,Bayern,UEFA CL,0.40,0.35,0.25
...
```

### CSV Completo (Opcional):
```csv
home,away,liga,prob_local,prob_empate,prob_visitante,forma_diferencia,lesiones_impact,es_final,es_derbi,es_playoff
Real Madrid,Barcelona,La Liga,0.45,0.30,0.25,0.5,-0.2,true,true,false
...
```

## ✅ Reglas de Validación (Obligatorias)

1. **Distribución Global**: 35-41% L, 25-33% E, 30-36% V
2. **Empates Individuales**: 4-6 empates por quiniela
3. **Concentración Máxima**: ≤70% mismo signo general, ≤60% en partidos 1-3
4. **Arquitectura**: 4 Core + 26 Satélites en 13 pares
5. **Correlación Jaccard**: ≤ 0.57 entre pares de satélites
6. **Distribución Equilibrada**: Sin dominancia excesiva de un resultado

## 🔧 Configuración Avanzada

Modificar `progol_optimizer/config/constants.py`:

```python
PROGOL_CONFIG = {
    "CALIBRACION_COEFICIENTES": {
        "k1_forma": 0.15,      # Factor forma
        "k2_lesiones": 0.10,   # Factor lesiones
        "k3_contexto": 0.20    # Factor contexto
    },
    "UMBRALES_CLASIFICACION": {
        "ancla_prob_min": 0.60,     # Umbral Ancla
        "divisor_prob_min": 0.40,   # Umbral Divisor
        "divisor_prob_max": 0.60
    },
    # ... más configuraciones
}
```

## 📈 Archivos de Salida

La aplicación genera automáticamente:

- **CSV Quinielas**: `quinielas_progol_YYYYMMDD_HHMMSS.csv`
- **JSON Completo**: `progol_completo_YYYYMMDD_HHMMSS.json`
- **Reporte Texto**: `reporte_progol_YYYYMMDD_HHMMSS.txt`
- **CSV Partidos**: `partidos_YYYYMMDD_HHMMSS.csv`
- **Configuración**: `configuracion_YYYYMMDD_HHMMSS.json`

## 🐛 Troubleshooting

### Error: "Se requieren exactamente 14 partidos"
**Solución**: Verificar que el CSV tenga exactamente 14 filas de partidos.

### Error: "Probabilidades no suman 1.0"
**Solución**: Verificar que para cada partido: `prob_local + prob_empate + prob_visitante = 1.0`

### Error: "Portafolio no válido"
**Solución**: Revisar la pestaña "Validación" para ver qué regla específica falla.

### Error de importación
**Solución**: 
```bash
pip install --upgrade -r requirements.txt
```

## 📚 Referencias

- **Documento técnico**: "Metodología Definitiva Progol" (7 partes)
- **Página 3**: Calibración Bayesiana y Draw-Propensity Rule
- **Página 4**: Taxonomía de partidos y arquitectura Core + Satélites
- **Página 5**: Optimización GRASP-Annealing

## 📞 Soporte

Para problemas técnicos:
1. Verificar que todos los archivos estén en la estructura correcta
2. Revisar los logs en `progol_optimizer.log`
3. Usar "Modo Debug" en la interfaz gráfica

---

**⚽ ¡Listo para optimizar tus quinielas Progol con metodología científica!**
"""
