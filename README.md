# Sistema Experto de Diagnóstico de Enfermedades

## Descripción

Este proyecto es un sistema experto para el diagnóstico de enfermedades basado en los síntomas ingresados por el usuario.
Utiliza un modelo de clasificación de texto simple construido con `scikit-learn` y `Naive Bayes`. La interfaz es con la linea de comandos de la consola, 
y el sistema proporciona probabilidades para cada enfermedad basada en los síntomas ingresados.

## Características

- Diagnóstico de enfermedades basado en síntomas ingresados.
- Muestra la probabilidad de cada enfermedad.
- Ejemplos de síntomas proporcionados para facilitar el ingreso de datos.

## Instalación

1. **Clona el repositorio**:

   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd <NOMBRE_DEL_REPOSITORIO>
   
2. **Crea un entorno virtual**:

   ```bash
    python -m venv .venv
    source .venv/bin/activate  
3. **Instala las dependencias**:

   ```bash
   pip install scikit-learn joblib
   
## USO
1. **Ejecuta el scriot**:
   
   ```bash
    python main.py
2. **Prueba el programa**:
Ingresa los síntomas separados por coma cuando se te solicite (ej. fiebre,tos,dolor de garganta).
El sistema proporcionará la probabilidad de cada enfermedad basada en los síntomas ingresados.
## EJEMPLOS

[![ejemplo.png](https://i.postimg.cc/QCqPNjjb/ejemplo.png)](https://postimg.cc/p5pkqbhh)
