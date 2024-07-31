import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
import joblib

# Enfermerdades y Sintomas
sintomas = [
    "fiebre,tos,dolor de garganta,congestion nasal",
    "mucosidad nasal,dolor de garganta,estornudos",
    "fiebre,tos seca,dificultad para respirar,fatiga",
    "fiebre alta,tos con esputo,dificultad para respirar,dolor en el pecho",
    "dolor de cabeza,tos,fatiga",
    "dolor muscular,fiebre,escalofrios",
    "dolor abdominal,vomitos,diarrea",
    "picazon,erupcion roja,fiebre",
    "dolor de garganta,escalofrios,fiebre",
    "tos seca,dolor muscular,fatiga",
    "fiebre,escalofrios,sudoracion nocturna,p perdidapérdida de peso",
    "dolor en el abdomen,nauseas,vomitos",
    "dificultad para respirar,opresion en el pecho",
    "fatiga,dificultad para concentrarse,irritabilidad",
    "erupcion en la piel,picazon,hinchazon",
    "dolor en el pecho,disnea,tos con sangre",
    "dolor de cabeza,secrecion nasal,dolor en los senos nasales"
]

enfermedades = [
    "Gripe",
    "Resfriado",
    "COVID-19",
    "Neumonia",
    "Sinusitis",
    "Influenza",
    "Gastroenteritis",
    "Alergia",
    "Amigdalitis",
    "Bronquitis",
    "Tuberculosis",
    "Candidiasis",
    "Asma",
    "Dermatitis",
    "Enfermedad Pulmonar Obstructiva Cronica (EPOC)",
    "Hepatitis",
    "Celulitis"
]

modelo = make_pipeline(CountVectorizer(), MultinomialNB())
modelo.fit(sintomas, enfermedades)

joblib.dump(modelo, 'modelo_diagnostico.pkl')

# Función para hacer predicciones con probabilidades
def diagnosticar(sintomas_usuario):
    modelo = joblib.load('modelo_diagnostico.pkl')
    predicciones = modelo.predict_proba([sintomas_usuario])[0]
    enfermedades_pred = modelo.classes_
    
    resultados = {enfermedades_pred[i]: predicciones[i] for i in range(len(enfermedades_pred))}
    return resultados

# Función principal para interactuar en la terminal
def main():
    print("Sistema Experto de Diagnóstico de Enfermedades")
    print("Ejemplos de sintomas: fiebre, tos, dolor de garganta, congestion nasal, etc.")
    while True:
        sintomas_usuario = input("Ingrese los sintomas separados por coma (ej. fiebre,tos,dolor de garganta): ").strip()
        if sintomas_usuario.lower() in ["salir", "exit"]:
            break
        resultados = diagnosticar(sintomas_usuario)
        if resultados:
            for enfermedad, probabilidad in resultados.items():
                print(f'Probabilidad de {enfermedad}: {probabilidad * 100:.2f}%')
        else:
            print("No se pudieron generar resultados. Verifica la entrada de sintomas.")

if __name__ == "__main__":
    main()
