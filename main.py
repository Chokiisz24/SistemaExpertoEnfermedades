import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
import joblib

# Datos de entrenamiento (enfermedades y síntomas)
sintomas = [
    "fiebre,tos,dolor de garganta,congestión nasal",
    "mucosidad nasal,dolor de garganta,estornudos",
    "fiebre,tos seca,dificultad para respirar,fatiga",
    "fiebre alta,tos con esputo,dificultad para respirar,dolor en el pecho",
    "dolor de cabeza,tos,fatiga",
    "dolor muscular,fiebre,escalofríos",
    "dolor abdominal,vómitos,diarrea",
    "picazón,erupción roja,fiebre",
    "dolor de garganta,escalofríos,fiebre",
    "tos seca,dolor muscular,fatiga",
    "fiebre,escalofríos,sudoración nocturna,pérdida de peso",
    "dolor en el abdomen,náuseas,vómitos",
    "dificultad para respirar,opresión en el pecho",
    "fatiga,dificultad para concentrarse,irritabilidad",
    "erupción en la piel,picazón,hinchazón",
    "dolor en el pecho,disnea,tos con sangre",
    "dolor de cabeza,secreción nasal,dolor en los senos nasales"
]

enfermedades = [
    "Gripe",
    "Resfriado",
    "COVID-19",
    "Neumonía",
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
    "Enfermedad Pulmonar Obstructiva Crónica (EPOC)",
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
    print("Ejemplos de síntomas: fiebre, tos, dolor de garganta, congestión nasal, etc.")
    while True:
        sintomas_usuario = input("Ingrese los síntomas separados por coma (ej. fiebre,tos,dolor de garganta): ").strip()
        if sintomas_usuario.lower() in ["salir", "exit"]:
            break
        resultados = diagnosticar(sintomas_usuario)
        if resultados:
            for enfermedad, probabilidad in resultados.items():
                print(f'Probabilidad de {enfermedad}: {probabilidad * 100:.2f}%')
        else:
            print("No se pudieron generar resultados. Verifica la entrada de síntomas.")

if __name__ == "__main__":
    main()
