import pandas as pd
import matplotlib.pyplot as plt

#Leer el archivo
file_path = 'threshold_report_0.7.txt'
scores = []

with open(file_path, 'r') as file:
    for line in file:
        #Redondear a decimales
        scores.append(round(float(line.strip()), 2))
        #scores.append(float(line.strip()))

#Crear un DataFrame de pandas
df = pd.DataFrame(scores, columns=['Score'])

#Contar la frecuencia de cada score
frequency = df['Score'].value_counts().head(30)

#Imprimir los 5 valores más repetidos
print("Los 5 valores más repetidos son:")
for score, count in frequency.items():
    print(f"Valor: {score}, Repeticiones: {count}")

#Graficar la frecuencia de los scores
plt.figure(figsize=(10, 6))
df['Score'].hist(bins=20, edgecolor='black')
plt.title('Frecuencia de Scores')
plt.xlabel('Score')
plt.ylabel('Frecuencia')
plt.grid(axis='y', alpha=0.75)

#Guardar la gráfica como imagen
plt.savefig('threshold_frecuency.png')

#Mostrar la gráfica
plt.show()