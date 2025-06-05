import os
import sys

# Agrega la carpeta padre (raíz del proyecto) al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify
from plagiarism_detector import algorithm
from flask_cors import CORS
from werkzeug.utils import secure_filename  # ✅ Import necesario para limpiar nombres

# Crear instancia de la app
app = Flask(__name__)
CORS(app)  # Esto permite todas las solicitudes cross-origin (útil para pruebas locales)

# Crear carpeta de subida si no existe
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('files[]')
    saved = []

    for file in files:
        # ✅ Elimina cualquier ruta de subcarpetas del nombre del archivo
        filename = secure_filename(os.path.basename(file.filename))
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        file.save(filepath)
        saved.append(filename)

    return jsonify({'uploaded': saved})

@app.route('/list', methods=['GET'])
def list_files():
    archivos = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.py')]
    return jsonify(archivos)

@app.route('/compare', methods=['POST'])
def compare_uploaded_files():
    data = request.get_json()
    file1 = data.get('file1')
    file2 = data.get('file2')
    if not file1 or not file2:
        return jsonify({'error': 'Faltan archivos'}), 400

    path1 = os.path.join(UPLOAD_FOLDER, file1)
    path2 = os.path.join(UPLOAD_FOLDER, file2)

    result = algorithm(path1, path2)
    if not result:
        return jsonify({'error': 'No se pudo comparar'}), 500

    types, sa, ted, ast_flag_0, ast_flag_1, edit_distance, n1, f1, l1 = result
    return jsonify({
        'tipos': types,
        'sa': sa,
        'ted': ted,
        'edit_distance': edit_distance,
        'plagio_0': ast_flag_0,
        'plagio_1': ast_flag_1,
        'nodos_diff': n1,
        'funciones_diff': f1,
        'bucles_diff': l1
    })

if __name__ == '__main__':
    app.run(debug=True)
