import os
import sys
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from plagiarism_detector import algorithm

# Agrega la carpeta padre (raíz del proyecto) al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Crear instancia de la app
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Función auxiliar para tipos sueltos
def to_serializable(obj):
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    return obj

@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('files[]')
    saved = []
    for file in files:
        filename = secure_filename(os.path.basename(file.filename))
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        saved.append(filename)
    return jsonify({'uploaded': saved})

@app.route('/list', methods=['GET'])
def list_files():
    archivos = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.py')]
    return jsonify(archivos)

@app.route('/compare-all', methods=['GET'])
def compare_all():
    archivos = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.py')]
    rutas = [os.path.join(UPLOAD_FOLDER, f) for f in archivos]

    from plagiarism_detector import compare_all_pairs
    matriz = compare_all_pairs(rutas)

    # Función recursiva robusta para convertir todo a tipos JSON-safe
    def convert(val):
        if isinstance(val, (int, float, str, bool)):
            return val
        elif isinstance(val, tuple):
            return [convert(v) for v in val]
        elif isinstance(val, list):
            return [convert(v) for v in val]
        elif hasattr(val, "tolist"):
            return val.tolist()
        else:
            return str(val)

    matriz_convertida = [
        [convert(valor) for valor in fila]
        for fila in matriz
    ]
    return jsonify({'comparaciones': matriz_convertida})

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
        'tipos': to_serializable(types),
        'sa': to_serializable(sa),
        'ted': to_serializable(ted),
        'edit_distance': to_serializable(edit_distance),
        'plagio_0': bool(ast_flag_0),
        'plagio_1': bool(ast_flag_1),
        'nodos_diff': to_serializable(n1),
        'funciones_diff': to_serializable(f1),
        'bucles_diff': to_serializable(l1)
    })

if __name__ == '__main__':
    app.run(debug=True)
