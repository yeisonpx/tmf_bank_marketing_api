"""
API Flask para predicción de ventas y contratación de clientes
IMF M7 - Práctica Final
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import os
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# Configuración
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Cargar modelos al iniciar la aplicación
print("Cargando modelos...")
try:
    with open(os.path.join(MODELS_DIR, 'future_sales.pkl'), 'rb') as f:
        sales_model = pickle.load(f)
    print("[OK] Modelo de ventas cargado")
except FileNotFoundError:
    print("[WARNING] No se encontró el modelo de ventas")
    sales_model = None

try:
    with open(os.path.join(MODELS_DIR, 'best_pipeline.pkl'), 'rb') as f:
        classification_pipeline = pickle.load(f)
    print("[OK] Pipeline de clasificación cargado")
except FileNotFoundError:
    print("[WARNING] No se encontró el pipeline de clasificación")
    classification_pipeline = None

@app.route('/')
def home():
    """
    Endpoint raíz con información de la API
    """
    return jsonify({
        "api": "IMF M7 - Predicción de Ventas y Contratación",
        "version": "1.0",
        "endpoints": {
            "/api/predict/sales": {
                "method": "POST",
                "description": "Predice ventas futuras para N días",
                "parameters": {
                    "days": "Número de días a predecir (int)"
                },
                "example": {
                    "days": 7
                }
            },
            "/api/predict/contract": {
                "method": "POST",
                "description": "Predice si un cliente contratará el producto",
                "parameters": {
                    "age": "Edad del cliente (int)",
                    "job": "Trabajo (str)",
                    "marital": "Estado civil: married, single, divorced (str)",
                    "education": "Nivel educativo: primary, secondary, tertiary, unknown (str)",
                    "default": "Tiene crédito en default: yes, no (str)",
                    "balance": "Balance de cuenta (int)",
                    "housing": "Tiene préstamo de vivienda: yes, no (str)",
                    "loan": "Tiene préstamo personal: yes, no (str)",
                    "contact": "Tipo de contacto: cellular, telephone, unknown (str)",
                    "day": "Día del mes (int)",
                    "month": "Mes: jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec (str)",
                    "campaign": "Número de contactos en esta campaña (int)",
                    "pdays": "Días desde último contacto (-1 si nunca) (int)",
                    "previous": "Número de contactos previos (int)",
                    "poutcome": "Resultado campaña anterior: success, failure, other, unknown (str)"
                },
                "example": {
                    "age": 35,
                    "job": "technician",
                    "marital": "married",
                    "education": "secondary",
                    "default": "no",
                    "balance": 1500,
                    "housing": "yes",
                    "loan": "no",
                    "contact": "cellular",
                    "day": 15,
                    "month": "may",
                    "campaign": 2,
                    "pdays": -1,
                    "previous": 0,
                    "poutcome": "unknown"
                }
            },
            "/health": {
                "method": "GET",
                "description": "Verifica el estado de la API y los modelos"
            }
        }
    })

@app.route('/health')
def health():
    """
    Endpoint de salud para verificar que los modelos estén cargados
    """
    return jsonify({
        "status": "healthy",
        "models": {
            "sales_forecast": sales_model is not None,
            "contract_prediction": classification_pipeline is not None
        }
    })

@app.route('/api/predict/sales', methods=['POST'])
def predict_sales():
    """
    Predice las ventas futuras para los próximos N días

    Request JSON:
        {
            "days": 7
        }

    Response JSON:
        {
            "status": "success",
            "days_predicted": 7,
            "predictions": [
                {
                    "date": "2025-01-01",
                    "sales_forecast": 7.56,
                    "lower_bound": -13.90,
                    "upper_bound": 27.98
                },
                ...
            ]
        }
    """
    if sales_model is None:
        return jsonify({
            "status": "error",
            "message": "Modelo de ventas no disponible"
        }), 500

    try:
        data = request.get_json()

        if 'days' not in data:
            return jsonify({
                "status": "error",
                "message": "Parámetro 'days' es requerido"
            }), 400

        days = int(data['days'])

        if days <= 0 or days > 365:
            return jsonify({
                "status": "error",
                "message": "El número de días debe estar entre 1 y 365"
            }), 400

        # Hacer predicción
        future = sales_model.make_future_dataframe(periods=days, freq='D')
        forecast = sales_model.predict(future)

        # Obtener solo las predicciones futuras
        future_forecast = forecast.tail(days)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

        # Convertir a lista de diccionarios
        predictions = []
        for _, row in future_forecast.iterrows():
            predictions.append({
                "date": row['ds'].strftime('%Y-%m-%d'),
                "sales_forecast": round(max(0, row['yhat']), 2),  # No puede ser negativo
                "lower_bound": round(max(0, row['yhat_lower']), 2),
                "upper_bound": round(max(0, row['yhat_upper']), 2)
            })

        return jsonify({
            "status": "success",
            "days_predicted": days,
            "predictions": predictions
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/predict/contract', methods=['POST'])
def predict_contract():
    """
    Predice si un cliente contratará el producto basándose en sus características

    Request JSON:
        {
            "age": 35,
            "job": "technician",
            "marital": "married",
            "education": "secondary",
            "default": "no",
            "balance": 1500,
            "housing": "yes",
            "loan": "no",
            "contact": "cellular",
            "day": 15,
            "month": "may",
            "campaign": 2,
            "pdays": -1,
            "previous": 0,
            "poutcome": "unknown"
        }

    Response JSON:
        {
            "status": "success",
            "prediction": {
                "will_contract": true,
                "probability": 0.75,
                "confidence": "high"
            }
        }
    """
    if classification_pipeline is None:
        return jsonify({
            "status": "error",
            "message": "Pipeline de clasificación no disponible"
        }), 500

    try:
        data = request.get_json()

        # Validar campos requeridos
        required_fields = [
            'age', 'job', 'marital', 'education', 'default', 'balance',
            'housing', 'loan', 'contact', 'day', 'month', 'campaign',
            'pdays', 'previous', 'poutcome'
        ]

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                "status": "error",
                "message": f"Campos faltantes: {', '.join(missing_fields)}"
            }), 400

        # Crear DataFrame con los datos del cliente
        # Nota: NO incluimos 'duration' porque no está disponible antes de la llamada
        client_data = pd.DataFrame([{
            'age': int(data['age']),
            'job': str(data['job']),
            'marital': str(data['marital']),
            'education': str(data['education']),
            'default': str(data['default']),
            'balance': int(data['balance']),
            'housing': str(data['housing']),
            'loan': str(data['loan']),
            'contact': str(data['contact']),
            'day': int(data['day']),
            'month': str(data['month']),
            'campaign': int(data['campaign']),
            'pdays': int(data['pdays']),
            'previous': int(data['previous']),
            'poutcome': str(data['poutcome'])
        }])

        # Crear características adicionales (igual que en el entrenamiento)
        client_data['age_group'] = pd.cut(client_data['age'], bins=[0, 30, 50, 100], labels=['joven', 'adulto', 'mayor'])
        client_data['has_credit'] = (client_data['loan'] == 'yes').astype(int)
        client_data['has_housing'] = (client_data['housing'] == 'yes').astype(int)
        client_data['contact_digital'] = client_data['contact'].isin(['cellular', 'telephone']).astype(int)

        # Hacer predicción
        prediction_proba = classification_pipeline.predict_proba(client_data)[0]
        prediction = classification_pipeline.predict(client_data)[0]

        # Calcular nivel de confianza
        prob_yes = prediction_proba[1]
        if prob_yes >= 0.7 or prob_yes <= 0.3:
            confidence = "high"
        elif prob_yes >= 0.6 or prob_yes <= 0.4:
            confidence = "medium"
        else:
            confidence = "low"

        return jsonify({
            "status": "success",
            "prediction": {
                "will_contract": bool(prediction),
                "probability": round(float(prob_yes), 4),
                "confidence": confidence
            },
            "client_summary": {
                "age": data['age'],
                "job": data['job'],
                "education": data['education'],
                "balance": data['balance']
            }
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "status": "error",
        "message": "Endpoint no encontrado"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "status": "error",
        "message": "Error interno del servidor"
    }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("API DE PREDICCIÓN - IMF M7")
    print("="*60)
    print("\nEndpoints disponibles:")
    print("  GET  /           - Información de la API")
    print("  GET  /health     - Estado de salud")
    print("  POST /api/predict/sales     - Predicción de ventas")
    print("  POST /api/predict/contract  - Predicción de contratación")
    print("\n" + "="*60)
    print("Servidor iniciado en http://localhost:8000")
    print("="*60 + "\n")

    app.run(host='0.0.0.0', port=8000, debug=True)
