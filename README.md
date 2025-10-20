# API de Predicción - IMF M7

API Flask para predicción de ventas futuras y probabilidad de contratación de clientes, desarrollada como parte de la Práctica Final del Módulo 7 de IMF.

## Características

- **Predicción de Ventas Futuras**: Utiliza Prophet para predecir ventas para N días en el futuro
- **Predicción de Contratación**: Clasifica clientes según su probabilidad de contratar un producto
- **API RESTful**: Endpoints simples y bien documentados
- **Validación de datos**: Manejo de errores y validación de entrada

## Estructura del Proyecto

```
imf_m7_api/
├── app.py                  # API Flask principal
├── requirements.txt        # Dependencias del proyecto
├── README.md              # Este archivo
├── models/                # Modelos entrenados
│   ├── future_sales.pkl   # Modelo Prophet para ventas
│   └── best_pipeline.pkl  # Pipeline de clasificación
└── notebooks/             # Jupyter Notebooks
    └── IMF_M7_Practica_Final.ipynb  # Análisis y entrenamiento de modelos
```

## Instalación

### 1. Crear entorno virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

Los modelos pre-entrenados ya están incluidos en el directorio `models/`

## Uso de la API

### Iniciar el servidor

```bash
python app.py
```

El servidor se iniciará en `http://localhost:8000`

### Endpoints Disponibles

#### 1. Health Check

```bash
GET /health
```

Verifica el estado de la API y los modelos cargados.

**Ejemplo:**
```bash
curl http://localhost:8000/health
```

**Respuesta:**
```json
{
  "status": "healthy",
  "models": {
    "sales_forecast": true,
    "contract_prediction": true
  }
}
```

---

#### 2. Predicción de Ventas Futuras

```bash
POST /api/predict/sales
```

Predice las ventas para los próximos N días.

**Parámetros:**
- `days` (int): Número de días a predecir (1-365)

**Ejemplo:**
```bash
curl -X POST http://localhost:8000/api/predict/sales \
  -H "Content-Type: application/json" \
  -d '{"days": 7}'
```

**Respuesta:**
```json
{
  "status": "success",
  "days_predicted": 7,
  "predictions": [
    {
      "date": "2025-01-01",
      "sales_forecast": 7.56,
      "lower_bound": 0.0,
      "upper_bound": 27.98
    },
    ...
  ]
}
```

---

#### 3. Predicción de Contratación de Cliente

```bash
POST /api/predict/contract
```

Predice si un cliente contratará el producto basándose en sus características.

**Parámetros:**

| Campo | Tipo | Descripción | Valores Posibles |
|-------|------|-------------|------------------|
| age | int | Edad del cliente | 18-100 |
| job | string | Profesión | admin., blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed, unknown |
| marital | string | Estado civil | married, single, divorced |
| education | string | Nivel educativo | primary, secondary, tertiary, unknown |
| default | string | ¿Tiene crédito en default? | yes, no |
| balance | int | Balance de cuenta | cualquier entero |
| housing | string | ¿Tiene préstamo de vivienda? | yes, no |
| loan | string | ¿Tiene préstamo personal? | yes, no |
| contact | string | Tipo de contacto | cellular, telephone, unknown |
| day | int | Día del mes | 1-31 |
| month | string | Mes | jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec |
| campaign | int | Contactos en esta campaña | cualquier entero positivo |
| pdays | int | Días desde último contacto | -1 si nunca contactado, o número positivo |
| previous | int | Contactos previos | cualquier entero >= 0 |
| poutcome | string | Resultado campaña anterior | success, failure, other, unknown |

**Ejemplo:**
```bash
curl -X POST http://localhost:8000/api/predict/contract \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Respuesta:**
```json
{
  "status": "success",
  "prediction": {
    "will_contract": false,
    "probability": 0.0856,
    "confidence": "high"
  },
  "client_summary": {
    "age": 35,
    "job": "technician",
    "education": "secondary",
    "balance": 1500
  }
}
```

**Niveles de Confianza:**
- `high`: probabilidad >= 70% o <= 30%
- `medium`: probabilidad >= 60% o <= 40%
- `low`: probabilidad entre 40% y 60%

---

## Notas Importantes

### Variables Excluidas en Producción

Siguiendo las indicaciones de la práctica, la variable `duration` (duración de la llamada) **NO** se incluye en el modelo de clasificación porque:

1. **No está disponible antes de la llamada**: La duración solo se conoce después de realizar la llamada
2. **Sería data leakage**: Usar esta variable en producción sería imposible ya que necesitamos predecir ANTES de hacer la llamada
3. **El objetivo es predecir previo a la llamada**: El modelo debe funcionar con información disponible antes del contacto

### Modelos Entrenados

Los modelos incluidos han sido entrenados y optimizados usando:
- **Modelo de ventas**: Prophet con estacionalidad diaria
- **Modelo de clasificación**: Gradient Boosting (mejor AUC en validación)

## Despliegue en Producción

Para desplegar en producción, se recomienda usar Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

O usar Docker (crea un `Dockerfile`):

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app:app"]
```

## Licencia

Este proyecto fue desarrollado como práctica educativa para IMF Business School.

## Autor

Práctica Final - Módulo 7: Introducción a Process Mining
