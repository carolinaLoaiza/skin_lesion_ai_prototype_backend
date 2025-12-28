# Guía de Pruebas - API con MongoDB

Esta guía te muestra cómo probar la integración completa de MongoDB con la API.

## Requisitos Previos

1. **MongoDB Atlas** configurado con tu URI en `.env`
2. **Dependencias instaladas**: `pip install -r requirements.txt`
3. **API corriendo**: `python main.py`

---

## Método 1: Script de Prueba Automático

### Paso 1: Iniciar la API

```bash
python main.py
```

Deberías ver:
```
INFO: Starting skin_lesion_ai_prototype_backend v1.0.0
INFO: MongoDB connection established
INFO: Application startup complete
INFO: Uvicorn running on http://0.0.0.0:8000
```

### Paso 2: Ejecutar el Script de Prueba

En otra terminal:

```bash
python test_mongodb_api.py
```

Este script probará automáticamente:
- ✓ Conexión a MongoDB
- ✓ Crear paciente
- ✓ Obtener paciente
- ✓ Crear lesión
- ✓ Listar lesiones del paciente
- ✓ Instrucciones para predicción con imagen
- ✓ Listar análisis
- ✓ Progresión temporal

---

## Método 2: Pruebas Manuales con cURL

### 1. Verificar que la API está corriendo

```bash
curl http://localhost:8000/
```

Respuesta esperada:
```json
{
  "name": "skin_lesion_ai_prototype_backend",
  "version": "1.0.0",
  "status": "running",
  "database_connected": true,
  "endpoints": {...}
}
```

### 2. Crear un Paciente

```bash
curl -X POST "http://localhost:8000/api/patients" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "PAT-001",
    "patient_full_name": "María García",
    "sex": "female",
    "date_of_birth": "23/07/1978"
  }'
```

### 3. Obtener el Paciente

```bash
curl http://localhost:8000/api/patients/PAT-001
```

### 4. Crear una Lesión

```bash
curl -X POST "http://localhost:8000/api/lesions" \
  -H "Content-Type: application/json" \
  -d '{
    "lesion_id": "LES-001",
    "patient_id": "PAT-001",
    "lesion_location": "back",
    "initial_size_mm": 12.4
  }'
```

### 5. Listar Lesiones del Paciente

```bash
curl http://localhost:8000/api/patients/PAT-001/lesions
```

### 6. Hacer una Predicción (CON imagen y guardado en BD)

```bash
curl -X POST "http://localhost:8000/api/predict" \
  -F "image=@path/to/your/lesion_image.jpg" \
  -F "age=45" \
  -F "sex=female" \
  -F "location=back" \
  -F "diameter=12.4" \
  -F "patient_id=PAT-001" \
  -F "lesion_id=LES-001"
```

**IMPORTANTE**: Reemplaza `path/to/your/lesion_image.jpg` con la ruta a una imagen real de lesión.

Respuesta esperada:
```json
{
  "model_a_probability": 0.72,
  "model_c_probability": 0.65,
  "extracted_features": [...],
  "metadata": {...},
  "analysis_id": "AN-20250101123045-a1b2c3d4"
}
```

### 7. Obtener el Análisis Guardado

Usando el `analysis_id` de la respuesta anterior:

```bash
curl http://localhost:8000/api/analyses/AN-20250101123045-a1b2c3d4
```

### 8. Ver Progresión Temporal de la Lesión

```bash
curl http://localhost:8000/api/lesions/LES-001/progression
```

### 9. Obtener Análisis de Alto Riesgo

```bash
curl "http://localhost:8000/api/analyses/high-risk/list?threshold=0.5"
```

---

## Método 3: Interfaz Swagger (Recomendado para Principiantes)

### Paso 1: Abrir Swagger UI

Abre en tu navegador: http://localhost:8000/docs

### Paso 2: Probar Endpoints Interactivamente

1. **Crear Paciente**:
   - Clic en `POST /api/patients`
   - Clic en "Try it out"
   - Pega este JSON en el cuerpo:
     ```json
     {
       "patient_id": "PAT-002",
       "patient_full_name": "Carlos López",
       "sex": "male",
       "date_of_birth": "10/03/1985"
     }
     ```
   - Clic en "Execute"

2. **Crear Lesión**:
   - Similar al paso anterior en `POST /api/lesions`

3. **Hacer Predicción con Guardado**:
   - Clic en `POST /api/predict`
   - Clic en "Try it out"
   - Sube una imagen
   - Llena los campos (age, sex, location, diameter)
   - **IMPORTANTE**: Llena también `patient_id` y `lesion_id`
   - Clic en "Execute"

4. **Ver Resultados**:
   - Usa `GET /api/patients/{patient_id}/analyses` para ver todos los análisis del paciente

---

## Método 4: Python con Requests

### Instalación

```bash
pip install requests
```

### Script de Prueba

```python
import requests

BASE_URL = "http://localhost:8000/api"

# 1. Crear paciente
patient = {
    "patient_id": "PAT-003",
    "patient_full_name": "Ana Martínez",
    "sex": "female",
    "date_of_birth": "15/08/1990"
}
response = requests.post(f"{BASE_URL}/patients", json=patient)
print(f"Paciente creado: {response.status_code}")
print(response.json())

# 2. Crear lesión
lesion = {
    "lesion_id": "LES-003",
    "patient_id": "PAT-003",
    "lesion_location": "left arm",
    "initial_size_mm": 8.5
}
response = requests.post(f"{BASE_URL}/lesions", json=lesion)
print(f"\nLesión creada: {response.status_code}")
print(response.json())

# 3. Hacer predicción con imagen
files = {'image': open('path/to/lesion_image.jpg', 'rb')}
data = {
    'age': 35,
    'sex': 'female',
    'location': 'left arm',
    'diameter': 8.5,
    'patient_id': 'PAT-003',
    'lesion_id': 'LES-003'
}
response = requests.post('http://localhost:8000/api/predict', files=files, data=data)
print(f"\nPredicción: {response.status_code}")
result = response.json()
print(f"Model A: {result['model_a_probability']:.2%}")
print(f"Model C: {result['model_c_probability']:.2%}")
print(f"Analysis ID: {result['analysis_id']}")

# 4. Obtener progresión temporal
response = requests.get(f"{BASE_URL}/lesions/LES-003/progression")
print(f"\nProgresión temporal: {response.status_code}")
print(f"Número de análisis: {len(response.json())}")
```

---

## Verificar en MongoDB

### Opción A: MongoDB Compass

1. Descarga MongoDB Compass: https://www.mongodb.com/products/compass
2. Conecta con tu URI: `mongodb+srv://vcloaizacarvajal:UYMdwsv8cK4I0T9f@projectdb.cw3vtth.mongodb.net/`
3. Selecciona la base de datos: `skin_lesion_triage_db`
4. Verás 3 colecciones:
   - `patients`
   - `lesions`
   - `analysis_cases`

### Opción B: MongoDB Atlas Web UI

1. Ingresa a https://cloud.mongodb.com/
2. Ve a "Database" → "Browse Collections"
3. Selecciona `skin_lesion_triage_db`
4. Navega por las colecciones

---

## Endpoints Disponibles

### Pacientes (Patients)
- `POST /api/patients` - Crear paciente
- `GET /api/patients/{patient_id}` - Obtener paciente
- `PUT /api/patients/{patient_id}` - Actualizar paciente
- `DELETE /api/patients/{patient_id}` - Eliminar paciente
- `GET /api/patients` - Listar pacientes (paginado)
- `GET /api/patients/search/by-name?name=...` - Buscar por nombre
- `GET /api/patients/stats/count` - Estadísticas

### Lesiones (Lesions)
- `POST /api/lesions` - Crear lesión
- `GET /api/lesions/{lesion_id}` - Obtener lesión
- `PUT /api/lesions/{lesion_id}` - Actualizar lesión
- `DELETE /api/lesions/{lesion_id}` - Eliminar lesión
- `GET /api/lesions` - Listar lesiones (paginado)
- `GET /api/patients/{patient_id}/lesions` - Lesiones de un paciente
- `GET /api/patients/{patient_id}/lesions/count` - Contar lesiones
- `GET /api/lesions/location/{location}` - Buscar por ubicación

### Análisis (Analyses)
- `POST /api/analyses` - Crear análisis
- `GET /api/analyses/{analysis_id}` - Obtener análisis
- `PUT /api/analyses/{analysis_id}` - Actualizar análisis
- `DELETE /api/analyses/{analysis_id}` - Eliminar análisis
- `GET /api/analyses` - Listar análisis (paginado)
- `GET /api/analyses/high-risk/list?threshold=0.5` - Alto riesgo
- `GET /api/patients/{patient_id}/analyses` - Análisis de un paciente
- `GET /api/lesions/{lesion_id}/analyses` - Análisis de una lesión
- `GET /api/lesions/{lesion_id}/progression` - **Progresión temporal**
- `GET /api/lesions/{lesion_id}/analyses/latest` - Análisis más reciente

### Predicción (Prediction)
- `POST /api/predict` - Predicción (con opciones patient_id y lesion_id)
- `POST /api/explain` - Explicación SHAP

---

## Solución de Problemas

### Error: "database_connected": false

**Causa**: MongoDB no se puede conectar

**Solución**:
1. Verifica tu archivo `.env`:
   ```
   MONGODB_URI=mongodb+srv://vcloaizacarvajal:UYMdwsv8cK4I0T9f@projectdb.cw3vtth.mongodb.net/
   MONGODB_DB_NAME=skin_lesion_triage_db
   ```
2. Verifica que tu IP esté en la lista blanca de MongoDB Atlas
3. Revisa los logs de la aplicación

### Error: "Patient 'PAT-001' already exists"

**Causa**: El paciente ya existe en la base de datos

**Solución**:
- Usa otro patient_id, o
- Elimina el paciente existente: `DELETE /api/patients/PAT-001`

### Error: "No module named 'motor'"

**Causa**: Dependencias no instaladas

**Solución**:
```bash
pip install -r requirements.txt
```

### Error: "Analysis created but could not be retrieved"

**Causa**: Problema con MongoDB durante el guardado

**Solución**:
1. Verifica la conexión a MongoDB
2. Revisa los logs de la aplicación
3. Verifica que las colecciones existan

---

## Limpieza de Datos de Prueba

Para eliminar los datos de prueba:

```bash
# Eliminar paciente (y sus relaciones)
curl -X DELETE http://localhost:8000/api/patients/PAT-001

# Eliminar lesión
curl -X DELETE http://localhost:8000/api/lesions/LES-001

# Eliminar análisis
curl -X DELETE http://localhost:8000/api/analyses/AN-xxx
```

---

## Próximos Pasos

1. ✓ Prueba la API con datos reales
2. ✓ Verifica los datos en MongoDB
3. ✓ Prueba la progresión temporal con múltiples análisis de la misma lesión
4. ✓ Integra con tu frontend
5. ✓ Implementa autenticación (próxima fase)

---

## Recursos Adicionales

- **Documentación Interactiva**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **MongoDB Atlas**: https://cloud.mongodb.com/
- **MongoDB Compass**: https://www.mongodb.com/products/compass
