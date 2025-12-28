"""
Script de prueba manual para la API con integración MongoDB.
Ejecutar con: python test_mongodb_api.py
"""

import requests
import json
from datetime import datetime

# Configuración
BASE_URL = "http://localhost:8000"
API_URL = f"{BASE_URL}/api"


def print_section(title):
    """Imprime un título de sección."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def print_response(response):
    """Imprime una respuesta HTTP formateada."""
    print(f"\nStatus Code: {response.status_code}")
    try:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except:
        print(f"Response: {response.text}")


def test_health_check():
    """Prueba el endpoint de health check."""
    print_section("1. HEALTH CHECK")

    response = requests.get(f"{BASE_URL}/")
    print_response(response)

    if response.status_code == 200:
        data = response.json()
        if data.get("database_connected"):
            print("\n✓ MongoDB conectado correctamente")
        else:
            print("\n✗ MongoDB NO está conectado")
            return False
    return True


def test_create_patient():
    """Prueba crear un paciente."""
    print_section("2. CREAR PACIENTE")

    patient_data = {
        "patient_id": "PAT-001",
        "patient_full_name": "Juan Pérez",
        "sex": "male",
        "date_of_birth": "15/05/1980"
    }

    print(f"\nDatos del paciente:")
    print(json.dumps(patient_data, indent=2))

    response = requests.post(f"{API_URL}/patients", json=patient_data)
    print_response(response)

    return response.status_code == 201


def test_get_patient():
    """Prueba obtener un paciente."""
    print_section("3. OBTENER PACIENTE")

    patient_id = "PAT-001"
    response = requests.get(f"{API_URL}/patients/{patient_id}")
    print_response(response)

    return response.status_code == 200


def test_create_lesion():
    """Prueba crear una lesión."""
    print_section("4. CREAR LESIÓN")

    lesion_data = {
        "lesion_id": "LES-001",
        "patient_id": "PAT-001",
        "lesion_location": "back",
        "initial_size_mm": 12.5
    }

    print(f"\nDatos de la lesión:")
    print(json.dumps(lesion_data, indent=2))

    response = requests.post(f"{API_URL}/lesions", json=lesion_data)
    print_response(response)

    return response.status_code == 201


def test_get_patient_lesions():
    """Prueba obtener lesiones de un paciente."""
    print_section("5. OBTENER LESIONES DEL PACIENTE")

    patient_id = "PAT-001"
    response = requests.get(f"{API_URL}/patients/{patient_id}/lesions")
    print_response(response)

    return response.status_code == 200


def test_predict_with_database():
    """Prueba hacer una predicción que se guarda en la base de datos."""
    print_section("6. PREDICCIÓN CON GUARDADO EN BD")

    print("\nNOTA: Este test requiere una imagen. Por ahora solo mostramos el formato.")
    print("\nEjemplo de cURL para hacer la predicción:")

    curl_example = """
curl -X POST "http://localhost:8000/api/predict" \\
  -H "Content-Type: multipart/form-data" \\
  -F "image=@path/to/lesion_image.jpg" \\
  -F "age=45" \\
  -F "sex=male" \\
  -F "location=back" \\
  -F "diameter=12.5" \\
  -F "patient_id=PAT-001" \\
  -F "lesion_id=LES-001"
"""

    print(curl_example)

    print("\nEjemplo de Python con requests:")

    python_example = """
import requests

files = {'image': open('path/to/lesion_image.jpg', 'rb')}
data = {
    'age': 45,
    'sex': 'male',
    'location': 'back',
    'diameter': 12.5,
    'patient_id': 'PAT-001',
    'lesion_id': 'LES-001'
}

response = requests.post('http://localhost:8000/api/predict', files=files, data=data)
print(response.json())
"""

    print(python_example)

    return True


def test_list_analyses():
    """Prueba listar análisis."""
    print_section("7. LISTAR ANÁLISIS")

    response = requests.get(f"{API_URL}/analyses")
    print_response(response)

    return response.status_code == 200


def test_get_lesion_progression():
    """Prueba obtener progresión temporal de una lesión."""
    print_section("8. PROGRESIÓN TEMPORAL DE LESIÓN")

    lesion_id = "LES-001"
    response = requests.get(f"{API_URL}/lesions/{lesion_id}/progression")
    print_response(response)

    return response.status_code == 200


def cleanup():
    """Limpia los datos de prueba (opcional)."""
    print_section("9. LIMPIEZA (OPCIONAL)")

    print("\n¿Deseas eliminar los datos de prueba? (s/n): ", end="")
    choice = input().lower()

    if choice == 's':
        # Eliminar paciente (esto debería eliminar también lesiones y análisis en cascada)
        response = requests.delete(f"{API_URL}/patients/PAT-001")
        print(f"\nEliminando paciente PAT-001: {response.status_code}")

        # Eliminar lesión
        response = requests.delete(f"{API_URL}/lesions/LES-001")
        print(f"Eliminando lesión LES-001: {response.status_code}")

        print("\n✓ Limpieza completada")
    else:
        print("\n- Datos de prueba conservados")


def main():
    """Ejecuta todas las pruebas."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "TEST MONGODB API INTEGRATION" + " "*30 + "║")
    print("╚" + "="*78 + "╝")

    print(f"\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base URL: {BASE_URL}")

    try:
        # Ejecutar pruebas
        if not test_health_check():
            print("\n✗ PRUEBA FALLIDA: MongoDB no conectado")
            print("Asegúrate de que:")
            print("  1. La API está corriendo (python main.py)")
            print("  2. MongoDB está accesible")
            print("  3. La variable MONGODB_URI en .env es correcta")
            return

        test_create_patient()
        test_get_patient()
        test_create_lesion()
        test_get_patient_lesions()
        test_predict_with_database()
        test_list_analyses()
        test_get_lesion_progression()

        # Limpieza opcional
        cleanup()

        print_section("RESUMEN")
        print("\n✓ Todas las pruebas básicas completadas")
        print("\nPróximos pasos:")
        print("  1. Prueba el endpoint /api/predict con una imagen real")
        print("  2. Verifica los datos en MongoDB Compass o Atlas")
        print("  3. Revisa la documentación interactiva en http://localhost:8000/docs")

    except requests.exceptions.ConnectionError:
        print("\n✗ ERROR: No se pudo conectar a la API")
        print("Asegúrate de que la API está corriendo:")
        print("  python main.py")
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
