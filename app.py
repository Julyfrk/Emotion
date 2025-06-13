from flask import Flask, request, jsonify
import os
import json
import mediapipe as mp
import numpy as np
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageEnhance
import io
import base64
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from dotenv import load_dotenv
from fer import FER
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

app = Flask(__name__)
CORS(app)

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Configura las credenciales de Google Drive desde la variable de entorno
CLIENT_SECRET_JSON = os.getenv('GOOGLE_DRIVE_CREDENTIALS')
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# ID de la carpeta donde deseas subir la imagen
FOLDER_ID = '1RLHKFduSGrOZNQM__5LF1HRAiduhyHMl'

# Traducción de emociones
TRADUCCION_EMOCIONES = {
    "angry": "enojado",
    "disgust": "disgustado",
    "fear": "miedo",
    "happy": "feliz",
    "sad": "triste",
    "surprise": "sorprendido",
    "neutral": "neutral"
}


def obtener_servicio_drive():
    """Inicializa el servicio de Google Drive."""
    try:
        creds = service_account.Credentials.from_service_account_info(
            json.loads(CLIENT_SECRET_JSON), scopes=SCOPES)
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        raise Exception(f"Error al cargar las credenciales: {e}")


def convertir_a_base64(imagen):
    """Convierte una imagen PIL a Base64."""
    buffered = io.BytesIO()
    imagen.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def procesar_imagen_con_puntos(image_np):
    """Procesa la imagen y añade puntos faciales usando Mediapipe."""
    imagen = Image.fromarray(image_np)
    mp_face_mesh = mp.solutions.face_mesh
    puntos_deseados = [70, 55, 285, 300, 33, 468, 133, 362, 473, 263, 4, 185, 0, 306, 17]

    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(image_np)
        if results.multi_face_landmarks:
            draw = ImageDraw.Draw(imagen)
            for face_landmarks in results.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    if idx in puntos_deseados:
                        h, w, _ = image_np.shape
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        draw.line((x - 4, y - 4, x + 4, y + 4), fill=(255, 0, 0), width=2)
                        draw.line((x - 4, y + 4, x + 4, y - 4), fill=(255, 0, 0), width=2)
    return imagen


@app.route('/upload', methods=['POST'])
def detectar_puntos_y_procesar_imagenes():
    """Procesa la imagen, detecta puntos faciales y emociones, y genera un PDF."""
    if 'file' not in request.files:
        return jsonify({'error': 'No se recibió correctamente la imagen'})

    archivo = request.files['file']
    if archivo.filename == '':
        return jsonify({'error': 'No se cargó ninguna imagen'})

    try:
        # --- Lectura y preprocesado ---
        imagen_pil = Image.open(archivo).convert('RGB')
        imagen_pil = imagen_pil.resize((300, 300))
        imagen_np = np.array(imagen_pil)

        # Mejora de contraste y nitidez
        imagen_mejorada = ImageEnhance.Contrast(imagen_pil).enhance(1.5)
        imagen_mejorada = ImageEnhance.Sharpness(imagen_mejorada).enhance(2.0)

        # Detectar puntos faciales
        imagen_con_puntos = procesar_imagen_con_puntos(imagen_np)

        # Detectar emoción dominante
        detector = FER(mtcnn=False)
        emociones = detector.detect_emotions(np.array(imagen_mejorada))
        if emociones:
            en = max(emociones[0]["emotions"], key=emociones[0]["emotions"].get)
            emocion_principal = TRADUCCION_EMOCIONES.get(en, en)
        else:
            emocion_principal = "No detectada"

        # (Opcional) Subida a Drive...
        service = obtener_servicio_drive()
        buf_drive = io.BytesIO()
        imagen_pil.save(buf_drive, format='PNG')
        buf_drive.seek(0)
        media = MediaIoBaseUpload(buf_drive, mimetype='image/png')
        meta = {'name': archivo.filename, 'parents':[FOLDER_ID]}
        drive_file = service.files().create(body=meta, media_body=media).execute()
        drive_id = drive_file.get('id')

        # --- Generar PDF en memoria ---
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        width, height = letter

        # Dibujar imagen: primero convertimos a un buffer de bytes
        img_buffer = io.BytesIO()
        imagen_con_puntos.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        # Ajusta tamaño y posición según te convenga:
        img_width = 300
        img_height = 300
        x = (width - img_width) / 2
        y = height - img_height - 100
        c.drawImage(img_buffer, x, y, img_width, img_height)

        # Escribir texto de la emoción
        text_x = 50
        text_y = y - 50
        c.setFont("Helvetica-Bold", 14)
        c.drawString(text_x, text_y, f"Emoción dominante: {emocion_principal}")

        # (Opcional) ID de Drive
        c.setFont("Helvetica", 10)
        c.drawString(text_x, text_y - 20, f"Drive file ID: {drive_id}")

        c.showPage()
        c.save()
        pdf_buffer.seek(0)

        # Devolver PDF
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name='resultado.pdf',
            mimetype='application/pdf'
        )

    except Exception as e:
        return jsonify({'error': f"Error al procesar la imagen: {e}"})



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
