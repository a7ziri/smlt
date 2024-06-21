from ultralytics import YOLO
from flask import request, Response, Flask, jsonify
from waitress import serve
from PIL import Image
import json
import torch
import cv2


app = Flask(__name__)


# Определение устройства: GPU, если доступен, иначе CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = YOLO("best.pt")

# Переместите модель на нужное устройство
model.model = model.model.to(device)


@app.route("/")
def root():
    with open("templates/index.html", encoding="utf-8") as file:
        return file.read()
    
@app.route("/stream")
def stream():
    with open("templates/stream.html", encoding="utf-8") as file:
        return file.read()


@app.route('/video_feed')
def video_feed():
    camera_index = request.args.get('camera_index', default=0, type=int)
    return Response(gen_frames(camera_index), mimetype='multipart/x-mixed-replace; boundary=frame')


# Функция для получения списка доступных видеоустройств
def get_available_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr


@app.route('/list_cameras')
def list_cameras():
    cameras = get_available_cameras()
    return jsonify(cameras)


@app.route('/raw_video_feed')
def raw_video_feed():
    camera_index = request.args.get('camera_index', default=0, type=int)
    return Response(gen_raw_frames(camera_index), mimetype='multipart/x-mixed-replace; boundary=frame')


def gen_raw_frames(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open video stream for camera {camera_index}.")
        return

    print(camera_index)
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame from camera {camera_index}.")
            break

        # Просто отправка кадра на веб-страницу без обработки
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()  # Закрытие видеопотока после завершения


def gen_frames(camera_index):
    cap = cv2.VideoCapture(camera_index)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Преобразование кадра в нужный формат
        frame_resized = cv2.resize(frame, (640, 640))  # Изменение размера до 640x640
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0  # HWC -> CHW -> BCHW

        # Выполнение детекции объектов
        results = model(frame_tensor)

        # Отображение результатов
        frame = frame_resized.copy()
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf)
                class_id = int(box.cls)
                
                if class_id < len(model.names):
                    label = f'{model.names[class_id]} {confidence:.2f}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    label = f'Неизвестный класс {class_id} {confidence:.2f}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route("/detect", methods=["POST"])
def detect():
    buf = request.files["image_file"]
    boxes = detect_objects_on_image(Image.open(buf.stream))
    return Response(
        json.dumps(boxes),  
        mimetype='application/json'
    )


def detect_objects_on_image(buf):
    results = model.predict(buf)
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [
            round(x) for x in box.xyxy[0].tolist()
        ]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([
            x1, y1, x2, y2, result.names[class_id], prob
        ])
    return output


if __name__ == '__main__':
    import os
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5000'))
    except ValueError:
        PORT = 5000
    app.run(HOST, PORT)
