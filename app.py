import gradio as gr
import PIL.Image as Image
import requests
from io import BytesIO
from PIL import ImageDraw, ImageFont
import cv2
import tempfile
import os
import time

print('base start')

def predict_image(img):
    # Конвертируем изображение в байты
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()

    # Отправляем изображение на API
    try:
        response = requests.post(
            "http://localhost:5000/detect",
            files={"image_file": ("image.jpg", img_bytes, "image/jpeg")}
        )
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"

    # Проверяем статус ответа
    if response.status_code != 200:
        return f"Error: Received status code {response.status_code} from API"

    # Напечатаем ответ для отладки
    print("Response text:", response.text)

    # Получаем результаты детекции
    try:
        boxes = response.json()
    except requests.exceptions.JSONDecodeError:
        return "Error: Unable to decode JSON response"

    # Отрисовываем результаты на изображении
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 70)
    except IOError:
        font = ImageFont.load_default()

    for box in boxes:
        x1, y1, x2, y2, label, conf = box
        draw.rectangle([x1, y1, x2, y2], outline="orange", width=4)
        draw.text((x1, y1), f"{label} {conf:.2f}", fill="orange", font=font)

    return img


def predict_images(files):
    results = []
    for file in files:
        # Открываем изображение
        img = Image.open(file.name)

        # Конвертируем изображение в байты
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()

        # Отправляем изображение на API
        try:
            response = requests.post(
                "http://localhost:5000/detect",
                files={"image_file": ("image.jpg", img_bytes, "image/jpeg")}
            )
        except requests.exceptions.RequestException as e:
            results.append(f"Request failed for {file.name}: {e}")
            continue

        # Проверяем статус ответа
        if response.status_code != 200:
            results.append(f"Error: Received status code {response.status_code} from API for {file.name}")
            continue

        # Получаем результаты детекции
        try:
            boxes = response.json()
        except requests.exceptions.JSONDecodeError:
            results.append(f"Error: Unable to decode JSON response for {file.name}")
            continue

        # Отрисовываем результаты на изображении
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 70)
        except IOError:
            font = ImageFont.load_default()

        for box in boxes:
            x1, y1, x2, y2, label, conf = box
            draw.rectangle([x1, y1, x2, y2], outline="yellow", width=4)
            draw.text((x1, y1), f"{label} {conf:.2f}", fill="yellow", font=font)

        results.append(img)

    return results


# Функция для отправки кадра на API
def send_frame_to_api(frame):
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()

    # Отправляем изображение на API
    try:
        response = requests.post(
            "http://localhost:5000/detect",
            files={"frame": ("frame.jpg", img_bytes, "image/jpeg")}
        )
        if response.status_code == 200:
            print(f"Frame processed successfully: {response.json()}")
        else:
            print(f"Error processing frame: Status code {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")


def convert_frame_to_pil_image(frame):
    # Преобразуем кадр в изображение формата PIL
    pil_image = Image.fromarray(frame)
    return pil_image


def process_video(video_file):
    # Создаем папку для сохранения кадров в формате JPEG
    output_folder = "frames"
    os.makedirs(output_folder, exist_ok=True)

    print(video_file, type(video_file))

    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_file}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Total frames: {frame_count}, FPS: {fps}")

    frame_number = 0
    processed_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Преобразуем кадр в формат JPEG в оперативной памяти
        pil_image = convert_frame_to_pil_image(frame)

        # Отправляем кадр на API
        processed_image = predict_image(pil_image)

        # Добавляем обработанное изображение в список
        processed_frames.append(processed_image)

        frame_number += 1

    cap.release()
    return processed_frames


with gr.Blocks() as iface:
    with gr.Tabs():
        with gr.TabItem("Photo"):
            with gr.Row():
                photo_to_load = gr.Image(type="pil", label='Choose your photo')
                photo_result = gr.Image(type="pil")
            detect_button = gr.Button("Scan")
            detect_button.click(predict_image, inputs=photo_to_load, outputs=photo_result)
        with gr.TabItem("Multiply"):
            files_to_load = gr.Files(label='Choose your files', type='filepath', file_count='multiple')
            detect_button = gr.Button("Scan")
            files_result = gr.Gallery()
            detect_button.click(predict_images, inputs=files_to_load, outputs=files_result)
            files_result
        with gr.TabItem("Video"):
            video_to_load = gr.Video(label='Choose your fideo', value='Path')
            detect_button = gr.Button("Scan")
            frames_result = gr.Gallery()
            detect_button.click(process_video, inputs=video_to_load, outputs=frames_result)
            frames_result


if __name__ == "__main__":
    iface.launch(share=True)
