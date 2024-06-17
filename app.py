import gradio as gr
import PIL.Image as Image
from ultralytics import YOLO


model = YOLO("best.pt")


def predict_image(img):
    results = model.predict(
        source=img,
        show_labels=True,
        show_conf=True,
        imgsz=640,
    )

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])

    return im


iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image")
    ],
    outputs=gr.Image(type="pil", label="Result"),
    title="Ultralytics Gradio",
    description="Upload images for inference. The Ultralytics YOLOv8n model is used by default.",
)

if __name__ == "__main__":
    iface.launch()
