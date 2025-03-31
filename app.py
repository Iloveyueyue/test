import torch
import gradio as gr
from PIL import Image
import pandas as pd
import cv2
from ultralytics import YOLO
from torchvision import transforms

# 加载预训练模型
yolo_model = YOLO('best.pt')  # YOLO物体检测模型
ssp_model = torch.load('ssp.pt', map_location=torch.device('cpu'))  # AI生成检测模型
event_model = torch.load('model.pt', map_location=torch.device('cpu'))  # 事件检测模型


# 事件检测预处理函数
def event_preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image.convert("RGB")).unsqueeze(0)


# YOLO物体检测函数
def yolo_predict(image_path, confidence_threshold=0.5):
    results = yolo_model(image_path)
    detections = []
    class_names = ['handgun', 'fire', 'long-barrelled gun', 'knife']

    for box in results[0].boxes:
        conf = float(box.conf)
        if conf >= confidence_threshold:
            detections.append({
                "Category": class_names[int(box.cls)],
                "Confidence": round(conf, 2)
            })

    # 生成标注图像
    annotated_img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_img), pd.DataFrame(detections)


# SSP AI检测函数
def ssp_predict(image_path):
    image = Image.open(image_path)
    # 此处添加实际推理逻辑
    return image, pd.DataFrame([{"Category": "AI生成内容", "Confidence": 0.92}])


# 事件检测函数
def event_predict(image_path):
    image = Image.open(image_path)
    input_tensor = event_preprocess(image)

    with torch.no_grad():
        outputs = event_model(input_tensor)

    # 定义事件类别（根据实际模型输出修改）
    event_classes = [
        "暴力行为", "火灾", "交通事故", "盗窃事件",
        "人群聚集", "武器出现", "打架斗殴", "异常奔跑",
        "设施损坏", "区域入侵", "可疑物品", "人员跌倒"
    ]

    # 处理模型输出
    probs = torch.sigmoid(outputs).cpu().numpy()[0]
    results = sorted(
        [{"Category": cls, "Confidence": round(float(prob), 2)}
         for cls, prob in zip(event_classes, probs)],
        key=lambda x: x["Confidence"],
        reverse=True
    )[:12]  # 取置信度最高的12个结果

    return image, pd.DataFrame(results)


# 统一预测函数
def model_predict(image_path, model_type, confidence):
    if model_type == "YOLOv11 物体检测":
        return yolo_predict(image_path, confidence)
    elif model_type == "SSP AI 检测":
        return ssp_predict(image_path)
    elif model_type == "事件检测":
        return event_predict(image_path)


# 创建Gradio界面
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🚀 智检视界 - 多模态图像分析平台")

    with gr.Row():
        model_selector = gr.Radio(
            label="选择检测模式",
            choices=["YOLOv11 物体检测", "SSP AI 检测", "事件检测"],
            value="YOLOv11 物体检测"
        )
        confidence_slider = gr.Slider(
            label="置信度阈值",
            minimum=0,
            maximum=1,
            value=0.5,
            step=0.05
        )

    with gr.Row():
        input_image = gr.Image(type="filepath", label="输入图像", height=400)
        output_image = gr.Image(type="pil", label="检测结果", height=400)

    results_table = gr.Dataframe(
        headers=["检测类别", "置信度"],
        datatype=["str", "number"],
        label="检测结果明细",
        interactive=False
    )

    with gr.Row():
        predict_btn = gr.Button("开始检测", variant="primary")
        clear_btn = gr.Button("清空结果")

    # 事件处理
    predict_btn.click(
        fn=model_predict,
        inputs=[input_image, model_selector, confidence_slider],
        outputs=[output_image, results_table]
    )

    clear_btn.click(
        fn=lambda: [None, None, pd.DataFrame(columns=["检测类别", "置信度"])],
        outputs=[input_image, output_image, results_table]
    )

# 启动应用
if __name__ == "__main__":
    app.launch(share=True, server_port=7860)