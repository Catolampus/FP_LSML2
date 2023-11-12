import streamlit as st
from pathlib import Path
import gdown
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision as tv
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import cv2


def plot_preds(pilimg, predict, accuracy):
    """
    Model predicitons
    """
    numimg = np.array(pilimg)
    boxes = predict[0]["boxes"][predict[0]["scores"] > accuracy].cpu().detach().numpy()
    labels = (
        predict[0]["labels"][predict[0]["scores"] > accuracy].cpu().detach().numpy()
    )
    # Numbers to labels
    label_code = ["__background__", "apple", "orange", "banana"]
    # Show
    counter = 0
    for i in boxes:
        final = cv2.rectangle(
            numimg,
            (int(i[0]), int(i[1])),
            (int(i[2] + 30), int(i[3])),
            color=(128, 0, 128),
            thickness=3,
        )
        cv2.putText(
            numimg,
            str(label_code[labels[counter]]),
            (int(i[0]), int(i[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            (0, 0, 0),
            2,
        )
        counter += 1
    plt.figure(figsize=(15, 10))
    return final


def get_weights():
    st.info("Downloading weights!")
    gdown.download(
        "https://drive.google.com/uc?id=1UWKvyQqtnpG8gJeKbvuFCJP723GajKZr",
        "model_params.pth",
    )
    return


if __name__ == "__main__":
    st.header("Model FasterRCNN")
    st.subheader("Upload your image with oranges, apples or bananas")
    if not Path("./model_params.pth").exists():
        get_weights()

    model = tv.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    for param in model.parameters():
        param.requires_grad = False
    num_class = 4
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_class)
    path_for_load = "model_params.pth"
    if Path(path_for_load).exists():
        model.load_state_dict(torch.load(path_for_load, map_location="cpu"))
    model.eval()
    acc = st.selectbox(
        "Desired accuracy:", [0.8, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.1, 0.9, 1.0]
    )
    uploaded_file = st.file_uploader("Upload your image here...")

    if uploaded_file is not None:
        pilimg = Image.open(uploaded_file)
        img = tv.transforms.ToTensor()(pilimg).unsqueeze(0)
        # img = img.to(device)
        predict = model(img)
        try:
            file = plot_preds(pilimg, predict, acc)
            st.image(file)
        except Exception:
            st.error("No such objects :( ")
