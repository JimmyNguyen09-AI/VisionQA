#author: Jimmy Nguyen
import os, tempfile, asyncio, torch
import streamlit as st
from dotenv import load_dotenv

from tools import Yolo, Blip
from prompt import SYSTEM_PROMPT

load_dotenv()

st.set_page_config(page_title="Image Q&A - AI", layout="wide")
st.title("Image Q&A Yolo and Blip")

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def get_blip() -> Blip:
    blip = Blip(device=device)
    blip.load_blip()
    return blip

@st.cache_resource
def get_yolo() -> Yolo:
    yolo = Yolo(device=device)
    yolo.load_model()
    return yolo

blip = get_blip()
yolo = get_yolo()

def blip_caption(img_path: str) -> str:
    return blip.blip_caption(img_path)

def yolo_detect(img_path: str, conf: float = 0.5, iou: float = 0.45) -> str:
    return yolo.yolo_detect(img_path, conf, iou)

from pydantic_ai import Agent
agent = Agent("openai:gpt-4o-mini", system_prompt=SYSTEM_PROMPT)

@agent.tool_plain
def blip_caption_tool(img_path: str) -> str:
    return blip_caption(img_path)

@agent.tool_plain
def yolo_detect_tool(img_path: str) -> str:
    return yolo_detect(
        img_path,
        conf=st.session_state.get("conf", 0.5),
        iou=st.session_state.get("iou", 0.45)
    )

with st.sidebar:
    st.header("Setting parameters for Yolo")
    st.session_state.conf = st.slider("Confidence", 0.10, 0.95, 0.50, 0.05)
    st.session_state.iou = st.slider("IoU", 0.10, 0.90, 0.45, 0.05)
    st.caption(f"Device: **{device}**")

uploaded = st.file_uploader("Upload an image here", type=["jpg", "jpeg", "png"])
question = st.chat_input("Ask a question about the image...")

if uploaded:
    st.image(uploaded, width=300)

if uploaded and question:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded.read())
        img_path = tmp.name

    st.chat_message("user").write(question)

    user_msg = (
        f"{question}\n"
        f"Use this local image path when calling tools: {img_path}\n"
        f"If you run yolo_detect, prefer conf={st.session_state.conf} and iou={st.session_state.iou}."
    )
    with st.spinner("Thinking..."):
        result = asyncio.run(agent.run(user_msg))

    st.chat_message("assistant").write(result.output)
    dets, boxed_path = yolo.yolo_detect_draw(
        img_path,
        conf=st.session_state.conf,
        iou=st.session_state.iou
    )
    st.image(boxed_path, caption="Detections (YOLOv5)")
    st.json(dets)