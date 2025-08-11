#author: Jimmy Nguyen
import torch
import os, uuid
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
class Yolo():
    def __init__(self,device=None):
        self.device = device
        self.model = None
    def load_model(self):
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5m", pretrained=True).to(self.device)
        return self.model
    def yolo_detect(self,img_path: str, conf: float=0.5, iou:float=0.45) -> str:
        self.model.conf = conf
        self.model.iou = iou
        res = self.model(img_path)
        output = res.xyxy[0].tolist()
        if len(output) == 0:
            return "No objects above threshold"
        names = res.names
        lines = []
        for *xyxy,score,cls in output:
            x1,y1,x2,y2 = map(int,xyxy)
            lines.append(f"[{x1}, {y1}, {x2}, {y2}] {names[int(cls)]} {float(score):.3f}")
        return "\n".join(lines)

    def yolo_detect_draw(self, img_path: str, conf: float = 0.5, iou: float = 0.45, save_path: str = None):

        self.model.conf = conf
        self.model.iou = iou
        res = self.model(img_path)
        rendered = res.render()[0]
        if save_path is None:
            root, ext = os.path.splitext(img_path)
            save_path = f"{root}_yolo{ext}"

        from PIL import Image
        Image.fromarray(rendered).save(save_path)
        dets = []
        names = res.names
        for *xyxy, score, cls in res.xyxy[0].tolist():
            x1, y1, x2, y2 = map(int, xyxy)
            dets.append({"bbox": [x1, y1, x2, y2], "label": names[int(cls)], "score": float(score)})
        return dets, save_path
class Blip():
    def __init__(self, device=None):
        self.device = device
        self.proc = None
        self.blip = None
    def load_blip(self, model_name = "Salesforce/blip-image-captioning-base"):
        self.proc = BlipProcessor.from_pretrained(model_name)
        self.blip = BlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        return self.proc, self.blip
    def blip_caption(self,img_path:str)->str:
        image = Image.open(img_path).convert("RGB")
        inputs = self.proc(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out_ids = self.blip.generate(**inputs, max_new_tokens=30, num_beams=5)
        return self.proc.decode(out_ids[0], skip_special_tokens=True)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Yolo = Yolo(device=device)
    Yolo.load_model()
    print(Yolo.yolo_detect("./cat1.jpg"))
    Blip = Blip(device=device)
    Blip.load_blip()
    print(Blip.blip_caption("./cat1.jpg"))






