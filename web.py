from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import base64

from detect_cam import detect_sign_from_frame, draw_label_on_frame

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def index():
    return FileResponse("static/index.html")

@app.post("/detect-sign")
async def detect_sign_api(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            print("Ảnh decode không thành công.")
            return JSONResponse(content={"label": "Ảnh không hợp lệ", "image": ""})

        print("Kích thước ảnh nhận được:", img.shape)

        label = detect_sign_from_frame(img)
        print("Nhãn dự đoán:", label)

        img = draw_label_on_frame(img, label)

        _, img_encoded = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(img_encoded).decode("utf-8")

        return JSONResponse(content={
            "label": label or "Không phát hiện",
            "image": img_base64
        })

    except Exception as e:
        print("Lỗi xử lý:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})
