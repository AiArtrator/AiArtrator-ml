import base64
import io
import os
from typing import Optional
from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Response, File
from fastapi.responses import FileResponse


import boto3
import numpy as np
from PIL import Image

import onnxruntime as ort
import dotenv

dotenv.load_dotenv()

BACKEND_ADD = os.getenv("BACKEND_ADD")
AWS_ID = os.getenv("S3_AWS_ID")
AWS_SECRET_KEY = os.getenv("S3_AWS_SECRET_KEY")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

s3r = boto3.resource(
    "s3", aws_access_key_id=AWS_ID, aws_secret_access_key=AWS_SECRET_KEY
)
BUCKET = s3r.Bucket(BUCKET_NAME)


class MockBucket:
    def download_file(self, _, filename):
        print(filename)


if os.getenv("MOCKING") == "True":
    BUCKET = MockBucket()
    MOD = 1
else:
    MOD = 10000
    # push gan to s3
    BUCKET.upload_file("tmp/model1.onnx", "model1.onnx")
    print("uploaded model to s3")

app = FastAPI()

origins = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=[BACKEND_ADD],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get(
    "/image", responses={200: {"content": {"image/png": {}}}}, response_class=Response
)
def getimg(model_address: str):
    # assert if model is not available at s3
    uuid = hash(model_address) % (MOD) + 1

    # check if the model exists locally
    if not os.path.exists(f"tmp/model{uuid}.onnx"):
        BUCKET.download_file(model_address, f"./tmp/model{uuid}.onnx")

    try:
        model = ort.InferenceSession(f"./tmp/model{uuid}.onnx")
    except:
        # internal server error, model is not available
        return Response(status_code=500, content="Model is not available")

    # timeout
    x = model.run(None, {"x": np.random.randn(1, 120).astype(np.float32)})
    # make image into numpy array
    img = (np.array(x[0]).squeeze(0) + 1) / 2
    # c h w -> h w c
    img = img.transpose(1, 2, 0)

    img = img * 255
    img = img.astype(np.uint8)

    img = Image.fromarray(img)

    # save img
    img.save(f"./tmp/img{uuid}.png")

    return FileResponse(f"./tmp/img{uuid}.png", media_type="image/png")


@app.get("/validate_model")
def validate(model_address: str):

    # download model from s3
    uuid = hash(model_address) % (MOD) + 1
    BUCKET.download_file(model_address, f"./tmp/model{uuid}.onnx")

    # validate model

    model = ort.InferenceSession(f"./tmp/model{uuid}.onnx")

    # timeout

    x = model.run(None, np.randn(1, 120))

    return {"status": "success"}


@app.post("/upload_model")
def upload_model(model_address: str, model: bytes = File(...)):
    # upload model to s3
    # change model to .read()

    model_io = io.BytesIO(model)
    # upload
    BUCKET.upload_fileobj(model_io, model_address)
    return {"status": "success"}
