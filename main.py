import io
import os

import boto3
import dotenv
import numpy as np
import onnxruntime as ort
import torch
from diffusers import LMSDiscreteScheduler
from fastapi import FastAPI, File, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from paint_with_words import paint_with_words, pww_load_tools
import os
import wget
import random
import requests

dotenv.load_dotenv()



loaded = pww_load_tools(
    "cuda:0",
    scheduler_type=LMSDiscreteScheduler,
    hf_model_path="CompVis/stable-diffusion-v1-4"
)

vae, unet, text_encoder, tokenizer, scheduler = loaded

def load_learned_embed_in_clip(
    learned_embeds_path, text_encoder, tokenizer, token=None
):
    loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")

    # separate token and the embeds
    trained_token = list(loaded_learned_embeds.keys())[0]
    embeds = loaded_learned_embeds[trained_token]

    # cast to dtype of text_encoder
    text_encoder.get_input_embeddings().weight.dtype

    # add the token in tokenizer
    token = token if token is not None else trained_token
    num_added_tokens = tokenizer.add_tokens(token)
    i = 1
    while num_added_tokens == 0:
        print(f"The tokenizer already contains the token {token}.")
        token = f"{token[:-1]}-{i}>"
        print(f"Attempting to add the token {token}.")
        num_added_tokens = tokenizer.add_tokens(token)
        i += 1

    # resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))

    # get the id for the token and assign the embeds
    token_id = tokenizer.convert_tokens_to_ids(token)
    text_encoder.get_input_embeddings().weight.data[token_id] = embeds
    return token





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

    def upload_fileobj(self, _a, _b):
        print(_a, _b)


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
        model_id = "sd-concepts-library/midjourney-style"


        # CODE FROM https://huggingface.co/spaces/sd-concepts-library/stable-diffusion-conceptualizer/blob/main/app.py.
        # MIT Licensed

        embeds_url = f"https://huggingface.co/{model_id}/resolve/main/learned_embeds.bin"
        os.makedirs(model_id,exist_ok = True)
        if not os.path.exists(f"{model_id}/learned_embeds.bin"):
            try:
                wget.download(embeds_url, out=model_id)
            except:
                print("Download failed. Trying with requests.")

        token_identifier = f"https://huggingface.co/{model_id}/raw/main/token_identifier.txt"
        response = requests.get(token_identifier)
        response.text

        concept_type = f"https://huggingface.co/{model_id}/raw/main/type_of_concept.txt"
        response = requests.get(concept_type)
        response.text

        load_learned_embed_in_clip(
            f"{model_id}/learned_embeds.bin", text_encoder, tokenizer, token=None
        )

    except:
        # internal server error, model is not available
        return Response(status_code=500, content="Model is not available")
    
    img = paint_with_words(
            color_context={}, # Change here
            color_map_image=None, # Change here
            input_prompt= "<midjourney-style>", # change here
            preloaded_utils=loaded,
            seed = random.randint(0, 100000),
        )
    
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

    model.run(None, np.randn(1, 120))

    return {"status": "success"}


@app.post("/upload_model")
def upload_model(model_address: str, model: bytes = File(...)):
    # upload model to s3
    # change model to .read()

    model_io = io.BytesIO(model)
    # upload
    BUCKET.upload_fileobj(model_io, model_address)
    return {"status": "success"}
