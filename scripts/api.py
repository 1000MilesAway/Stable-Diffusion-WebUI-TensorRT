from modules.script_callbacks import on_app_started
from modules.api.models import *
from modules.api import api

from fastapi import FastAPI, Body
import gradio as gr

from ui_trt import export_unet_to_trt


def trt_api(_: gr.Blocks, app: FastAPI):

    @app.post("/trt/convert")
    async def convert(
        height: int = Body(..., description="Height of the image to convert"),
        width: int = Body(..., description="Width of the image to convert"),
        ):
        export_unet_to_trt(
            batch_min=1,
            batch_opt=1,
            batch_max=1,
            height_min=height,
            height_opt=height,
            height_max=height,
            width_min=width,
            width_opt=width,
            width_max=width,
            token_count_min=75,
            token_count_opt=75,
            token_count_max=75,
            force_export=False,
            static_shapes=True,
            preset=None,)
        return {}


on_app_started(trt_api)