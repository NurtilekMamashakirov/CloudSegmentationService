from fastapi import FastAPI
from src.api.schemas import FilePath
from src.service.image_segmentation import segmentate_image

app = FastAPI()


@app.post("/segment-clouds")
async def root(file_path: FilePath):
    path = file_path.path
    segmentate_image_path = segmentate_image(path)
    return {"path": segmentate_image_path}
