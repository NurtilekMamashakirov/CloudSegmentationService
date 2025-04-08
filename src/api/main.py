from fastapi import FastAPI
from src.api.schemas import FilePath
from src.service.image_segmentation import segmentate_image

app = FastAPI()


@app.post("/segment-clouds")
async def root(file_path: FilePath):
    path = file_path.path
    segmentated_image = segmentate_image(path)
    path = segmentated_image["path"]
    cloud_percentage = segmentated_image["cloud_percentage"]
    return {"path": path, "cloud_percentage": cloud_percentage}
