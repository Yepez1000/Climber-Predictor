from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from fastapi import FastAPI, UploadFile
import uvicorn
import io
from fastapi import Form
from fastapi.responses import JSONResponse



app = FastAPI()

class RouteGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)

@app.on_event("startup")
def load_models():
    global color_model, type_model, gnn_model
  
    color_model = YOLO("Pytrochfiles/colorbestv1.pt")
    type_model = YOLO("Pytrochfiles/holdsbestv2.pt")
    
    gnn_model = RouteGNN(in_channels=8, hidden_channels=64, num_classes=10)
    gnn_model.load_state_dict(torch.load("Pytrochfiles/route_gnn_weights.pt", map_location="cpu"))
    gnn_model.eval()


def classify_holds(img_url, target_color):
    # image = Image.open(img_path).convert("RGB")
    response = requests.get(img_path)
    response.raise_for_status()  # will raise error if the request fails
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    temp_path = "temp_downloaded_image.jpg"
    image.save(temp_path)
    color_result = color_model.predict(temp_path)[0]
    color_result = color_model.predict(img_path)[0]

    holds = []
    for i, box in enumerate(color_result.boxes):
        if color_result.names[int(box.cls[0])] != target_color:
            continue

        coords = [int(x) for x in box.xyxy[0].tolist()]
        x1, y1, x2, y2 = coords
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width, height = x2 - x1, y2 - y1

        # Crop and run through hold type model
        crop = image.crop((x1, y1, x2, y2))
        # crop.save("temp_crop.jpg")
        type_result = type_model.predict("temp_crop.jpg")[0]

        if len(type_result.boxes) > 0:
            label = type_result.names[int(type_result.boxes[0].cls[0])]
        else:
            label = "unknown"

        holds.append({
            "x": center_x,
            "y": center_y,
            "type": label.lower(),
            "width": width,
            "height": height
        })

    return holds, image.size  # (width, height)


from torch_geometric.data import Data
import torch.nn.functional as F

HOLD_TYPES = ['jug', 'crimp', 'sloper', 'pinch', 'edge']
type_to_onehot = {t: [int(i == j) for i in range(len(HOLD_TYPES))] for j, t in enumerate(HOLD_TYPES)}

def normalize(val, max_val):
    return val / max_val

def build_graph(holds, wall_size):
    wall_w, wall_h = wall_size
    x_list, edge_index = [], []

    for h in holds:
        x = normalize(h["x"], wall_w)
        y = normalize(h["y"], wall_h)
        w = normalize(h["width"], wall_w)
        h_ = normalize(h["height"], wall_h)
        type_oh = type_to_onehot.get(h["type"], [0]*len(HOLD_TYPES))
        area = normalize(h["width"] * h["height"], wall_w * wall_h)
        x_list.append([x, y, area] + type_oh)

    for i in range(len(holds)):
        for j in range(len(holds)):
            if i != j:
                dx = holds[i]["x"] - holds[j]["x"]
                dy = holds[i]["y"] - holds[j]["y"]
                dist = (dx**2 + dy**2) ** 0.5
                if dist / wall_w < 0.25:
                    edge_index.append([i, j])

    x = torch.tensor(x_list, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    graph = Data(x=x, edge_index=edge_index)
    return graph




@app.get("/")
def home():
    return {"message": "FastAPI is running on EC2"}


@app.post("/api/upload")
async def predict(img_url: str = Form(...), target_color: str = Form(...)):
    try:
        contents = await file.read()
       

        # Run your classification logic
        holds, wall_size = classify_holds(img_url, target_color)
        graph = build_graph(holds, wall_size)

        if graph.edge_index.numel() == 0:
            return JSONResponse(content={
                "grade": "Unpredictable - insufficient graph data"
            })

        with torch.no_grad():
            graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
            output = gnn_model(graph.x, graph.edge_index, graph.batch)
            prediction = output.argmax(dim=1).item()

        return JSONResponse(content={
            "grade": f"V{prediction}"
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})