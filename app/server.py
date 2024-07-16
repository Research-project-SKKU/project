import argparse
import json
from multiprocessing import Pipe, Process

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from websockets import ConnectionClosed, connect

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

app.mount("/static", StaticFiles(directory="app/static"), name="static")

clients = []


@app.get("/")
async def get_index_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            print(data)
            for client in clients:
                await client.send_text(data)
    except WebSocketDisconnect:
        clients.remove(websocket)
        print("WebSocket client disconnected")
    except ConnectionClosed:
        clients.remove(websocket)
        print("WebSocket connection closed")


class Server:
    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        self.host = host
        self.port = port
        self.uri = f"ws://{self.host}:{self.port}/ws"
        self.process = None
        self.parent_conn, self.child_conn = Pipe()

    async def send_model_info(self, model_info):
        model_info = {k: v.tolist() for k, v in model_info.items()}
        async with connect(self.uri) as websocket:
            await websocket.send(json.dumps(model_info))

    def run_server(self, conn):
        conn.send("ready")
        conn.close()
        uvicorn.run(app, host=self.host, port=self.port)

    def start(self):
        self.process = Process(target=self.run_server, args=(self.child_conn,))
        self.process.start()
        self.parent_conn.recv()

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start the Pytorch visualization server"
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host for the server"
    )
    parser.add_argument("--port", type=int, default=8000, help="Port for the server")
    args = parser.parse_args()

    server = Server(host=args.host, port=args.port)
    server.start()
