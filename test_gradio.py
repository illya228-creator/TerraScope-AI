
import time
import threading
import urllib.request

import gradio as gr
import uvicorn

with gr.Blocks() as demo:
    gr.Markdown("# TerraScope AI")
    gr.Markdown("Проверка интерфейса Gradio")

# Исправляем None в конфигурации страницы
demo.config["body_css"] = {}

app = demo.app


def run_server():
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=7861,
        log_level="info",
    )


thread = threading.Thread(target=run_server, daemon=True)
thread.start()

for attempt in range(15):
    time.sleep(1)

    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:7861/",
            timeout=3,
        ) as response:
            print("HTTP status:", response.status)

            if response.status == 200:
                print("SUCCESS: Gradio page responds.")
            break

    except Exception as exc:
        print(f"Attempt {attempt + 1}: {exc}")
else:
    print("ERROR: Server did not respond.")
