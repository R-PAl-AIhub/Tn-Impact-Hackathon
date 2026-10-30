import webbrowser
import threading
import uvicorn
import main

def open_browser():
    webbrowser.open("http://127.0.0.1:8000")

threading.Timer(1.5, open_browser).start()

if __name__ == "__main__":
  uvicorn.run(main.app, host="127.0.0.1", port=8000)
