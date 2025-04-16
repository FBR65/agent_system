import os
import uvicorn

os.environ["SERVER_HOST"] = "127.0.0.1"
os.environ["SERVER_PORT"] = str(8504)

if __name__ == "__main__":
    uvicorn.run(
        "app.app:app", host="0.0.0.0", port=int(os.environ["SERVER_PORT"])
    )  # ,workers=5)
