from fastapi import FastAPI
from fastapi.responses import HTMLResponse


APP_ROOT_MARKER = "app-root"
APP_SHELL_MARKER = "training-manager-dashboard"


def build_shell_html() -> str:
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Training Manager</title>
  </head>
  <body>
    <div id="{APP_ROOT_MARKER}" data-shell="{APP_SHELL_MARKER}"></div>
  </body>
</html>
"""


def create_shell_app() -> FastAPI:
    app = FastAPI(title="Training Manager Frontend Shell")

    @app.get("/", response_class=HTMLResponse)
    async def shell() -> str:
        return build_shell_html()

    return app


app = create_shell_app()

