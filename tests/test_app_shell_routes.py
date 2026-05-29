from fastapi.testclient import TestClient

from app.main import create_app
from frontend.app_shell import APP_ROOT_MARKER, APP_SHELL_MARKER


def test_app_shell_mount_returns_html_root_marker() -> None:
    client = TestClient(create_app())
    response = client.get("/app")

    assert response.status_code == 200
    assert f'id="{APP_ROOT_MARKER}"' in response.text
    assert f'data-shell="{APP_SHELL_MARKER}"' in response.text
