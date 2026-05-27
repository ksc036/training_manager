from fastapi.testclient import TestClient

from app.main import create_app


def test_core_pages_are_available() -> None:
    client = TestClient(create_app())

    assert client.get("/datasets").status_code == 200
    assert client.get("/train").status_code == 200
    assert client.get("/runs").status_code == 200
    assert client.get("/compare").status_code == 200
