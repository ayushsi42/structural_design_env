from fastapi.testclient import TestClient

from server.app import app


client = TestClient(app)


def test_root_serves_interactive_demo():
    response = client.get("/")

    assert response.status_code == 200
    assert "StructuralDesignEnv Interactive Demo" in response.text
    assert "Site Grid" in response.text
    assert 'id="gridSvg"' in response.text


def test_demo_alias_serves_interactive_demo():
    response = client.get("/demo")

    assert response.status_code == 200
    assert "Element Inspector" in response.text
    assert "Live Results" in response.text
    assert "data-start-episode" in response.text


def test_tasks_endpoint_still_lists_registry():
    response = client.get("/tasks")

    assert response.status_code == 200
    payload = response.json()
    task_ids = {task["id"] for task in payload["tasks"]}
    assert {"task1_warehouse", "task2_office", "task3_hospital"} <= task_ids
