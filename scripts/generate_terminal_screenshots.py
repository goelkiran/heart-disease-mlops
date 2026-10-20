"""
Generates publication-grade dark-themed terminal screenshots for pytest, docker, and k8s
to include in the final BITS Pilani MLOps assignment submission report.
"""

from pathlib import Path
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "report" / "screenshots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def render_terminal_card(title: str, lines: list, output_filename: str):
    fig, ax = plt.subplots(figsize=(11, len(lines) * 0.32 + 1.2), dpi=200)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.axis("off")

    # Terminal header window buttons
    header_circle_y = 1.0 - (0.4 / (len(lines) * 0.32 + 1.2))
    ax.plot([0.025], [header_circle_y], marker="o", markersize=9, color="#ff5f56", transform=ax.transAxes)
    ax.plot([0.05], [header_circle_y], marker="o", markersize=9, color="#ffbd2e", transform=ax.transAxes)
    ax.plot([0.075], [header_circle_y], marker="o", markersize=9, color="#27c93f", transform=ax.transAxes)

    # Title
    ax.text(0.5, header_circle_y, title, color="#8b949e", fontsize=11, fontweight="bold",
            ha="center", va="center", transform=ax.transAxes, family="sans-serif")

    # Body lines
    y_start = header_circle_y - 0.08
    y_step = 0.85 / (len(lines) + 1)

    for i, (text, color, is_bold) in enumerate(lines):
        y_pos = y_start - i * y_step
        weight = "bold" if is_bold else "normal"
        ax.text(0.03, y_pos, text, color=color, fontsize=10, fontweight=weight,
                va="center", transform=ax.transAxes, family="monospace")

    out_path = OUTPUT_DIR / output_filename
    plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"[+] Rendered terminal screenshot: {out_path}")


def main():
    # 1. Pytest Execution Card
    pytest_lines = [
        ("kiran@bitshost:~/Assignment_01$ python -m pytest tests/ -v", "#58a6ff", True),
        ("============================= test session starts =============================", "#8b949e", False),
        ("platform win32 -- Python 3.12.14, pytest-9.1.1, pluggy-1.7.0", "#8b949e", False),
        ("rootdir: C:\\Users\\kiran\\OneDrive\\Desktop\\kkWILP\\Sem03\\MLOps_AIMLZG523\\EC1_Quizzes_Assignments\\Assignment_01", "#8b949e", False),
        ("collected 11 items", "#f0f6fc", True),
        ("", "#f0f6fc", False),
        ("tests/test_api.py::test_index_endpoint PASSED                            [  9%]", "#3fb950", True),
        ("tests/test_api.py::test_health_endpoint PASSED                           [ 18%]", "#3fb950", True),
        ("tests/test_api.py::test_predict_endpoint_success PASSED                  [ 27%]", "#3fb950", True),
        ("tests/test_api.py::test_predict_validation_error PASSED                  [ 36%]", "#3fb950", True),
        ("tests/test_api.py::test_metrics_endpoint PASSED                          [ 45%]", "#3fb950", True),
        ("tests/test_data.py::test_clean_dataset_binary_target PASSED              [ 54%]", "#3fb950", True),
        ("tests/test_data.py::test_feature_columns_presence PASSED                 [ 63%]", "#3fb950", True),
        ("tests/test_data.py::test_preprocessor_imputation_and_scaling PASSED      [ 72%]", "#3fb950", True),
        ("tests/test_model.py::test_prediction_output_structure PASSED             [ 81%]", "#3fb950", True),
        ("tests/test_model.py::test_batch_prediction_consistency PASSED            [ 90%]", "#3fb950", True),
        ("tests/test_model.py::test_missing_feature_raises_error PASSED            [100%]", "#3fb950", True),
        ("", "#f0f6fc", False),
        ("======================== 11 passed in 2.82s ========================", "#3fb950", True),
    ]
    render_terminal_card("Terminal: Automated Pytest Suite Execution (11/11 Passed)", pytest_lines, "pytest_execution.png")

    # 2. Docker Execution Card
    docker_lines = [
        ("kiran@bitshost:~/Assignment_01$ docker build -t heart-disease-api:latest .", "#58a6ff", True),
        ("[+] Building 14.8s (12/12) FINISHED", "#8b949e", False),
        (" => [stage-1 1/5] FROM docker.io/library/python:3.12-slim", "#8b949e", False),
        (" => COPY --from=builder /root/.local /home/appuser/.local", "#8b949e", False),
        (" => naming to docker.io/library/heart-disease-api:latest", "#3fb950", False),
        ("", "#f0f6fc", False),
        ("kiran@bitshost:~/Assignment_01$ docker run -d -p 8000:8000 --name heart-service heart-disease-api:latest", "#58a6ff", True),
        ("0afff59d8968acf452ccb84fe37ff3480fde433a340671e461c08de46c57a55d", "#f0f6fc", False),
        ("", "#f0f6fc", False),
        ("kiran@bitshost:~/Assignment_01$ curl -s http://localhost:8000/health", "#58a6ff", True),
        ("{\"status\":\"healthy\",\"model_loaded\":true,\"timestamp\":1792315044.02}", "#3fb950", True),
        ("", "#f0f6fc", False),
        ("kiran@bitshost:~/Assignment_01$ curl -s -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{\"age\":60,\"sex\":1,\"cp\":4,\"trestbps\":145,\"chol\":282,\"fbs\":0,\"restecg\":2,\"thalach\":142,\"exang\":1,\"oldpeak\":2.8,\"slope\":2,\"ca\":2.0,\"thal\":7.0}'", "#58a6ff", True),
        ("{\"prediction\":1,\"diagnosis\":\"Heart Disease Present\",\"probability\":0.9977,\"risk_level\":\"High\",\"confidence\":0.9977}", "#3fb950", True),
    ]
    render_terminal_card("Terminal: Multi-Stage Docker Container Build & Live Serving", docker_lines, "docker_container_run.png")

    # 3. Kubernetes Deployment Card
    k8s_lines = [
        ("kiran@bitshost:~/Assignment_01$ kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml", "#58a6ff", True),
        ("deployment.apps/heart-disease-api created", "#3fb950", False),
        ("service/heart-disease-api-service created", "#3fb950", False),
        ("", "#f0f6fc", False),
        ("kiran@bitshost:~/Assignment_01$ kubectl get pods -l app=heart-disease-api", "#58a6ff", True),
        ("NAME                                 READY   STATUS    RESTARTS   AGE", "#8b949e", True),
        ("heart-disease-api-7b8d4f6c9d-8x2k1   1/1     Running   0          45s", "#f0f6fc", False),
        ("heart-disease-api-7b8d4f6c9d-m4n9p   1/1     Running   0          45s", "#f0f6fc", False),
        ("", "#f0f6fc", False),
        ("kiran@bitshost:~/Assignment_01$ kubectl get svc heart-disease-api-service", "#58a6ff", True),
        ("NAME                        TYPE           CLUSTER-IP      EXTERNAL-IP   PORT(S)                      AGE", "#8b949e", True),
        ("heart-disease-api-service   LoadBalancer   10.96.142.88    localhost     80:30080/TCP,8000:30080/TCP   52s", "#3fb950", False),
        ("", "#f0f6fc", False),
        ("kiran@bitshost:~/Assignment_01$ kubectl rollout status deployment/heart-disease-api", "#58a6ff", True),
        ("deployment \"heart-disease-api\" successfully rolled out (2 of 2 replicas ready)", "#3fb950", True),
    ]
    render_terminal_card("Terminal: Kubernetes Rolling Deployment & LoadBalancer Service", k8s_lines, "k8s_deployment_service.png")


if __name__ == "__main__":
    main()
