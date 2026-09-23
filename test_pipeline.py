import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

from fastapi.testclient import TestClient
from backend.main import app
from backend.database.session import SessionLocal, init_db
from backend.models.database_models import Repository, AnalysisResult, MetricRecord, Prediction

def run_verification():
    print("=" * 70)
    print("STARTING END-TO-END VERIFICATION: MAINTAINABILITY RISK PREDICTOR")
    print("=" * 70)

    # 1. Initialize SQLite Database
    print("\n[Step 1] Initializing SQLite Database...")
    init_db()
    print("  -> Database tables created successfully.")

    client = TestClient(app)

    # 2. Test Health Endpoint
    print("\n[Step 2] Testing /health endpoint...")
    health_resp = client.get("/health")
    assert health_resp.status_code == 200, f"Health check failed: {health_resp.text}"
    print(f"  -> Health check OK: {health_resp.json()}")

    # 3. Test Repository Registration / Clone (Using generated sample repository)
    print("\n[Step 3] Testing /clone with sample git repository...")
    sample_path = str(root_dir / "sample_repo")
    clone_payload = {
        "url": sample_path,
        "branch": "master",
        "name": "ECommerceSample"
    }
    clone_resp = client.post("/clone", json=clone_payload)
    assert clone_resp.status_code == 201, f"Clone failed: {clone_resp.text}"
    repo_data = clone_resp.json()
    repo_id = repo_data["id"]
    print(f"  -> Repository registered successfully: ID={repo_id}, Name={repo_data['name']}")

    # 4. Test Repository Analysis Pipeline
    print(f"\n[Step 4] Testing /analyze for Repository ID={repo_id}...")
    analyze_payload = {"repository_id": repo_id}
    analyze_resp = client.post("/analyze", json=analyze_payload)
    assert analyze_resp.status_code == 200, f"Analysis failed: {analyze_resp.text}"
    analysis_data = analyze_resp.json()
    analysis_id = analysis_data["id"]
    print(f"  -> Analysis completed: Analysis ID={analysis_id}")
    print(f"     * Total Files: {analysis_data['total_files']}")
    print(f"     * Total LOC: {analysis_data['total_loc']}")
    print(f"     * Avg Cyclomatic Complexity: {analysis_data['avg_complexity']}")
    print(f"     * Avg Cohesion: {analysis_data['avg_cohesion']}")
    print(f"     * Maintainability Index: {analysis_data['maintainability_index']}")
    print(f"     * Overall Risk Score: {analysis_data['overall_risk_score']} ({analysis_data['risk_class']})")

    # 5. Test Metrics Retrieval
    print(f"\n[Step 5] Testing /metrics endpoints for Analysis ID={analysis_id}...")
    metrics_resp = client.get(f"/metrics/{analysis_id}")
    assert metrics_resp.status_code == 200, f"Metrics failed: {metrics_resp.text}"
    metrics_list = metrics_resp.json()
    print(f"  -> Retrieved {len(metrics_list)} file metric records.")
    sample = metrics_list[0]
    print(f"     * Sample File: {sample['file_path']}")
    print(f"       LOC: {sample['loc']}, CC: {sample['cyclomatic_complexity']}, "
          f"Ca: {sample['coupling_afferent']}, Ce: {sample['coupling_efferent']}, "
          f"LCOM: {sample['cohesion']}, Churn: {sample['code_churn']}, Risk: {sample['risk_score']}")

    summary_resp = client.get(f"/metrics/{analysis_id}/summary")
    assert summary_resp.status_code == 200
    print("  -> Metrics summary endpoint OK.")

    # 6. Test Evolution Graph Retrieval
    print(f"\n[Step 6] Testing /graph endpoints for Analysis ID={analysis_id}...")
    graph_resp = client.get(f"/graph/{analysis_id}")
    assert graph_resp.status_code == 200, f"Graph failed: {graph_resp.text}"
    graph_data = graph_resp.json()
    print(f"  -> React Flow Graph Nodes: {graph_data['node_count']}, Edges: {graph_data['edge_count']}")
    assert graph_data['node_count'] > 0, "Graph should have nodes"

    # 7. Test PyTorch Geometric Tensor Structure
    print(f"\n[Step 7] Testing /graph/{analysis_id}/pyg-summary...")
    pyg_resp = client.get(f"/graph/{analysis_id}/pyg-summary")
    assert pyg_resp.status_code == 200, f"PyG summary failed: {pyg_resp.text}"
    pyg_data = pyg_resp.json()
    print(f"  -> PyG Data Tensors Verified:")
    print(f"     * Nodes (N): {pyg_data['num_nodes']}")
    print(f"     * Edges (E): {pyg_data['num_edges']}")
    print(f"     * Node Feature Dim: {pyg_data['node_feature_dim']} features ({pyg_data['feature_names'][:4]}...)")
    print(f"     * Edge Feature Dim: {pyg_data['edge_feature_dim']}")

    # 8. Test Explainable AI & Predictions
    print(f"\n[Step 8] Testing /predict endpoint...")
    pred_payload = {"analysis_id": analysis_id}
    pred_resp = client.post("/predict", json=pred_payload)
    assert pred_resp.status_code == 200, f"Predict failed: {pred_resp.text}"
    pred_data = pred_resp.json()
    print(f"  -> Prediction Result:")
    print(f"     * Risk Score: {pred_data['risk_score']} ({pred_data['risk_class']})")
    print(f"     * Confidence: {pred_data['confidence']}")
    print(f"     * Probabilities: {pred_data['probabilities']}")
    print(f"     * XAI Factors ({len(pred_data['explanations'])}):")
    for exp in pred_data["explanations"]:
        print(f"       - [{exp['impact']}] {exp['factor']}: {exp['description']}")
    print(f"     * Recommendations ({len(pred_data['recommendations'])}):")
    for rec in pred_data["recommendations"][:2]:
        print(f"       - {rec}")

    # 9. Test Training Pipeline Orchestration Endpoint
    print(f"\n[Step 9] Testing /train endpoint...")
    train_payload = {"epochs": 20, "learning_rate": 0.005, "hidden_dim": 64}
    train_resp = client.post("/train", json=train_payload)
    assert train_resp.status_code == 200, f"Train failed: {train_resp.text}"
    train_data = train_resp.json()
    print(f"  -> GAT Training Orchestration OK: Model={train_data['model_name']}, Accuracy={train_data['accuracy']}, F1={train_data['f1_score']}")

    # 10. Test Dashboard Aggregation Endpoint
    print(f"\n[Step 10] Testing /dashboard endpoint...")
    dash_resp = client.get(f"/dashboard/{analysis_id}")
    assert dash_resp.status_code == 200, f"Dashboard failed: {dash_resp.text}"
    dash_data = dash_resp.json()
    print(f"  -> Dashboard payload validated:")
    print(f"     * Gauge: {dash_data['maintainability_gauge']}")
    print(f"     * Commit Trends: {len(dash_data['commit_trends'])} items")
    print(f"     * Developer Activity: {len(dash_data['developer_activity'])} items")

    # 11. Test History Endpoint
    print(f"\n[Step 11] Testing /history endpoint...")
    hist_resp = client.get("/history")
    assert hist_resp.status_code == 200, f"History failed: {hist_resp.text}"
    hist_list = hist_resp.json()
    print(f"  -> Analysis History contains {len(hist_list)} records.")

    # 12. Test Report Exports (JSON, CSV, PDF)
    print(f"\n[Step 12] Testing Report Exports (/reports/{analysis_id}/...)")
    
    # JSON
    json_rep_resp = client.get(f"/reports/{analysis_id}/json")
    assert json_rep_resp.status_code == 200
    print("  -> JSON Report export: SUCCESS")

    # CSV
    csv_rep_resp = client.get(f"/reports/{analysis_id}/csv")
    assert csv_rep_resp.status_code == 200
    assert "file_path,loc,sloc" in csv_rep_resp.text
    print("  -> CSV Report export: SUCCESS")

    # PDF
    pdf_rep_resp = client.get(f"/reports/{analysis_id}/pdf")
    assert pdf_rep_resp.status_code == 200
    assert pdf_rep_resp.content.startswith(b"%PDF")
    print(f"  -> PDF Report export: SUCCESS (Size: {len(pdf_rep_resp.content):,} bytes)")

    print("\n" + "=" * 70)
    print("ALL 12 END-TO-END VERIFICATION CHECKS PASSED WITH ZERO ERRORS!")
    print("=" * 70)

if __name__ == "__main__":
    run_verification()
