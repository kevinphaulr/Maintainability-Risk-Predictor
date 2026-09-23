# Predicting Software Maintainability Risk Using Repository Evolution Graph Intelligence and Explainable AI

> **Final Year Engineering Capstone Project**  
> An enterprise-grade, academic software analysis platform combining Git mining, multi-language AST parsing, Graph Theory (NetworkX), Graph Neural Networks (PyTorch Geometric), and Explainable AI (XAI) to quantify, visualize, and mitigate repository maintainability risk.

---

## 📌 Executive Summary & Project Objective

Modern software repositories undergo continuous evolution characterized by frequent code changes, multi-contributor turnover, and expanding dependency topologies. Traditional static analysis tools evaluate code files in isolation, failing to capture the **temporal evolution dynamics** and **structural graph coupling** across interrelated modules.

This platform bridges this gap by:
1. **Cloning and Mining Repositories**: Extracting Git commit logs, code churn, contributor activity, and temporal co-change coupling.
2. **Deep Source Code AST Parsing**: Analyzing multi-language code structures using Python AST, Tree-Sitter (`tree-sitter-python`, `tree-sitter-javascript`), and heuristic parsers.
3. **Multi-Dimensional Metrics Extraction**: Quantifying 13+ software engineering metrics per module (McCabe Cyclomatic Complexity, Afferent/Efferent Coupling, LCOM Cohesion, Churn, Bug Density, Maintainability Index).
4. **Constructing Repository Evolution Graphs**: Modeling files as nodes (rich in extracted metric feature vectors) and semantic/temporal dependencies (imports, function calls, co-change) as multi-relational edges.
5. **PyTorch Geometric Tensor Representation**: Converting evolution graphs into PyG `Data` tensors ($x \in \mathbb{R}^{N \times 13}$, $\text{edge\_index} \in \mathbb{Z}^{2 \times E}$, $\text{edge\_attr} \in \mathbb{R}^{E \times 4}$) ready for Graph Attention Network (GAT) classification.
6. **Explainable AI (XAI)**: Decomposing risk drivers into high complexity, volatile churn, tight coupling, and contributor fragmentation with actionable refactoring advice.
7. **Comprehensive Reporting**: Multi-format exports in JSON, CSV, and academic PDF reports via ReportLab.

---

## 🏛️ System Architecture

```
                                      +--------------------------+
                                      |   Git Remote Repository  |
                                      +--------------------------+
                                                   |
                                            git clone / pull
                                                   v
+--------------------------------------------------------------------------------------------------+
|                                    BACKEND PIPELINE (FastAPI)                                    |
|                                                                                                  |
|  +--------------------+   +------------------------+   +---------------------------------------+ |
|  |     GitService     |   |     ParserService      |   |            MetricsService             | |
|  | - Commit history   |   | - Tree-sitter parsers  |   | - Cyclomatic Complexity (McCabe)      | |
|  | - Code churn       |   | - Import extraction    |   | - Coupling (Afferent Ca, Efferent Ce) | |
|  | - Contributor logs |   | - Call graph mapping   |   | - Instability Metric (I = Ce/(Ca+Ce)) | |
|  | - Temporal co-change|  | - LCOM Cohesion        |   | - Maintainability Index (MI)          | |
|  +--------------------+   +------------------------+   +---------------------------------------+ |
|             \                         |                                   /                      |
|              \                        |                                  /                       |
|               v                       v                                 v                        |
|  +---------------------------------------------------------------------------------------------+ |
|  |                              Repository Evolution Graph Builder                             | |
|  |  * Nodes = Source Files (with 13-feature normalized metric vectors)                         | |
|  |  * Edges = Imports, Function Calls, and Git Co-Change Coupling                               | |
|  |  * Layout = NetworkX 2D Spring Algorithm for interactive canvas                             | |
|  |  * PyG Conversion = PyTorch Geometric Data(x, edge_index, edge_attr, y)                     | |
|  +---------------------------------------------------------------------------------------------+ |
|                                           |                                                      |
|                                           v                                                      |
|  +---------------------------------------------------------------------------------------------+ |
|  |                           Explainable AI & Report Engine                                    | |
|  |  * Multi-Class Risk Classifier: Low (<35), Medium (35-70), High (>70)                       | |
|  |  * Factor Attribution (Complexity, Churn, Coupling, Contributor Turnover, Bug Density)     | |
|  |  * Prescriptive Refactoring Recommendations                                                  | |
|  |  * Multi-format Exporters: PDF (ReportLab), CSV (Pandas/CSV), JSON                          | |
|  +---------------------------------------------------------------------------------------------+ |
+--------------------------------------------------------------------------------------------------+
                                           |
                                           v
+--------------------------------------------------------------------------------------------------+
|                              PERSISTENCE & API INTERFACES                                        |
|  * SQLite Database: repositories, analysis_results, metrics, predictions, training_history        |
|  * REST Endpoints: /clone, /analyze, /metrics, /graph, /predict, /train, /dashboard, /history    |
+--------------------------------------------------------------------------------------------------+
```

---

## 📐 Mathematical Formulations of Software Metrics

### 1. Cyclomatic Complexity ($CC$ / McCabe)
Measures the number of linearly independent paths through a module's source code:
$$CC = E - N + 2P$$
Where $E$ is the number of edges in the control-flow graph, $N$ is the number of nodes, and $P$ is the number of connected components (or decision points $+ 1$).

### 2. Module Coupling & Instability ($I$)
- **Afferent Coupling ($C_a$)**: Number of external modules depending on this module.
- **Efferent Coupling ($C_e$)**: Number of modules this module depends upon.
- **Instability Metric ($I$)**:
  $$I = \frac{C_e}{C_a + C_e}, \quad I \in [0, 1]$$
  $I = 0$ indicates a completely stable, core module; $I = 1$ indicates a completely unstable, dependent module.

### 3. Lack of Cohesion in Methods ($LCOM$)
For a class with $M$ methods and attributes $A$:
$$\text{Cohesion} = \frac{|\{(m_i, m_j) \mid A(m_i) \cap A(m_j) \neq \emptyset\}|}{\binom{|M|}{2}}$$

### 4. Maintainability Index ($MI$)
Standard Software Engineering Institute (SEI) composite polynomial:
$$MI = \max\left(0, \min\left(100, \left(171 - 5.2 \ln(V) - 0.23 CC - 16.2 \ln(\text{LOC})\right) \times \frac{100}{171} + \text{Bonus}(\text{Comments})\right)\right)$$

### 5. Multi-Feature Node Vector ($x_i \in \mathbb{R}^{13}$)
Each node in the evolution graph is parameterized by:
$$\mathbf{x}_i = [\log(1+\text{LOC}), \log(1+\text{SLOC}), \widetilde{CC}, \text{Cohesion}, \widetilde{C_a}, \widetilde{C_e}, I, \log(1+D), \log(1+\text{Churn}), \log(1+\text{Commits}), \widetilde{\text{Devs}}, \log(1+\text{Age}), \widetilde{\text{BugDensity}}]^T$$

---

## 📁 Repository Directory Structure

```
Maintainability-Risk-Predictor/
├── backend/
│   ├── database/
│   │   ├── __init__.py
│   │   └── session.py            # SQLite engine, sessionmaker, Base, init_db
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── builder.py            # NetworkX Evolution Graph builder & 2D layout
│   │   └── pyg_converter.py      # PyG Data tensor converter and serializer
│   ├── ml/
│   │   ├── __init__.py
│   │   └── dataset_loader.py     # PyG dataset batching and loader utilities
│   ├── models/
│   │   ├── __init__.py
│   │   └── database_models.py    # SQLAlchemy models for repositories, metrics, etc.
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── analysis_router.py    # POST /analyze, GET /analysis/{id}
│   │   ├── dashboard_router.py   # GET /dashboard, GET /health
│   │   ├── graph_router.py       # GET /graph, GET /graph/{id}/pyg-summary
│   │   ├── history_router.py     # GET /history, GET /history/{id}
│   │   ├── metrics_router.py     # GET /metrics, GET /metrics/{id}/summary
│   │   ├── ml_router.py          # POST /predict, POST /train, GET /models
│   │   ├── report_router.py      # GET /reports/{id}/json, /csv, /pdf
│   │   └── repository_router.py  # POST /clone, POST /upload, GET /repositories
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── pydantic_schemas.py   # Pydantic V2 schemas for validation & responses
│   ├── services/
│   │   ├── __init__.py
│   │   ├── evolution_graph_service.py # Orchestrator for full analysis flow
│   │   ├── git_service.py        # Git cloning, commit history, churn, bug mining
│   │   ├── metrics_service.py    # Coupling, cohesion, MI, and risk calculations
│   │   ├── parser_service.py     # Tree-sitter and AST multi-language parsing
│   │   ├── report_service.py     # ReportLab PDF, CSV, and JSON generators
│   │   └── xai_service.py        # Factor attribution and prescriptive recommendations
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py             # App configurations, paths, thresholds
│   │   ├── file_utils.py         # Source scanning and path normalization
│   │   └── logger.py             # Formatted system logger
│   ├── repositories/             # Cloned local repository storage
│   ├── uploads/                  # Temporary archive uploads
│   └── main.py                   # FastAPI app entry point & CORS
├── dataset/                      # PyG (.pt) and JSON evolution graphs
├── trained_models/               # GAT model weights & checkpoints
├── reports/                      # Exported PDF, CSV, and JSON reports
├── requirements.txt              # Pinned Python dependencies
├── package.json                  # Root npm project definition
└── README.md                     # Comprehensive academic documentation
```

---

## 🚀 Installation & Setup Guide

### 1. Prerequisites
- Python 3.10+ (Verified on Python 3.10 - 3.13)
- Git 2.30+

### 2. Clone or Enter the Project
```powershell
cd "Maintainability Risk Predictor"
```

### 3. Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

---

## 🏃 Running the Application

### Start the FastAPI Backend Server
```powershell
python -m uvicorn backend.main:app --reload --port 8000
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- ReDoc Documentation: `http://localhost:8000/redoc`

---

## 📡 REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/clone` | Clones a GitHub repo or registers local directory (`{ "url": "..." }`) |
| `POST` | `/upload` | Uploads a `.zip` archive of a repository |
| `GET` | `/repositories` | Lists all registered repositories |
| `POST` | `/analyze` | Executes complete analysis pipeline for `{ "repository_id": int }` |
| `GET` | `/analysis/{id}` | Retrieves summary of an analysis run |
| `GET` | `/metrics` | Retrieves 13+ file-level metrics (latest or specified `analysis_id`) |
| `GET` | `/metrics/{id}/summary` | Retrieves aggregate metrics, MI, and top complex/churned files |
| `GET` | `/graph` | Retrieves React Flow JSON graph (nodes, edges, metrics, positions) |
| `GET` | `/graph/{id}/pyg-summary` | Inspects PyTorch Geometric tensor dimensions and metadata |
| `POST` | `/predict` | Generates risk score, confidence, XAI explanations, recommendations |
| `POST` | `/train` | Triggers Graph Attention Network (GAT) training pipeline |
| `GET` | `/dashboard` | Complete aggregated dashboard payload (gauges, trends, predictions) |
| `GET` | `/history` | Historical timeline of past repository analyses |
| `GET` | `/reports/{id}/pdf` | Downloads generated academic PDF report |
| `GET` | `/reports/{id}/csv` | Downloads metrics in CSV format |
| `GET` | `/reports/{id}/json` | Downloads complete analysis in JSON format |
| `GET` | `/health` | Health check endpoint |

---

## 🧪 Verification & Testing

Run the included verification suite to clone/analyze a repository and validate all database records, graphs, and reports:
```powershell
python -m pytest backend/tests/ -v
# Or run direct verification test:
python test_pipeline.py
```
