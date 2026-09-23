from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Any, Optional
from datetime import datetime


# --- Repository Schemas ---
class RepositoryCloneRequest(BaseModel):
    url: str = Field(..., description="GitHub repository URL or local repository path")
    branch: Optional[str] = Field("main", description="Branch to checkout")
    name: Optional[str] = Field(None, description="Optional custom name for repository")


class RepositoryResponse(BaseModel):
    id: int
    name: str
    url: Optional[str]
    default_branch: str
    local_path: str
    cloned_at: datetime
    status: str
    total_commits: int
    total_developers: int
    repo_age_days: float

    class Config:
        from_attributes = True


# --- Analysis Schemas ---
class AnalysisTriggerRequest(BaseModel):
    repository_id: int = Field(..., description="ID of the repository to analyze")


class AnalysisStatusResponse(BaseModel):
    id: int
    repository_id: int
    analyzed_at: datetime
    status: str = "completed"
    total_files: int
    total_loc: int
    avg_complexity: float
    avg_cohesion: float = 1.0
    avg_coupling: float = 0.0
    overall_risk_score: float
    risk_class: str
    maintainability_index: float

    class Config:
        from_attributes = True


# --- Metrics Schemas ---
class MetricItemSchema(BaseModel):
    file_path: str
    loc: int
    sloc: int
    comments: int
    cyclomatic_complexity: float
    cohesion: float
    coupling_afferent: int
    coupling_efferent: int
    coupling_instability: float
    dependency_count: int
    code_churn: int
    commit_count: int
    developer_count: int
    file_age_days: float
    commit_frequency: float
    commit_interval_avg: float
    bug_density: float
    maintainability_index: float
    risk_score: float
    risk_class: str

    class Config:
        from_attributes = True


class MetricsSummarySchema(BaseModel):
    analysis_id: int
    repository_name: str
    total_files: int
    total_loc: int
    avg_complexity: float
    avg_cohesion: float
    avg_instability: float
    maintainability_index: float
    high_risk_files_count: int
    medium_risk_files_count: int
    low_risk_files_count: int
    top_complex_files: List[MetricItemSchema]
    top_churned_files: List[MetricItemSchema]


# --- Graph Schemas ---
class GraphNodePosition(BaseModel):
    x: float
    y: float


class GraphNodeSchema(BaseModel):
    id: str
    label: str
    file_type: str
    risk_class: str
    risk_score: float
    metrics: Dict[str, Any]
    position: GraphNodePosition


class GraphEdgeSchema(BaseModel):
    id: str
    source: str
    target: str
    type: str  # "import", "function_call", "co_change"
    weight: float


class GraphDataResponse(BaseModel):
    analysis_id: int
    node_count: int
    edge_count: int
    nodes: List[GraphNodeSchema]
    edges: List[GraphEdgeSchema]
    summary: Dict[str, Any]


class PyGDataSummaryResponse(BaseModel):
    analysis_id: int
    num_nodes: int
    num_edges: int
    node_feature_dim: int
    edge_feature_dim: int
    feature_names: List[str]
    has_isolated_nodes: bool
    is_directed: bool


# --- ML & Prediction Schemas ---
class TrainRequest(BaseModel):
    epochs: int = Field(50, ge=1, le=500, description="Number of training epochs")
    learning_rate: float = Field(0.005, gt=0, lt=1, description="Learning rate")
    hidden_dim: int = Field(64, ge=16, le=512, description="Hidden dimensions of GAT layers")


class TrainResponse(BaseModel):
    status: str
    model_name: str
    epochs: int
    train_loss: float
    val_loss: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]
    training_curves: Dict[str, List[float]]


class PredictRequest(BaseModel):
    analysis_id: int = Field(..., description="Analysis ID to generate maintainability prediction for")


class RiskProbabilitySchema(BaseModel):
    low: float
    medium: float
    high: float


class RiskExplanationFactor(BaseModel):
    factor: str
    impact: str  # High, Medium, Low
    description: str
    affected_files: List[str]


class PredictionResponse(BaseModel):
    analysis_id: int
    risk_score: float
    risk_class: str
    confidence: float
    probabilities: RiskProbabilitySchema
    explanations: List[RiskExplanationFactor]
    recommendations: List[str]


# --- Dashboard & History Schemas ---
class CommitTrendItem(BaseModel):
    date: str
    commit_count: int
    churn: int


class DeveloperActivityItem(BaseModel):
    author: str
    commits: int
    impact_score: float


class DashboardResponse(BaseModel):
    repository: RepositoryResponse
    analysis: AnalysisStatusResponse
    maintainability_gauge: Dict[str, Any]
    metrics_summary: MetricsSummarySchema
    commit_trends: List[CommitTrendItem]
    developer_activity: List[DeveloperActivityItem]
    prediction: Optional[PredictionResponse]
    quick_recommendations: List[str]


class HistoryItemResponse(BaseModel):
    id: int
    repository_id: int
    repository_name: str
    repository_url: Optional[str]
    analyzed_at: datetime
    total_files: int
    total_loc: int
    overall_risk_score: float
    risk_class: str
    maintainability_index: float

    class Config:
        from_attributes = True
