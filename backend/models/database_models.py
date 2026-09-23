import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    Text,
    Boolean,
)
from sqlalchemy.orm import relationship
from backend.database.session import Base


class Repository(Base):
    """Represents a cloned or uploaded source code repository."""
    __tablename__ = "repositories"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    url = Column(String(1024), nullable=True)
    default_branch = Column(String(100), default="main")
    local_path = Column(String(1024), nullable=False)
    cloned_at = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String(50), default="cloned")  # cloned, analyzing, completed, failed
    total_commits = Column(Integer, default=0)
    total_developers = Column(Integer, default=0)
    repo_age_days = Column(Float, default=0.0)

    # Relationships
    analyses = relationship(
        "AnalysisResult", back_populates="repository", cascade="all, delete-orphan"
    )


class AnalysisResult(Base):
    """Represents an execution run of the repository analyzer and graph builder."""
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    repository_id = Column(Integer, ForeignKey("repositories.id"), nullable=False)
    analyzed_at = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String(50), default="completed")
    
    total_files = Column(Integer, default=0)
    total_loc = Column(Integer, default=0)
    avg_complexity = Column(Float, default=0.0)
    avg_cohesion = Column(Float, default=0.0)
    avg_coupling = Column(Float, default=0.0)
    overall_risk_score = Column(Float, default=0.0)  # 0 to 100 scale
    risk_class = Column(String(20), default="Low")    # Low, Medium, High
    maintainability_index = Column(Float, default=100.0)  # Standard 0-100 scale
    
    graph_node_count = Column(Integer, default=0)
    graph_edge_count = Column(Integer, default=0)
    graph_json_path = Column(String(1024), nullable=True)
    pyg_data_path = Column(String(1024), nullable=True)
    summary = Column(Text, nullable=True)

    # Relationships
    repository = relationship("Repository", back_populates="analyses")
    metrics = relationship(
        "MetricRecord", back_populates="analysis", cascade="all, delete-orphan"
    )
    predictions = relationship(
        "Prediction", back_populates="analysis", cascade="all, delete-orphan"
    )


class MetricRecord(Base):
    """Contains all 13+ extracted software & evolution metrics for an individual file node."""
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    analysis_id = Column(Integer, ForeignKey("analysis_results.id"), nullable=False)
    file_path = Column(String(1024), nullable=False, index=True)
    
    # 1. Size & Structure
    loc = Column(Integer, default=0)
    sloc = Column(Integer, default=0)
    comments = Column(Integer, default=0)
    
    # 2. Complexity & Quality
    cyclomatic_complexity = Column(Float, default=1.0)
    cohesion = Column(Float, default=1.0)  # Normalized LCOM / cohesion index
    coupling_afferent = Column(Integer, default=0)  # Ca
    coupling_efferent = Column(Integer, default=0)  # Ce
    coupling_instability = Column(Float, default=0.0)  # I = Ce / (Ca + Ce)
    dependency_count = Column(Integer, default=0)
    
    # 3. Git Evolution Metrics
    code_churn = Column(Integer, default=0)  # Additions + Deletions
    commit_count = Column(Integer, default=0)
    developer_count = Column(Integer, default=0)
    file_age_days = Column(Float, default=0.0)
    commit_frequency = Column(Float, default=0.0)  # Commits / month
    commit_interval_avg = Column(Float, default=0.0)  # Days
    bug_density = Column(Float, default=0.0)  # Bug fix commits / LOC
    
    # Composite maintainability evaluation
    maintainability_index = Column(Float, default=100.0)
    risk_score = Column(Float, default=0.0)
    risk_class = Column(String(20), default="Low")

    # Relationships
    analysis = relationship("AnalysisResult", back_populates="metrics")


class Prediction(Base):
    """Represents a maintainability risk prediction outcome and XAI explanations."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    analysis_id = Column(Integer, ForeignKey("analysis_results.id"), nullable=False)
    
    risk_score = Column(Float, nullable=False)
    risk_class = Column(String(20), nullable=False)  # Low, Medium, High
    confidence = Column(Float, nullable=False)
    probability_low = Column(Float, default=0.0)
    probability_medium = Column(Float, default=0.0)
    probability_high = Column(Float, default=0.0)
    
    explanations_json = Column(Text, nullable=False)  # Serialized feature contributions & reasons
    recommendations_json = Column(Text, nullable=False)  # Serialized actionable guidance
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    analysis = relationship("AnalysisResult", back_populates="predictions")


class TrainingHistory(Base):
    """Tracks training runs, metrics, and checkpoint metadata."""
    __tablename__ = "training_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_name = Column(String(255), default="GAT-MaintainabilityPredictor")
    epochs = Column(Integer, nullable=False)
    train_loss = Column(Float, nullable=False)
    val_loss = Column(Float, nullable=False)
    accuracy = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    
    confusion_matrix_json = Column(Text, nullable=True)
    training_curves_json = Column(Text, nullable=True)
    model_checkpoint_path = Column(String(1024), nullable=True)
    trained_at = Column(DateTime, default=datetime.datetime.utcnow)
