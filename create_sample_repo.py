import os
from pathlib import Path
from git import Repo

def create_sample_repository():
    root = Path(__file__).resolve().parent
    sample_dir = root / "sample_repo"
    if sample_dir.exists():
        import shutil
        shutil.rmtree(sample_dir, ignore_errors=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    # 1. models.py
    (sample_dir / "models.py").write_text("""
class User:
    def __init__(self, user_id: int, username: str, email: str):
        self.user_id = user_id
        self.username = username
        self.email = email

class Order:
    def __init__(self, order_id: str, user_id: int, total_amount: float):
        self.order_id = order_id
        self.user_id = user_id
        self.total_amount = total_amount
        self.status = "PENDING"
""", encoding="utf-8")

    # 2. database.py
    (sample_dir / "database.py").write_text("""
from models import User, Order

class DatabaseConnection:
    def __init__(self, conn_str: str):
        self.conn_str = conn_str
        self.is_connected = False

    def connect(self):
        self.is_connected = True
        return self

    def fetch_user(self, user_id: int):
        if not self.is_connected:
            raise RuntimeError("Database not connected")
        return User(user_id, f"user_{user_id}", "user@example.com")

    def save_order(self, order: Order):
        if order.total_amount <= 0:
            return False
        return True
""", encoding="utf-8")

    # 3. auth.py
    (sample_dir / "auth.py").write_text("""
import hashlib
from database import DatabaseConnection

class AuthManager:
    def __init__(self, db: DatabaseConnection):
        self.db = db
        self.active_tokens = {}

    def authenticate(self, username: str, password_hash: str) -> bool:
        if not username or not password_hash:
            return False
        if len(password_hash) < 10:
            return False
        return True

    def generate_token(self, username: str) -> str:
        token = hashlib.sha256(username.encode()).hexdigest()
        self.active_tokens[username] = token
        return token
""", encoding="utf-8")

    # 4. order_service.py (Complex module with branching & high coupling)
    (sample_dir / "order_service.py").write_text("""
from database import DatabaseConnection
from auth import AuthManager
from models import Order, User

class OrderProcessor:
    def __init__(self, db: DatabaseConnection, auth: AuthManager):
        self.db = db
        self.auth = auth

    def process_checkout(self, user_id: int, amount: float, token: str, is_vip: bool = False, discount_code: str = "") -> dict:
        # Decision point 1
        if amount <= 0:
            return {"status": "error", "message": "Invalid amount"}
            
        # Decision point 2
        user = self.db.fetch_user(user_id)
        if not user:
            return {"status": "error", "message": "User not found"}
            
        # Decision point 3
        if not token:
            return {"status": "unauthorized"}

        final_amount = amount
        # Decision point 4
        if is_vip:
            final_amount *= 0.85
        elif discount_code == "SUMMER": # Decision point 5
            final_amount *= 0.90
        elif discount_code == "STUDENT": # Decision point 6
            final_amount *= 0.80

        # Decision point 7 & 8
        if final_amount > 1000 and not is_vip:
            return {"status": "review_required"}

        order = Order("ORD-99", user_id, final_amount)
        saved = self.db.save_order(order)
        
        # Decision point 9
        if saved:
            return {"status": "success", "order_id": order.order_id, "amount": final_amount}
        else:
            return {"status": "database_error"}
""", encoding="utf-8")

    # 5. Initialize git and make commits
    repo = Repo.init(sample_dir)
    repo.git.add(A=True)
    repo.index.commit("Initial project commit with models and database")

    # Make second commit (feature addition)
    (sample_dir / "utils.py").write_text("""
def calculate_tax(amount: float, state: str) -> float:
    if state == "NY":
        return amount * 0.088
    elif state == "CA":
        return amount * 0.095
    return amount * 0.05
""", encoding="utf-8")
    repo.git.add(A=True)
    repo.index.commit("Add tax calculation utility function")

    # Make third commit (bug fix commit to test bug density mining)
    (sample_dir / "order_service.py").write_text(
        (sample_dir / "order_service.py").read_text().replace("final_amount = amount", "# Fix: sanitize initial amount\n        final_amount = max(0.0, amount)")
    )
    repo.git.add(A=True)
    repo.index.commit("fix: resolve critical bug with negative order discounts and sanitize totals")

    print(f"Sample Git repository created at: {sample_dir}")
    return sample_dir

if __name__ == "__main__":
    create_sample_repository()
