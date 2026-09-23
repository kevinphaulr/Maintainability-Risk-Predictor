import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional

import tree_sitter_python as tsp
import tree_sitter_javascript as tsj
from tree_sitter import Language, Parser

from backend.utils.logger import get_logger

logger = get_logger("parser_service")

# Initialize Tree-Sitter Parsers
try:
    PY_LANGUAGE = Language(tsp.language())
    py_parser = Parser(PY_LANGUAGE)
except Exception as e:
    logger.warning(f"Failed to initialize tree-sitter Python: {e}")
    py_parser = None

try:
    JS_LANGUAGE = Language(tsj.language())
    js_parser = Parser(JS_LANGUAGE)
except Exception as e:
    logger.warning(f"Failed to initialize tree-sitter JavaScript: {e}")
    js_parser = None


class ParsedFileInfo:
    """Structured result of parsing a source code file."""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.total_lines: int = 0
        self.sloc: int = 0
        self.comment_lines: int = 0
        self.blank_lines: int = 0
        self.cyclomatic_complexity: float = 1.0
        self.imports: List[str] = []
        self.internal_imported_files: Set[str] = set()
        self.function_defs: List[str] = []
        self.function_calls: List[str] = []
        self.classes: List[Dict[str, Any]] = []
        self.cohesion: float = 1.0  # Cohesion index (1.0 = highly cohesive, 0.0 = completely uncohesive)
        self.dependency_count: int = 0


class ParserService:
    """
    Multi-language AST and source code analyzer using Tree-sitter and Python AST.
    Extracts imports, function calls, classes, LOC, and decision points.
    """

    @staticmethod
    def parse_file(file_path: Path, all_repo_rel_paths: Set[str], repo_root: Path) -> ParsedFileInfo:
        """Parses an individual source code file and extracts all structural AST features."""
        rel_path = file_path.resolve().relative_to(repo_root.resolve()).as_posix()
        info = ParsedFileInfo(rel_path)

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return info

        lines = content.splitlines()
        info.total_lines = len(lines)

        # Count blank and basic comment lines
        for line in lines:
            stripped = line.strip()
            if not stripped:
                info.blank_lines += 1
            elif stripped.startswith(("#", "//", "/*", "*")):
                info.comment_lines += 1
            else:
                info.sloc += 1

        suffix = file_path.suffix.lower()

        if suffix == ".py":
            ParserService._parse_python(content, info, all_repo_rel_paths, rel_path)
        elif suffix in {".js", ".jsx", ".ts", ".tsx"}:
            ParserService._parse_javascript(content, info, all_repo_rel_paths, rel_path)
        else:
            ParserService._parse_generic(content, info, all_repo_rel_paths, rel_path)

        info.dependency_count = len(info.imports)
        return info

    @staticmethod
    def _parse_python(content: str, info: ParsedFileInfo, all_repo_files: Set[str], current_rel_path: str) -> None:
        """Deep AST parsing for Python using Python's native AST and Tree-Sitter."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            # Fallback to tree-sitter or regex if syntax error exists (e.g. incomplete code)
            ParserService._parse_python_treesitter(content, info, all_repo_files, current_rel_path)
            return

        branches = 1  # Base cyclomatic complexity
        classes_data: List[Dict[str, Any]] = []

        class ASTVisitor(ast.NodeVisitor):
            def __init__(self):
                self.branches = 1
                self.imports: List[str] = []
                self.calls: List[str] = []
                self.functions: List[str] = []

            def visit_Import(self, node: ast.Import):
                for alias in node.names:
                    self.imports.append(alias.name)
                self.generic_visit(node)

            def visit_ImportFrom(self, node: ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    full_name = f"{module}.{alias.name}" if module else alias.name
                    self.imports.append(full_name)
                self.generic_visit(node)

            def visit_FunctionDef(self, node: ast.FunctionDef):
                self.functions.append(node.name)
                self.branches += 1
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                self.functions.append(node.name)
                self.branches += 1
                self.generic_visit(node)

            def visit_If(self, node: ast.If):
                self.branches += 1
                self.generic_visit(node)

            def visit_For(self, node: ast.For):
                self.branches += 1
                self.generic_visit(node)

            def visit_While(self, node: ast.While):
                self.branches += 1
                self.generic_visit(node)

            def visit_ExceptHandler(self, node: ast.ExceptHandler):
                self.branches += 1
                self.generic_visit(node)

            def visit_BoolOp(self, node: ast.BoolOp):
                self.branches += len(node.values) - 1
                self.generic_visit(node)

            def visit_Call(self, node: ast.Call):
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                if func_name:
                    self.calls.append(func_name)
                self.generic_visit(node)

            def visit_ClassDef(self, node: ast.ClassDef):
                methods: List[str] = []
                method_attrs: Dict[str, Set[str]] = {}
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append(item.name)
                        # Collect accessed attributes on 'self'
                        attrs = set()
                        for subnode in ast.walk(item):
                            if (
                                isinstance(subnode, ast.Attribute)
                                and isinstance(subnode.value, ast.Name)
                                and subnode.value.id == "self"
                            ):
                                attrs.add(subnode.attr)
                        method_attrs[item.name] = attrs

                classes_data.append({
                    "name": node.name,
                    "methods": methods,
                    "method_attrs": method_attrs,
                })
                self.generic_visit(node)

        visitor = ASTVisitor()
        visitor.visit(tree)

        info.imports = visitor.imports
        info.function_defs = visitor.functions
        info.function_calls = visitor.calls
        info.cyclomatic_complexity = float(visitor.branches)
        info.classes = classes_data

        # Compute Cohesion (LCOM - Lack of Cohesion in Methods)
        info.cohesion = ParserService._calculate_lcom_cohesion(classes_data)

        # Resolve internal imports to other repository files
        info.internal_imported_files = ParserService._resolve_internal_imports(
            info.imports, all_repo_files, current_rel_path
        )

    @staticmethod
    def _parse_python_treesitter(content: str, info: ParsedFileInfo, all_repo_files: Set[str], current_rel_path: str) -> None:
        """Tree-sitter fallback parser for Python."""
        if not py_parser:
            ParserService._parse_generic(content, info, all_repo_files, current_rel_path)
            return

        tree = py_parser.parse(bytes(content, "utf-8"))
        root = tree.root_node
        
        # Traverse tree nodes
        branches = 1
        imports: List[str] = []
        functions: List[str] = []
        calls: List[str] = []

        def traverse(node):
            nonlocal branches
            ntype = node.type
            if ntype in {"if_statement", "for_statement", "while_statement", "except_clause", "conditional_expression"}:
                branches += 1
            elif ntype in {"function_definition"}:
                branches += 1
                for child in node.children:
                    if child.type == "identifier":
                        functions.append(content[child.start_byte:child.end_byte])
                        break
            elif ntype == "import_statement":
                imports.append(content[node.start_byte:node.end_byte])
            elif ntype == "import_from_statement":
                imports.append(content[node.start_byte:node.end_byte])
            elif ntype == "call":
                for child in node.children:
                    if child.type in {"identifier", "attribute"}:
                        calls.append(content[child.start_byte:child.end_byte])
                        break
            for child in node.children:
                traverse(child)

        traverse(root)
        info.imports = imports
        info.function_defs = functions
        info.function_calls = calls
        info.cyclomatic_complexity = float(branches)
        info.internal_imported_files = ParserService._resolve_internal_imports(imports, all_repo_files, current_rel_path)

    @staticmethod
    def _parse_javascript(content: str, info: ParsedFileInfo, all_repo_files: Set[str], current_rel_path: str) -> None:
        """Parses JavaScript / TypeScript using Tree-Sitter and regex."""
        branches = 1
        imports: List[str] = []
        functions: List[str] = []
        calls: List[str] = []

        if js_parser:
            tree = js_parser.parse(bytes(content, "utf-8"))
            root = tree.root_node

            def traverse(node):
                nonlocal branches
                ntype = node.type
                if ntype in {"if_statement", "for_statement", "while_statement", "catch_clause", "ternary_expression"}:
                    branches += 1
                elif ntype in {"function_declaration", "method_definition", "arrow_function"}:
                    branches += 1
                    for child in node.children:
                        if child.type in {"identifier", "property_identifier"}:
                            functions.append(content[child.start_byte:child.end_byte])
                            break
                elif ntype == "import_statement":
                    imports.append(content[node.start_byte:node.end_byte])
                elif ntype == "call_expression":
                    for child in node.children:
                        if child.type in {"identifier", "member_expression"}:
                            calls.append(content[child.start_byte:child.end_byte])
                            break
                for child in node.children:
                    traverse(child)

            traverse(root)
        else:
            # Regex fallback
            import_matches = re.findall(r"import\s+.*?from\s+['\"](.*?)['\"]", content)
            require_matches = re.findall(r"require\(['\"](.*?)['\"]\)", content)
            imports = import_matches + require_matches
            functions = re.findall(r"function\s+([a-zA-Z0-9_]+)", content)
            branches += len(re.findall(r"\b(if|for|while|catch)\b", content))

        info.imports = imports
        info.function_defs = functions
        info.function_calls = calls
        info.cyclomatic_complexity = float(branches)
        info.internal_imported_files = ParserService._resolve_internal_imports(imports, all_repo_files, current_rel_path)

    @staticmethod
    def _parse_generic(content: str, info: ParsedFileInfo, all_repo_files: Set[str], current_rel_path: str) -> None:
        """Generic heuristic parser for Java, C++, Go, etc."""
        # Find import patterns
        import_patterns = re.findall(r"^\s*(?:import|include|require)\s+[\"<]?([^\"<>\n;]+)[\">]?", content, re.MULTILINE)
        info.imports = import_patterns
        
        # Decision points
        branches = 1 + len(re.findall(r"\b(if|else if|for|while|switch|case|catch)\b", content))
        info.cyclomatic_complexity = float(branches)
        
        # Function pattern heuristic
        func_patterns = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*\{", content)
        info.function_defs = func_patterns
        
        info.internal_imported_files = ParserService._resolve_internal_imports(import_patterns, all_repo_files, current_rel_path)

    @staticmethod
    def _resolve_internal_imports(import_strings: List[str], all_repo_files: Set[str], current_file: str) -> Set[str]:
        """
        Maps imported module names / relative path imports to known repository files.
        E.g. 'backend.utils.config' -> 'backend/utils/config.py'
        './database/session' -> 'backend/database/session.py'
        """
        internal_matches: Set[str] = set()
        curr_dir = Path(current_file).parent

        for imp in import_strings:
            # Clean string
            cleaned = imp.strip().replace(";", "").replace('"', "").replace("'", "")
            # Extract module if it's an import statement string like "from foo import bar"
            from_match = re.search(r"from\s+([a-zA-Z0-9_\.]+)", cleaned)
            if from_match:
                cleaned = from_match.group(1)

            # Check relative paths
            if cleaned.startswith("."):
                candidate = (curr_dir / cleaned).resolve()
                cand_str = candidate.as_posix()
                for rf in all_repo_files:
                    if rf.startswith(cand_str) or Path(rf).stem == Path(cleaned).name:
                        internal_matches.add(rf)
                continue

            # Check dot-separated python package imports
            path_converted = cleaned.replace(".", "/")
            for rf in all_repo_files:
                rf_no_ext = Path(rf).with_suffix("").as_posix()
                if rf_no_ext.endswith(path_converted) or Path(rf).stem == Path(path_converted).name:
                    if rf != current_file:
                        internal_matches.add(rf)

        return internal_matches

    @staticmethod
    def _calculate_lcom_cohesion(classes_data: List[Dict[str, Any]]) -> float:
        """
        Calculates normalized cohesion index based on Lack of Cohesion in Methods (LCOM).
        Cohesion ranges from 0.0 (low cohesion) to 1.0 (high cohesion).
        """
        if not classes_data:
            return 1.0  # Functional files without classes default to cohesive

        cohesion_scores: List[float] = []
        for cls in classes_data:
            methods = cls["methods"]
            method_attrs = cls["method_attrs"]
            num_methods = len(methods)

            if num_methods <= 1:
                cohesion_scores.append(1.0)
                continue

            # Compare all pairs of methods: do they share attributes?
            shared_pairs = 0
            distinct_pairs = 0
            for i in range(num_methods):
                for j in range(i + 1, num_methods):
                    attrs_i = method_attrs.get(methods[i], set())
                    attrs_j = method_attrs.get(methods[j], set())
                    if attrs_i.intersection(attrs_j):
                        shared_pairs += 1
                    else:
                        distinct_pairs += 1

            total_pairs = shared_pairs + distinct_pairs
            if total_pairs == 0:
                cohesion_scores.append(1.0)
            else:
                # Cohesion ratio = shared pairs / total pairs
                score = shared_pairs / total_pairs
                cohesion_scores.append(max(0.0, min(1.0, score)))

        return round(sum(cohesion_scores) / len(cohesion_scores), 2)
