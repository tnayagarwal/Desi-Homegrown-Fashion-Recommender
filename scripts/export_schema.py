"""
Export FastAPI OpenAPI schema to a static JSON file.

Usage:
    python scripts/export_schema.py [output.json]
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.api.server import app

def export_schema(output_path: str = "openapi.json") -> None:
    schema = app.openapi()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    print(f"Schema exported to '{output_path}' "
          f"({len(schema.get('paths', {}))} endpoints, "
          f"{len(schema.get('components', {}).get('schemas', {}))} schemas).")

if __name__ == "__main__":
    export_schema(sys.argv[1] if len(sys.argv) > 1 else "openapi.json")
