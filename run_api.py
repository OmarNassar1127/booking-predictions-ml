"""
Run the FastAPI server

Usage:
    python run_api.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

import uvicorn
from src.utils.config_loader import config
from src.utils.logger import setup_logger


def main():
    """Run the API server"""

    # Setup logging
    log_file = Path("logs") / "api.log"
    log_file.parent.mkdir(exist_ok=True)
    setup_logger(log_file=str(log_file))

    # Get config
    host = config.get('api.host', '0.0.0.0')
    port = config.get('api.port', 8000)
    reload = config.get('api.reload', True)

    print("=" * 80)
    print("BATTERY PREDICTION API SERVER")
    print("=" * 80)
    print(f"\n  Starting API server at http://{host}:{port}")
    print(f"  API Documentation: http://{host}:{port}/docs")
    print(f"  Health Check: http://{host}:{port}/health")
    print("\n" + "=" * 80)

    # Run server
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    main()
