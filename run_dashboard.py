"""
Run the Gradio dashboard

Usage:
    python run_dashboard.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.dashboard.app import launch_dashboard
from src.utils.logger import setup_logger


def main():
    """Run the dashboard"""

    # Setup logging
    log_file = Path("logs") / "dashboard.log"
    log_file.parent.mkdir(exist_ok=True)
    setup_logger(log_file=str(log_file))

    print("=" * 80)
    print("BATTERY PREDICTION DASHBOARD")
    print("=" * 80)
    print("\n  Starting Gradio dashboard at http://localhost:7860")
    print("\n" + "=" * 80)

    launch_dashboard()


if __name__ == "__main__":
    main()
