"""
Entry point for running the AgentQuant Streamlit application.
"""
import argparse
from pathlib import Path


def ensure_directories():
    """Ensure required directories exist"""
    # Create figures directory
    project_root = Path(__file__).parent
    figures_dir = project_root / "figures"
    figures_dir.mkdir(exist_ok=True)

    print(f"Created figures directory: {figures_dir.absolute()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgentQuant offline demo")
    parser.parse_args()
    from scripts.demo_run import main
    raise SystemExit(main())
