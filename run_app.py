"""
Entry point for running the AgentQuant Streamlit application.
"""
import os
import subprocess
import sys
from pathlib import Path

def ensure_directories():
    """Ensure required directories exist"""
    # Create figures directory
    project_root = Path(__file__).parent
    figures_dir = project_root / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    print(f"Created figures directory: {figures_dir.absolute()}")

def run_streamlit():
    """Run the Streamlit application"""
    # Ensure we're in the correct directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Create required directories
    ensure_directories()
    
    # Get the path to the Streamlit app
    app_path = project_root / "src" / "app" / "streamlit_app.py"
    
    print(f"Starting Streamlit app: {app_path}")
    
    # Run Streamlit
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run", 
        str(app_path),
        "--server.port", "8501",
        "--browser.serverAddress", "localhost",
        "--server.headless", "true"
    ]
    
    try:
        subprocess.run(streamlit_cmd)
    except KeyboardInterrupt:
        print("\nStreamlit app stopped by user")
    except Exception as e:
        print(f"Error running Streamlit app: {e}")

if __name__ == "__main__":
    run_streamlit()
