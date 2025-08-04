#!/usr/bin/env python3
"""
Main script to run the complete tree health classification pipeline.

This script executes all steps of the ML pipeline in the correct order:
1. Data loading and preprocessing
2. Feature evaluation and selection
3. Feature range calculation for UI
4. Model training and evaluation
5. Launch Gradio web interface

Usage:
    python main.py [--skip-app]
    
Options:
    --skip-app    Skip launching the Gradio app at the end
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_script(script_path, description):
    """Run a Python script and handle errors."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Script: {script_path}")
    print('='*60)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✅ SUCCESS: {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: {description} failed with exit code {e.returncode}")
        print("Pipeline execution stopped.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run the complete tree health classification pipeline")
    parser.add_argument("--skip-app", action="store_true", help="Skip launching the Gradio app")
    args = parser.parse_args()
    
    # Get project root and src directory
    project_root = Path(__file__).parent
    src_dir = project_root / "src"
    
    # Verify src directory exists
    if not src_dir.exists():
        print(f"❌ ERROR: Source directory not found: {src_dir}")
        sys.exit(1)
    
    print("🌳 Tree Health Classification Pipeline")
    print("=====================================")
    print(f"Project root: {project_root}")
    print(f"Source directory: {src_dir}")
    
    # Define pipeline steps
    pipeline_steps = [
        (src_dir / "data_loading.py", "Data Loading and Preprocessing"),
        (src_dir / "feature_evaluation.py", "Feature Evaluation and Selection"),
        (src_dir / "calculate_feature_ranges.py", "Feature Range Calculation"),
        (src_dir / "train.py", "Model Training and Evaluation"),
    ]
    
    # Execute pipeline steps
    for script_path, description in pipeline_steps:
        if not script_path.exists():
            print(f"❌ ERROR: Script not found: {script_path}")
            sys.exit(1)
        
        success = run_script(script_path, description)
        if not success:
            sys.exit(1)
    
    print(f"\n{'='*60}")
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print('='*60)
    print("\nAll steps completed:")
    print("✅ Data loaded and preprocessed")
    print("✅ Features evaluated and selected")
    print("✅ Feature ranges calculated")
    print("✅ Model trained and saved")
    
    # Launch Gradio app unless skipped
    if not args.skip_app:
        app_script = src_dir / "app.py"
        if app_script.exists():
            print(f"\n{'='*60}")
            print("🚀 LAUNCHING GRADIO WEB INTERFACE")
            print('='*60)
            print("\nStarting the web interface...")
            print("Press Ctrl+C to stop the server")
            
            try:
                subprocess.run([sys.executable, str(app_script)], check=True)
            except KeyboardInterrupt:
                print("\n\n👋 Gradio app stopped by user")
            except subprocess.CalledProcessError as e:
                print(f"❌ ERROR: Failed to launch Gradio app (exit code {e.returncode})")
        else:
            print(f"❌ WARNING: Gradio app script not found: {app_script}")
    else:
        print("\n📝 To launch the web interface manually, run:")
        print("   python src/app.py")
    
    print("\n🌳 Pipeline execution complete!")


if __name__ == "__main__":
    main()