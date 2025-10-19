#!/usr/bin/env python3
"""Entry point for the retirement planner application."""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    import streamlit.web.cli as stcli

    # Run the main app
    sys.argv = ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=localhost"]

    sys.exit(stcli.main())
