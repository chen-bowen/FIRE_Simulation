"""Root-level entry point for Streamlit Cloud deployment.

This file serves as the entry point for Streamlit Cloud. It imports and runs
the main application from the app package.
"""

from app.main import main

if __name__ == "__main__":
    main()
