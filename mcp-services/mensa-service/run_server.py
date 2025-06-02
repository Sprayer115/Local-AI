#!/usr/bin/env python3
"""
Run script for Mensa MCP Server
Compatible with MCPO (MCP OpenAPI Proxy)
"""

import os
import sys
import logging
import asyncio

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the server"""
    logger.info("🍽️ Starting Mensa MCP Server via run_server.py...")
    
    # Debug environment
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'not set')}")
    
    # Check if directories exist
    if os.path.exists('/app/src'):
        logger.info(f"Files in /app/src: {os.listdir('/app/src')}")
    else:
        logger.warning("/app/src directory not found")
    
    # Add src to Python path if not already there
    src_path = os.path.join(os.getcwd(), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        logger.info(f"Added to Python path: {src_path}")
    
    try:
        # Import and run the server
        logger.info("Importing mensa_server module...")
        from src.mensa_server import main as server_main
        logger.info("✅ Successfully imported from src.mensa_server")
        
        logger.info("Starting server main function...")
        asyncio.run(server_main())
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Available files in src/:")
        try:
            for file in os.listdir('/app/src'):
                if file.endswith('.py'):
                    logger.error(f"  - {file}")
        except Exception as ex:
            logger.error(f"Could not list src/ directory: {ex}")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Server error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()