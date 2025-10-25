#!/usr/bin/env python3
"""
MCP Time Server - Entry point for running the time service
"""
import asyncio
import os
from src.time_server import serve


def main():
    """Main entry point for the time server"""
    # Get timezone from environment variable
    local_timezone = os.getenv("LOCAL_TIMEZONE")
    
    # Run the async server
    asyncio.run(serve(local_timezone))


if __name__ == "__main__":
    main()
