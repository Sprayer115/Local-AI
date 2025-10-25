#!/usr/bin/env python3
"""
SSE (Server-Sent Events) HTTP Server for MCP Time Service
Provides HTTP/SSE transport for the MCP protocol
"""
import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.sse import SseServerTransport
from mcp.server import Server
from mcp.types import Tool, TextContent

from src.time_server import TimeServer, get_local_tz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Store active transports
transports: Dict[str, SseServerTransport] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    logger.info("Starting MCP Time Service SSE Server...")
    yield
    logger.info("Shutting down MCP Time Service SSE Server...")
    # Clean up transports
    for transport in transports.values():
        try:
            await transport.close()
        except Exception as e:
            logger.error(f"Error closing transport: {e}")


# Create FastAPI app
app = FastAPI(
    title="MCP Time Service",
    description="Time and timezone conversion service via MCP protocol",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "MCP Time Service",
        "version": "1.0.0",
        "protocol": "Model Context Protocol (MCP)",
        "transport": "SSE (Server-Sent Events)",
        "endpoints": {
            "sse": "/sse",
            "health": "/health",
            "docs": "/docs"
        },
        "tools": [
            {
                "name": "get_current_time",
                "description": "Get current time in a specific timezone"
            },
            {
                "name": "convert_time",
                "description": "Convert time between timezones"
            }
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "service": "mcp-time"}


# ============================================================================
# REST API Endpoints for Open WebUI Integration
# ============================================================================

@app.get("/api/current-time")
async def api_current_time(timezone: str = "UTC"):
    """
    REST API: Get current time in a specific timezone
    
    Args:
        timezone: IANA timezone name (e.g., 'Europe/Berlin', 'America/New_York')
    
    Returns:
        JSON with current time information
    """
    try:
        time_server = TimeServer()
        result = time_server.get_current_time(timezone)
        return result.model_dump()
    except Exception as e:
        logger.error(f"Error in api_current_time: {e}", exc_info=True)
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )


@app.get("/api/future-date")
async def api_future_date(days: int, timezone: str = "UTC"):
    """
    REST API: Get future date information
    
    Args:
        days: Number of days to add to current date
        timezone: IANA timezone name
    
    Returns:
        JSON with future date information
    """
    try:
        time_server = TimeServer()
        result = time_server.get_future_date(days, timezone)
        return result.model_dump()
    except Exception as e:
        logger.error(f"Error in api_future_date: {e}", exc_info=True)
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )


@app.get("/api/past-date")
async def api_past_date(days: int, timezone: str = "UTC"):
    """
    REST API: Get past date information
    
    Args:
        days: Number of days to subtract from current date
        timezone: IANA timezone name
    
    Returns:
        JSON with past date information
    """
    try:
        time_server = TimeServer()
        result = time_server.get_past_date(days, timezone)
        return result.model_dump()
    except Exception as e:
        logger.error(f"Error in api_past_date: {e}", exc_info=True)
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )


@app.get("/api/date-info")
async def api_date_info(date: str, timezone: str = "UTC"):
    """
    REST API: Get information about a specific date
    
    Args:
        date: Date in YYYY-MM-DD format
        timezone: IANA timezone name
    
    Returns:
        JSON with date information
    """
    try:
        time_server = TimeServer()
        result = time_server.get_date_info(date, timezone)
        return result.model_dump()
    except Exception as e:
        logger.error(f"Error in api_date_info: {e}", exc_info=True)
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )


@app.get("/api/convert-time")
async def api_convert_time(
    source: str,
    time: str,
    target: str
):
    """
    REST API: Convert time between timezones
    
    Args:
        source: Source IANA timezone name
        time: Time in HH:MM format (24-hour)
        target: Target IANA timezone name
    
    Returns:
        JSON with conversion result
    """
    try:
        time_server = TimeServer()
        result = time_server.convert_time(source, time, target)
        return result.model_dump()
    except Exception as e:
        logger.error(f"Error in api_convert_time: {e}", exc_info=True)
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )


@app.get("/api/tools")
async def api_list_tools():
    """
    REST API: List available tools (OpenAPI format for Open WebUI)
    
    Returns:
        JSON with list of available tools and their schemas
    """
    return {
        "tools": [
            {
                "name": "get_current_time",
                "description": "Get current time and date in a specific timezone",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "IANA timezone name (e.g., 'Europe/Berlin', 'America/New_York')",
                            "default": "UTC"
                        }
                    },
                    "required": []
                },
                "endpoint": "/api/current-time",
                "method": "GET"
            },
            {
                "name": "get_future_date",
                "description": "Calculate a future date by adding days to today. Perfect for weather forecasts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "Number of days to add (e.g., 1 for tomorrow, 7 for next week)"
                        },
                        "timezone": {
                            "type": "string",
                            "description": "IANA timezone name",
                            "default": "UTC"
                        }
                    },
                    "required": ["days"]
                },
                "endpoint": "/api/future-date",
                "method": "GET"
            },
            {
                "name": "get_past_date",
                "description": "Calculate a past date by subtracting days from today",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "Number of days to subtract"
                        },
                        "timezone": {
                            "type": "string",
                            "description": "IANA timezone name",
                            "default": "UTC"
                        }
                    },
                    "required": ["days"]
                },
                "endpoint": "/api/past-date",
                "method": "GET"
            },
            {
                "name": "get_date_info",
                "description": "Get detailed information about a specific date",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "Date in YYYY-MM-DD format"
                        },
                        "timezone": {
                            "type": "string",
                            "description": "IANA timezone name",
                            "default": "UTC"
                        }
                    },
                    "required": ["date"]
                },
                "endpoint": "/api/date-info",
                "method": "GET"
            },
            {
                "name": "convert_time",
                "description": "Convert time between timezones",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Source IANA timezone name"
                        },
                        "time": {
                            "type": "string",
                            "description": "Time in HH:MM format (24-hour)",
                            "pattern": "^([0-1][0-9]|2[0-3]):[0-5][0-9]$"
                        },
                        "target": {
                            "type": "string",
                            "description": "Target IANA timezone name"
                        }
                    },
                    "required": ["source", "time", "target"]
                },
                "endpoint": "/api/convert-time",
                "method": "GET"
            }
        ]
    }


# ============================================================================
# MCP SSE Endpoints (for MCP-compatible clients)
# ============================================================================


@app.get("/sse")
async def sse_endpoint(request: Request):
    """SSE endpoint for MCP protocol communication"""
    logger.info("New SSE connection request")
    
    # Get local timezone
    local_timezone = os.getenv("LOCAL_TIMEZONE")
    local_tz = str(get_local_tz(local_timezone))
    
    # Create MCP server instance
    server = Server("mcp-time")
    time_server = TimeServer()
    
    # Register tools
    @server.list_tools()
    async def list_tools():
        """List available time tools"""
        return [
            Tool(
                name="get_current_time",
                description="Get current time in a specific timezone",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": f"IANA timezone name (e.g., 'America/New_York', 'Europe/London'). Use '{local_tz}' as local timezone if no timezone provided by the user.",
                        }
                    },
                    "required": ["timezone"],
                },
            ),
            Tool(
                name="convert_time",
                description="Convert time between timezones",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source_timezone": {
                            "type": "string",
                            "description": f"Source IANA timezone name (e.g., 'America/New_York', 'Europe/London'). Use '{local_tz}' as local timezone if no source timezone provided by the user.",
                        },
                        "time": {
                            "type": "string",
                            "description": "Time to convert in 24-hour format (HH:MM)",
                        },
                        "target_timezone": {
                            "type": "string",
                            "description": f"Target IANA timezone name (e.g., 'Asia/Tokyo', 'America/San_Francisco'). Use '{local_tz}' as local timezone if no target timezone provided by the user.",
                        },
                    },
                    "required": ["source_timezone", "time", "target_timezone"],
                },
            ),
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        """Handle tool calls"""
        try:
            logger.info(f"Tool call: {name} with arguments: {arguments}")
            
            if name == "get_current_time":
                timezone = arguments.get("timezone")
                if not timezone:
                    raise ValueError("Missing required argument: timezone")
                result = time_server.get_current_time(timezone)
            
            elif name == "convert_time":
                if not all(k in arguments for k in ["source_timezone", "time", "target_timezone"]):
                    raise ValueError("Missing required arguments")
                result = time_server.convert_time(
                    arguments["source_timezone"],
                    arguments["time"],
                    arguments["target_timezone"],
                )
            else:
                raise ValueError(f"Unknown tool: {name}")
            
            logger.info(f"Tool result: {result}")
            return [
                TextContent(
                    type="text",
                    text=json.dumps(result.model_dump(), indent=2)
                )
            ]
        
        except Exception as e:
            logger.error(f"Error in tool call: {e}", exc_info=True)
            raise ValueError(f"Error processing mcp-server-time query: {str(e)}")
    
    # Create SSE transport
    transport = SseServerTransport("/messages")
    
    # Store session ID for later cleanup
    session_id = request.query_params.get("sessionId", "default")
    transports[session_id] = transport
    
    # Return SSE response using the transport's built-in handler
    return await transport.handle_sse(
        request.scope,
        request.receive,
        request._send,
        lambda read, write: server.run(read, write, server.create_initialization_options())
    )


@app.post("/messages")
async def messages_endpoint(request: Request):
    """Messages endpoint for client-to-server communication"""
    try:
        data = await request.json()
        logger.info(f"Received message: {data}")
        
        # Find the transport for this session
        session_id = request.query_params.get("sessionId", "default")
        transport = transports.get(session_id)
        
        if transport:
            # Forward the message to the transport
            await transport.send_message(data)
            return JSONResponse({"status": "received"})
        else:
            logger.warning(f"No transport found for session {session_id}")
            return JSONResponse({"error": "Session not found"}, status_code=404)
            
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=400)


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

