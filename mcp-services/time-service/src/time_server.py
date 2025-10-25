"""
MCP Time Server
Provides time and timezone conversion capabilities through MCP protocol
"""
from datetime import datetime, timedelta, date
from enum import Enum
import json
from typing import Sequence, Optional
from zoneinfo import ZoneInfo
from tzlocal import get_localzone_name

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
from mcp.shared.exceptions import McpError
from pydantic import BaseModel


class TimeTools(str, Enum):
    GET_CURRENT_TIME = "get_current_time"
    CONVERT_TIME = "convert_time"
    GET_FUTURE_DATE = "get_future_date"
    GET_PAST_DATE = "get_past_date"
    GET_DATE_INFO = "get_date_info"


class TimeResult(BaseModel):
    timezone: str
    datetime: str
    day_of_week: str
    is_dst: bool


class TimeConversionResult(BaseModel):
    source: TimeResult
    target: TimeResult
    time_difference: str


class DateResult(BaseModel):
    date: str  # ISO format YYYY-MM-DD
    day_of_week: str
    week_number: int
    is_weekend: bool
    timezone: str
    full_datetime: str


class TimeConversionInput(BaseModel):
    source_tz: str
    time: str
    target_tz_list: list[str]


def get_local_tz(local_tz_override: str | None = None) -> ZoneInfo:
    """Get local timezone with optional override"""
    if local_tz_override:
        return ZoneInfo(local_tz_override)

    # Get local timezone from datetime.now()
    local_tzname = get_localzone_name()
    if local_tzname is not None:
        return ZoneInfo(local_tzname)
    # Default to UTC if local timezone cannot be determined
    return ZoneInfo("UTC")


def get_zoneinfo(timezone_name: str) -> ZoneInfo:
    """Get ZoneInfo with error handling"""
    try:
        return ZoneInfo(timezone_name)
    except Exception as e:
        raise McpError(f"Invalid timezone: {str(e)}")


class TimeServer:
    """Time server implementation"""
    
    def get_current_time(self, timezone_name: str) -> TimeResult:
        """Get current time in specified timezone"""
        timezone = get_zoneinfo(timezone_name)
        current_time = datetime.now(timezone)

        return TimeResult(
            timezone=timezone_name,
            datetime=current_time.isoformat(timespec="seconds"),
            day_of_week=current_time.strftime("%A"),
            is_dst=bool(current_time.dst()),
        )
    
    def get_future_date(self, days: int, timezone_name: str = "UTC") -> DateResult:
        """Get date information for a future date"""
        timezone = get_zoneinfo(timezone_name)
        current_time = datetime.now(timezone)
        future_time = current_time + timedelta(days=days)
        
        return DateResult(
            date=future_time.strftime("%Y-%m-%d"),
            day_of_week=future_time.strftime("%A"),
            week_number=future_time.isocalendar()[1],
            is_weekend=future_time.weekday() >= 5,
            timezone=timezone_name,
            full_datetime=future_time.isoformat(timespec="seconds")
        )
    
    def get_past_date(self, days: int, timezone_name: str = "UTC") -> DateResult:
        """Get date information for a past date"""
        timezone = get_zoneinfo(timezone_name)
        current_time = datetime.now(timezone)
        past_time = current_time - timedelta(days=days)
        
        return DateResult(
            date=past_time.strftime("%Y-%m-%d"),
            day_of_week=past_time.strftime("%A"),
            week_number=past_time.isocalendar()[1],
            is_weekend=past_time.weekday() >= 5,
            timezone=timezone_name,
            full_datetime=past_time.isoformat(timespec="seconds")
        )
    
    def get_date_info(self, date_str: str, timezone_name: str = "UTC") -> DateResult:
        """Get information about a specific date (YYYY-MM-DD format)"""
        timezone = get_zoneinfo(timezone_name)
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            # Add timezone info
            date_with_tz = date_obj.replace(tzinfo=timezone)
            
            return DateResult(
                date=date_with_tz.strftime("%Y-%m-%d"),
                day_of_week=date_with_tz.strftime("%A"),
                week_number=date_with_tz.isocalendar()[1],
                is_weekend=date_with_tz.weekday() >= 5,
                timezone=timezone_name,
                full_datetime=date_with_tz.isoformat(timespec="seconds")
            )
        except ValueError:
            raise ValueError("Invalid date format. Expected YYYY-MM-DD")

    def convert_time(
        self, source_tz: str, time_str: str, target_tz: str
    ) -> TimeConversionResult:
        """Convert time between timezones"""
        source_timezone = get_zoneinfo(source_tz)
        target_timezone = get_zoneinfo(target_tz)

        try:
            parsed_time = datetime.strptime(time_str, "%H:%M").time()
        except ValueError:
            raise ValueError("Invalid time format. Expected HH:MM [24-hour format]")

        now = datetime.now(source_timezone)
        source_time = datetime(
            now.year,
            now.month,
            now.day,
            parsed_time.hour,
            parsed_time.minute,
            tzinfo=source_timezone,
        )

        target_time = source_time.astimezone(target_timezone)
        source_offset = source_time.utcoffset() or timedelta()
        target_offset = target_time.utcoffset() or timedelta()
        hours_difference = (target_offset - source_offset).total_seconds() / 3600

        if hours_difference.is_integer():
            time_diff_str = f"{hours_difference:+.1f}h"
        else:
            # For fractional hours like Nepal's UTC+5:45
            time_diff_str = f"{hours_difference:+.2f}".rstrip("0").rstrip(".") + "h"

        return TimeConversionResult(
            source=TimeResult(
                timezone=source_tz,
                datetime=source_time.isoformat(timespec="seconds"),
                day_of_week=source_time.strftime("%A"),
                is_dst=bool(source_time.dst()),
            ),
            target=TimeResult(
                timezone=target_tz,
                datetime=target_time.isoformat(timespec="seconds"),
                day_of_week=target_time.strftime("%A"),
                is_dst=bool(target_time.dst()),
            ),
            time_difference=time_diff_str,
        )


async def serve(local_timezone: str | None = None) -> None:
    """Start the MCP time server"""
    server = Server("mcp-time")
    time_server = TimeServer()
    local_tz = str(get_local_tz(local_timezone))

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available time tools."""
        return [
            Tool(
                name=TimeTools.GET_CURRENT_TIME.value,
                description="Get current time and date in a specific timezone. Use this to get today's date and current time.",
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
                name=TimeTools.CONVERT_TIME.value,
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
            Tool(
                name=TimeTools.GET_FUTURE_DATE.value,
                description="Calculate a future date by adding days to today. Perfect for weather forecasts and planning. Returns the date, day of week, and full datetime.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "Number of days to add to current date (e.g., 1 for tomorrow, 7 for next week)",
                            "minimum": 0
                        },
                        "timezone": {
                            "type": "string",
                            "description": f"IANA timezone name (e.g., 'Europe/Berlin'). Use '{local_tz}' as default.",
                            "default": local_tz
                        }
                    },
                    "required": ["days"],
                },
            ),
            Tool(
                name=TimeTools.GET_PAST_DATE.value,
                description="Calculate a past date by subtracting days from today. Returns the date, day of week, and full datetime.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "Number of days to subtract from current date",
                            "minimum": 0
                        },
                        "timezone": {
                            "type": "string",
                            "description": f"IANA timezone name. Use '{local_tz}' as default.",
                            "default": local_tz
                        }
                    },
                    "required": ["days"],
                },
            ),
            Tool(
                name=TimeTools.GET_DATE_INFO.value,
                description="Get detailed information about a specific date (day of week, week number, weekend status).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "Date in YYYY-MM-DD format (e.g., '2025-12-31')",
                            "pattern": "^\\d{4}-\\d{2}-\\d{2}$"
                        },
                        "timezone": {
                            "type": "string",
                            "description": f"IANA timezone name. Use '{local_tz}' as default.",
                            "default": local_tz
                        }
                    },
                    "required": ["date"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(
        name: str, arguments: dict
    ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        """Handle tool calls for time queries."""
        try:
            match name:
                case TimeTools.GET_CURRENT_TIME.value:
                    timezone = arguments.get("timezone", local_tz)
                    result = time_server.get_current_time(timezone)

                case TimeTools.CONVERT_TIME.value:
                    if not all(
                        k in arguments
                        for k in ["source_timezone", "time", "target_timezone"]
                    ):
                        raise ValueError("Missing required arguments")

                    result = time_server.convert_time(
                        arguments["source_timezone"],
                        arguments["time"],
                        arguments["target_timezone"],
                    )
                
                case TimeTools.GET_FUTURE_DATE.value:
                    days = arguments.get("days")
                    if days is None:
                        raise ValueError("Missing required argument: days")
                    timezone = arguments.get("timezone", local_tz)
                    result = time_server.get_future_date(int(days), timezone)
                
                case TimeTools.GET_PAST_DATE.value:
                    days = arguments.get("days")
                    if days is None:
                        raise ValueError("Missing required argument: days")
                    timezone = arguments.get("timezone", local_tz)
                    result = time_server.get_past_date(int(days), timezone)
                
                case TimeTools.GET_DATE_INFO.value:
                    date_str = arguments.get("date")
                    if not date_str:
                        raise ValueError("Missing required argument: date")
                    timezone = arguments.get("timezone", local_tz)
                    result = time_server.get_date_info(date_str, timezone)
                
                case _:
                    raise ValueError(f"Unknown tool: {name}")

            return [
                TextContent(type="text", text=json.dumps(result.model_dump(), indent=2))
            ]

        except Exception as e:
            raise ValueError(f"Error processing mcp-server-time query: {str(e)}")

    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options)
