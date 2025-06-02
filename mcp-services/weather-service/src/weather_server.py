#!/usr/bin/env python3
"""
MCP Weather Server - Liefert Wetterdaten für verschiedene Städte
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# OpenWeatherMap API Key aus Umgebungsvariable
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", None)
if OPENWEATHER_API_KEY:
    logger.info("OpenWeatherMap API Key gefunden in Umgebungsvariablen")
else:
    logger.info("Kein API Key in Umgebungsvariablen gefunden. Verwende Mock-Daten oder gib den Key manuell ein.")

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Server initialisieren
app = Server("weather-server")

# Mock-Daten für Demo (falls keine API verfügbar)
MOCK_WEATHER_DATA = {
    "berlin": {
        "city": "Berlin",
        "country": "DE",
        "temperature": 15.2,
        "humidity": 68,
        "description": "Bewölkt",
        "wind_speed": 12.5,
        "pressure": 1013
    },
    "munich": {
        "city": "München", 
        "country": "DE",
        "temperature": 12.8,
        "humidity": 75,
        "description": "Leichter Regen",
        "wind_speed": 8.3,
        "pressure": 1008
    },
    "hamburg": {
        "city": "Hamburg",
        "country": "DE", 
        "temperature": 18.1,
        "humidity": 82,
        "description": "Sonnig",
        "wind_speed": 15.2,
        "pressure": 1018
    }
}

# Pydantic Models für strukturierte Daten
class WeatherData(BaseModel):
    city: str
    country: str
    temperature: float
    humidity: int
    description: str
    wind_speed: float
    pressure: int

class WeatherRequest(BaseModel):
    city: str
    use_real_api: bool = False
    api_key: Optional[str] = None

@app.list_resources()
async def list_resources() -> List[Resource]:
    """Liste verfügbare Wetter-Ressourcen"""
    return [
        Resource(
            uri="weather://cities",
            name="Verfügbare Städte",
            description="Liste aller verfügbaren Städte für Wetterabfragen",
            mimeType="application/json"
        ),
        Resource(
            uri="weather://current/berlin",
            name="Berlin Wetter",
            description="Aktuelles Wetter in Berlin",
            mimeType="application/json"
        ),
        Resource(
            uri="weather://current/munich", 
            name="München Wetter",
            description="Aktuelles Wetter in München",
            mimeType="application/json"
        ),
        Resource(
            uri="weather://current/hamburg",
            name="Hamburg Wetter", 
            description="Aktuelles Wetter in Hamburg",
            mimeType="application/json"
        )
    ]

@app.read_resource()
async def read_resource(uri: str) -> str:
    """Lese spezifische Wetter-Ressource"""
    if uri == "weather://cities":
        cities = list(MOCK_WEATHER_DATA.keys())
        return json.dumps({
            "available_cities": cities,
            "total": len(cities)
        }, indent=2)
    
    if uri.startswith("weather://current/"):
        city = uri.split("/")[-1].lower()
        if city in MOCK_WEATHER_DATA:
            return json.dumps(MOCK_WEATHER_DATA[city], indent=2)
        else:
            return json.dumps({"error": f"Stadt '{city}' nicht verfügbar"})
    
    return json.dumps({"error": "Ressource nicht gefunden"})

@app.list_tools()
async def list_tools() -> List[Tool]:
    """Liste verfügbare Weather Tools"""
    return [
        Tool(
            name="get_weather",
            description="Ruft aktuelle Wetterdaten für eine Stadt ab",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string", 
                        "description": "Name der Stadt (z.B. 'berlin', 'munich', 'hamburg')"
                    },
                    "use_real_api": {
                        "type": "boolean",
                        "description": "Verwende echte API",
                        "default": False
                    },
                    "api_key": {
                        "type": "string",
                        "description": "OpenWeatherMap API Key (optional)",
                        "default": None
                    }
                },
                "required": ["city"]
            }
        ),
        Tool(
            name="get_hourly_forecast",
            description="Ruft stündliche Wettervorhersage für bis zu 4 Tage ab (echte Daten via API)", 
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Name der Stadt"
                    },
                    "hours": {
                        "type": "integer", 
                        "description": "Anzahl Stunden (1-96)",
                        "default": 24,
                        "minimum": 1,
                        "maximum": 96
                    },
                    "use_real_api": {
                        "type": "boolean",
                        "description": "Verwende echte API",
                        "default": True
                    },
                    "api_key": {
                        "type": "string",
                        "description": "OpenWeatherMap API Key (optional)",
                        "default": None
                    }
                },
                "required": ["city"]
            }
        ),
        Tool(
            name="compare_weather",
            description="Vergleicht Wetter zwischen zwei Städten",
            inputSchema={
                "type": "object", 
                "properties": {
                    "city1": {
                        "type": "string",
                        "description": "Erste Stadt"
                    },
                    "city2": {
                        "type": "string", 
                        "description": "Zweite Stadt"
                    },
                    "use_real_api": {
                        "type": "boolean",
                        "description": "Verwende echte API (benötigt API Key)",
                        "default": False
                    },
                    "api_key": {
                        "type": "string",
                        "description": "OpenWeatherMap API Key (optional)",
                        "default": None
                    }
                },
                "required": ["city1", "city2"]
            }
        )
    ]

async def fetch_real_weather(city: str, api_key: str) -> Dict[str, Any]:
    """Ruft echte Wetterdaten von OpenWeatherMap ab mit Geocoding-Unterstützung"""
    
    # Schritt 1: Geocoding API verwenden, um Koordinaten zu ermitteln
    geocode_url = "http://api.openweathermap.org/geo/1.0/direct"
    geocode_params = {
        "q": city,
        "limit": 1,
        "appid": api_key
    }
    
    try:
        async with httpx.AsyncClient() as client:
            # Erst Geocoding-Anfrage durchführen, um Koordinaten zu bekommen
            logger.info(f"Führe Geocoding für Stadt '{city}' durch...")
            geocode_response = await client.get(geocode_url, params=geocode_params)
            geocode_response.raise_for_status()
            geocode_data = geocode_response.json()
            
            if not geocode_data:
                logger.warning(f"Keine Geocoding-Ergebnisse für '{city}'")
                return {"error": f"Stadt '{city}' konnte nicht gefunden werden"}
            
            # Koordinaten aus der Geocoding-Antwort extrahieren
            location = geocode_data[0]
            lat = location.get("lat")
            lon = location.get("lon")
            city_name = location.get("name")
            country = location.get("country")
            
            logger.info(f"Geocoding erfolgreich: {city_name}, {country} ({lat}, {lon})")
            
            # Schritt 2: Wetterdaten mit Koordinaten abrufen
            weather_url = "https://api.openweathermap.org/data/2.5/weather"
            weather_params = {
                "lat": lat,
                "lon": lon,
                "appid": api_key,
                "units": "metric",
                "lang": "de"
            }
            
            logger.info(f"Rufe Wetterdaten ab für Koordinaten ({lat}, {lon})...")
            weather_response = await client.get(weather_url, params=weather_params)
            weather_response.raise_for_status()
            weather_data = weather_response.json()
            
            return {
                "city": city_name or weather_data.get("name", city),
                "country": country or weather_data.get("sys", {}).get("country", ""),
                "temperature": weather_data.get("main", {}).get("temp", 0),
                "humidity": weather_data.get("main", {}).get("humidity", 0),
                "description": weather_data.get("weather", [{}])[0].get("description", "Keine Beschreibung"),
                "wind_speed": weather_data.get("wind", {}).get("speed", 0),
                "pressure": weather_data.get("main", {}).get("pressure", 0)
            }
    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Wetterdaten: {e}")
        return {"error": f"API-Fehler: {str(e)}"}

async def fetch_hourly_forecast(city: str, api_key: str, hours: int = 24) -> Dict[str, Any]:
    """Ruft stündliche Wettervorhersagedaten von OpenWeatherMap ab"""
    
    # Schritt 1: Geocoding API verwenden, um Koordinaten zu ermitteln
    geocode_url = "http://api.openweathermap.org/geo/1.0/direct"
    geocode_params = {
        "q": city,
        "limit": 1,
        "appid": api_key
    }
    
    try:
        async with httpx.AsyncClient() as client:
            # Erst Geocoding-Anfrage durchführen
            logger.info(f"Führe Geocoding für Stadt '{city}' durch...")
            geocode_response = await client.get(geocode_url, params=geocode_params)
            geocode_response.raise_for_status()
            geocode_data = geocode_response.json()
            
            if not geocode_data:
                logger.warning(f"Keine Geocoding-Ergebnisse für '{city}'")
                return {"error": f"Stadt '{city}' konnte nicht gefunden werden"}
            
            # Koordinaten aus der Geocoding-Antwort extrahieren
            location = geocode_data[0]
            lat = location.get("lat")
            lon = location.get("lon")
            city_name = location.get("name")
            country = location.get("country")
            
            # Schritt 2: Stündliche Vorhersage abrufen
            # Verwende die pro-API URL mit Fallback auf die normale API
            forecast_url = "https://api.openweathermap.org/data/2.5/forecast" 
            forecast_params = {
                "lat": lat,
                "lon": lon,
                "appid": api_key,
                "units": "metric",
                "lang": "de",
                "cnt": min(hours, 96)  # Maximal 96 Stunden (4 Tage) verfügbar
            }
            
            logger.info(f"Rufe stündliche Vorhersage ab für {city_name} ({lat}, {lon})...")
            forecast_response = await client.get(forecast_url, params=forecast_params)
            forecast_response.raise_for_status()
            forecast_data = forecast_response.json()
            
            # Stündliche Daten extrahieren und formatieren
            hourly_forecast = []
            for item in forecast_data.get("list", [])[:hours]:
                timestamp = item.get("dt")
                temp = item.get("main", {}).get("temp")
                description = item.get("weather", [{}])[0].get("description", "")
                humidity = item.get("main", {}).get("humidity")
                wind_speed = item.get("wind", {}).get("speed")
                
                # Wir formatieren das Datum aus Unix-Timestamp
                from datetime import datetime
                dt_obj = datetime.fromtimestamp(timestamp)
                time_str = dt_obj.strftime("%d.%m.%Y %H:%M")
                
                hourly_forecast.append({
                    "time": time_str,
                    "temperature": temp,
                    "description": description,
                    "humidity": humidity,
                    "wind_speed": wind_speed
                })
            
            return {
                "city": city_name,
                "country": country,
                "forecast": hourly_forecast
            }
            
    except Exception as e:
        logger.error(f"Fehler beim Abrufen der stündlichen Vorhersage: {e}")
        return {"error": f"API-Fehler: {str(e)}"}

@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Führt Weather Tool-Aufrufe aus"""
    
    if name == "get_weather":
        city = arguments.get("city", "").lower()
        use_real_api = arguments.get("use_real_api", False)
        # Verwende den API-Key aus Argumenten oder aus Umgebungsvariablen
        api_key = arguments.get("api_key") or os.getenv("OPENWEATHER_API_KEY")
        
        if use_real_api and api_key:
            weather_data = await fetch_real_weather(city, api_key)
        else:
            # Wenn use_real_api=True aber kein API-Key vorhanden, warnen
            if use_real_api and not api_key:
                return [TextContent(
                    type="text",
                    text="⚠️ 'use_real_api' ist aktiviert, aber kein API-Key wurde angegeben oder in Umgebungsvariablen gefunden."
                )]
            
            weather_data = MOCK_WEATHER_DATA.get(city)
            if not weather_data:
                available_cities = ", ".join(MOCK_WEATHER_DATA.keys())
                return [TextContent(
                    type="text",
                    text=f"❌ Stadt '{city}' nicht in Mock-Daten verfügbar.\n"
                         f"Verfügbare Städte: {available_cities}\n"
                         f"Oder nutze use_real_api=true mit API Key."
                )]
        
        if "error" in weather_data:
            return [TextContent(type="text", text=f"❌ {weather_data['error']}")]
        
        # Formatierte Ausgabe
        result = f"""🌤️ **Wetter in {weather_data['city']}, {weather_data['country']}**

🌡️ **Temperatur:** {weather_data['temperature']}°C
💧 **Luftfeuchtigkeit:** {weather_data['humidity']}%
📖 **Beschreibung:** {weather_data['description']}
💨 **Windgeschwindigkeit:** {weather_data['wind_speed']} km/h
📊 **Luftdruck:** {weather_data['pressure']} hPa"""

        return [TextContent(type="text", text=result)]
    
    elif name == "get_weather_forecast":
        city = arguments.get("city", "").lower()
        days = arguments.get("days", 3)
        
        # Mock Forecast-Daten generieren
        base_weather = MOCK_WEATHER_DATA.get(city)
        if not base_weather:
            return [TextContent(
                type="text", 
                text=f"❌ Stadt '{city}' nicht verfügbar für Vorhersage"
            )]
        
        forecast_text = f"📅 **{days}-Tage Wettervorhersage für {base_weather['city']}**\n\n"
        
        for day in range(1, days + 1):
            temp_variation = (-2 + (day * 0.5)) if day % 2 else (1 + (day * 0.3))
            temp = round(base_weather['temperature'] + temp_variation, 1)
            
            forecast_text += f"**Tag {day}:** {temp}°C, {base_weather['description']}\n"
        
        return [TextContent(type="text", text=forecast_text)]
    
    elif name == "get_hourly_forecast":
        city = arguments.get("city", "").lower()
        hours = arguments.get("hours", 24)
        use_real_api = arguments.get("use_real_api", True)  # Standardmäßig aktiviert
        api_key = arguments.get("api_key") or os.getenv("OPENWEATHER_API_KEY")
        
        if not use_real_api:
            # Wenn API nicht verwendet werden soll, einfache Mock-Daten zurückgeben
            base_weather = MOCK_WEATHER_DATA.get(city)
            if not base_weather:
                return [TextContent(
                    type="text", 
                    text=f"❌ Stadt '{city}' nicht verfügbar für stündliche Vorhersage"
                )]
            
            forecast_text = f"⏱️ **Stündliche Vorhersage für {base_weather['city']} (Mock-Daten)**\n\n"
            from datetime import datetime, timedelta
            now = datetime.now()
            
            for hour in range(hours):
                time = now + timedelta(hours=hour)
                time_str = time.strftime("%d.%m.%Y %H:%M")
                temp_variation = (-1 + (hour * 0.2)) if hour % 4 == 0 else (0.5 + (hour * 0.1))
                temp = round(base_weather['temperature'] + temp_variation, 1)
                
                forecast_text += f"**{time_str}:** {temp}°C, {base_weather['description']}\n"
            
            return [TextContent(type="text", text=forecast_text)]
        
        # Echte API Daten verwenden
        if not api_key:
            return [TextContent(
                type="text",
                text="⚠️ Für die stündliche Vorhersage wird ein API-Key benötigt, aber keiner wurde angegeben oder in Umgebungsvariablen gefunden."
            )]
        
        # Echte stündliche Vorhersagedaten abrufen
        forecast_data = await fetch_hourly_forecast(city, api_key, hours)
        
        if "error" in forecast_data:
            return [TextContent(
                type="text", 
                text=f"❌ {forecast_data['error']}"
            )]
        
        forecast_text = f"⏱️ **Stündliche Vorhersage für {forecast_data['city']}, {forecast_data['country']}**\n\n"
        
        # Die erste Vorhersage detailliert anzeigen
        if forecast_data["forecast"]:
            first_forecast = forecast_data["forecast"][0]
            forecast_text += f"**Erste Vorhersage ({first_forecast['time']}):**\n"
            forecast_text += f"🌡️ Temperatur: {first_forecast['temperature']}°C\n"
            forecast_text += f"💧 Luftfeuchtigkeit: {first_forecast['humidity']}%\n"
            forecast_text += f"📖 Beschreibung: {first_forecast['description']}\n"
            forecast_text += f"💨 Windgeschwindigkeit: {first_forecast['wind_speed']} km/h\n\n"
        
        # Die weiteren Vorhersagen kompakter anzeigen
        forecast_text += "**Weitere Vorhersagen:**\n"
        for item in forecast_data["forecast"][1:]:
            forecast_text += f"• {item['time']}: {item['temperature']}°C, {item['description']}\n"
        
        return [TextContent(type="text", text=forecast_text)]
    
    elif name == "compare_weather":
        city1 = arguments.get("city1", "").lower()
        city2 = arguments.get("city2", "").lower()
        use_real_api = arguments.get("use_real_api", False)
        api_key = arguments.get("api_key") or os.getenv("OPENWEATHER_API_KEY")
        
        if use_real_api and api_key:
            # Echte API nutzen für beide Städte
            logger.info(f"Verwende echte API für Wettervergleich zwischen {city1} und {city2}")
            weather1 = await fetch_real_weather(city1, api_key)
            weather2 = await fetch_real_weather(city2, api_key)
            
            if "error" in weather1 or "error" in weather2:
                error_text = ""
                if "error" in weather1:
                    error_text += f"❌ Stadt 1 ({city1}): {weather1['error']}\n"
                if "error" in weather2:
                    error_text += f"❌ Stadt 2 ({city2}): {weather2['error']}"
                return [TextContent(type="text", text=error_text)]
        else:
            # Mock-Daten verwenden
            if use_real_api and not api_key:
                logger.warning("use_real_api=true aber kein API-Key vorhanden für Wettervergleich")
                
            weather1 = MOCK_WEATHER_DATA.get(city1)
            weather2 = MOCK_WEATHER_DATA.get(city2)
            
            if not weather1 or not weather2:
                available_cities = ", ".join(MOCK_WEATHER_DATA.keys())
                error_text = f"❌ Eine oder beide Städte nicht in Mock-Daten verfügbar.\nVerfügbare Städte: {available_cities}"
                if not api_key:
                    error_text += "\nOder nutze use_real_api=true mit API Key für weitere Städte."
                return [TextContent(type="text", text=error_text)]
        
        # Vergleichsdaten berechnen
        temp_diff = abs(weather1['temperature'] - weather2['temperature'])
        hum_diff = abs(weather1['humidity'] - weather2['humidity'])
        warmer_city = weather1['city'] if weather1['temperature'] > weather2['temperature'] else weather2['city']
        
        comparison = f"""⚖️ **Wettervergleich**

🏙️ **{weather1['city']}, {weather1['country']}**
🌡️ Temperatur: {weather1['temperature']}°C
💧 Luftfeuchtigkeit: {weather1['humidity']}%
📖 Beschreibung: {weather1['description']}

🏙️ **{weather2['city']}, {weather2['country']}**
🌡️ Temperatur: {weather2['temperature']}°C
💧 Luftfeuchtigkeit: {weather2['humidity']}%
📖 Beschreibung: {weather2['description']}

📊 **Vergleich:**
🌡️ Temperaturdifferenz: {temp_diff}°C
💧 Feuchtigkeitsdifferenz: {hum_diff}%
🥇 Wärmer: {warmer_city}
"""

        data_source = "Echtzeit-API" if use_real_api and api_key else "Mock-Daten"
        comparison += f"ℹ️ *Datenquelle: {data_source}*"

        return [TextContent(type="text", text=comparison)]
    
    else:
        return [TextContent(type="text", text=f"❌ Unbekanntes Tool: {name}")]

async def main():
    """Hauptfunktion - startet den MCP Server"""
    logger.info("🌤️ Weather MCP Server wird gestartet...")
    logger.info("📡 MCP stdio Modus")
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream, 
            app.create_initialization_options()
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
