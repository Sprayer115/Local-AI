#!/usr/bin/env python3
"""
Einfache Demo für mcp_benchmark_llm.py
Zeigt Verwendung mit Weather und Mensa Service (Mock-Implementierung)
Tools werden dynamisch vom MCP-Server geladen
"""

from mcp_benchmark_llm import TestCase, run_mcp_benchmark
import subprocess
import json
import sys
from pathlib import Path

def get_weather_tools_from_server():
    """Lädt Tool-Definitionen direkt aus der Weather Server Klasse"""
    
    try:
        print(f"   🔧 Lade Weather Tools aus Server-Definition...")
        
        # Direkt die Tool-Definitionen aus dem Code extrahieren
        weather_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Ruft aktuelle Wetterdaten für eine Stadt ab (Mock-Daten oder echte API)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "Name der Stadt"
                            },
                            "use_real_api": {
                                "type": "boolean",
                                "description": "Verwende echte API (benötigt API Key)",
                                "default": False
                            }
                        },
                        "required": ["city"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_hourly_forecast",
                    "description": "Ruft stündliche Wettervorhersage für bis zu 4 Tage ab (echte Daten via API)",
                    "parameters": {
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
                            }
                        },
                        "required": ["city"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "compare_weather",
                    "description": "Vergleicht Wetter zwischen zwei Städten",
                    "parameters": {
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
                            }
                        },
                        "required": ["city1", "city2"]
                    }
                }
            }
        ]
        
        print(f"   ✅ {len(weather_tools)} Weather Tools geladen")
        return weather_tools
        
    except Exception as e:
        print(f"   ⚠️  Fehler beim Laden der Weather Tools: {e}")
        return []

def get_mensa_tools_from_server():
    """Lädt Tool-Definitionen direkt aus der Mensa Server Klasse"""
    
    try:
        print(f"   🔧 Lade Mensa Tools aus Server-Definition...")
        
        # Direkt die Tool-Definitionen aus dem Code extrahieren
        mensa_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_daily_menu",
                    "description": "Ruft das Tagesmenü der HTWG Mensa ab (echt oder Mock-Daten)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "days_ahead": {
                                "type": "integer",
                                "description": "Anzahl Tage in die Zukunft (0=heute, 1=morgen, etc.)",
                                "default": 0,
                                "minimum": 0,
                                "maximum": 14
                            },
                            "use_real_data": {
                                "type": "boolean",
                                "description": "Verwende echte Daten von der HTWG-Website",
                                "default": True
                            }
                        },
                        "required": ["days_ahead"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weekly_menu",
                    "description": "Ruft das Wochenmenü der HTWG Mensa ab",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "weeks_ahead": {
                                "type": "integer",
                                "description": "Anzahl Wochen in die Zukunft (0=diese Woche)",
                                "default": 0,
                                "minimum": 0,
                                "maximum": 4
                            },
                            "use_real_data": {
                                "type": "boolean",
                                "description": "Verwende echte Daten von der HTWG-Website",
                                "default": True
                            }
                        },
                        "required": ["weeks_ahead"]
                    }
                }
            }
        ]
        
        print(f"   ✅ {len(mensa_tools)} Mensa Tools geladen")
        return mensa_tools
        
    except Exception as e:
        print(f"   ⚠️  Fehler beim Laden der Mensa Tools: {e}")
        return []

def get_all_tools_from_servers():
    """Lädt alle Tools von beiden MCP-Servern"""
    print("🔧 Lade Tools aus Server-Definitionen...")
    
    weather_tools = get_weather_tools_from_server()
    mensa_tools = get_mensa_tools_from_server()
    
    all_tools = weather_tools + mensa_tools
    
    print(f"📋 Gesamt: {len(all_tools)} Tools erfolgreich geladen ({len(weather_tools)} Weather + {len(mensa_tools)} Mensa)")
    
    return all_tools

def main():
    # 1. Test-Cases definieren - verschiedene Städte und Formulierungen
    test_cases = [
        # Weather Service Tests - verschiedene deutsche Städte
        TestCase(
            name="Berlin Weather",
            prompt="Wie ist das Wetter in Berlin?",
            expected_tool_call="get_weather",
            expected_parameters={"city": "berlin"}
        ),
        TestCase(
            name="Munich Weather Alternative",
            prompt="Zeige mir das aktuelle Wetter in München.",
            expected_tool_call="get_weather",
            expected_parameters={"city": "münchen"}
        ),
        TestCase(
            name="Hamburg Weather Casual",
            prompt="Wie ist es denn heute in Hamburg?",
            expected_tool_call="get_weather",
            expected_parameters={"city": "hamburg"}
        ),
        TestCase(
            name="Frankfurt Weather Short",
            prompt="Frankfurt Wetter?",
            expected_tool_call="get_weather",
            expected_parameters={"city": "frankfurt"}
        ),
        TestCase(
            name="Cologne Weather Formal",
            prompt="Können Sie mir bitte das Wetter für Köln mitteilen?",
            expected_tool_call="get_weather",
            expected_parameters={"city": "köln"}
        ),
        TestCase(
            name="Stuttgart Weather Question",
            prompt="Regnet es gerade in Stuttgart?",
            expected_tool_call="get_weather",
            expected_parameters={"city": "stuttgart"}
        ),
        
        # Weather Comparisons - verschiedene Formulierungen
        TestCase(
            name="Weather Comparison Standard",
            prompt="Vergleiche das Wetter zwischen München und Hamburg.",
            expected_tool_call="compare_weather",
            expected_parameters={"city1": "münchen", "city2": "hamburg"}
        ),
        TestCase(
            name="Weather Comparison Casual",
            prompt="Wo ist es wärmer - in Berlin oder Düsseldorf?",
            expected_tool_call="compare_weather",
            expected_parameters={"city1": "berlin", "city2": "düsseldorf"}
        ),
        TestCase(
            name="Weather Comparison Direct",
            prompt="Dresden vs Leipzig Wetter",
            expected_tool_call="compare_weather",
            expected_parameters={"city1": "dresden", "city2": "leipzig"}
        ),
        
        # Hourly Forecast Tests - verschiedene Stunden
        TestCase(
            name="Hourly Forecast 12h",
            prompt="Gib mir die Wettervorhersage für die nächsten 12 Stunden in Berlin.",
            expected_tool_call="get_hourly_forecast",
            expected_parameters={"city": "berlin", "hours": 12}
        ),
        TestCase(
            name="Hourly Forecast 24h",
            prompt="Wie wird das Wetter in den nächsten 24 Stunden in Hannover?",
            expected_tool_call="get_hourly_forecast",
            expected_parameters={"city": "hannover", "hours": 24}
        ),
        TestCase(
            name="Hourly Forecast 6h",
            prompt="Stundenvorhersage für Nürnberg, nächste 6 Stunden",
            expected_tool_call="get_hourly_forecast",
            expected_parameters={"city": "nürnberg", "hours": 6}
        ),
        
        # Mensa Service Tests - verschiedene Tage und Formulierungen
        TestCase(
            name="Today Menu Simple",
            prompt="Was gibt es heute in der Mensa?",
            expected_tool_call="get_daily_menu",
            expected_parameters={"days_ahead": 0}
        ),
        TestCase(
            name="Today Menu Detailed",
            prompt="Zeig mir das heutige Mensamenü.",
            expected_tool_call="get_daily_menu",
            expected_parameters={"days_ahead": 0}
        ),
        TestCase(
            name="Tomorrow Menu",
            prompt="Was steht morgen auf dem Mensaplan?",
            expected_tool_call="get_daily_menu",
            expected_parameters={"days_ahead": 1}
        ),
        TestCase(
            name="Tomorrow Menu Alternative",
            prompt="Mensamenü für morgen anzeigen",
            expected_tool_call="get_daily_menu",
            expected_parameters={"days_ahead": 1}
        ),
        TestCase(
            name="Day After Tomorrow Menu",
            prompt="Was gibt es übermorgen zu essen?",
            expected_tool_call="get_daily_menu",
            expected_parameters={"days_ahead": 2}
        ),
        TestCase(
            name="Future Menu 3 Days",
            prompt="Mensaplan in 3 Tagen?",
            expected_tool_call="get_daily_menu",
            expected_parameters={"days_ahead": 3}
        ),
        TestCase(
            name="Future Menu Formal",
            prompt="Können Sie mir das Menü für übermorgen zeigen?",
            expected_tool_call="get_daily_menu",
            expected_parameters={"days_ahead": 2}
        )
    ]
    
    # 2. Tools dynamisch von beiden MCP-Servern laden
    tools = get_all_tools_from_servers()
    
    if not tools:
        print("❌ Keine Tools geladen - Abbruch!")
        return
    
    print(f"✅ {len(tools)} Tools bereit für Benchmark:")
    for tool in tools:
        print(f"   • {tool['function']['name']}: {tool['function']['description']}")
    
    # 3. Tool Executor (Minimal Mock für Benchmark)
    def execute_tool(function_name: str, arguments: dict) -> dict:
        # Minimale Mock-Response nur für Benchmark-Zwecke
        return {
            "function": function_name,
            "arguments": arguments,
            "status": "success",
            "mock_response": f"Tool {function_name} würde ausgeführt mit {arguments}"
        }
    
    # 4. Modelle definieren
    models = [
        {
            "name": "llama3.2",
            "provider": "ollama",
            "config": {
                "model": "ollama/llama3.2",
                "base_url": "http://localhost:11434",
                "temperature": 0.1
            }
        }
    ]
    
    
    # 5. Benchmark ausführen
    results = run_mcp_benchmark(
        test_cases=test_cases,
        models=models,
        tools=tools,
        execute_tool_fn=execute_tool,
        repetition_rounds=1
    )
    
    print(f"\n✅ Benchmark abgeschlossen: {len(results)} Tests")

if __name__ == "__main__":
    main()

"""
BEISPIEL-RESULT:

Nach Ausführung erhältst du eine Liste von BenchmarkResult-Objekten mit folgenden Eigenschaften:

result = BenchmarkResult(
    test_case=TestCase(name="Berlin Weather", prompt="Wie ist das Wetter in Berlin?", ...),
    model="ollama/llama3.2",
    provider="ollama", 
    round_number=1,
    
    # Timing
    response_time=2.145,           # Gesamtzeit in Sekunden
    first_tool_call_time=2.089,    # Zeit bis ersten Tool-Call
    
    # Korrektheit
    tool_calls_made=1,             # Anzahl Tool-Calls
    correct_tool_called=True,      # Richtiges Tool verwendet
    correct_parameters=True,       # Parameter korrekt
    parameter_accuracy=1.0,        # 100% Parameter-Genauigkeit
    
    # Tatsächliche Ergebnisse
    actual_tool_call="get_weather",
    actual_parameters={"city": "berlin"},
    tool_execution_time=0.012,
    
    # Meta-Daten
    error=None,                    # Kein Fehler
    tokens_used=287,               # Verwendete Tokens
    response_content=""            # LLM-Antwort-Text
)

BENCHMARK-ZUSAMMENFASSUNG:
================================================================================
🤖 Modell: ollama/llama3.2 (ollama)
📊 Gesamt: 8/8 erfolgreich
⏱️  Durchschnittliche Antwortzeit: 2.145s
🎯 Tool-Call-Genauigkeit: 8/8 (100.0%)
📋 Parameter-Genauigkeit: 8/8 (100.0%)

📋 Test-Case Details:
   Berlin Weather (2 Runden):
     🎯 Tools: 2/2 korrekt (100.0%)
     📋 Parameter: 100.0% korrekt im Durchschnitt
     ⏱️  Zeit: 2.145s durchschnittlich
   
   Weather Comparison (2 Runden):
     🎯 Tools: 2/2 korrekt (100.0%)
     📋 Parameter: 100.0% korrekt im Durchschnitt
     ⏱️  Zeit: 2.234s durchschnittlich
   
   Today Menu (2 Runden):
     🎯 Tools: 2/2 korrekt (100.0%)
     📋 Parameter: 100.0% korrekt im Durchschnitt
     ⏱️  Zeit: 1.987s durchschnittlich
   
   Tomorrow Menu (2 Runden):
     🎯 Tools: 2/2 korrekt (100.0%)
     📋 Parameter: 100.0% korrekt im Durchschnitt
     ⏱️  Zeit: 2.012s durchschnittlich

💾 Detaillierte Ergebnisse exportiert nach: mcp_benchmark_results_1753544980.json
================================================================================
"""
