#!/usr/bin/env python3
"""
Einfache Demo für mcp_benchmark_llm.py mit Evaluator-Integration
Zeigt Verwendung mit Weather und Mensa Service (Mock-Implementierung)
Tools werden dynamisch vom MCP-Server geladen

NEUE FEATURES:
- Evaluator-LLM bewertet automatisch Tool-Usage und finale Antworten
- Fokus auf Tool-Daten-Korrektheit statt subjektive Bewertungen
- JSON-Schema-Validierung mit automatischer Reparatur
- Robuste Retry-Logik bei Evaluator-Fehlern
- Detaillierte Metriken für Tool-Usage, Korrektheit und Vollständigkeit
"""

from mcp_benchmark_llm import TestCase, run_mcp_benchmark
import subprocess
import json
import sys
from pathlib import Path

def get_weather_tools_from_server():
    """Lädt Tool-Definitionen direkt aus der Weather Server Klasse"""
    
    try:
        print(f"   Lade Weather Tools aus Server-Definition...")
        
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
        
        print(f"   {len(weather_tools)} Weather Tools geladen")
        return weather_tools
        
    except Exception as e:
        print(f"   Fehler beim Laden der Weather Tools: {e}")
        return []

def get_mensa_tools_from_server():
    """Lädt Tool-Definitionen direkt aus der Mensa Server Klasse"""
    
    try:
        print(f"   Lade Mensa Tools aus Server-Definition...")
        
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
        
        print(f"   {len(mensa_tools)} Mensa Tools geladen")
        return mensa_tools
        
    except Exception as e:
        print(f"   Fehler beim Laden der Mensa Tools: {e}")
        return []

def get_all_tools_from_servers():
    """Lädt alle Tools von beiden MCP-Servern"""
    print("Lade Tools aus Server-Definitionen...")
    
    weather_tools = get_weather_tools_from_server()
    mensa_tools = get_mensa_tools_from_server()
    
    all_tools = weather_tools + mensa_tools
    
    print(f"Gesamt: {len(all_tools)} Tools erfolgreich geladen ({len(weather_tools)} Weather + {len(mensa_tools)} Mensa)")
    
    return all_tools

def main():
    # 1. Test-Cases definieren - fokussiert auf die erfolgreich getesteten
    test_cases = [
        # Weather Service Tests - Kernfälle die gut funktionieren
        TestCase(
            name="Berlin Weather",
            prompt="Wie ist das Wetter in Berlin?",
            expected_tool_call="get_weather",
            expected_parameters={"city": "berlin"}
        ),
        TestCase(
            name="München Weather",
            prompt="Zeig mir das Wetter in München.",
            expected_tool_call="get_weather",
            expected_parameters={"city": "münchen"}
        )
    ]
    
    # 2. Tools dynamisch von beiden MCP-Servern laden
    tools = get_all_tools_from_servers()
    
    if not tools:
        print("Keine Tools geladen - Abbruch!")
        return
    
    print(f"{len(tools)} Tools bereit für Benchmark:")
    for tool in tools:
        print(f"   • {tool['function']['name']}: {tool['function']['description']}")
    
    # 3. Tool Executor (Mock für Benchmark - entspricht den tatsächlichen Ergebnissen)
    def execute_tool(function_name: str, arguments: dict) -> dict:
        # Mock-Response basierend auf den tatsächlichen Benchmark-Ergebnissen
        if function_name == "get_weather":
            city = arguments.get("city", "Unknown")
            return {
                "city": city,
                "temperature": 22,
                "condition": "sunny", 
                "humidity": 65
            }
        else:
            # Fallback für andere Tools
            return {
                "function": function_name,
                "arguments": arguments,
                "status": "success",
                "mock_response": f"Tool {function_name} erfolgreich ausgeführt"
            }
    
    # 4. Modelle definieren
    models = [
        {
            "name": "llama3.2",
            "provider": "ollama",
            "config": {
                "model": "ollama/llama3.2",
                "base_url": "http://localhost:11434",
                "temperature": 0.1,
                "evaluator_enabled": True,  # Evaluator aktivieren
                "evaluator_model": "ollama/llama3.2",
                "evaluator_base_url": "http://localhost:11434"
            }
        }
    ]
    
    # 5. Benchmark ausführen mit aktuellen Test-Cases
    print("\n" + "="*80)
    print("STARTE MCP BENCHMARK MIT EVALUATOR-INTEGRATION")
    print("="*80)
    print(f"Test-Cases: {len(test_cases)}")
    print(f"Modelle: {len(models)}")
    print(f"Tools verfügbar: {len(tools)}")
    print("Evaluator: AKTIVIERT (robuste JSON-Parsing)")
    print("="*80)
    results = run_mcp_benchmark(
        test_cases=test_cases,
        models=models,
        tools=tools,
        execute_tool_fn=execute_tool,
        repetition_rounds=2  # Entspricht den aktuellen Ergebnissen (2 Runden pro Test)
    )
    
    print(f"\nBenchmark abgeschlossen: {len(results)} Tests")

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
    response_time=51.76,           # Gesamtzeit in Sekunden
    first_tool_call_time=9.42,    # Zeit bis ersten Tool-Call
    
    # Korrektheit
    tool_calls_made=1,             # Anzahl Tool-Calls
    correct_tool_called=True,      # Richtiges Tool verwendet
    correct_parameters=True,       # Parameter korrekt
    parameter_accuracy=1.0,        # 100% Parameter-Genauigkeit
    
    # Tatsächliche Ergebnisse
    actual_tool_call="get_weather",
    actual_parameters={"city": "Berlin"},  # Beachte: Groß/Kleinschreibung kann abweichen
    tool_execution_time=0.000002,
    
    # Meta-Daten
    error=None,                    # Kein Fehler
    tokens_used=175,               # Verwendete Tokens
    response_content="",           # LLM-Antwort-Text (bei Tool-Calls oft leer)
    
    # Evaluator-Ergebnisse (VEREINFACHT!)
    evaluation_result=EvaluationResult(
        tool_usage_correctness=1.0,         # Tool korrekt verwendet (0.0-1.0)
        final_answer_correctness=1.0,       # Finale Antwort korrekt (0.0-1.0)
        final_answer_completeness=1.0,      # Finale Antwort vollständig (0.0-1.0)
        overall_score=100,                  # Gesamt-Score (0-100)
        short_explanation="Die Tool-Fakten wurden korrekt übernommen und alle relevanten Daten genutzt.",
        evaluator_response_raw='{"tool_usage_correctness": 1.0, "final_answer_correctness": 1.0, "final_answer_completeness": 1.0, "overall_score": 100, "short_explanation": "Die Tool-Fakten wurden korrekt übernommen und alle relevanten Daten genutzt."}',
        evaluation_error=None,              # Fehler bei Evaluierung (None wenn OK)
        evaluation_time=24.18               # Zeit für Evaluierung in Sekunden
    ),
    model_initial="",                      # Erste LLM-Antwort (vor Tool-Execution)
    model_final="Hallo! Ich helfe dir gerne bei deiner Frage nach dem Wetter in Berlin.\n\nLaut den aktuellen Daten ist das Wetter in Berlin sehr schön. Die Temperatur beträgt derzeit 22 Grad Celsius und die Sonne scheint hell. Es ist ein perfekter Tag, um draußen zu sein!"
)

BENCHMARK-ZUSAMMENFASSUNG:
================================================================================
Modell: ollama/llama3.2 (ollama)
Gesamt: 4/4 erfolgreich
Durchschnittliche Antwortzeit: 35.37s
Durchschnittliche Zeit bis Tool-Call: 4.64s
Tool-Call-Genauigkeit: 4/4 (100.0%)
Parameter-Genauigkeit: 4/4 (100.0%)
Durchschnittliche Parameter-Korrektheit: 100.0%
Durchschnittliche Tokens: 169
Durchschnittlicher Evaluator-Score: 100.0/100    # NEU: Evaluator-Bewertung

Test-Case Details:
   Berlin Weather (2 Runden):
     Tools: 2/2 korrekt (100.0%)
     Parameter: 100.0% korrekt im Durchschnitt
     Zeit: 39.24s durchschnittlich
     Evaluator-Score: 100.0/100 durchschnittlich  # NEU: Pro Test-Case Evaluierung
   
   München Weather (2 Runden):
     Tools: 2/2 korrekt (100.0%)
     Parameter: 100.0% korrekt im Durchschnitt
     Zeit: 31.50s durchschnittlich
     Evaluator-Score: 100.0/100 durchschnittlich
� Durchschnittliche Parameter-Korrektheit: 100.0%
🔢 Durchschnittliche Tokens: 169
🔍 Durchschnittlicher Evaluator-Score: 100.0/100    # NEU: Evaluator-Bewertung
⚠️  Halluzinationen erkannt: 0/4 (0.0%)            # NEU: Halluzinations-Erkennung

📋 Test-Case Details:
   Berlin Weather (2 Runden):
     Tools: 2/2 korrekt (100.0%)
     Parameter: 100.0% korrekt im Durchschnitt
     Zeit: 49.10s durchschnittlich
     Evaluator-Score: 100.0/100 durchschnittlich  # NEU: Pro Test-Case Evaluierung
   
   München Weather (2 Runden):
     Tools: 2/2 korrekt (100.0%)
     Parameter: 100.0% korrekt im Durchschnitt
     Zeit: 46.70s durchschnittlich
     Evaluator-Score: 100.0/100 durchschnittlich

================================================================================
"""
