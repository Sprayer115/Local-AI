#!/usr/bin/env python3
"""
Demonstration der MCP Multi-Model-Evaluierung mit ECHTEN LLM-Calls
Zeigt die neue Multi-Model-Evaluierungsfunktionalität aus mcp_benchmark_llm.py

WICHTIG: Diese Demo verwendet ECHTE LLM-Calls, keine Mock-Daten!
- MCPBenchmarkLLM generiert echte Tool-Calls
- Tools werden mit LLM-generierten Parametern ausgeführt
- Multi-Model-Evaluatoren bewerten die echten Ergebnisse
"""

import json
import time
from typing import Dict, Any, List
from dataclasses import asdict

# Import der Multi-Model-Klassen UND der Benchmark-Klasse
from mcp_benchmark_llm import (
    MCPMultiModelEvaluator,
    MCPBenchmarkLLM,
    BenchmarkResult,
    EvaluatorFactory,
    TestCase,
    MultiModelEvaluationResult
)

def export_multi_model_results(multi_model_results: List[Dict[str, Any]], filename: str | None = None) -> str:
    """
    Speichert die Multi-Model-Ergebnisse als JSON-Datei und gibt den Dateinamen zurück.
    
    Struktur:
    - test_case: Erwartete Eingaben/Outputs
    - benchmark_result: Echte LLM-Outputs (actual_tool_call, model_initial, model_final, etc.)
    - multi_model_result: Evaluierungen von mehreren Modellen
    """
    # Modelle aggregieren
    models_used: List[str] = []
    for entry in multi_model_results:
        mmr = entry.get("multi_model_result")
        if hasattr(mmr, "models_used") and isinstance(mmr.models_used, list):
            for m in mmr.models_used:
                if m not in models_used:
                    models_used.append(m)

    data = {
        "timestamp": int(time.time()),
        "type": "multi_model_mcp_evaluation",
        "total_test_cases": len(multi_model_results),
        "models": models_used,
        "description": "Multi-Model MCP Evaluation with REAL LLM calls (not mocked)",
        "structure": {
            "test_case": "Expected inputs/outputs for the test",
            "benchmark_result": "REAL LLM outputs: actual_tool_call, model_initial, model_final, parameters, etc.",
            "multi_model_result": "Evaluations from multiple evaluator models"
        },
        "results": []
    }

    for entry in multi_model_results:
        test_case = entry.get("test_case", {})
        benchmark_result = entry.get("benchmark_result")  # ✅ Benchmark-Ergebnis holen
        mmr = entry.get("multi_model_result")
        mmr_dict = mmr.to_dict() if hasattr(mmr, "to_dict") else {}
        
        # TestCase robust serialisieren (Objekt -> Dict)
        if isinstance(test_case, TestCase) or hasattr(test_case, "__dataclass_fields__"):
            test_case_serialized = asdict(test_case)
        elif isinstance(test_case, dict):
            test_case_serialized = test_case
        else:
            test_case_serialized = {"name": getattr(test_case, "name", str(test_case))}
        
        # ✅ Benchmark-Result serialisieren (enthält echte LLM-Daten)
        benchmark_result_serialized = None
        if benchmark_result:
            if hasattr(benchmark_result, "to_dict"):
                benchmark_result_serialized = benchmark_result.to_dict()
            elif hasattr(benchmark_result, "__dataclass_fields__"):
                benchmark_result_serialized = asdict(benchmark_result)
            elif isinstance(benchmark_result, dict):
                benchmark_result_serialized = benchmark_result

        result_entry = {
            "test_case": test_case_serialized,
            "multi_model_result": mmr_dict
        }
        
        # ✅ Benchmark-Result hinzufügen (falls vorhanden)
        if benchmark_result_serialized:
            result_entry["benchmark_result"] = benchmark_result_serialized
        
        data["results"].append(result_entry)

    if not filename:
        filename = f"mcp_benchmark_results_{data['timestamp']}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Ergebnisse gespeichert unter: {filename}")
    return filename

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
                    "description": "Ruft aktuelle Wetterdaten für eine Stadt ab ",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "Name der Stadt"
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
                    "description": "Ruft stündliche Wettervorhersage für bis zu 4 Tage ab",
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
                    "description": "Ruft das Tagesmenü der HTWG Mensa ab",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "days_ahead": {
                                "type": "integer",
                                "description": "Anzahl Tage in die Zukunft (0=heute, 1=morgen, etc.)",
                                "default": 0,
                                "minimum": 0,
                                "maximum": 14
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

def demonstrate_multi_model_evaluator():
    """
    Demonstriert die Multi-Model-Evaluierung mit ECHTEN LLM-Calls
    
    Flow:
    1. Erstellt MCPBenchmarkLLM für das zu testende Modell
    2. Führt ECHTE Tool-Calls durch (nicht Mock!)
    3. Evaluiert mit mehreren Evaluator-Modellen
    """
    
    print("=== MCP Multi-Model-Evaluator Demonstration ===\n")
    print("⚠️  WICHTIG: Diese Demo macht ECHTE LLM-Calls!\n")
    
    # 1. Zu testendes Modell konfigurieren
    test_model_config = {
        "model": "ollama/llama3.2",
        "base_url": "http://localhost:11434",
        "temperature": 0.2,
        "timeout": 60
    }
    
    print(f"🤖 Zu testendes Modell: {test_model_config['model']}")
    
    # 2. Multi-Model-Evaluator-Konfiguration
    evaluation_models = [
        {"name": "llama3.2", "provider": "ollama", "base_url": "http://localhost:11434"},
        {"name": "mistral", "provider": "ollama", "base_url": "http://localhost:11434"},
    ]
    
    print(f"\n📊 Evaluator-Modelle: {len(evaluation_models)}")
    for model in evaluation_models:
        print(f"   • {model['provider']}/{model['name']}")
    
    # 3. MCPBenchmarkLLM erstellen (für echte Tool-Calls)
    print(f"\n🔧 Erstelle MCPBenchmarkLLM...")
    try:
        llm = MCPBenchmarkLLM(**test_model_config)
        print("✅ MCPBenchmarkLLM erfolgreich erstellt")
    except Exception as e:
        print(f"❌ Fehler beim Erstellen von MCPBenchmarkLLM: {e}")
        return []
    
    # 4. Multi-Model-Evaluator erstellen
    print(f"\n🔧 Erstelle Multi-Model-Evaluator...")
    
    try:
        multi_evaluator = EvaluatorFactory.create_multi_model_evaluator(
            models=evaluation_models,
            base_config={
                "max_retries": 2,
                "temperature": 0.0,
                "timeout": 30
            },
            show_progress=True,
            timeout=90
        )
        print("✅ Multi-Model-Evaluator erfolgreich erstellt")
    except Exception as e:
        print(f"❌ Fehler beim Erstellen des Multi-Model-Evaluators: {e}")
        return []
    
    # 3. TestCases: Nutze die echten TestCases wie in demonstrate_advanced_evaluator_with_standard_testcases
    test_cases = [
        # Weather Service Tests - verschiedene deutsche Städte
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
    
    # 5. Tools laden
    print(f"\n🔧 Lade MCP Tools...")
    tools = get_all_tools_from_servers()
    print(f"✅ {len(tools)} Tools geladen")
    
    # 6. Tool-Execution-Funktion (Mock - simuliert echte API-Calls)
    def execute_tool(function_name: str, arguments: dict) -> dict:
        """Simuliert Tool-Execution (in Produktion würde hier echter API-Call stattfinden)"""
        if function_name == "get_weather":
            city = arguments.get("city", "Unknown")
            return {
                "city": city,
                "temperature": 22,
                "condition": "sunny",
                "humidity": 65
            }
        elif function_name == "get_hourly_forecast":
            city = arguments.get("city", "Unknown")
            hours = arguments.get("hours", 24)
            try:
                if isinstance(hours, str):
                    hours = int(hours)
                hours = int(hours)
            except (ValueError, TypeError):
                hours = 24
            return {
                "city": city,
                "forecast": [
                    {"hour": i, "temp": 22 + i % 3, "condition": "sunny" if i % 2 == 0 else "cloudy"}
                    for i in range(min(hours, 12))
                ],
                "hours_requested": hours
            }
        elif function_name == "compare_weather":
            city1 = arguments.get("city1", "City1")
            city2 = arguments.get("city2", "City2")
            return {
                "city1": {"name": city1, "temperature": 22, "condition": "sunny"},
                "city2": {"name": city2, "temperature": 18, "condition": "cloudy"},
                "comparison": f"{city1} ist wärmer als {city2}"
            }
        elif function_name == "get_daily_menu":
            days_ahead = arguments.get("days_ahead", 0)
            try:
                if isinstance(days_ahead, str):
                    days_ahead = int(days_ahead)
                days_ahead = int(days_ahead)
            except (ValueError, TypeError):
                days_ahead = 0
            return {
                "date": f"2024-08-{11 + days_ahead}",
                "menu": [
                    {"item": "Schnitzel mit Pommes", "price": "8.50€"},
                    {"item": "Vegetarisches Curry", "price": "7.20€"}
                ],
                "days_ahead": days_ahead
            }
        elif function_name == "get_weekly_menu":
            weeks_ahead = arguments.get("weeks_ahead", 0)
            try:
                if isinstance(weeks_ahead, str):
                    weeks_ahead = int(weeks_ahead)
                weeks_ahead = int(weeks_ahead)
            except (ValueError, TypeError):
                weeks_ahead = 0
            return {
                "week": f"KW {32 + weeks_ahead}",
                "menu": {"Mo": "Schnitzel", "Di": "Curry", "Mi": "Pizza"},
                "weeks_ahead": weeks_ahead
            }
        else:
            return {
                "function": function_name,
                "arguments": arguments,
                "status": "success",
                "mock_response": f"Tool {function_name} erfolgreich ausgeführt"
            }
    
    # 7. ECHTE LLM-Benchmarks durchführen
    print(f"\n{'='*80}")
    print("STARTE ECHTE LLM-BENCHMARKS MIT MULTI-MODEL-EVALUIERUNG")
    print(f"{'='*80}")
    print(f"TestCases: {len(test_cases)}")
    print(f"Zu testendes Modell: {test_model_config['model']}")
    print(f"Evaluator-Modelle: {len(evaluation_models)}")
    print(f"{'='*80}\n")

    multi_model_results = []
    benchmark_results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TestCase {i}/{len(test_cases)}: {test_case.name}")
        print(f"{'='*60}")
        
        try:
            # ========== ECHTE LLM-BENCHMARK DURCHFÜHREN ==========
            # Das LLM generiert selbst den Tool-Call (keine Mock-Daten!)
            benchmark_result = llm.benchmark_test_case(
                test_case=test_case,
                tools=tools,
                execute_tool_fn=execute_tool,
                provider="ollama",
                round_number=1
            )
            benchmark_results.append(benchmark_result)
            
            # Ergebnisse des echten LLM-Calls
            print(f"\n📋 LLM-Benchmark-Ergebnis:")
            print(f"   Tool gewählt: {benchmark_result.actual_tool_call}")
            print(f"   Parameter: {benchmark_result.actual_parameters}")
            print(f"   Korrekt: {'✅' if benchmark_result.correct_tool_called else '❌'}")
            print(f"   Parameter-Genauigkeit: {benchmark_result.parameter_accuracy:.1%}")
            
            # Prüfe ob LLM überhaupt einen Tool-Call gemacht hat
            if not benchmark_result.actual_tool_call:
                print(f"   ⚠️ Kein Tool-Call vom LLM generiert - überspringe Evaluierung")
                continue
            
            # ========== MULTI-MODEL-EVALUIERUNG DER ECHTEN DATEN ==========
            print(f"\n📊 Starte Multi-Model-Evaluierung...")
            
            # Tool-Call-JSON aus Benchmark-Ergebnis erstellen
            tool_call_json = {
                "function": {
                    "name": benchmark_result.actual_tool_call,
                    "arguments": benchmark_result.actual_parameters
                }
            }
            
            # Tool-Response aus dem Benchmark holen (wurde bereits ausgeführt)
            # Da execute_tool deterministisch ist, können wir es erneut aufrufen
            tool_response = execute_tool(
                benchmark_result.actual_tool_call,
                benchmark_result.actual_parameters
            )
            
            multi_result = multi_evaluator.evaluate_interaction_multi_model(
                original_prompt=test_case.prompt,
                model_initial=benchmark_result.model_initial,
                tool_call_json=tool_call_json,
                tool_response=tool_response,
                model_final=benchmark_result.model_final,
                expected_tool_call=test_case.expected_tool_call,
                expected_parameters=test_case.expected_parameters
            )

            test_result = {
                "test_case": test_case,
                "benchmark_result": benchmark_result,  # Echte LLM-Daten
                "multi_model_result": multi_result     # Multi-Model-Bewertung
            }
            multi_model_results.append(test_result)
            
            # Kurze Zusammenfassung
            successful_evals = [e for e in multi_result.model_evaluations if e.evaluation_error is None]
            if successful_evals:
                avg_score = sum(e.overall_score for e in successful_evals) / len(successful_evals)
                print(f"   Durchschnittlicher Evaluierungs-Score: {avg_score:.1f}/100")
            
        except Exception as e:
            print(f"   ❌ Fehler bei TestCase {test_case.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # 8. Multi-Model-Ergebnisse analysieren
    print(f"\n{'='*80}")
    print("ZUSAMMENFASSUNG: MULTI-MODEL EVALUIERUNG")
    print(f"{'='*80}")
    
    # Benchmark-Statistiken
    if benchmark_results:
        successful_benchmarks = [r for r in benchmark_results if r.error is None and r.actual_tool_call]
        print(f"\n🎯 LLM-BENCHMARK-STATISTIKEN:")
        print(f"   Durchgeführt: {len(benchmark_results)} Tests")
        print(f"   Erfolgreich: {len(successful_benchmarks)} Tests")
        print(f"   Tool-Calls korrekt: {sum(1 for r in successful_benchmarks if r.correct_tool_called)}/{len(successful_benchmarks)}")
        
        if successful_benchmarks:
            avg_param_accuracy = sum(r.parameter_accuracy for r in successful_benchmarks) / len(successful_benchmarks)
            avg_response_time = sum(r.response_time for r in successful_benchmarks) / len(successful_benchmarks)
            print(f"   Durchschn. Parameter-Genauigkeit: {avg_param_accuracy:.1%}")
            print(f"   Durchschn. Response-Zeit: {avg_response_time:.2f}s")
    
    # 5. Multi-Model-Ergebnisse analysieren
    print(f"\n{'='*80}")
    print("MULTI-MODEL EVALUIERUNG ERGEBNISSE")
    print(f"{'='*80}")
    
    if multi_model_results:
        print(f"📊 Erfolgreich evaluiert: {len(multi_model_results)} TestCases")
        
        # Statistiken pro Modell
        all_evaluations = []
        for result in multi_model_results:
            all_evaluations.extend(result["multi_model_result"].model_evaluations)
        
        # Gruppiere nach Modell
        model_stats = {}
        for evaluation in all_evaluations:
            model_name = evaluation.evaluator_model
            if model_name not in model_stats:
                model_stats[model_name] = {
                    "scores": [],
                    "tool_scores": [],
                    "answer_scores": [],
                    "times": [],
                    "errors": 0
                }
            
            if evaluation.evaluation_error is None:
                model_stats[model_name]["scores"].append(evaluation.overall_score)
                model_stats[model_name]["tool_scores"].append(evaluation.tool_usage_correctness)
                model_stats[model_name]["answer_scores"].append(evaluation.answer_correctness)
                model_stats[model_name]["times"].append(evaluation.evaluation_time)
            else:
                model_stats[model_name]["errors"] += 1
        
        # Ergebnisse pro Modell anzeigen
        print(f"\n📈 MODELL-VERGLEICH:")
        for model_name, stats in model_stats.items():
            if stats["scores"]:
                avg_score = sum(stats["scores"]) / len(stats["scores"])
                avg_tool = sum(stats["tool_scores"]) / len(stats["tool_scores"])
                avg_answer = sum(stats["answer_scores"]) / len(stats["answer_scores"])
                avg_time = sum(stats["times"]) / len(stats["times"])
                
                print(f"\n   🤖 {model_name}:")
                print(f"      Overall Score: {avg_score:.1f}/100")
                print(f"      Tool Usage: {avg_tool:.2f}")
                print(f"      Answer Correctness: {avg_answer:.2f}")
                print(f"      Avg. Zeit: {avg_time:.2f}s")
                print(f"      Evaluierungen: {len(stats['scores'])} erfolgreich, {stats['errors']} Fehler")
        
        # TestCase-Details
        print(f"\n📋 TESTCASE-DETAILS:")
        for result in multi_model_results:
            test_case = result["test_case"]
            multi_result = result["multi_model_result"]

            # Unterstütze sowohl Dataclass-Objekte als auch Dicts
            tc_name = getattr(test_case, "name", None)
            if tc_name is None and isinstance(test_case, dict):
                tc_name = test_case.get("name")
            tc_prompt = getattr(test_case, "prompt", None)
            if tc_prompt is None and isinstance(test_case, dict):
                tc_prompt = test_case.get("original_prompt") or test_case.get("prompt")

            print(f"\n   📝 {tc_name}:")
            print(f"      Prompt: {tc_prompt}")
            
            successful_evals = [e for e in multi_result.model_evaluations if e.evaluation_error is None]
            if successful_evals:
                avg_score = sum(e.overall_score for e in successful_evals) / len(successful_evals)
                print(f"      Durchschnittlicher Score: {avg_score:.1f}/100")
                print(f"      Erfolgreiche Evaluierungen: {len(successful_evals)}/{len(multi_result.model_evaluations)}")
            
            # Zeige pro Modell
            for evaluation in multi_result.model_evaluations:
                status = "✅" if evaluation.evaluation_error is None else "❌"
                score = f"{evaluation.overall_score:.1f}" if evaluation.evaluation_error is None else "ERROR"
                print(f"        {status} {evaluation.evaluator_model}: {score}/100")
    
    print(f"\n{'='*80}")
    print("MULTI-MODEL EVALUIERUNG ABGESCHLOSSEN")
    print(f"{'='*80}")
    
    return multi_model_results

def demonstrate_advanced_evaluator_with_standard_testcases():
    """
    Demonstriert Advanced-Evaluator mit Standard-TestCases und ECHTEN LLM-Calls
    
    Diese Funktion zeigt den vollständigen Workflow:
    1. MCPBenchmarkLLM macht echte Tool-Calls
    2. MCPAdvancedEvaluator bewertet die Ergebnisse
    """
    
    print("=== MCP Advanced-Evaluator mit Standard TestCases (ECHTE LLM-Calls) ===\n")
    
    # 1. Test-Cases definieren - identisch mit mcp_demo.py
    test_cases = [
        # Weather Service Tests - verschiedene deutsche Städte
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
    
    # 2. Tools dynamisch laden (identisch mit mcp_demo.py)
    tools = get_all_tools_from_servers()
    
    if not tools:
        print("Keine Tools geladen - Abbruch!")
        return []
    
    print(f"{len(tools)} Tools bereit für Advanced-Benchmark:")
    for tool in tools:
        print(f"   • {tool['function']['name']}: {tool['function']['description']}")
    
    # 3. Tool Executor (identisch mit mcp_demo.py)
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
        elif function_name == "get_hourly_forecast":
            city = arguments.get("city", "Unknown")
            hours = arguments.get("hours", 24)
            
            # Handle string-to-int conversion robustly
            try:
                if isinstance(hours, str):
                    hours = int(hours)
                hours = int(hours)  # Ensure it's always an integer
            except (ValueError, TypeError):
                hours = 24  # Default fallback
            
            return {
                "city": city,
                "forecast": [
                    {"hour": i, "temp": 22 + i % 3, "condition": "sunny" if i % 2 == 0 else "cloudy"}
                    for i in range(min(hours, 12))  # Now guaranteed to work
                ],
                "hours_requested": hours
            }
        elif function_name == "compare_weather":
            city1 = arguments.get("city1", "City1")
            city2 = arguments.get("city2", "City2")
            return {
                "city1": {"name": city1, "temperature": 22, "condition": "sunny"},
                "city2": {"name": city2, "temperature": 18, "condition": "cloudy"},
                "comparison": f"{city1} ist wärmer als {city2}"
            }
        elif function_name == "get_daily_menu":
            days_ahead = arguments.get("days_ahead", 0)
            
            # Handle string-to-int conversion robustly
            try:
                if isinstance(days_ahead, str):
                    days_ahead = int(days_ahead)
                days_ahead = int(days_ahead)  # Ensure it's always an integer
            except (ValueError, TypeError):
                days_ahead = 0  # Default fallback
            
            return {
                "date": f"2024-08-{11 + days_ahead}",
                "menu": [
                    {"item": "Schnitzel mit Pommes", "price": "8.50€"},
                    {"item": "Vegetarisches Curry", "price": "7.20€"}
                ],
                "days_ahead": days_ahead
            }
        elif function_name == "get_weekly_menu":
            weeks_ahead = arguments.get("weeks_ahead", 0)
            
            # Handle string-to-int conversion robustly
            try:
                if isinstance(weeks_ahead, str):
                    weeks_ahead = int(weeks_ahead)
                weeks_ahead = int(weeks_ahead)  # Ensure it's always an integer
            except (ValueError, TypeError):
                weeks_ahead = 0  # Default fallback
            
            return {
                "week": f"KW {32 + weeks_ahead}",
                "menu": {"Mo": "Schnitzel", "Di": "Curry", "Mi": "Pizza"},
                "weeks_ahead": weeks_ahead
            }
        else:
            # Fallback für andere Tools
            return {
                "function": function_name,
                "arguments": arguments,
                "status": "success",
                "mock_response": f"Tool {function_name} erfolgreich ausgeführt"
            }
    
    # 4. Advanced-Evaluator erstellen (anstatt normalem Evaluator)
    print("\n1. Erstelle Advanced-Evaluator...")
    
    try:
        advanced_evaluator = EvaluatorFactory.create_ollama_evaluator(
            model="llama3.2",
            base_url="http://localhost:11434",
            max_retries=3,
            temperature=0.0  # Maximale Determinismus für bessere Konsistenz
        )
        print("✓ Advanced-Ollama-Evaluator erstellt")
    except Exception as e:
        print(f"❌ Fehler beim Erstellen des Advanced-Evaluators: {e}")
        return []
    
    # 5. Modelle definieren (mit Advanced-Integration)
    models = [
        {
            "name": "llama3.2",
            "provider": "ollama",
            "config": {
                "model": "ollama/llama3.2",
                "base_url": "http://localhost:11434",
                "temperature": 0.1,
                # DISABLED: Standard evaluator deaktiviert, da wir Advanced-Evaluator verwenden
                "evaluator_enabled": False,  # Standard-Evaluator deaktiviert
                # Advanced-Evaluator wird manuell über eigene Integration verwendet
            }
        }
    ]
    
    # 6. Direkter Advanced-Benchmark (ohne Standard-Evaluator)
    print("\n" + "="*80)
    print("STARTE MCP ADVANCED-BENCHMARK MIT STANDARD TESTCASES")
    print("="*80)
    print(f"Test-Cases: {len(test_cases)}")
    print(f"Modelle: {len(models)}")
    print(f"Tools verfügbar: {len(tools)}")
    print("Evaluator: NUR ADVANCED (MCPAdvancedEvaluator)")
    print("="*80)
    
    # MCPBenchmarkLLM aus der neuen Datei verwenden
    llm = MCPBenchmarkLLM(**models[0]["config"])
    
    results = []
    total_tests = len(test_cases) * 2  # 2 Runden pro Test
    current_test = 0
    
    print(f"\nFühre {total_tests} Tests mit Advanced-Evaluator durch...")
    
    for test_case in test_cases:
        print(f"\n--- Test-Case: {test_case.name} ---")
        
        for round_num in range(1, 3):  # 2 Runden pro Test
            current_test += 1
            print(f"  Runde {round_num}/2 ({current_test}/{total_tests})")
            
            try:
                # Standard-Benchmark-Test durchführen (ohne Standard-Evaluator)
                benchmark_result = llm.benchmark_test_case(
                    test_case=test_case,
                    tools=tools,
                    execute_tool_fn=execute_tool,
                    provider=models[0]["provider"],
                    round_number=round_num
                )
                
                # Advanced-Evaluator direkt anwenden
                if benchmark_result.error is None and benchmark_result.actual_tool_call:
                    print(f"    Tool-Call: {benchmark_result.actual_tool_call}")
                    print(f"    Parameter: {benchmark_result.actual_parameters}")
                    
                    # Tool-Response für Evaluator abrufen
                    tool_response = execute_tool(benchmark_result.actual_tool_call, benchmark_result.actual_parameters)
                    
                    # Advanced-Evaluator anwenden
                    advanced_evaluation = advanced_evaluator.evaluate_mcp_interaction(
                        original_prompt=test_case.prompt,
                        model_initial=benchmark_result.model_initial,
                        tool_call_json={"function": {"name": benchmark_result.actual_tool_call, "arguments": benchmark_result.actual_parameters}},
                        tool_response=tool_response,
                        model_final=benchmark_result.model_final,
                        expected_tool_call=test_case.expected_tool_call,
                        expected_parameters=test_case.expected_parameters
                    )
                    
                    # Advanced-Evaluierung zum Ergebnis hinzufügen (direkt als MCPEvaluationResult)
                    benchmark_result.evaluation_result = advanced_evaluation
                    
                    print(f"    ✓ Advanced-Score: {advanced_evaluation.overall_score:.1f}/100")
                    
                else:
                    print(f"    ❌ Fehler oder kein Tool-Call: {benchmark_result.error}")
                
                results.append(benchmark_result)
                
            except Exception as e:
                print(f"    ❌ Test fehlgeschlagen: {e}")
                # Erstelle Fallback-Ergebnis
                fallback_result = BenchmarkResult(
                    test_case=test_case,
                    model=models[0]["config"]["model"],
                    provider=models[0]["provider"],
                    round_number=round_num,
                    response_time=0.0,
                    error=str(e)
                )
                results.append(fallback_result)
    
    print(f"\nAdvanced-Benchmark abgeschlossen: {len(results)} Tests")
    
    # 7. Detaillierte Ergebnisauswertung (ähnlich mcp_demo.py)
    if results:
        print("\n" + "="*80)
        print("ADVANCED-BENCHMARK ZUSAMMENFASSUNG")
        print("="*80)
        
        # Grundlegende Statistiken
        successful_results = [r for r in results if r.error is None]
        total_tests = len(results)
        successful_tests = len(successful_results)
        
        if successful_results:
            avg_response_time = sum(r.response_time for r in successful_results) / len(successful_results)
            avg_tool_call_time = sum(r.first_tool_call_time for r in successful_results if r.first_tool_call_time) / len([r for r in successful_results if r.first_tool_call_time])
            
            correct_tool_calls = sum(1 for r in successful_results if r.correct_tool_called)
            correct_parameters = sum(1 for r in successful_results if r.correct_parameters)
            avg_parameter_accuracy = sum(r.parameter_accuracy for r in successful_results) / len(successful_results)
            avg_tokens = sum(r.tokens_used for r in successful_results if r.tokens_used) / len([r for r in successful_results if r.tokens_used])
            
            # Evaluator-spezifische Statistiken
            eval_results = [r for r in successful_results if r.evaluation_result is not None]
            if eval_results:
                avg_eval_score = sum(r.evaluation_result.overall_score for r in eval_results) / len(eval_results)
                avg_eval_time = sum(r.evaluation_result.evaluation_time for r in eval_results if r.evaluation_result.evaluation_time) / len([r for r in eval_results if r.evaluation_result.evaluation_time])
                eval_errors = sum(1 for r in eval_results if r.evaluation_result.evaluation_error is not None)
            else:
                avg_eval_score = 0
                avg_eval_time = 0
                eval_errors = 0
            
            print(f"🔄 Modell: ollama/llama3.2 (ollama) mit Advanced-Evaluator")
            print(f"✅ Gesamt: {successful_tests}/{total_tests} erfolgreich")
            print(f"⏱️  Durchschnittliche Antwortzeit: {avg_response_time:.2f}s")
            print(f"🔧 Durchschnittliche Zeit bis Tool-Call: {avg_tool_call_time:.2f}s")
            print(f"🎯 Tool-Call-Genauigkeit: {correct_tool_calls}/{len(successful_results)} ({(correct_tool_calls/len(successful_results)*100 if len(successful_results) > 0 else 0):.1f}%)")
            print(f"📋 Parameter-Genauigkeit: {correct_parameters}/{len(successful_results)} ({(correct_parameters/len(successful_results)*100 if len(successful_results) > 0 else 0):.1f}%)")
            print(f"🔢 Durchschnittliche Parameter-Korrektheit: {avg_parameter_accuracy:.1f}%")
            print(f"🔤 Durchschnittliche Tokens: {avg_tokens:.0f}")
            
            if eval_results:
                avg_eval_score = sum(r.evaluation_result.overall_score for r in eval_results if r.evaluation_result.overall_score is not None) / len([r for r in eval_results if r.evaluation_result.overall_score is not None])
                avg_eval_time = sum(r.evaluation_result.evaluation_time for r in eval_results if r.evaluation_result.evaluation_time is not None) / len([r for r in eval_results if r.evaluation_result.evaluation_time is not None])
                eval_errors = len([r for r in eval_results if r.evaluation_result.evaluation_error is not None])
                
                print(f"🔍 Durchschnittlicher Advanced-Score: {avg_eval_score:.1f}/100")
                print(f"⏲️  Durchschnittliche Advanced-Zeit: {avg_eval_time:.2f}s")
                print(f"⚠️  Advanced-Fehler: {eval_errors}/{len(eval_results)} ({(eval_errors/len(eval_results)*100 if eval_results else 0):.1f}%)")
            
            # Test-Case Details gruppiert
            print(f"\n📋 Test-Case Details:")
            
            # Gruppiere Ergebnisse nach Test-Case
            test_case_groups = {}
            for result in successful_results:
                test_name = result.test_case.name
                if test_name not in test_case_groups:
                    test_case_groups[test_name] = []
                test_case_groups[test_name].append(result)
            
            for test_name, group_results in test_case_groups.items():
                tool_correct = sum(1 for r in group_results if r.correct_tool_called)
                param_accuracy = sum(r.parameter_accuracy for r in group_results) / len(group_results)
                avg_time = sum(r.response_time for r in group_results) / len(group_results)
                
                eval_group = [r for r in group_results if r.evaluation_result is not None]
                if eval_group:
                    avg_eval_score_group = sum(r.evaluation_result.overall_score for r in eval_group) / len(eval_group)
                else:
                    avg_eval_score_group = 0
                
                print(f"   {test_name} ({len(group_results)} Runden):")
                print(f"     Tools: {tool_correct}/{len(group_results)} korrekt ({(tool_correct/len(group_results)*100):.1f}%)")
                print(f"     Parameter: {param_accuracy*100:.1f}% korrekt im Durchschnitt")
                print(f"     Zeit: {avg_time:.2f}s durchschnittlich")
                print(f"     Advanced-Score: {avg_eval_score_group:.1f}/100 durchschnittlich")
        
        print("\n" + "="*80)
    
    return results



if __name__ == "__main__":
    print("🚀 MCP Multi-Model-Evaluator Demonstration")
    print("=" * 50)
    
    # Direkte Multi-Model-Evaluierung
    multi_results = demonstrate_multi_model_evaluator()
    
    # Ergebnisse speichern (JSON)
    if multi_results:
        export_multi_model_results(multi_results)
    
    # Finale Zusammenfassung
    print("\n" + "=" * 50)
    print("MULTI-MODEL-DEMONSTRATION ABGESCHLOSSEN")
    print("=" * 50)
    
    if multi_results:
        total_multi_evaluations = sum(len(r["multi_model_result"].model_evaluations) for r in multi_results)
        successful_multi_evaluations = sum(r["multi_model_result"].successful_evaluations for r in multi_results)
        
        print(f"✅ Multi-Model-Evaluator: {successful_multi_evaluations}/{total_multi_evaluations} erfolgreich")
        print(f"   TestCases evaluiert: {len(multi_results)}")
        print(f"   Modelle verwendet: {len(multi_results[0]['multi_model_result'].models_used) if multi_results else 0}")
    
    print(f"\n🎯 NEUE MULTI-MODEL-FUNKTIONALITÄT:")
    print("• Array von Evaluierungen pro TestCase")
    print("• Sequentielle Ausführung mit detailliertem Progress")
    print("• Erweiterte Fehlerbehandlung pro Modell")
    print("• Unterstützung verschiedener Provider (Ollama, OpenAI, etc.)")
    print("• Aggregierte Statistiken können aus Ergebnis-Arrays berechnet werden")
    
    # Beispiel für die Nutzung in anderen Systemen
    print(f"\n📋 INTEGRATION IN BESTEHENDE SYSTEME:")
    print("""
# Multi-Model-Evaluator direkt verwenden:
from mcp_benchmark_llm import EvaluatorFactory

models = [
    {"name": "llama3.2", "provider": "ollama", "base_url": "http://localhost:11434"},
    {"name": "mistral", "provider": "ollama", "base_url": "http://localhost:11434"}
]

evaluator = EvaluatorFactory.create_multi_model_evaluator(
    models=models,
    base_config={"max_retries": 3, "temperature": 0.0}
)

# Multi-Model-Evaluierung - gibt MultiModelEvaluationResult zurück
result = evaluator.evaluate_interaction_multi_model(
    original_prompt="Wie ist das Wetter?",
    model_initial="Tool wird aufgerufen...",
    tool_call_json={"function": {"name": "get_weather", "arguments": '{"city": "berlin"}'}},
    tool_response={"temperature": 22, "condition": "sunny"},
    model_final="Das Wetter in Berlin ist sonnig bei 22°C."
)

# result.model_evaluations enthält Array aller Evaluierungen
# result.model_evaluations[0] = Evaluierung mit llama3.2
# result.model_evaluations[1] = Evaluierung mit mistral
    """)
    
    print(f"\nDemonstration erfolgreich abgeschlossen! 🚀")
    print("\nVorteile des Advanced-basierten MCP-Benchmarks:")
    print("• Nutzt die gleichen TestCases wie der Standard-Benchmark")
    print("• Fortgeschrittenes Framework für robuste und konsistente Evaluierung")
    print("• Lokale Ollama-basierte Evaluierung ohne externe API-Kosten")
    print("• Detaillierte Dokumentation der Ergebnisse wie im Standard-Benchmark")
    print("• Kompatibel mit bestehender MCP-Benchmark-Infrastruktur")
    print("• Erweiterte JSON-Parsing mit automatischer Reparatur")
    print("• Unterstützung für verschiedene Tool-Typen (Weather, Mensa, etc.)")
    print("• Vollständige Kompatibilität mit TestCase-Format")
    
    # Beispiel-Result-Format Dokumentation
    print("\n" + "="*50)
    print("BEISPIEL ADVANCED-RESULT FORMAT:")
    print("="*50)
    print("""
BENCHMARK-RESULT (ähnlich mcp_demo.py):
================================================================================
result = BenchmarkResult(
    test_case=TestCase(name="Berlin Weather", prompt="Wie ist das Wetter in Berlin?", ...),
    model="ollama/llama3.2",
    provider="ollama", 
    round_number=1,
    
    # Timing
    response_time=45.23,           # Gesamtzeit in Sekunden
    first_tool_call_time=8.12,    # Zeit bis ersten Tool-Call
    
    # Korrektheit
    tool_calls_made=1,             # Anzahl Tool-Calls
    correct_tool_called=True,      # Richtiges Tool verwendet
    correct_parameters=True,       # Parameter korrekt
    parameter_accuracy=1.0,        # 100% Parameter-Genauigkeit
    
    # Tatsächliche Ergebnisse
    actual_tool_call="get_weather",
    actual_parameters={"city": "berlin"},
    tool_execution_time=0.000002,
    
    # Meta-Daten
    error=None,                    # Kein Fehler
    tokens_used=182,               # Verwendete Tokens
    
    # ...eigene Evaluator-Resultate...
    model_initial="",                      # Erste LLM-Antwort (vor Tool-Execution)
    model_final="Das Wetter in Berlin ist heute sonnig mit 22°C und 65% Luftfeuchtigkeit."
)

################################################################################
## Benchmark-Zusammenfassung (angepasst, ohne Deepeval/Labeling)
################################################################################
    """)


