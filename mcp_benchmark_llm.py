#!/usr/bin/env python3
"""
Vereinfachte LLM-Implementierung für MCP-Benchmarking
Basiert auf LiteLLM, optimiert für erste Tool-Calls und Performance-Messung
"""

import logging
import os
import time
import json
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
import warnings

# LiteLLM-Import mit Telemetrie-Deaktivierung
os.environ["LITELLM_TELEMETRY"] = "False"
import litellm

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestCase:
    """Einzelner Test-Case mit erwarteten Ergebnissen"""
    prompt: str
    expected_tool_call: str
    expected_parameters: Dict[str, Any]
    system_prompt: Optional[str] = None
    name: Optional[str] = None

@dataclass
class BenchmarkResult:
    """Benchmark-Ergebnis für einen einzelnen Test"""
    test_case: TestCase
    model: str
    provider: str
    round_number: int
    
    # Timing-Metriken
    response_time: float
    first_tool_call_time: Optional[float] = None
    
    # Tool-Call-Analyse
    tool_calls_made: int = 0
    correct_tool_called: bool = False
    correct_parameters: bool = False
    parameter_accuracy: float = 0.0  # Prozentsatz korrekte Parameter
    
    # Tatsächliche Ergebnisse
    actual_tool_call: Optional[str] = None
    actual_parameters: Dict[str, Any] = field(default_factory=dict)
    tool_execution_time: Optional[float] = None
    
    # Meta-Daten
    error: Optional[str] = None
    tokens_used: Optional[int] = None
    response_content: str = ""
    
    def to_dict(self) -> Dict:
        """Konvertiert zu Dictionary für JSON-Export"""
        return {
            "test_case": {
                "name": self.test_case.name,
                "prompt": self.test_case.prompt,
                "expected_tool_call": self.test_case.expected_tool_call,
                "expected_parameters": self.test_case.expected_parameters
            },
            "model": self.model,
            "provider": self.provider,
            "round_number": self.round_number,
            "response_time": self.response_time,
            "first_tool_call_time": self.first_tool_call_time,
            "tool_calls_made": self.tool_calls_made,
            "correct_tool_called": self.correct_tool_called,
            "correct_parameters": self.correct_parameters,
            "parameter_accuracy": self.parameter_accuracy,
            "actual_tool_call": self.actual_tool_call,
            "actual_parameters": self.actual_parameters,
            "tool_execution_time": self.tool_execution_time,
            "error": self.error,
            "tokens_used": self.tokens_used,
            "response_content": self.response_content
        }

class MCPBenchmarkLLM:
    """
    Vereinfachte LLM-Klasse für MCP-Benchmarking
    Fokus auf erste Tool-Calls und Performance-Messung
    """
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = 60,
        **extra_settings
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_settings = extra_settings
        
        # LiteLLM konfigurieren
        litellm.telemetry = False
        litellm.set_verbose = False
        litellm.suppress_debug_messages = True
        litellm.drop_params = True
        litellm.modify_params = True
        
        # Warnings unterdrücken
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        logger.info(f"MCPBenchmarkLLM initialisiert für Modell: {model}")
    
    def _build_completion_params(self, **override_params) -> Dict[str, Any]:
        """Baut Parameter für LiteLLM completion calls"""
        params = {
            "model": self.model,
            "temperature": self.temperature,
        }
        
        # Optionale Parameter hinzufügen
        if self.base_url:
            params["base_url"] = self.base_url
        if self.api_key:
            params["api_key"] = self.api_key
        if self.timeout:
            params["timeout"] = self.timeout
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
            
        # Extra-Einstellungen (z.B. num_ctx für Ollama)
        if self.extra_settings:
            params.update(self.extra_settings)
            
        # Override-Parameter anwenden
        params.update(override_params)
        
        return params
    
    def _is_ollama_provider(self) -> bool:
        """Erkennt Ollama-Provider unabhängig von der Namenskonvention"""
        if not self.model:
            return False
            
        # Direkter ollama/ Prefix
        if self.model.startswith("ollama/"):
            return True
            
        # Prüfe Umgebungsvariablen für Ollama Base URL
        base_url = os.getenv("OPENAI_BASE_URL", "")
        api_base = os.getenv("OPENAI_API_BASE", "")
        
        # Typische Ollama-Endpunkte
        ollama_endpoints = ["localhost:11434", "127.0.0.1:11434", ":11434"]
        
        return any(endpoint in base_url or endpoint in api_base for endpoint in ollama_endpoints)
    
    def _calculate_parameter_accuracy(
        self,
        expected_params: Dict[str, Any],
        actual_params: Dict[str, Any]
    ) -> float:
        """
        Berechnet die Genauigkeit der Parameter als Prozentsatz
        """
        if not expected_params:
            return 1.0 if not actual_params else 0.0
        
        correct_params = 0
        total_params = len(expected_params)
        
        for key, expected_value in expected_params.items():
            if key in actual_params:
                actual_value = actual_params[key]
                # Flexible Vergleiche für verschiedene Datentypen
                if isinstance(expected_value, str) and isinstance(actual_value, str):
                    if expected_value.lower() == actual_value.lower():
                        correct_params += 1
                elif expected_value == actual_value:
                    correct_params += 1
        
        return correct_params / total_params if total_params > 0 else 0.0
    
    def _parse_tool_call_arguments(self, tool_call: Dict, is_ollama: bool = False) -> tuple:
        """
        Parst Tool-Call-Argumente sicher
        
        Returns:
            tuple: (function_name, arguments, tool_call_id)
        """
        try:
            if is_ollama:
                # Spezielle Behandlung für Ollama
                if "function" in tool_call and isinstance(tool_call["function"], dict):
                    function_name = tool_call["function"]["name"]
                    arguments = json.loads(tool_call["function"]["arguments"])
                else:
                    function_name = tool_call.get("name", "unknown_function")
                    arguments_str = tool_call.get("arguments", "{}")
                    arguments = json.loads(arguments_str) if arguments_str else {}
                tool_call_id = tool_call.get("id", f"tool_{id(tool_call)}")
            else:
                # Standard-Format für andere Provider
                function_name = tool_call["function"]["name"]
                arguments_str = tool_call["function"]["arguments"]
                arguments = json.loads(arguments_str) if arguments_str else {}
                tool_call_id = tool_call["id"]
                
        except (KeyError, json.JSONDecodeError, TypeError) as e:
            logger.error(f"Fehler beim Parsen der Tool-Call-Argumente: {e}")
            function_name = tool_call.get("name", "unknown_function")
            arguments = {}
            tool_call_id = tool_call.get("id", f"tool_{id(tool_call)}")
            
        return function_name, arguments, tool_call_id
    
    def benchmark_test_case(
        self,
        test_case: TestCase,
        tools: List[Dict],
        execute_tool_fn: Callable,
        provider: str,
        round_number: int = 1
    ) -> BenchmarkResult:
        """
        Führt einen einzelnen Test-Case durch und vergleicht mit erwarteten Ergebnissen
        
        Args:
            test_case: Der zu testende TestCase mit erwarteten Ergebnissen
            tools: Liste der verfügbaren Tools (OpenAI-Format)
            execute_tool_fn: Funktion zum Ausführen der Tools
            provider: Name des Providers (z.B. "ollama", "openai")
            round_number: Nummer der Wiederholung
            
        Returns:
            BenchmarkResult: Detaillierte Ergebnisse mit Vergleich zu erwarteten Werten
        """
        start_time = time.time()
        first_tool_call_time = None
        tool_execution_time = None
        error = None
        
        # Ergebnis-Objekt initialisieren
        result = BenchmarkResult(
            test_case=test_case,
            model=self.model,
            provider=provider,
            round_number=round_number,
            response_time=0.0
        )
        
        try:
            # Nachrichten aufbauen
            messages = []
            if test_case.system_prompt:
                messages.append({"role": "system", "content": test_case.system_prompt})
            messages.append({"role": "user", "content": test_case.prompt})
            
            logger.info(f"Test '{test_case.name}' - Runde {round_number} - Modell: {self.model}")
            logger.debug(f"Erwarteter Tool-Call: {test_case.expected_tool_call}")
            logger.debug(f"Erwartete Parameter: {test_case.expected_parameters}")
            
            # LLM-Aufruf mit Tools
            completion_params = self._build_completion_params(
                messages=messages,
                tools=tools,
                stream=False
            )
            
            response = litellm.completion(**completion_params)
            response_content = response["choices"][0]["message"]["content"] or ""
            tool_calls = response["choices"][0]["message"].get("tool_calls")
            
            result.response_content = response_content
            result.tool_calls_made = len(tool_calls) if tool_calls else 0
            
            # Token-Zählung (falls verfügbar)
            if hasattr(response, 'usage') and response.usage:
                result.tokens_used = response.usage.total_tokens
            
            if tool_calls:
                first_tool_call_time = time.time() - start_time
                result.first_tool_call_time = first_tool_call_time
                
                # Ersten Tool-Call analysieren (für Benchmark relevanter)
                first_tool_call = tool_calls[0]
                is_ollama = self._is_ollama_provider()
                
                if isinstance(first_tool_call, dict):
                    function_name, arguments, tool_call_id = self._parse_tool_call_arguments(
                        first_tool_call, is_ollama
                    )
                else:
                    # Objekt-Style Tool-Calls
                    try:
                        function_name = first_tool_call.function.name
                        arguments = json.loads(first_tool_call.function.arguments) if first_tool_call.function.arguments else {}
                        tool_call_id = first_tool_call.id
                    except (json.JSONDecodeError, AttributeError) as e:
                        logger.error(f"Fehler beim Parsen des Tool-Calls: {e}")
                        function_name = "unknown_function"
                        arguments = {}
                        tool_call_id = f"tool_{id(first_tool_call)}"
                
                result.actual_tool_call = function_name
                result.actual_parameters = arguments
                
                # Tool-Call-Korrektheit prüfen
                result.correct_tool_called = (function_name == test_case.expected_tool_call)
                
                # Parameter-Korrektheit prüfen
                result.parameter_accuracy = self._calculate_parameter_accuracy(
                    test_case.expected_parameters, arguments
                )
                result.correct_parameters = (result.parameter_accuracy == 1.0)
                
                # Tool ausführen (für realistische Timing-Messung)
                if execute_tool_fn:
                    tool_start_time = time.time()
                    try:
                        execute_tool_fn(function_name, arguments)
                        result.tool_execution_time = time.time() - tool_start_time
                    except Exception as e:
                        logger.warning(f"Tool-Ausführung fehlgeschlagen: {e}")
                        result.tool_execution_time = time.time() - tool_start_time
                
                logger.info(f"✅ Tool-Call: {function_name} {'✓' if result.correct_tool_called else '✗'}")
                logger.info(f"✅ Parameter: {result.parameter_accuracy:.1%} korrekt {'✓' if result.correct_parameters else '✗'}")
            else:
                logger.warning("❌ Kein Tool-Call erkannt")
                result.actual_tool_call = None
                result.actual_parameters = {}
            
        except Exception as e:
            error = str(e)
            result.error = error
            logger.error(f"❌ Fehler im Test: {error}")
        
        # Gesamt-Timing setzen
        result.response_time = time.time() - start_time
        
        logger.info(f"Test abgeschlossen in {result.response_time:.3f}s")
        return result
    
    def simple_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Einfache Completion ohne Tools (für Vergleichszwecke)
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            completion_params = self._build_completion_params(
                messages=messages,
                stream=False,
                **kwargs
            )
            
            response = litellm.completion(**completion_params)
            return response["choices"][0]["message"]["content"] or ""
            
        except Exception as e:
            logger.error(f"Fehler bei einfacher Completion: {e}")
            return f"Fehler: {str(e)}"

    def run_benchmark_suite(
        self,
        test_cases: List[TestCase],
        tools: List[Dict],
        execute_tool_fn: Callable,
        provider: str,
        repetition_rounds: int = 3
    ) -> List[BenchmarkResult]:
        """
        Führt eine vollständige Benchmark-Suite für dieses Modell durch
        
        Args:
            test_cases: Liste der Test-Cases mit erwarteten Ergebnissen
            tools: MCP-Tools im OpenAI-Format
            execute_tool_fn: Tool-Execution-Funktion
            provider: Provider-Name (z.B. "ollama", "openai")
            repetition_rounds: Anzahl der Wiederholungen pro Test
            
        Returns:
            Liste aller Benchmark-Ergebnisse
        """
        all_results = []
        
        logger.info(f"\n🚀 Starte Benchmark-Suite für {self.model}")
        logger.info(f"   Provider: {provider}")
        logger.info(f"   Test-Cases: {len(test_cases)}")
        logger.info(f"   Wiederholungen: {repetition_rounds}")
        logger.info(f"   Gesamt-Tests: {len(test_cases) * repetition_rounds}")
        
        for test_case in test_cases:
            logger.info(f"\n📋 Test-Case: {test_case.name or 'Unnamed'}")
            
            for round_num in range(1, repetition_rounds + 1):
                logger.debug(f"   Runde {round_num}/{repetition_rounds}")
                
                result = self.benchmark_test_case(
                    test_case=test_case,
                    tools=tools,
                    execute_tool_fn=execute_tool_fn,
                    provider=provider,
                    round_number=round_num
                )
                
                all_results.append(result)
                
                # Kurze Pause zwischen Tests
                if round_num < repetition_rounds:
                    time.sleep(0.5)
        
        logger.info(f"\n✅ Benchmark-Suite abgeschlossen: {len(all_results)} Tests")
        return all_results

class MCPBenchmarkSuite:
    """
    Benchmark-Suite für mehrere Modelle und MCP-Szenarien
    """
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        
    def add_model(
        self,
        model: str,
        **llm_kwargs
    ) -> MCPBenchmarkLLM:
        """Fügt ein Modell zur Benchmark-Suite hinzu"""
        return MCPBenchmarkLLM(model=model, **llm_kwargs)
    
    def export_results(self, filename: str = "mcp_benchmark_results.json"):
        """Exportiert Ergebnisse als JSON"""
        data = {
            "timestamp": time.time(),
            "total_tests": len(self.results),
            "results": [result.to_dict() for result in self.results]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Benchmark-Ergebnisse exportiert nach: {filename}")
    
    def print_summary(self):
        """Druckt eine detaillierte Zusammenfassung der Benchmark-Ergebnisse"""
        if not self.results:
            logger.warning("Keine Benchmark-Ergebnisse vorhanden")
            return
        
        print("\n" + "="*80)
        print("MCP BENCHMARK ZUSAMMENFASSUNG")
        print("="*80)
        
        # Nach Modell und Test-Case gruppieren
        model_results = {}
        for result in self.results:
            model_key = f"{result.model} ({result.provider})"
            if model_key not in model_results:
                model_results[model_key] = {}
            
            test_name = result.test_case.name or "Unnamed"
            if test_name not in model_results[model_key]:
                model_results[model_key][test_name] = []
            
            model_results[model_key][test_name].append(result)
        
        for model_key, test_results in model_results.items():
            print(f"\n🤖 Modell: {model_key}")
            print("-" * 60)
            
            all_model_results = [r for test_list in test_results.values() for r in test_list]
            successful = [r for r in all_model_results if r.error is None]
            failed = [r for r in all_model_results if r.error is not None]
            
            print(f"📊 Gesamt: {len(successful)}/{len(all_model_results)} erfolgreich")
            
            if successful:
                # Timing-Statistiken
                avg_response_time = sum(r.response_time for r in successful) / len(successful)
                first_tool_times = [r.first_tool_call_time for r in successful if r.first_tool_call_time]
                avg_first_tool_time = sum(first_tool_times) / len(first_tool_times) if first_tool_times else 0
                
                # Tool-Call-Genauigkeit
                correct_tools = sum(1 for r in successful if r.correct_tool_called)
                correct_params = sum(1 for r in successful if r.correct_parameters)
                avg_param_accuracy = sum(r.parameter_accuracy for r in successful) / len(successful)
                
                print(f"⏱️  Durchschnittliche Antwortzeit: {avg_response_time:.3f}s")
                print(f"⚡ Durchschnittliche Zeit bis Tool-Call: {avg_first_tool_time:.3f}s")
                print(f"🎯 Tool-Call-Genauigkeit: {correct_tools}/{len(successful)} ({correct_tools/len(successful):.1%})")
                print(f"📋 Parameter-Genauigkeit: {correct_params}/{len(successful)} ({correct_params/len(successful):.1%})")
                print(f"📊 Durchschnittliche Parameter-Korrektheit: {avg_param_accuracy:.1%}")
                
                # Token-Statistiken (falls verfügbar)
                token_results = [r for r in successful if r.tokens_used is not None]
                if token_results:
                    avg_tokens = sum(r.tokens_used for r in token_results) / len(token_results)
                    print(f"🔢 Durchschnittliche Tokens: {avg_tokens:.0f}")
            
            # Detaillierte Test-Case-Statistiken
            print(f"\n📋 Test-Case Details:")
            for test_name, test_case_results in test_results.items():
                test_successful = [r for r in test_case_results if r.error is None]
                if not test_successful:
                    continue
                
                rounds = len(test_case_results)
                correct_tools = sum(1 for r in test_successful if r.correct_tool_called)
                correct_params = sum(1 for r in test_successful if r.correct_parameters)
                avg_time = sum(r.response_time for r in test_successful) / len(test_successful)
                avg_param_acc = sum(r.parameter_accuracy for r in test_successful) / len(test_successful)
                
                print(f"   {test_name} ({rounds} Runden):")
                print(f"     🎯 Tools: {correct_tools}/{len(test_successful)} korrekt ({correct_tools/len(test_successful):.1%})")
                print(f"     📋 Parameter: {avg_param_acc:.1%} korrekt im Durchschnitt")
                print(f"     ⏱️  Zeit: {avg_time:.3f}s durchschnittlich")
                
                # Zeige häufigste Fehler bei Tool-Calls
                if correct_tools < len(test_successful):
                    wrong_tools = [r.actual_tool_call for r in test_successful if not r.correct_tool_called and r.actual_tool_call]
                    if wrong_tools:
                        from collections import Counter
                        most_common_wrong = Counter(wrong_tools).most_common(2)
                        wrong_tools_str = ", ".join([f"{tool} ({count}x)" for tool, count in most_common_wrong])
                        print(f"     ❌ Häufige falsche Tools: {wrong_tools_str}")
            
            if failed:
                print(f"\n❌ Fehler ({len(failed)}):")
                error_counts = {}
                for result in failed:
                    error_key = result.error[:50] + "..." if len(result.error) > 50 else result.error
                    error_counts[error_key] = error_counts.get(error_key, 0) + 1
                
                for error, count in error_counts.items():
                    print(f"   - {error} ({count}x)")
        
        print("\n" + "="*80)

def run_mcp_benchmark(
    test_cases: List[TestCase],
    models: List[Dict[str, Any]],
    tools: List[Dict],
    execute_tool_fn: Callable,
    repetition_rounds: int = 3
) -> List[BenchmarkResult]:
    """
    Haupt-Benchmark-Funktion - führt Tests für mehrere Modelle durch
    
    Args:
        test_cases: Liste der TestCase-Objekte mit erwarteten Ergebnissen
        models: Liste der Modell-Konfigurationen, Format:
                [{"name": "model_name", "provider": "provider", "config": {...}}]
        tools: MCP-Tools im OpenAI-Format
        execute_tool_fn: Funktion zum Ausführen der Tools
        repetition_rounds: Anzahl der Wiederholungen pro Test
        
    Returns:
        Liste aller Benchmark-Ergebnisse
        
    Example:
        test_cases = [
            TestCase(
                name="Weather Berlin",
                prompt="Wie ist das Wetter in Berlin?",
                expected_tool_call="get_weather",
                expected_parameters={"city": "berlin"}
            )
        ]
        
        models = [
            {
                "name": "llama3.2",
                "provider": "ollama", 
                "config": {
                    "model": "ollama/llama3.2",
                    "base_url": "http://localhost:11434"
                }
            }
        ]
        
        results = run_mcp_benchmark(test_cases, models, tools, execute_tool_fn, 5)
    """
    suite = MCPBenchmarkSuite()
    all_results = []
    
    logger.info(f"\n🚀 STARTE MCP BENCHMARK SUITE")
    logger.info(f"   Modelle: {len(models)}")
    logger.info(f"   Test-Cases: {len(test_cases)}")
    logger.info(f"   Wiederholungen: {repetition_rounds}")
    logger.info(f"   Gesamt-Tests: {len(models) * len(test_cases) * repetition_rounds}")
    
    for model_config in models:
        try:
            # LLM-Instanz erstellen
            llm = MCPBenchmarkLLM(**model_config["config"])
            logger.info(f"\n✅ Modell geladen: {model_config['name']}")
            
            # Benchmark für dieses Modell ausführen - jetzt mit der korrekten Methode
            model_results = llm.run_benchmark_suite(
                test_cases=test_cases,
                tools=tools,
                execute_tool_fn=execute_tool_fn,
                provider=model_config["provider"],
                repetition_rounds=repetition_rounds
            )
            
            all_results.extend(model_results)
            suite.results.extend(model_results)
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Modell {model_config['name']}: {e}")
    
    # Ergebnisse anzeigen
    suite.print_summary()
    
    # Ergebnisse exportieren
    timestamp = int(time.time())
    filename = f"mcp_benchmark_results_{timestamp}.json"
    suite.export_results(filename)
    logger.info(f"\n💾 Detaillierte Ergebnisse exportiert nach: {filename}")
    
    return all_results

if __name__ == "__main__":
    # Einfaches Test-Beispiel der neuen Benchmark-Funktionalität
    logger.info("🧪 MCPBenchmarkLLM Test - Fokus auf erwartete Tool-Calls")
    
    # Test-Cases mit erwarteten Ergebnissen definieren
    test_cases = [
        TestCase(
            name="Berlin Weather",
            prompt="Wie ist das Wetter in Berlin?",
            expected_tool_call="get_weather",
            expected_parameters={"city": "berlin"},
            system_prompt="Du bist ein Wetterassistent. Verwende die verfügbaren Tools."
        ),
        TestCase(
            name="München Weather",
            prompt="Zeig mir das Wetter in München.",
            expected_tool_call="get_weather", 
            expected_parameters={"city": "münchen"}
        )
    ]
    
    # Mock Tools definieren
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Ruft Wetterdaten ab",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "Stadtname"}
                    },
                    "required": ["city"]
                }
            }
        }
    ]
    
    # Mock Tool Executor
    def mock_execute_tool(function_name: str, arguments: dict) -> dict:
        return {"result": f"Mock result for {function_name} with {arguments}"}
    
    # Modell-Konfiguration
    models = [
        {
            "name": "llama3.2",
            "provider": "ollama",
            "config": {
                "model": "ollama/llama3.2",
                "base_url": "http://localhost:11434"
            }
        }
    ]
    
    # Benchmark ausführen
    try:
        results = run_mcp_benchmark(
            test_cases=test_cases,
            models=models,
            tools=tools,
            execute_tool_fn=mock_execute_tool,
            repetition_rounds=2
        )
        print(f"\n✅ Test abgeschlossen mit {len(results)} Ergebnissen")
        
    except Exception as e:
        print(f"❌ Test-Fehler: {e}")
        import traceback
        traceback.print_exc()
