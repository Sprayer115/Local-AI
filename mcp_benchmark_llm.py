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
class EvaluationResult:
    """Ergebnis der Evaluierung durch das externe Evaluator-Modell"""
    tool_usage_correctness: float  # 0.0 - 1.0
    final_answer_correctness: float  # 0.0 - 1.0
    final_answer_completeness: float  # 0.0 - 1.0
    overall_score: float  # 0 - 100
    short_explanation: str
    
    # Meta-Informationen für Debugging
    evaluator_response_raw: Optional[str] = None
    evaluation_error: Optional[str] = None
    evaluation_time: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Konvertiert zu Dictionary für JSON-Export"""
        return {
            "tool_usage_correctness": self.tool_usage_correctness,
            "final_answer_correctness": self.final_answer_correctness,
            "final_answer_completeness": self.final_answer_completeness,
            "overall_score": self.overall_score,
            "short_explanation": self.short_explanation,
            "evaluator_response_raw": self.evaluator_response_raw,
            "evaluation_error": self.evaluation_error,
            "evaluation_time": self.evaluation_time
        }

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
    
    # Evaluierung durch externes Evaluator-Modell
    evaluation_result: Optional[EvaluationResult] = None
    model_initial: str = ""
    model_final: str = ""
    
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
            "response_content": self.response_content,
            "evaluation_result": self.evaluation_result.to_dict() if self.evaluation_result else None,
            "model_initial": self.model_initial,
            "model_final": self.model_final
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
        
        # Evaluator-Konfiguration
        self.evaluator_enabled = extra_settings.get("evaluator_enabled", False)
        self.evaluator_model = extra_settings.get("evaluator_model", "ollama/llama3.2")
        self.evaluator_base_url = extra_settings.get("evaluator_base_url", self.base_url)
        
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
    
    def call_evaluator(self, payload: dict, prompt: str, settings: dict = None, max_retries: int = 5) -> str:
        """
        Ruft das externe Evaluator-LLM auf mit robuster Retry-Logik für JSON-Validierung
        
        Args:
            payload: Dictionary mit den zu evaluierenden Daten
            prompt: System-Prompt für das Evaluator-LLM
            settings: Zusätzliche LLM-Einstellungen
            max_retries: Maximale Anzahl von Wiederholungsversuchen
            
        Returns:
            Raw response string vom Evaluator-LLM (validiertes JSON)
        """
        
        def repair_json_syntax(json_str: str) -> str:
            """Repariert häufige JSON-Syntaxfehler"""
            import re
            
            # Entferne trailing commas
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
            
            # Füge fehlende Anführungszeichen um Feldnamen hinzu
            json_str = re.sub(r'(\w+)(\s*:)', r'"\1"\2', json_str)
            
            # Stelle sicher, dass String-Werte in Anführungszeichen stehen
            # Aber nicht double-quote bereits quoted strings
            json_str = re.sub(r':\s*([^"\d\[\]{},\s][^,\}\]]*)', r': "\1"', json_str)
            
            # Korrigiere falsche Anführungszeichen (curly quotes zu straight quotes)
            json_str = json_str.replace('"', '"').replace('"', '"')
            json_str = json_str.replace(''', "'").replace(''', "'")
            
            # Entferne doppelte Anführungszeichen 
            json_str = re.sub(r'""([^"]*?)""', r'"\1"', json_str)
            
            return json_str
        
        if settings is None:
            settings = {}
            
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                # Deterministische Einstellungen für Evaluator
                evaluator_params = {
                    "model": self.evaluator_model,
                    "temperature": 0.0,  # Maximale Determinismus
                    "top_p": 0.1,  # Sehr fokussierte Antworten
                    "repeat_penalty": 1.1,  # Verhindere Wiederholungen
                    "max_tokens": 300,  # Mehr Platz für sauberes JSON
                    "timeout": self.timeout or 60,
                    "stop": ["\n\n", "```"]  # Stoppe bei Markdown oder doppelten Newlines
                }
                
                if self.evaluator_base_url:
                    evaluator_params["base_url"] = self.evaluator_base_url
                if self.api_key:
                    evaluator_params["api_key"] = self.api_key
                    
                # Zusätzliche Settings übernehmen
                evaluator_params.update(settings)
                
                # Messages aufbauen - User Message betont nochmal JSON-only
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"""ANTWORTE NUR MIT REINEM JSON - NICHTS ANDERES!

Das JSON muss EXAKT diese 5 Felder haben:
- tool_usage_correctness (Zahl zwischen 0.0 und 1.0)
- final_answer_correctness (Zahl zwischen 0.0 und 1.0)  
- final_answer_completeness (Zahl zwischen 0.0 und 1.0)
- overall_score (Ganzzahl zwischen 0 und 100)
- short_explanation (String auf Deutsch)

WICHTIG: Verwende ANFÜHRUNGSZEICHEN um alle String-Werte!

Bewerte diese Interaktion:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Antwort als gültiges JSON:"""}
                ]
                
                response = litellm.completion(
                    messages=messages,
                    stream=False,
                    **evaluator_params
                )
                
                content = response["choices"][0]["message"]["content"] or ""
                
                # Bereinige Response von möglichen Formatierungen
                content = content.strip()
                
                # Entferne Markdown-Code-Blöcke falls vorhanden
                if content.startswith("```json"):
                    content = content[7:]
                elif content.startswith("```"):
                    content = content[3:]
                    
                if content.endswith("```"):
                    content = content[:-3]
                    
                content = content.strip()
                
                # JSON-Validierung - das ist der kritische Teil!
                try:
                    import re  # Import für JSON-Bereinigung
                    
                    # Zusätzliche Bereinigung vor JSON-Parsing
                    content_cleaned = content.strip()
                    
                    # Entferne häufige Störungen
                    content_cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content_cleaned)  # Steuerzeichen
                    content_cleaned = content_cleaned.replace('\r\n', '\n').replace('\r', '\n')  # Zeilenenden
                    
                    # Spezielle Bereinigung für LiteLLM Debug-Ausgaben
                    lines = content_cleaned.split('\n')
                    json_lines = []
                    in_json = False
                    brace_count = 0
                    
                    for line in lines:
                        line = line.strip()
                        # Skip debug lines
                        if ('11111111' in line or 'RAW response' in line or 'transform=' in line or
                            'INFO:' in line or 'HTTP Request:' in line):
                            continue
                            
                        # Find JSON start
                        if line.startswith('{') and not in_json:
                            in_json = True
                            json_lines = [line]
                            brace_count = line.count('{') - line.count('}')
                        elif in_json:
                            json_lines.append(line)
                            brace_count += line.count('{') - line.count('}')
                            if brace_count <= 0:
                                break
                    
                    if json_lines:
                        content_cleaned = '\n'.join(json_lines)
                    
                    # Remove trailing "None" or similar artifacts
                    content_cleaned = re.sub(r'\s*None\s*$', '', content_cleaned).strip()
                    
                    # Zusätzliche JSON-Reparatur für häufige Fehler
                    content_cleaned = repair_json_syntax(content_cleaned)
                    
                    logger.debug(f"JSON vor Parsing (Versuch {attempt + 1}): '{content_cleaned}'")
                    
                    parsed_json = json.loads(content_cleaned)
                    
                    # Überprüfe ob alle erforderlichen Felder vorhanden sind
                    required_fields = [
                        "tool_usage_correctness", "final_answer_correctness", 
                        "final_answer_completeness", "overall_score", "short_explanation"
                    ]
                    
                    missing_fields = [field for field in required_fields if field not in parsed_json]
                    
                    if missing_fields:
                        if attempt < max_retries:
                            logger.warning(f"Evaluator-JSON unvollständig (Versuch {attempt + 1}/{max_retries + 1}). Fehlende Felder: {missing_fields}")
                            time.sleep(0.5)
                            continue
                        else:
                            logger.error(f"Evaluator-JSON nach {max_retries + 1} Versuchen unvollständig: {missing_fields}")
                            return content  # Geben zurück was wir haben
                    
                    # JSON ist gültig und vollständig
                    logger.debug(f"Evaluator-JSON erfolgreich validiert (Versuch {attempt + 1})")
                    return content_cleaned  # Gib das bereinigte JSON zurück
                    
                except json.JSONDecodeError as json_err:
                    if attempt < max_retries:
                        logger.warning(f"Evaluator-Antwort ungültiges JSON (Versuch {attempt + 1}/{max_retries + 1}): {str(json_err)}")
                        logger.debug(f"Rohe Antwort: {content[:200]}...")
                        logger.debug(f"Bereinigte Antwort: {content_cleaned[:200] if 'content_cleaned' in locals() else 'N/A'}...")
                        time.sleep(0.5)  # Kurz warten vor nächstem Versuch
                        continue
                    else:
                        logger.error(f"Evaluator-JSON nach {max_retries + 1} Versuchen ungültig: {str(json_err)}")
                        logger.error(f"Letzte rohe Antwort: {content}")
                        logger.error(f"Letzte bereinigte Antwort: {content_cleaned if 'content_cleaned' in locals() else 'N/A'}")
                        last_error = f"JSON Parse Error: {str(json_err)}"
                
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    logger.warning(f"Evaluator-Fehler (Versuch {attempt + 1}/{max_retries + 1}): {e}")
                    time.sleep(1.0)  # Längere Pause bei echten Fehlern
                    continue
                else:
                    logger.error(f"Evaluator fehlgeschlagen nach {max_retries + 1} Versuchen: {e}")
        
        return f"EVALUATOR_ERROR: {last_error or 'Unbekannter Fehler nach mehreren Versuchen'}"
    
    def evaluate_interaction(
        self,
        original_prompt: str,
        model_initial: str,
        tool_call_json: Union[dict, str],
        tool_response: Union[dict, str],
        model_final: str
    ) -> EvaluationResult:
        """
        Evaluiert eine vollständige LLM-Tool-Interaktion durch externes Evaluator-LLM
        
        Args:
            original_prompt: Ursprünglicher User-Prompt
            model_initial: Erste LLM-Antwort (die Tool-Aufruf initiiert)
            tool_call_json: JSON des Tool-Aufrufs
            tool_response: Antwort des Tools
            model_final: Finale LLM-Antwort nach Tool-Nutzung
            
        Returns:
            EvaluationResult mit detaillierter Bewertung
        """
        start_time = time.time()
        
        # Evaluator-Prompt (fokussiert nur auf Tool-Nutzung und Datenverwendung)
        evaluator_system_prompt = """Du bist ein JSON-Evaluator. Deine einzige Aufgabe ist es, EXAKT gültiges JSON zu produzieren.

KRITISCH: Du MUSST mit genau diesem JSON-Format antworten:

{"tool_usage_correctness": 1.0, "final_answer_correctness": 1.0, "final_answer_completeness": 1.0, "overall_score": 100, "short_explanation": "Deutsche Erklärung hier"}

REGELN:
1. EXAKT 5 Felder, nichts anderes
2. Zahlen ohne Anführungszeichen 
3. Strings MIT Anführungszeichen
4. Kein Text außerhalb des JSON
5. Keine Kommentare, kein Markdown

BEWERTUNG:
- tool_usage_correctness: Hat LLM Tool richtig genutzt? (0.0-1.0)
- final_answer_correctness: Alle Tool-Fakten korrekt übernommen? (0.0-1.0) 
- final_answer_completeness: Alle relevanten Tool-Daten genutzt? (0.0-1.0)
- overall_score: Gewichteter Score 0-100: 40% correctness + 35% completeness + 25% tool_usage
- short_explanation: Deutsche Kurzerklärung

WICHTIG: Sprachliche Ergänzungen sind NORMAL - bewerte nur Tool-Daten-Korrektheit!

Antworte NUR mit dem JSON, nichts anderes!"""
        
        # Payload für Evaluator aufbauen
        payload = {
            "original_prompt": original_prompt,
            "model_initial": model_initial,
            "tool_call_json": tool_call_json,
            "tool_response": tool_response,
            "model_final": model_final
        }
        
        # Evaluator aufrufen
        evaluator_response = self.call_evaluator(payload, evaluator_system_prompt)
        evaluation_time = time.time() - start_time
        
        # Debug: Zeige Evaluator Input und Output
        logger.debug(f"Evaluator Input: {json.dumps(payload, ensure_ascii=False, indent=2)}")
        logger.debug(f"Evaluator Raw Response: {evaluator_response}")
        
        # Antwort parsen
        try:
            # Versuche JSON zu parsen
            if evaluator_response.startswith("EVALUATOR_ERROR:"):
                # Fallback bei Evaluator-Fehler
                logger.warning(f"Evaluator-Fehler, verwende Fallback-Bewertung: {evaluator_response}")
                return EvaluationResult(
                    tool_usage_correctness=0.0,
                    final_answer_correctness=0.0,
                    final_answer_completeness=0.0,
                    overall_score=0.0,
                    short_explanation="Evaluator nicht verfügbar",
                    evaluator_response_raw=evaluator_response,
                    evaluation_error=evaluator_response,
                    evaluation_time=evaluation_time
                )
            
            # JSON parsen mit besserem Error Handling
            evaluator_response = evaluator_response.strip()
            
            # Entferne mögliche Markdown-Code-Blöcke
            if evaluator_response.startswith("```json"):
                evaluator_response = evaluator_response[7:]
            if evaluator_response.endswith("```"):
                evaluator_response = evaluator_response[:-3]
            evaluator_response = evaluator_response.strip()
            
            # Entferne potentielle unsichtbare Zeichen und normalisiere
            import re
            evaluator_response = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', evaluator_response)  # Entferne Steuerzeichen
            evaluator_response = evaluator_response.replace('\r\n', '\n').replace('\r', '\n')  # Normalisiere Zeilenenden
            
            # Zusätzliche Bereinigung für häufige LLM-Fehler
            evaluator_response = re.sub(r'(?m)^[^{]*({.*})[^}]*$', r'\1', evaluator_response)  # Extrahiere nur JSON-Block
            evaluator_response = evaluator_response.strip()
            
            # Spezielle Bereinigung für Debug-Ausgaben in LiteLLM
            # Entferne Zeilen mit "11111111111111" oder "RAW response" oder "transform="
            lines = evaluator_response.split('\n')
            clean_lines = []
            json_started = False
            json_content = []
            brace_count = 0
            
            for line in lines:
                line = line.strip()
                # Überspringe Debug-Zeilen
                if ('11111111' in line or 
                    'RAW response' in line or 
                    'transform=' in line or
                    'INFO:' in line or
                    'HTTP Request:' in line):
                    continue
                    
                # Finde JSON-Start
                if line.startswith('{') and not json_started:
                    json_started = True
                    json_content = [line]
                    brace_count = line.count('{') - line.count('}')
                elif json_started:
                    json_content.append(line)
                    brace_count += line.count('{') - line.count('}')
                    # JSON-Ende erreicht
                    if brace_count <= 0:
                        break
                        
            if json_content:
                evaluator_response = '\n'.join(json_content).strip()
            else:
                # Fallback: Normale Bereinigung
                for line in lines:
                    line = line.strip()
                    if ('11111111' not in line and 
                        'RAW response' not in line and 
                        'transform=' not in line and
                        'None' != line.strip() and
                        'INFO:' not in line and
                        line):  # Nicht-leere Zeilen
                        clean_lines.append(line)
                evaluator_response = '\n'.join(clean_lines).strip()
            
            # Final cleaning: Entferne "None" am Ende falls noch vorhanden
            evaluator_response = re.sub(r'\s*None\s*$', '', evaluator_response).strip()
            
            # Debug-Ausgabe für Problembehebung
            logger.debug(f"Bereinigte Evaluator-Antwort: '{evaluator_response}'")
            
            result_data = json.loads(evaluator_response)
            
            # Validiere erforderliche Felder mit Fallback-Werten
            required_fields = {
                "tool_usage_correctness": 0.0,
                "final_answer_correctness": 0.0,
                "final_answer_completeness": 0.0,
                "overall_score": 0.0,
                "short_explanation": "Unvollständige Evaluator-Antwort"
            }
            
            # Fehlende Felder mit Fallback-Werten füllen
            for field, default_value in required_fields.items():
                if field not in result_data:
                    logger.warning(f"Fehlendes Evaluator-Feld '{field}', verwende Fallback: {default_value}")
                    result_data[field] = default_value
            
            return EvaluationResult(
                tool_usage_correctness=result_data["tool_usage_correctness"],
                final_answer_correctness=result_data["final_answer_correctness"],
                final_answer_completeness=result_data["final_answer_completeness"],
                overall_score=result_data["overall_score"],
                short_explanation=result_data["short_explanation"],
                evaluator_response_raw=evaluator_response,
                evaluation_error=None,
                evaluation_time=evaluation_time
            )
            
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.error(f"Fehler beim Parsen der Evaluator-Antwort: {e}")
            logger.error(f"Rohe Evaluator-Antwort: '{evaluator_response}'")
            logger.error(f"Bereinigte Evaluator-Antwort: '{evaluator_response}'")
            
            # Versuche manuelle JSON-Reparatur bei häufigen Fehlern
            try:
                import re
                
                # Häufiger Fehler: Trailing comma oder fehlende quotes
                repaired = evaluator_response
                
                # Versuche zunächst die repair_json_syntax Funktion
                def repair_json_syntax_local(json_str: str) -> str:
                    """Lokale Kopie der JSON-Reparatur-Funktion"""
                    # Entferne trailing commas
                    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                    
                    # Korrigiere fehlerhafte Anführungszeichen
                    json_str = json_str.replace('"', '"').replace('"', '"')
                    json_str = json_str.replace(''', "'").replace(''', "'")
                    
                    # Template-basierte Reparatur für bekannte Fehler
                    if "tool_usage_correctness" in json_str:
                        # Extrahiere Werte mit Regex
                        tool_usage = re.search(r'"?tool_usage_correctness"?\s*:\s*([0-9.]+)', json_str)
                        final_correctness = re.search(r'"?final_answer_correctness"?\s*:\s*([0-9.]+)', json_str)  
                        final_completeness = re.search(r'"?final_answer_completeness"?\s*:\s*([0-9.]+)', json_str)
                        overall = re.search(r'"?overall_score"?\s*:\s*([0-9]+)', json_str)
                        explanation = re.search(r'"?short_explanation"?\s*:\s*"?([^"]*)"?', json_str)
                        
                        # Falls alle Werte gefunden wurden, baue sauberes JSON
                        if all([tool_usage, final_correctness, final_completeness, overall, explanation]):
                            return json.dumps({
                                "tool_usage_correctness": float(tool_usage.group(1)),
                                "final_answer_correctness": float(final_correctness.group(1)),
                                "final_answer_completeness": float(final_completeness.group(1)),
                                "overall_score": int(overall.group(1)),
                                "short_explanation": explanation.group(1).strip('"\'')
                            }, ensure_ascii=False)
                    
                    return json_str
                
                repaired = repair_json_syntax_local(repaired)
                
                # Versuche nochmal zu parsen
                result_data = json.loads(repaired)
                logger.info("JSON erfolgreich durch erweiterte Reparatur gefixt")
                
                # Validiere erforderliche Felder mit Fallback-Werten
                required_fields = {
                    "tool_usage_correctness": 0.0,
                    "final_answer_correctness": 0.0,
                    "final_answer_completeness": 0.0,
                    "overall_score": 0.0,
                    "short_explanation": "Reparierte Evaluator-Antwort"
                }
                
                # Fehlende Felder mit Fallback-Werten füllen
                for field, default_value in required_fields.items():
                    if field not in result_data:
                        logger.warning(f"Fehlendes Evaluator-Feld '{field}', verwende Fallback: {default_value}")
                        result_data[field] = default_value
                
                return EvaluationResult(
                    tool_usage_correctness=result_data["tool_usage_correctness"],
                    final_answer_correctness=result_data["final_answer_correctness"],
                    final_answer_completeness=result_data["final_answer_completeness"],
                    overall_score=result_data["overall_score"],
                    short_explanation=result_data["short_explanation"],
                    evaluator_response_raw=evaluator_response,
                    evaluation_error=f"Repariert: {str(e)}",
                    evaluation_time=evaluation_time
                )
                
            except Exception as repair_error:
                logger.error(f"JSON-Reparatur fehlgeschlagen: {repair_error}")
            
            # Fallback-Evaluierung bei Parse-Fehler
            return EvaluationResult(
                tool_usage_correctness=0.0,
                final_answer_correctness=0.0,
                final_answer_completeness=0.0,
                overall_score=0.0,
                short_explanation="Evaluator-Antwort ungültig",
                evaluator_response_raw=evaluator_response,
                evaluation_error=f"Parse-Fehler: {str(e)}",
                evaluation_time=evaluation_time
            )
        
        except Exception as e:
            # Absoluter Fallback für unerwartete Fehler
            logger.error(f"Unerwarteter Fehler in evaluate_interaction: {e}")
            return EvaluationResult(
                tool_usage_correctness=0.0,
                final_answer_correctness=0.0,
                final_answer_completeness=0.0,
                overall_score=0.0,
                short_explanation="Evaluierung fehlgeschlagen",
                evaluator_response_raw=evaluator_response if 'evaluator_response' in locals() else None,
                evaluation_error=str(e),
                evaluation_time=evaluation_time
            )
    
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
            
            # Model initial response speichern (vor Tool-Ausführung)
            # Bei Tool-Calls ist response_content oft leer - das ist normales Verhalten
            # da das LLM direkt zum Tool-Call übergeht ohne zusätzlichen Text
            result.model_initial = response_content if response_content else ""
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
                        tool_response = execute_tool_fn(function_name, arguments)
                        result.tool_execution_time = time.time() - tool_start_time
                    except Exception as e:
                        logger.warning(f"Tool-Ausführung fehlgeschlagen: {e}")
                        result.tool_execution_time = time.time() - tool_start_time
                        tool_response = {"error": str(e)}
                
                # Zweiten LLM-Aufruf für finale Antwort (mit Tool-Response)
                try:
                    # Conversation-History für finalen Aufruf aufbauen
                    final_messages = messages.copy()
                    
                    # Erste LLM-Antwort mit Tool-Call hinzufügen
                    assistant_msg = {"role": "assistant", "content": response_content}
                    if tool_calls:
                        assistant_msg["tool_calls"] = tool_calls
                    final_messages.append(assistant_msg)
                    
                    # Tool-Response hinzufügen
                    tool_msg = {
                        "role": "tool", 
                        "tool_call_id": tool_call_id,
                        "content": json.dumps(tool_response, ensure_ascii=False) if tool_response else "Tool execution failed"
                    }
                    final_messages.append(tool_msg)
                    
                    # Finalen LLM-Aufruf machen (ohne Tools, da schon verwendet)
                    final_params = self._build_completion_params(
                        messages=final_messages,
                        stream=False
                    )
                    
                    final_response = litellm.completion(**final_params)
                    result.model_final = final_response["choices"][0]["message"]["content"] or ""
                    
                    logger.debug(f"Finale LLM-Antwort: {result.model_final[:100]}...")
                    
                except Exception as e:
                    logger.warning(f"Finaler LLM-Aufruf fehlgeschlagen: {e}")
                    # Fallback: Generiere eine Antwort basierend auf Tool-Response
                    if tool_response:
                        if isinstance(tool_response, dict) and "error" not in tool_response:
                            result.model_final = f"Basierend auf den Tool-Daten: {json.dumps(tool_response, ensure_ascii=False)}"
                        else:
                            result.model_final = f"Das Tool meldete einen Fehler: {tool_response}"
                    else:
                        result.model_final = "Keine Tool-Response erhalten"
                
                # Evaluierung durch externes Evaluator-LLM (falls aktiviert)
                if self.evaluator_enabled:
                    try:
                        logger.debug("Starte Evaluator-Aufruf...")
                        evaluation = self.evaluate_interaction(
                            original_prompt=test_case.prompt,
                            model_initial=result.model_initial,
                            tool_call_json=first_tool_call if isinstance(first_tool_call, dict) else {"name": function_name, "arguments": arguments},
                            tool_response=tool_response,
                            model_final=result.model_final
                        )
                        result.evaluation_result = evaluation
                        
                        logger.info(f"Evaluator-Score: {evaluation.overall_score:.1f}/100")
                        
                    except Exception as e:
                        logger.error(f"Evaluierung fehlgeschlagen: {e}")
                        # Erstelle Fallback-Evaluierung, damit der Benchmark weiterläuft
                        result.evaluation_result = EvaluationResult(
                            tool_usage_correctness=0.0,
                            final_answer_correctness=0.0,
                            final_answer_completeness=0.0,
                            overall_score=0.0,
                            short_explanation="Evaluierung fehlgeschlagen",
                            evaluator_response_raw=None,
                            evaluation_error=str(e),
                            evaluation_time=0.0
                        )
                
                logger.info(f"Tool-Call: {function_name} {'✓' if result.correct_tool_called else '✗'}")
                logger.info(f"Parameter: {result.parameter_accuracy:.1%} korrekt {'✓' if result.correct_parameters else '✗'}")
            else:
                logger.warning("Kein Tool-Call erkannt")
                result.actual_tool_call = None
                result.actual_parameters = {}
                result.model_final = response_content
            
        except Exception as e:
            error = str(e)
            result.error = error
            logger.error(f"Fehler im Test: {error}")
        
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
        
        logger.info(f"\nStarte Benchmark-Suite für {self.model}")
        logger.info(f"   Provider: {provider}")
        logger.info(f"   Test-Cases: {len(test_cases)}")
        logger.info(f"   Wiederholungen: {repetition_rounds}")
        logger.info(f"   Gesamt-Tests: {len(test_cases) * repetition_rounds}")
        
        for test_case in test_cases:
            logger.info(f"\nTest-Case: {test_case.name or 'Unnamed'}")
            
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
        
        logger.info(f"\nBenchmark-Suite abgeschlossen: {len(all_results)} Tests")
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
            print(f"\nModell: {model_key}")
            print("-" * 60)
            
            all_model_results = [r for test_list in test_results.values() for r in test_list]
            successful = [r for r in all_model_results if r.error is None]
            failed = [r for r in all_model_results if r.error is not None]
            
            print(f"Gesamt: {len(successful)}/{len(all_model_results)} erfolgreich")
            
            if successful:
                # Timing-Statistiken
                avg_response_time = sum(r.response_time for r in successful) / len(successful)
                first_tool_times = [r.first_tool_call_time for r in successful if r.first_tool_call_time]
                avg_first_tool_time = sum(first_tool_times) / len(first_tool_times) if first_tool_times else 0
                
                # Tool-Call-Genauigkeit
                correct_tools = sum(1 for r in successful if r.correct_tool_called)
                correct_params = sum(1 for r in successful if r.correct_parameters)
                avg_param_accuracy = sum(r.parameter_accuracy for r in successful) / len(successful)
                
                print(f"Durchschnittliche Antwortzeit: {avg_response_time:.3f}s")
                print(f"Durchschnittliche Zeit bis Tool-Call: {avg_first_tool_time:.3f}s")
                print(f"Tool-Call-Genauigkeit: {correct_tools}/{len(successful)} ({correct_tools/len(successful):.1%})")
                print(f"Parameter-Genauigkeit: {correct_params}/{len(successful)} ({correct_params/len(successful):.1%})")
                print(f"Durchschnittliche Parameter-Korrektheit: {avg_param_accuracy:.1%}")
                
                # Token-Statistiken (falls verfügbar)
                token_results = [r for r in successful if r.tokens_used is not None]
                if token_results:
                    avg_tokens = sum(r.tokens_used for r in token_results) / len(token_results)
                    print(f"Durchschnittliche Tokens: {avg_tokens:.0f}")
                
                # Evaluator-Statistiken (falls verfügbar)
                eval_results = [r for r in successful if r.evaluation_result is not None]
                if eval_results:
                    avg_eval_score = sum(r.evaluation_result.overall_score for r in eval_results) / len(eval_results)
                    print(f"Durchschnittlicher Evaluator-Score: {avg_eval_score:.1f}/100")
            
            # Detaillierte Test-Case-Statistiken
            print(f"\nTest-Case Details:")
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
                print(f"     Tools: {correct_tools}/{len(test_successful)} korrekt ({correct_tools/len(test_successful):.1%})")
                print(f"     Parameter: {avg_param_acc:.1%} korrekt im Durchschnitt")
                print(f"     Zeit: {avg_time:.3f}s durchschnittlich")
                
                # Zeige häufigste Fehler bei Tool-Calls
                if correct_tools < len(test_successful):
                    wrong_tools = [r.actual_tool_call for r in test_successful if not r.correct_tool_called and r.actual_tool_call]
                    if wrong_tools:
                        from collections import Counter
                        most_common_wrong = Counter(wrong_tools).most_common(2)
                        wrong_tools_str = ", ".join([f"{tool} ({count}x)" for tool, count in most_common_wrong])
                        print(f"     Häufige falsche Tools: {wrong_tools_str}")
                
                # Evaluator-Details für diesen Test-Case (falls verfügbar)
                eval_results = [r for r in test_successful if r.evaluation_result is not None]
                if eval_results:
                    avg_eval = sum(r.evaluation_result.overall_score for r in eval_results) / len(eval_results)
                    print(f"     Evaluator-Score: {avg_eval:.1f}/100 durchschnittlich")
            
            if failed:
                print(f"\nFehler ({len(failed)}):")
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
    
    logger.info(f"\nSTARTE MCP BENCHMARK SUITE")
    logger.info(f"   Modelle: {len(models)}")
    logger.info(f"   Test-Cases: {len(test_cases)}")
    logger.info(f"   Wiederholungen: {repetition_rounds}")
    logger.info(f"   Gesamt-Tests: {len(models) * len(test_cases) * repetition_rounds}")
    
    for model_config in models:
        try:
            # LLM-Instanz erstellen
            llm = MCPBenchmarkLLM(**model_config["config"])
            logger.info(f"\nModell geladen: {model_config['name']}")
            
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
            logger.error(f"Fehler bei Modell {model_config['name']}: {e}")
    
    # Ergebnisse anzeigen
    suite.print_summary()
    
    # Ergebnisse exportieren
    timestamp = int(time.time())
    filename = f"mcp_benchmark_results_{timestamp}.json"
    suite.export_results(filename)
    logger.info(f"\nDetaillierte Ergebnisse exportiert nach: {filename}")
    
    return all_results

def test_evaluation_examples():
    """
    Test-Funktion für die Evaluator-Integration mit den spezifizierten Beispielen
    """
    print("\nTeste Evaluator-Integration mit Beispielen...")
    
    # Test-LLM mit Mock-Evaluator
    llm = MCPBenchmarkLLM(
        model="test/model",
        evaluator_enabled=True,
        evaluator_model="test/evaluator"
    )
    
    # Mock für den Evaluator-Aufruf
    def mock_evaluator_response(self, payload, prompt, settings=None):
        """Mock-Funktion für Evaluator-Tests"""
        original_prompt = payload.get("original_prompt", "")
        tool_response = payload.get("tool_response", {})
        model_final = payload.get("model_final", "")
        
        # Beispiel 1: Korrekte Interaktion
        if "example.com" in original_prompt and "93.184.216.34" in str(tool_response):
            return json.dumps({
                "tool_usage_correctness": 1.0,
                "final_answer_correctness": 1.0,
                "final_answer_completeness": 1.0,
                "overall_score": 100,
                "short_explanation": "Tool wurde korrekt aufgerufen und Antwort konsistent verwendet."
            })
        
        # Beispiel 2: Timeout (Tool-Fehler)
        elif "timeout" in str(tool_response) and "12.345" in model_final:
            return json.dumps({
                "tool_usage_correctness": 1.0,
                "final_answer_correctness": 0.0,
                "final_answer_completeness": 0.0,
                "overall_score": 25,
                "short_explanation": "Tool korrekt aufgerufen, aber falsche Daten in finaler Antwort verwendet."
            })
        
        # Beispiel 3: Ungültiges JSON
        else:
            return "I'm broken"  # Simuliert kaputten Evaluator
    
    # Mock-Funktion temporär einsetzen
    original_call_evaluator = llm.call_evaluator
    llm.call_evaluator = lambda payload, prompt, settings=None: mock_evaluator_response(llm, payload, prompt, settings)
    
    try:
        # Test 1: Korrekte Interaktion
        print("\nTest 1: Korrekte DNS-Lookup-Interaktion")
        result1 = llm.evaluate_interaction(
            original_prompt="Zeige mir die aktuelle IP einer Domain example.com via dns_lookup tool.",
            model_initial="Ich rufe dns_lookup mit domain='example.com' auf.",
            tool_call_json={"name":"dns_lookup","arguments":{"domain":"example.com"}},
            tool_response={"ip":"93.184.216.34","status":"ok"},
            model_final="Die IP von example.com ist 93.184.216.34."
        )
        print(f"Overall Score: {result1.overall_score}/100")
        print(f"Halluzination: Entfernt (nicht mehr bewertet)")
        assert result1.overall_score == 100, f"Erwarteter Score 100, erhalten: {result1.overall_score}"
        
        # Test 2: Timeout + falsche Datennutzung
        print("\nTest 2: Database-Query mit Timeout und falscher Datennutzung")
        result2 = llm.evaluate_interaction(
            original_prompt="Gib die Anzahl Nutzer aus DBX.",
            model_initial="Ich rufe db_query mit query='SELECT COUNT(*) FROM users' auf.",
            tool_call_json={"name":"db_query","arguments":{"query":"SELECT COUNT(*) FROM users"}},
            tool_response={"error":"timeout"},
            model_final="Die Anzahl Nutzer ist 12.345."
        )
        print(f"Overall Score: {result2.overall_score}/100")
        print(f"Tool Usage: {result2.tool_usage_correctness}")
        assert result2.overall_score == 25, f"Erwarteter Score 25, erhalten: {result2.overall_score}"
        
        # Test 3: Kaputte Evaluator-Antwort
        print("\nTest 3: Ungültiges Evaluator-JSON")
        result3 = llm.evaluate_interaction(
            original_prompt="Beliebiger Test",
            model_initial="Test",
            tool_call_json={"name":"test","arguments":{}},
            tool_response={"result":"ok"},
            model_final="Test Antwort"
        )
        print(f"Overall Score: {result3.overall_score}/100")
        print(f"Evaluation Error: {result3.evaluation_error is not None}")
        assert result3.overall_score == 0, f"Bei Parse-Fehler sollte Score 0 sein, erhalten: {result3.overall_score}"
        assert result3.evaluation_error is not None, "Evaluation Error sollte gesetzt sein"
        
        print("\nAlle Evaluator-Tests erfolgreich!")
        
    finally:
        # Mock-Funktion zurücksetzen
        llm.call_evaluator = original_call_evaluator

if __name__ == "__main__":
    # Einfaches Test-Beispiel der neuen Benchmark-Funktionalität
    logger.info("MCPBenchmarkLLM Test - Fokus auf erwartete Tool-Calls mit Evaluator")
    
    # Zuerst die Evaluator-Tests ausführen
    test_evaluation_examples()
    
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
        if function_name == "get_weather":
            city = arguments.get("city", "unknown")
            return {"temperature": 22, "condition": "sunny", "city": city}
        return {"result": f"Mock result for {function_name} with {arguments}"}
    
    # Modell-Konfiguration
    models = [
        {
            "name": "llama3.2",
            "provider": "ollama",
            "config": {
                "model": "ollama/llama3.2",
                "base_url": "http://localhost:11434",
                "evaluator_enabled": True,  # Evaluator aktivieren
                "evaluator_model": "ollama/llama3.2",
                "evaluator_base_url": "http://localhost:11434"
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
        print(f"\nTest abgeschlossen mit {len(results)} Ergebnissen")
        
    except Exception as e:
        print(f"Test-Fehler: {e}")
        import traceback
        traceback.print_exc()
