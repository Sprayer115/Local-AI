#!/usr/bin/env python3
"""
Robuster MCP-Evaluator mit fortgeschrittener LLM-basierter Evaluierung
Speziell optimiert für MCP Tool-Call-Evaluierung mit verbesserter Robustheit
"""

import json
import logging
import time
import re
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import warnings

# LiteLLM für LLM-Aufrufe
import os
os.environ["LITELLM_TELEMETRY"] = "False"
import litellm

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationCriteria(Enum):
    """Vordefinierte Evaluierungskriterien für MCP-Benchmarks"""
    TOOL_USAGE_CORRECTNESS = "tool_usage_correctness"
    ANSWER_CORRECTNESS = "answer_correctness"
    ANSWER_COMPLETENESS = "answer_completeness"
    FAITHFULNESS = "faithfulness"
    COHERENCE = "coherence"
    SAFETY = "safety"

@dataclass
class EvaluationStep:
    """Einzelner Evaluierungsschritt für strukturierte Bewertung"""
    step_number: int
    description: str
    weight: float = 1.0  # Gewichtung für finalen Score

@dataclass
class TestCase:
    """Einzelner Test-Case mit erwarteten Ergebnissen"""
    prompt: str
    expected_tool_call: str
    expected_parameters: Dict[str, Any]
    system_prompt: Optional[str] = None
    name: Optional[str] = None

@dataclass
class MCPEvaluationResult:
    """Detailliertes Evaluierungsergebnis mit strukturierter Bewertung"""
    
    # Hauptmetriken (0.0 - 1.0)
    tool_usage_correctness: float
    answer_correctness: float
    answer_completeness: float
    overall_score: float  # 0 - 100
    
    # Detaillierte Begründung
    reasoning: str
    evaluation_steps_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Meta-Informationen
    evaluator_model: str = ""
    evaluation_time: float = 0.0
    evaluation_steps: List[str] = field(default_factory=list)
    
    # Debug-Informationen
    raw_evaluator_response: Optional[str] = None
    evaluation_error: Optional[str] = None
    retry_count: int = 0
    
    # Tool-Interaktions-Daten
    tool_response: Optional[Union[Dict, str]] = None
    actual_tool_call: Optional[str] = None
    actual_parameters: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary für JSON-Export"""
        return {
            "tool_usage_correctness": self.tool_usage_correctness,
            "answer_correctness": self.answer_correctness,
            "answer_completeness": self.answer_completeness,
            "overall_score": self.overall_score,
            "reasoning": self.reasoning,
            "evaluation_steps_results": self.evaluation_steps_results,
            "evaluator_model": self.evaluator_model,
            "evaluation_time": self.evaluation_time,
            "evaluation_steps": self.evaluation_steps,
            "raw_evaluator_response": self.raw_evaluator_response,
            "evaluation_error": self.evaluation_error,
            "retry_count": self.retry_count,
            "tool_response": self.tool_response,
            "actual_tool_call": self.actual_tool_call,
            "actual_parameters": self.actual_parameters
        }

@dataclass
class MultiModelEvaluationResult:
    """Ergebnis-Container für Multi-Model-Evaluierung"""
    
    # Einzelne Evaluierungen pro Modell
    model_evaluations: List[MCPEvaluationResult] = field(default_factory=list)
    
    # Meta-Informationen
    models_used: List[str] = field(default_factory=list)
    successful_evaluations: int = 0
    failed_evaluations: int = 0
    total_evaluation_time: float = 0.0
    
    # Aggregierte Statistiken (optional, können später berechnet werden)
    average_scores: Optional[Dict[str, float]] = None
    
    def add_evaluation(self, result: MCPEvaluationResult, model_name: str):
        """Fügt eine Evaluierung hinzu"""
        self.model_evaluations.append(result)
        if model_name not in self.models_used:
            self.models_used.append(model_name)
        
        if result.evaluation_error is None:
            self.successful_evaluations += 1
        else:
            self.failed_evaluations += 1
            
        self.total_evaluation_time += result.evaluation_time
    
    def get_evaluation_by_model(self, model_name: str) -> Optional[MCPEvaluationResult]:
        """Holt Evaluierung für spezifisches Modell"""
        for eval_result in self.model_evaluations:
            if eval_result.evaluator_model == model_name:
                return eval_result
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary für JSON-Export"""
        return {
            "model_evaluations": [eval_result.to_dict() for eval_result in self.model_evaluations],
            "models_used": self.models_used,
            "successful_evaluations": self.successful_evaluations,
            "failed_evaluations": self.failed_evaluations,
            "total_evaluation_time": self.total_evaluation_time,
            "average_scores": self.average_scores
        }

class BaseEvaluationMetric(ABC):
    """Basis-Klasse für fortgeschrittene Evaluierungsmetriken"""
    
    def __init__(
        self,
        name: str,
        criteria: str,
        evaluation_steps: Optional[List[str]] = None,
        threshold: float = 0.5,
        model: str = "ollama/llama3.2"
    ):
        self.name = name
        self.criteria = criteria
        self.evaluation_steps = evaluation_steps or []
        self.threshold = threshold
        self.model = model
        
    @abstractmethod
    def evaluate(self, interaction_data: Dict[str, Any]) -> float:
        """Evaluiert die Interaktion und gibt Score zurück"""
        pass

class MCPToolCorrectnessMetric(BaseEvaluationMetric):
    """Spezielle Metrik für Tool-Call-Korrektheit"""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="Tool Usage Correctness",
            criteria="Evaluate if the correct tool was called with appropriate parameters",
            evaluation_steps=[
                "Analyze if the chosen tool matches the user's request",
                "Verify that parameters are correctly formatted and appropriate",
                "Check if the tool call would logically fulfill the user's intent"
            ],
            **kwargs
        )
    
    def evaluate(self, interaction_data: Dict[str, Any]) -> float:
        # Tool-spezifische Evaluierungslogik
        expected_tool = interaction_data.get("expected_tool_call")
        actual_tool = interaction_data.get("actual_tool_call")
        expected_params = interaction_data.get("expected_parameters", {})
        actual_params = interaction_data.get("actual_parameters", {})
        
        # Tool-Name-Vergleich
        tool_correct = expected_tool == actual_tool if expected_tool else 0.5
        
        # Parameter-Genauigkeit
        param_accuracy = self._calculate_parameter_accuracy(expected_params, actual_params)
        
        # Gewichtete Kombination
        return (tool_correct * 0.6 + param_accuracy * 0.4)
    
    def _calculate_parameter_accuracy(self, expected: Dict, actual: Dict) -> float:
        if not expected:
            return 1.0 if not actual else 0.8
        
        correct = 0
        total = len(expected)
        
        for key, expected_val in expected.items():
            # Case-insensitive key matching
            actual_key = None
            for ak in actual.keys():
                if str(ak).lower() == str(key).lower():
                    actual_key = ak
                    break
            
            if actual_key is not None:
                if self._values_match(expected_val, actual[actual_key]):
                    correct += 1
                elif self._fuzzy_match(str(expected_val), str(actual[actual_key])):
                    correct += 0.8
        
        return correct / total if total > 0 else 0.0
    
    def _fuzzy_match(self, expected: str, actual: str) -> bool:
        """Fuzzy String-Matching für Parameter"""
        return expected.lower() in actual.lower() or actual.lower() in expected.lower()
    
    def _values_match(self, expected, actual) -> bool:
        """Robustes Value-Matching mit Type-Konvertierung"""
        # Exakte Übereinstimmung
        if expected == actual:
            return True
        
        # String-Vergleich (case-insensitive)
        if str(expected).lower() == str(actual).lower():
            return True
        
        # Numerische Konvertierung für int/string Mischungen
        try:
            # Beide zu int konvertieren
            if int(expected) == int(actual):
                return True
        except (ValueError, TypeError):
            pass
        
        try:
            # Beide zu float konvertieren
            if float(expected) == float(actual):
                return True
        except (ValueError, TypeError):
            pass
        
        # Boolean-Konvertierung
        try:
            exp_bool = str(expected).lower() in ['true', '1', 'yes', 'on']
            act_bool = str(actual).lower() in ['true', '1', 'yes', 'on']
            if exp_bool == act_bool:
                return True
        except:
            pass
        
        return False

class MCPAdvancedEvaluator:
    """
    Haupt-Evaluator-Klasse mit fortgeschrittener LLM-basierter Bewertung
    Robuste LLM-basierte Evaluierung für MCP-Interaktionen
    """
    
    def __init__(
        self,
        model: str = "ollama/llama3.2",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 5,
        temperature: float = 0.0  # Deterministische Evaluierung
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        
        # Fortgeschrittene Evaluierungs-Konfiguration
        self.evaluation_steps_cache = {}
        self.metrics_registry = {}
        
        # LiteLLM konfigurieren
        litellm.telemetry = False
        litellm.set_verbose = False
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        # Standard-Metriken registrieren
        self._register_default_metrics()
        
        logger.info(f"MCPAdvancedEvaluator initialisiert mit Modell: {model}")
    
    def _register_default_metrics(self):
        """Registriert Standard-Metriken"""
        self.metrics_registry["tool_correctness"] = MCPToolCorrectnessMetric(
            model=self.model
        )
    
    def _build_llm_params(self, **override_params) -> Dict[str, Any]:
        """Baut LLM-Parameter für sichere Aufrufe"""
        # Sicherstellen, dass bei Ollama die Modellbezeichnung einen Provider-Prefix hat
        model_name = self.model
        if self.base_url and "/" not in model_name:
            # Annahme: base_url -> Ollama
            model_name = f"ollama/{model_name}"

        params = {
            "model": model_name,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "max_tokens": 500,  # Begrenzt für fokussierte Antworten
            "top_p": 0.1,  # Sehr deterministisch
            "stop": ["\n\n", "```", "---"]  # Klare Stopp-Signale
        }
        
        if self.base_url:
            params["base_url"] = self.base_url
        if self.api_key:
            params["api_key"] = self.api_key
            
        params.update(override_params)
        return params
    
    def generate_evaluation_steps(
        self,
        criteria: str,
        evaluation_context: Dict[str, Any]
    ) -> List[str]:
        """
        Generiert strukturierte Evaluierungsschritte basierend auf Kriterien
        """
        cache_key = f"{criteria}:{hash(str(evaluation_context))}"
        if cache_key in self.evaluation_steps_cache:
            return self.evaluation_steps_cache[cache_key]
        
        # System-Prompt für Schritt-Generierung
        system_prompt = """Du bist ein Experte für LLM-Evaluierung. Deine Aufgabe ist es, präzise Evaluierungsschritte zu erstellen.

Erstelle 3-5 spezifische, messbare Evaluierungsschritte basierend auf den gegebenen Kriterien.

Regeln:
1. Jeder Schritt muss objektiv messbar sein
2. Schritte sollen aufeinander aufbauen
3. Fokus auf Tool-Verwendung und Datenqualität
4. Verwende klare, präzise Sprache
5. Antworte NUR mit einer nummerierten Liste

Beispiel-Format:
1. Prüfe ob das richtige Tool aufgerufen wurde
2. Validiere die Parameter-Korrektheit
3. Bewerte die Vollständigkeit der Tool-Response
4. Analysiere die Integration der Tool-Daten in die finale Antwort"""
        
        user_prompt = f"""Kriterien: {criteria}

Kontext:
- User-Prompt: {evaluation_context.get('user_prompt', 'N/A')}
- Erwarteter Tool: {evaluation_context.get('expected_tool', 'N/A')}
- Tatsächlicher Tool: {evaluation_context.get('actual_tool', 'N/A')}

Erstelle spezifische Evaluierungsschritte:"""
        
        try:
            response = self._safe_llm_call(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            steps = self._parse_evaluation_steps(response)
            self.evaluation_steps_cache[cache_key] = steps
            return steps
            
        except Exception as e:
            logger.warning(f"Fehler bei Schritt-Generierung: {e}")
            # Fallback zu Standard-Schritten
            return [
                "Prüfe die Tool-Call-Korrektheit",
                "Validiere die Parameter-Angemessenheit",
                "Bewerte die Antwort-Qualität",
                "Analysiere die Vollständigkeit"
            ]
    
    def _parse_evaluation_steps(self, response: str) -> List[str]:
        """Parst Evaluierungsschritte aus LLM-Response"""
        steps = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # Erkenne numerierte Listen
            if re.match(r'^\d+\.', line):
                step = re.sub(r'^\d+\.\s*', '', line)
                steps.append(step)
            elif line and len(steps) < 6:  # Max 6 Schritte
                steps.append(line)
        
        return steps[:5] if steps else ["Evaluiere die Interaktion holistically"]
    
    def _safe_llm_call(
        self,
        messages: List[Dict[str, str]],
        max_retries: Optional[int] = None
    ) -> str:
        """
        Sicherer LLM-Aufruf mit Retry-Logik und Error-Handling
        """
        retries = max_retries or self.max_retries
        last_error = None
        
        for attempt in range(retries + 1):
            try:
                params = self._build_llm_params(messages=messages)
                # Zusätzliche Härtung: Falls Modell ohne Präfix durchrutscht und base_url gesetzt ist
                if self.base_url and isinstance(params.get("model"), str) and "/" not in params["model"]:
                    params["model"] = f"ollama/{params['model']}"
                    logger.debug(f"Normalisiere Modellbezeichnung für Ollama: {params['model']}")
                response = litellm.completion(**params)
                
                content = response["choices"][0]["message"]["content"]
                if not content or content.strip() == "":
                    raise ValueError("Leere LLM-Antwort")
                
                return content.strip()
                
            except Exception as e:
                last_error = e
                if attempt < retries:
                    wait_time = min(2 ** attempt, 10)  # Exponential backoff
                    logger.warning(f"LLM-Aufruf fehlgeschlagen (Versuch {attempt + 1}/{retries + 1}): {e}")
                    time.sleep(wait_time)
                    continue
        
        # Alle Versuche fehlgeschlagen
        raise Exception(f"LLM-Aufruf nach {retries + 1} Versuchen fehlgeschlagen: {last_error}")
    
    def _create_evaluation_prompt(
        self,
        criteria: str,
        evaluation_steps: List[str],
        interaction_data: Dict[str, Any]
    ) -> Tuple[str, str]:
        """Erstellt System- und User-Prompt für Evaluierung"""
        
        system_prompt = f"""Du bist ein präziser LLM-Evaluator. Deine Aufgabe ist es, MCP Tool-Interaktionen zu bewerten.

WICHTIG: Du MUSST mit einem gültigen JSON antworten, das EXAKT diese Struktur hat:

{{
    "tool_usage_correctness": 0.95,
    "answer_correctness": 0.88,
    "answer_completeness": 0.92,
    "overall_score": 89,
    "reasoning": "Detaillierte deutsche Erklärung der Bewertung",
    "step_results": [
        {{"step": "Schritt 1", "score": 0.9, "explanation": "Begründung"}},
        {{"step": "Schritt 2", "score": 0.8, "explanation": "Begründung"}}
    ]
}}

Bewertungskriterien: {criteria}

Folge diesen Evaluierungsschritten:"""

        for i, step in enumerate(evaluation_steps, 1):
            system_prompt += f"\n{i}. {step}"

        system_prompt += """

Bewertungsrichtlinien:
- tool_usage_correctness: 0.0-1.0 (Wurde das richtige Tool mit korrekten Parametern aufgerufen?)
- answer_correctness: 0.0-1.0 (Sind alle Tool-Daten korrekt in der Antwort verwendet?)
- answer_completeness: 0.0-1.0 (Wurden alle relevanten Tool-Daten verwendet?)
- overall_score: 0-100 (Gewichteter Gesamtscore: 40% tool_usage + 35% correctness + 25% completeness)

Antworte NUR mit gültigem JSON, nichts anderes!"""

        user_prompt = f"""Evaluiere diese MCP Tool-Interaktion:

ORIGINAL PROMPT: {interaction_data.get('original_prompt', 'N/A')}

TOOL-AUFRUF:
- Erwarteter Tool: {interaction_data.get('expected_tool_call', 'N/A')}
- Tatsächlicher Tool: {interaction_data.get('actual_tool_call', 'N/A')}
- Erwartete Parameter: {json.dumps(interaction_data.get('expected_parameters', {}), ensure_ascii=False)}
- Tatsächliche Parameter: {json.dumps(interaction_data.get('actual_parameters', {}), ensure_ascii=False)}

TOOL-RESPONSE: {json.dumps(interaction_data.get('tool_response', {}), ensure_ascii=False)}

FINALE LLM-ANTWORT: {interaction_data.get('model_final', 'N/A')}

Bewerte als JSON:"""

        return system_prompt, user_prompt
    
    def _parse_evaluation_response(
        self,
        response: str,
        interaction_data: Dict[str, Any]
    ) -> MCPEvaluationResult:
        """
        Parst und validiert Evaluator-Response
        Robustes Parsing mit Fallback-Mechanismen
        """
        try:
            # JSON-Bereinigung und -Extraktion
            cleaned_response = self._clean_json_response(response)
            result_data = json.loads(cleaned_response)
            
            # Validierung der Pflichtfelder
            required_fields = {
                "tool_usage_correctness": 0.0,
                "answer_correctness": 0.0,
                "answer_completeness": 0.0,
                "overall_score": 0.0,
                "reasoning": "Keine Begründung verfügbar"
            }
            
            for field, default in required_fields.items():
                if field not in result_data:
                    logger.warning(f"Fehlendes Feld '{field}' in Evaluator-Response, verwende Default: {default}")
                    result_data[field] = default
            
            # Score-Normalisierung und -Validierung
            tool_score = max(0.0, min(1.0, float(result_data["tool_usage_correctness"])))
            answer_score = max(0.0, min(1.0, float(result_data["answer_correctness"])))
            completeness_score = max(0.0, min(1.0, float(result_data["answer_completeness"])))
            overall_score = max(0.0, min(100.0, float(result_data["overall_score"])))
            
            return MCPEvaluationResult(
                tool_usage_correctness=tool_score,
                answer_correctness=answer_score,
                answer_completeness=completeness_score,
                overall_score=overall_score,
                reasoning=str(result_data["reasoning"]),
                evaluation_steps_results=result_data.get("step_results", []),
                evaluator_model=self.model,
                raw_evaluator_response=response,
                # Tool-Interaktions-Daten hinzufügen
                tool_response=interaction_data.get("tool_response"),
                actual_tool_call=interaction_data.get("actual_tool_call"),
                actual_parameters=interaction_data.get("actual_parameters")
            )
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"JSON-Parse-Fehler in Evaluator-Response: {e}")
            logger.debug(f"Rohe Response: {response[:200]}...")
            
            # Versuche regelbasierte Extraktion als Fallback
            return self._fallback_evaluation(response, interaction_data, str(e))
    
    def _clean_json_response(self, response: str) -> str:
        """Bereinigt LLM-Response für JSON-Parsing"""
        # Entferne Markdown-Code-Blöcke
        response = re.sub(r'```(?:json)?\s*', '', response)
        response = re.sub(r'```\s*$', '', response)
        
        # Suche JSON-Block
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            response = json_match.group()
        
        # Bereinige häufige Probleme
        response = re.sub(r',(\s*[}\]])', r'\1', response)  # Trailing commas
        response = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', response)  # Steuerzeichen
        
        return response.strip()
    
    def _fallback_evaluation(
        self,
        raw_response: str,
        interaction_data: Dict[str, Any],
        error_msg: str
    ) -> MCPEvaluationResult:
        """
        Fallback-Evaluierung basierend auf regelbasierten Heuristiken
        Wird verwendet wenn JSON-Parsing fehlschlägt
        """
        logger.warning("Verwende Fallback-Evaluierung aufgrund von Parse-Fehlern")
        
        # Einfache regelbasierte Bewertung
        expected_tool = interaction_data.get("expected_tool_call")
        actual_tool = interaction_data.get("actual_tool_call")
        
        # Tool-Korrektheit
        tool_score = 1.0 if expected_tool == actual_tool else 0.0
        
        # Parameter-Bewertung
        expected_params = interaction_data.get("expected_parameters", {})
        actual_params = interaction_data.get("actual_parameters", {})
        param_score = self._calculate_parameter_match(expected_params, actual_params)
        
        # Antwort-Bewertung (heuristisch)
        model_final = interaction_data.get("model_final", "")
        tool_response = interaction_data.get("tool_response", {})
        
        answer_score = 0.5  # Neutral bei Unsicherheit
        completeness_score = 0.7 if model_final and len(model_final) > 20 else 0.3
        
        # Gesamtscore berechnen
        overall_score = (tool_score * 0.4 + answer_score * 0.35 + completeness_score * 0.25) * 100
        
        return MCPEvaluationResult(
            tool_usage_correctness=tool_score,
            answer_correctness=answer_score,
            answer_completeness=completeness_score,
            overall_score=overall_score,
            reasoning=f"Fallback-Evaluierung aufgrund von Parse-Fehler: {error_msg}",
            evaluator_model=self.model,
            raw_evaluator_response=raw_response,
            evaluation_error=error_msg,
            # Tool-Interaktions-Daten hinzufügen
            tool_response=interaction_data.get("tool_response"),
            actual_tool_call=interaction_data.get("actual_tool_call"),
            actual_parameters=interaction_data.get("actual_parameters")
        )
    
    def _calculate_parameter_match(self, expected: Dict, actual: Dict) -> float:
        """Berechnet Parameter-Übereinstimmung mit case-insensitive Matching und Type-Konvertierung"""
        if not expected:
            return 1.0 if not actual else 0.8
        
        matches = 0
        total = len(expected)
        
        for key, exp_val in expected.items():
            # Case-insensitive key matching
            actual_key = None
            for ak in actual.keys():
                if str(ak).lower() == str(key).lower():
                    actual_key = ak
                    break
            
            if actual_key is not None:
                if self._values_match(exp_val, actual[actual_key]):
                    matches += 1
                elif str(exp_val).lower() in str(actual[actual_key]).lower():
                    matches += 0.8
        
        return matches / total if total > 0 else 0.0
    
    def evaluate_mcp_interaction(
        self,
        original_prompt: str,
        model_initial: str,
        tool_call_json: Union[Dict, str],
        tool_response: Union[Dict, str],
        model_final: str,
        expected_tool_call: Optional[str] = None,
        expected_parameters: Optional[Dict[str, Any]] = None,
        custom_criteria: Optional[str] = None,
        custom_evaluation_steps: Optional[List[str]] = None
    ) -> MCPEvaluationResult:
        """
        Haupt-Evaluierungsmethode für MCP-Interaktionen
        Fortgeschrittene, robuste LLM-basierte Evaluierung
        """
        start_time = time.time()
        
        # Interaktionsdaten strukturieren
        interaction_data = {
            "original_prompt": original_prompt,
            "model_initial": model_initial,
            "tool_call_json": tool_call_json,
            "tool_response": tool_response,
            "model_final": model_final,
            "expected_tool_call": expected_tool_call,
            "expected_parameters": expected_parameters or {},
            "actual_tool_call": self._extract_tool_name(tool_call_json),
            "actual_parameters": self._extract_tool_parameters(tool_call_json)
        }
        
        # Kriterien und Schritte bestimmen
        criteria = custom_criteria or """
        Evaluiere die Qualität dieser MCP Tool-Interaktion basierend auf:
        1. Korrektheit des Tool-Aufrufs
        2. Angemessenheit der Parameter
        3. Korrekte Verwendung der Tool-Response
        4. Vollständigkeit der finalen Antwort
        """
        
        evaluation_steps = custom_evaluation_steps
        if not evaluation_steps:
            evaluation_steps = self.generate_evaluation_steps(criteria, interaction_data)
        
        # Evaluierungs-Prompt erstellen
        system_prompt, user_prompt = self._create_evaluation_prompt(
            criteria, evaluation_steps, interaction_data
        )
        
        # LLM-basierte Evaluierung mit Retries
        for attempt in range(self.max_retries):
            try:
                response = self._safe_llm_call([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ], max_retries=1)  # Einzelner Versuch pro Attempt
                
                # Response parsen
                result = self._parse_evaluation_response(response, interaction_data)
                
                # Meta-Informationen hinzufügen
                result.evaluation_time = time.time() - start_time
                result.evaluation_steps = evaluation_steps
                result.retry_count = attempt
                
                logger.info(f"Evaluierung abgeschlossen: Score {result.overall_score:.1f}/100 (Versuch {attempt + 1})")
                return result
                
            except Exception as e:
                logger.warning(f"Evaluierung fehlgeschlagen (Versuch {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    # Letzter Versuch fehlgeschlagen - Fallback-Evaluierung
                    result = self._fallback_evaluation("", interaction_data, str(e))
                    result.evaluation_time = time.time() - start_time
                    result.retry_count = attempt + 1
                    return result
                
                time.sleep(min(2 ** attempt, 5))  # Exponential backoff
    
    def _extract_tool_name(self, tool_call_json: Union[Dict, str]) -> Optional[str]:
        """Extrahiert Tool-Name aus Tool-Call"""
        try:
            if isinstance(tool_call_json, str):
                tool_data = json.loads(tool_call_json)
            else:
                tool_data = tool_call_json
            
            # Verschiedene Formate unterstützen
            if "function" in tool_data:
                return tool_data["function"].get("name")
            elif "name" in tool_data:
                return tool_data["name"]
            else:
                return str(tool_data.get("tool", "unknown"))
        except:
            return None
    
    def _extract_tool_parameters(self, tool_call_json: Union[Dict, str]) -> Dict[str, Any]:
        """Extrahiert Parameter aus Tool-Call"""
        try:
            if isinstance(tool_call_json, str):
                tool_data = json.loads(tool_call_json)
            else:
                tool_data = tool_call_json
            
            # Verschiedene Formate unterstützen
            if "function" in tool_data:
                args_str = tool_data["function"].get("arguments", "{}")
                return json.loads(args_str) if isinstance(args_str, str) else args_str
            elif "arguments" in tool_data:
                args = tool_data["arguments"]
                return json.loads(args) if isinstance(args, str) else args
            else:
                return tool_data.get("parameters", {})
        except:
            return {}
    
    def batch_evaluate(
        self,
        interactions: List[Dict[str, Any]],
        criteria: Optional[str] = None,
        show_progress: bool = True
    ) -> List[MCPEvaluationResult]:
        """
        Batch-Evaluierung für mehrere Interaktionen
        Optimiert für Performance mit Parallel-Verarbeitung
        """
        results = []
        total = len(interactions)
        
        logger.info(f"Starte Batch-Evaluierung für {total} Interaktionen")
        
        for i, interaction in enumerate(interactions):
            if show_progress and i % 10 == 0:
                logger.info(f"Fortschritt: {i}/{total} ({i/total*100:.1f}%)")
            
            try:
                result = self.evaluate_mcp_interaction(
                    original_prompt=interaction["original_prompt"],
                    model_initial=interaction["model_initial"],
                    tool_call_json=interaction["tool_call_json"],
                    tool_response=interaction["tool_response"],
                    model_final=interaction["model_final"],
                    expected_tool_call=interaction.get("expected_tool_call"),
                    expected_parameters=interaction.get("expected_parameters"),
                    custom_criteria=criteria
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Fehler bei Interaktion {i}: {e}")
                # Fallback-Result für fehlgeschlagene Evaluierungen
                fallback = MCPEvaluationResult(
                    tool_usage_correctness=0.0,
                    answer_correctness=0.0,
                    answer_completeness=0.0,
                    overall_score=0.0,
                    reasoning=f"Evaluierung fehlgeschlagen: {str(e)}",
                    evaluator_model=self.model,
                    evaluation_error=str(e),
                    # Tool-Interaktions-Daten hinzufügen (falls verfügbar)
                    tool_response=interaction.get("tool_response"),
                    actual_tool_call=interaction.get("actual_tool_call"),
                    actual_parameters=interaction.get("actual_parameters")
                )
                results.append(fallback)
        
        logger.info(f"Batch-Evaluierung abgeschlossen: {len(results)} Ergebnisse")
        return results

class MCPMultiModelEvaluator:
    """
    Multi-Model-Evaluator für parallele Bewertung mit verschiedenen Modellen
    Sequentielle Ausführung mit detailliertem Progress-Reporting
    """
    
    def __init__(
        self,
        models: List[Dict[str, Any]],
        base_evaluator_config: Optional[Dict[str, Any]] = None,
        show_progress: bool = True
    ):
        """
        Initialisiert Multi-Model-Evaluator
        
        Args:
            models: Liste von Modell-Konfigurationen
                   Format: [{"name": "llama3.2", "provider": "ollama", "base_url": "...", ...}]
            base_evaluator_config: Basis-Konfiguration für alle Evaluatoren
            show_progress: Ob detaillierte Progress-Informationen angezeigt werden sollen
        """
        self.models = models
        self.base_config = base_evaluator_config or {}
        self.show_progress = show_progress
        self.evaluators = {}
        
        # Evaluatoren für alle Modelle erstellen
        self._initialize_evaluators()
        
        logger.info(f"MCPMultiModelEvaluator initialisiert mit {len(self.models)} Modellen")
        if self.show_progress:
            self._log_model_overview()
    
    def _initialize_evaluators(self):
        """Initialisiert Evaluatoren für alle konfigurierten Modelle"""
        for model_config in self.models:
            try:
                model_key = f"{model_config.get('provider', 'unknown')}/{model_config.get('name', 'unknown')}"
                
                # Merge base config mit model-spezifischer config
                evaluator_config = {**self.base_config, **model_config}
                # Entferne potentiell störenden 'model'-Key aus Konfiguration
                evaluator_config.pop("model", None)
                
                # Provider-spezifische Evaluator-Erstellung
                if model_config.get("provider") == "ollama":
                    evaluator = EvaluatorFactory.create_ollama_evaluator(
                        model=model_config["name"],
                        base_url=model_config.get("base_url", "http://localhost:11434"),
                        **{k: v for k, v in evaluator_config.items() 
                           if k not in ["name", "provider", "base_url"]}
                    )
                elif model_config.get("provider") == "openai":
                    evaluator = EvaluatorFactory.create_openai_evaluator(
                        model=model_config["name"],
                        api_key=model_config.get("api_key"),
                        **{k: v for k, v in evaluator_config.items() 
                           if k not in ["name", "provider", "api_key"]}
                    )
                else:
                    # Generischer Evaluator
                    normalized_model = model_config.get("name", "unknown")
                    provider = model_config.get('provider', 'generic')
                    if provider == "ollama" and "/" not in normalized_model:
                        normalized_model = f"ollama/{normalized_model}"
                    evaluator = MCPAdvancedEvaluator(
                        model=normalized_model if provider != "generic" else f"{provider}/{normalized_model}",
                        **{k: v for k, v in evaluator_config.items() if k not in ["name", "provider", "base_url", "api_key"]}
                    )
                
                self.evaluators[model_key] = evaluator
                
            except Exception as e:
                logger.error(f"Fehler beim Initialisieren des Evaluators für {model_config}: {e}")
                # Fehlgeschlagene Evaluatoren werden übersprungen
    
    def _log_model_overview(self):
        """Loggt Übersicht über konfigurierte Modelle"""
        print(f"\n📋 Multi-Model-Evaluator Konfiguration:")
        print(f"   Modelle: {len(self.models)} konfiguriert, {len(self.evaluators)} erfolgreich initialisiert")
        
        for i, model_config in enumerate(self.models, 1):
            model_key = f"{model_config.get('provider', 'unknown')}/{model_config.get('name', 'unknown')}"
            status = "✅" if model_key in self.evaluators else "❌"
            print(f"   {i}. {status} {model_key}")
            if model_config.get("base_url"):
                print(f"      └─ URL: {model_config['base_url']}")
    
    def evaluate_interaction_multi_model(
        self,
        original_prompt: str,
        model_initial: str,
        tool_call_json: Union[Dict, str],
        tool_response: Union[Dict, str],
        model_final: str,
        expected_tool_call: Optional[str] = None,
        expected_parameters: Optional[Dict[str, Any]] = None,
        custom_criteria: Optional[str] = None
    ) -> MultiModelEvaluationResult:
        """
        Evaluiert eine MCP-Interaktion mit allen konfigurierten Modellen
        
        Args:
            Alle Parameter wie bei MCPAdvancedEvaluator.evaluate_mcp_interaction()
            
        Returns:
            MultiModelEvaluationResult mit Evaluierungen aller Modelle
        """
        start_time = time.time()
        multi_result = MultiModelEvaluationResult()
        
        if self.show_progress:
            print(f"\n🔄 Multi-Model-Evaluierung gestartet ({len(self.evaluators)} Modelle)")
            print(f"   Prompt: {original_prompt[:60]}{'...' if len(original_prompt) > 60 else ''}")
        
        # Sequentielle Evaluierung mit allen Modellen
        for i, (model_key, evaluator) in enumerate(self.evaluators.items(), 1):
            if self.show_progress:
                print(f"\n   [{i}/{len(self.evaluators)}] Evaluiere mit {model_key}...")
            
            model_start_time = time.time()
            
            try:
                # Einzelne Evaluierung
                evaluation_result = evaluator.evaluate_mcp_interaction(
                    original_prompt=original_prompt,
                    model_initial=model_initial,
                    tool_call_json=tool_call_json,
                    tool_response=tool_response,
                    model_final=model_final,
                    expected_tool_call=expected_tool_call,
                    expected_parameters=expected_parameters,
                    custom_criteria=custom_criteria
                )
                
                # Modell-Identifikation sicherstellen
                evaluation_result.evaluator_model = model_key
                
                # Zu Multi-Result hinzufügen
                multi_result.add_evaluation(evaluation_result, model_key)
                
                if self.show_progress:
                    model_time = time.time() - model_start_time
                    print(f"      ✅ Score: {evaluation_result.overall_score:.1f}/100 "
                          f"(Tool: {evaluation_result.tool_usage_correctness:.2f}, "
                          f"Answer: {evaluation_result.answer_correctness:.2f}) "
                          f"[{model_time:.2f}s]")
                    if evaluation_result.evaluation_error:
                        print(f"      ⚠️  Warnung: {evaluation_result.evaluation_error}")
                
            except Exception as e:
                # Fehler-Evaluierung erstellen
                error_result = MCPEvaluationResult(
                    tool_usage_correctness=0.0,
                    answer_correctness=0.0,
                    answer_completeness=0.0,
                    overall_score=0.0,
                    reasoning=f"Evaluierung mit {model_key} fehlgeschlagen",
                    evaluator_model=model_key,
                    evaluation_error=str(e),
                    evaluation_time=time.time() - model_start_time
                )
                
                multi_result.add_evaluation(error_result, model_key)
                
                if self.show_progress:
                    model_time = time.time() - model_start_time
                    print(f"      ❌ Fehler: {str(e)[:50]}{'...' if len(str(e)) > 50 else ''} [{model_time:.2f}s]")
                
                logger.error(f"Multi-Model-Evaluierung fehlgeschlagen für {model_key}: {e}")
        
        # Gesamt-Zeit setzen
        multi_result.total_evaluation_time = time.time() - start_time
        
        if self.show_progress:
            self._log_multi_result_summary(multi_result)
        
        return multi_result
    
    def _log_multi_result_summary(self, multi_result: MultiModelEvaluationResult):
        """Loggt Zusammenfassung der Multi-Model-Evaluierung"""
        print(f"\n📊 Multi-Model-Evaluierung abgeschlossen:")
        print(f"   ✅ Erfolgreich: {multi_result.successful_evaluations}/{len(multi_result.models_used)}")
        print(f"   ❌ Fehlgeschlagen: {multi_result.failed_evaluations}/{len(multi_result.models_used)}")
        print(f"   ⏱️  Gesamtzeit: {multi_result.total_evaluation_time:.2f}s")
        
        # Score-Übersicht (nur erfolgreiche Evaluierungen)
        successful_evals = [eval_result for eval_result in multi_result.model_evaluations 
                           if eval_result.evaluation_error is None]
        
        if successful_evals:
            avg_overall = sum(e.overall_score for e in successful_evals) / len(successful_evals)
            avg_tool = sum(e.tool_usage_correctness for e in successful_evals) / len(successful_evals)
            avg_answer = sum(e.answer_correctness for e in successful_evals) / len(successful_evals)
            
            print(f"   📈 Durchschnittliche Scores:")
            print(f"      Overall: {avg_overall:.1f}/100")
            print(f"      Tool Usage: {avg_tool:.2f}")
            print(f"      Answer Correctness: {avg_answer:.2f}")
    
    def batch_evaluate_multi_model(
        self,
        interactions: List[Dict[str, Any]],
        criteria: Optional[str] = None
    ) -> List[MultiModelEvaluationResult]:
        """
        Batch-Evaluierung für mehrere Interaktionen mit allen Modellen
        
        Args:
            interactions: Liste von Interaktions-Dictionaries
            criteria: Optionale custom criteria
            
        Returns:
            Liste von MultiModelEvaluationResult-Objekten
        """
        results = []
        total = len(interactions)
        
        logger.info(f"Starte Multi-Model-Batch-Evaluierung für {total} Interaktionen mit {len(self.evaluators)} Modellen")
        
        if self.show_progress:
            print(f"\n🚀 Multi-Model-Batch-Evaluierung gestartet:")
            print(f"   Interaktionen: {total}")
            print(f"   Modelle: {len(self.evaluators)}")
            print(f"   Gesamt-Evaluierungen: {total * len(self.evaluators)}")
        
        for i, interaction in enumerate(interactions):
            if self.show_progress:
                print(f"\n{'='*60}")
                print(f"Interaktion {i+1}/{total} ({(i+1)/total*100:.1f}%)")
                print(f"{'='*60}")
            
            try:
                result = self.evaluate_interaction_multi_model(
                    original_prompt=interaction["original_prompt"],
                    model_initial=interaction["model_initial"],
                    tool_call_json=interaction["tool_call_json"],
                    tool_response=interaction["tool_response"],
                    model_final=interaction["model_final"],
                    expected_tool_call=interaction.get("expected_tool_call"),
                    expected_parameters=interaction.get("expected_parameters"),
                    custom_criteria=criteria
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Fehler bei Multi-Model-Batch-Evaluierung für Interaktion {i}: {e}")
                # Fallback-Result erstellen
                fallback_result = MultiModelEvaluationResult()
                fallback_result.failed_evaluations = len(self.evaluators)
                results.append(fallback_result)
        
        if self.show_progress:
            self._log_batch_summary(results)
        
        logger.info(f"Multi-Model-Batch-Evaluierung abgeschlossen: {len(results)} Interaktionen")
        return results
    
    def _log_batch_summary(self, results: List[MultiModelEvaluationResult]):
        """Loggt Batch-Zusammenfassung"""
        total_evaluations = sum(len(r.model_evaluations) for r in results)
        total_successful = sum(r.successful_evaluations for r in results)
        total_failed = sum(r.failed_evaluations for r in results)
        total_time = sum(r.total_evaluation_time for r in results)
        
        print(f"\n🎯 Batch-Evaluierung Zusammenfassung:")
        print(f"   Interaktionen: {len(results)}")
        print(f"   Gesamt-Evaluierungen: {total_evaluations}")
        print(f"   ✅ Erfolgreich: {total_successful} ({total_successful/total_evaluations*100:.1f}%)")
        print(f"   ❌ Fehlgeschlagen: {total_failed} ({total_failed/total_evaluations*100:.1f}%)")
        print(f"   ⏱️  Gesamtzeit: {total_time:.1f}s")
        print(f"   ⚡ Durchschnittliche Zeit pro Evaluierung: {total_time/total_evaluations:.2f}s")

# Factory-Klassen und Utilities

class EvaluatorFactory:
    """Factory für verschiedene Evaluator-Konfigurationen"""
    
    @staticmethod
    def create_ollama_evaluator(
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        **kwargs
    ) -> MCPAdvancedEvaluator:
        """Erstellt Ollama-basierten Evaluator"""
        # Doppelte Präfixe vermeiden
        full_model = model if model.startswith("ollama/") else f"ollama/{model}"
        return MCPAdvancedEvaluator(
            model=full_model,
            base_url=base_url,
            **kwargs
        )
    
    @staticmethod
    def create_openai_evaluator(
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        **kwargs
    ) -> MCPAdvancedEvaluator:
        """Erstellt OpenAI-basierten Evaluator"""
        return MCPAdvancedEvaluator(
            model=model,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            **kwargs
        )
    
    @staticmethod
    def create_multi_model_evaluator(
        models: List[Dict[str, Any]],
        base_config: Optional[Dict[str, Any]] = None,
        show_progress: bool = True,
        timeout: Optional[int] = None
    ) -> MCPMultiModelEvaluator:
        """
        Erstellt Multi-Model-Evaluator
        
        Args:
            models: Liste von Modell-Konfigurationen
                   Beispiel: [
                       {"name": "llama3.2", "provider": "ollama", "base_url": "http://localhost:11434"},
                       {"name": "llama3.1", "provider": "ollama", "base_url": "http://localhost:11434"},
                       {"name": "gpt-4", "provider": "openai", "api_key": "..."}
                   ]
            base_config: Basis-Konfiguration für alle Evaluatoren
            show_progress: Ob Progress-Informationen angezeigt werden sollen
            timeout: Optionaler Timeout (Sekunden) für Ollama/OpenAI-Aufrufe; wird in die Evaluator-Konfiguration gemerged
            
        Returns:
            MCPMultiModelEvaluator-Instanz
        """
        # Timeout (falls angegeben) in die Basis-Konfiguration übernehmen
        if timeout is not None:
            base_config = {**(base_config or {}), "timeout": timeout}
        return MCPMultiModelEvaluator(
            models=models,
            base_evaluator_config=base_config,
            show_progress=show_progress
        )

def create_custom_evaluation_criteria(
    domain: str,
    specific_requirements: List[str]
) -> str:
    """Erstellt domänen-spezifische Evaluierungskriterien"""
    criteria = f"""Evaluiere diese {domain}-spezifische MCP Tool-Interaktion basierend auf:

Allgemeine Kriterien:
1. Tool-Call-Korrektheit und Parameter-Angemessenheit
2. Korrekte Integration der Tool-Response
3. Vollständigkeit und Genauigkeit der finalen Antwort

Spezifische Anforderungen:"""
    
    for i, req in enumerate(specific_requirements, 4):
        criteria += f"\n{i}. {req}"
    
    return criteria

# Hinweis: Tests und Demos befinden sich in separaten Dateien (z.B. mcp_advanced_demo.py).
