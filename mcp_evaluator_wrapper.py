#!/usr/bin/env python3
"""
Wrapper für die Integration des robusten MCP-Evaluators mit bestehenden Systemen
Ermöglicht die Verwendung ohne Änderungen an mcp_benchmark_llm.py
Erweitert um Multi-Model-Support für parallele Evaluierung mit verschiedenen Modellen
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union

from mcp_benchmark_llm import (
    EvaluatorFactory, 
    MCPEvaluationResult, 
    MultiModelEvaluationResult,
    MCPMultiModelEvaluator
)
from mcp_benchmark_llm_old import EvaluationResult

logger = logging.getLogger(__name__)

class MCPEvaluatorWrapper:
    """
    Wrapper-Klasse für nahtlose Integration des robusten Evaluators
    """
    
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        temperature: float = 0.0,
        # Neue Multi-Model-Parameter
        models: Optional[List[Dict[str, Any]]] = None,
        multi_model_mode: bool = False
    ):
        """
        Initialisiert den Wrapper mit Evaluator(en)
        
        Args:
            model: Standard-Modell (z.B. "llama3.2", "llama3.1")
            base_url: Ollama-Server URL
            api_key: Optionaler API-Key (meist nicht nötig für Ollama)
            max_retries: Maximale Retry-Anzahl
            temperature: Temperatur für deterministische Evaluierung
            models: Liste von Modellen für Multi-Model-Modus (optional)
            multi_model_mode: Ob Multi-Model-Evaluierung verwendet werden soll
        """
        
        self.multi_model_mode = multi_model_mode
        
        if multi_model_mode and models:
            # Multi-Model-Evaluator erstellen
            self._init_multi_model_evaluator(models, max_retries, temperature)
        else:
            # Standard Single-Model-Evaluator erstellen
            self._init_single_model_evaluator(model, base_url, max_retries, temperature)
    
    def _init_single_model_evaluator(
        self, 
        model: str, 
        base_url: str, 
        max_retries: int, 
        temperature: float
    ):
        """Initialisiert Standard-Evaluator"""
        try:
            if model.startswith("ollama/"):
                model = model.replace("ollama/", "")
                
            self.evaluator = EvaluatorFactory.create_ollama_evaluator(
                model=model,
                base_url=base_url,
                max_retries=max_retries,
                temperature=temperature
            )
            
            self.model_name = f"ollama/{model}"
            logger.info(f"MCPEvaluatorWrapper initialisiert mit {self.model_name}")
            
        except Exception as e:
            logger.error(f"Fehler beim Initialisieren des Evaluators: {e}")
            self.evaluator = None
            self.model_name = model
    
    def _init_multi_model_evaluator(
        self,
        models: List[Dict[str, Any]],
        max_retries: int,
        temperature: float
    ):
        """Initialisiert Multi-Model-Evaluator"""
        try:
            base_config = {
                "max_retries": max_retries,
                "temperature": temperature
            }
            
            self.multi_evaluator = EvaluatorFactory.create_multi_model_evaluator(
                models=models,
                base_config=base_config,
                show_progress=False  # Reduziert Ausgabe für Wrapper-Nutzung
            )
            
            self.evaluator = None  # Single-Evaluator deaktiviert
            self.model_name = f"multi_model_{len(models)}_evaluators"
            
            logger.info(f"MCPEvaluatorWrapper initialisiert mit Multi-Model-Evaluator ({len(models)} Modelle)")
            
        except Exception as e:
            logger.error(f"Fehler beim Initialisieren des Multi-Model-Evaluators: {e}")
            self.multi_evaluator = None
            self.model_name = "multi_model_error"
    
    def evaluate_interaction(
        self,
        original_prompt: str,
        model_initial: str,
        tool_call_json: Union[dict, str],
        tool_response: Union[dict, str],
        model_final: str,
        expected_tool_call: Optional[str] = None,
        expected_parameters: Optional[Dict[str, Any]] = None
    ) -> Union[EvaluationResult, List[EvaluationResult]]:
        """
        Evaluiert MCP-Interaktion - unterstützt sowohl Single- als auch Multi-Model-Modus
        
        Args:
            original_prompt: Ursprünglicher User-Prompt
            model_initial: Erste LLM-Antwort
            tool_call_json: Tool-Call JSON
            tool_response: Tool-Response
            model_final: Finale LLM-Antwort
            expected_tool_call: Erwarteter Tool-Name (optional)
            expected_parameters: Erwartete Parameter (optional)
            
        Returns:
            EvaluationResult (Single-Model) oder Liste von EvaluationResult (Multi-Model)
        """
        
        if self.multi_model_mode and hasattr(self, 'multi_evaluator') and self.multi_evaluator:
            return self._evaluate_multi_model(
                original_prompt, model_initial, tool_call_json, 
                tool_response, model_final, expected_tool_call, expected_parameters
            )
        else:
            return self._evaluate_single_model(
                original_prompt, model_initial, tool_call_json,
                tool_response, model_final, expected_tool_call, expected_parameters
            )
    
    def _evaluate_single_model(
        self,
        original_prompt: str,
        model_initial: str,
        tool_call_json: Union[dict, str],
        tool_response: Union[dict, str],
        model_final: str,
        expected_tool_call: Optional[str] = None,
        expected_parameters: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Standard Single-Model-Evaluierung"""
        
        if not self.evaluator:
            logger.warning("Robuster Evaluator nicht verfügbar, verwende Fallback")
            return self._create_fallback_result("Robuster Evaluator nicht initialisiert")
        
        try:
            # Verwende den robusten Evaluator
            mcp_result = self.evaluator.evaluate_mcp_interaction(
                original_prompt=original_prompt,
                model_initial=model_initial,
                tool_call_json=tool_call_json,
                tool_response=tool_response,
                model_final=model_final,
                expected_tool_call=expected_tool_call,
                expected_parameters=expected_parameters
            )
            
            # Konvertiere zu kompatiblem EvaluationResult
            return EvaluationResult(
                tool_usage_correctness=mcp_result.tool_usage_correctness,
                final_answer_correctness=mcp_result.answer_correctness,
                final_answer_completeness=mcp_result.answer_completeness,
                overall_score=mcp_result.overall_score,
                short_explanation=mcp_result.reasoning,
                evaluator_response_raw=mcp_result.raw_evaluator_response,
                evaluation_error=mcp_result.evaluation_error,
                evaluation_time=mcp_result.evaluation_time
            )
            
        except Exception as e:
            logger.error(f"Fehler bei robuster Evaluierung: {e}")
            return self._create_fallback_result(str(e))
    
    def _evaluate_multi_model(
        self,
        original_prompt: str,
        model_initial: str,
        tool_call_json: Union[dict, str],
        tool_response: Union[dict, str],
        model_final: str,
        expected_tool_call: Optional[str] = None,
        expected_parameters: Optional[Dict[str, Any]] = None
    ) -> List[EvaluationResult]:
        """Multi-Model-Evaluierung - gibt Liste von EvaluationResult zurück"""
        
        if not hasattr(self, 'multi_evaluator') or not self.multi_evaluator:
            logger.error("Multi-Model-Evaluator nicht verfügbar")
            return [self._create_fallback_result("Multi-Model-Evaluator nicht initialisiert")]
        
        try:
            # Multi-Model-Evaluierung
            multi_result = self.multi_evaluator.evaluate_interaction_multi_model(
                original_prompt=original_prompt,
                model_initial=model_initial,
                tool_call_json=tool_call_json,
                tool_response=tool_response,
                model_final=model_final,
                expected_tool_call=expected_tool_call,
                expected_parameters=expected_parameters
            )
            
            # Konvertiere alle MCPEvaluationResult zu EvaluationResult
            evaluation_results = []
            for mcp_result in multi_result.model_evaluations:
                evaluation_result = EvaluationResult(
                    tool_usage_correctness=mcp_result.tool_usage_correctness,
                    final_answer_correctness=mcp_result.answer_correctness,
                    final_answer_completeness=mcp_result.answer_completeness,
                    overall_score=mcp_result.overall_score,
                    short_explanation=f"[{mcp_result.evaluator_model}] {mcp_result.reasoning}",
                    evaluator_response_raw=mcp_result.raw_evaluator_response,
                    evaluation_error=mcp_result.evaluation_error,
                    evaluation_time=mcp_result.evaluation_time
                )
                evaluation_results.append(evaluation_result)
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Fehler bei Multi-Model-Evaluierung: {e}")
            return [self._create_fallback_result(str(e))]
    
    def _create_fallback_result(self, error_msg: str) -> EvaluationResult:
        """Erstellt Fallback-Ergebnis bei Fehlern"""
        return EvaluationResult(
            tool_usage_correctness=0.0,
            final_answer_correctness=0.0,
            final_answer_completeness=0.0,
            overall_score=0.0,
            short_explanation="Evaluierung fehlgeschlagen (robuster Evaluator)",
            evaluator_response_raw=None,
            evaluation_error=error_msg,
            evaluation_time=0.0
        )
    
    def batch_evaluate_interactions(
        self,
        interactions: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[EvaluationResult]:
        """
        Batch-Evaluierung mehrerer Interaktionen
        
        Args:
            interactions: Liste von Interaktions-Dictionaries
            show_progress: Ob Fortschritt angezeigt werden soll
            
        Returns:
            Liste von EvaluationResult-Objekten
        """
        
        if not self.evaluator:
            logger.error("Robuster Evaluator nicht verfügbar für Batch-Evaluierung")
            return []
        
        results = []
        total = len(interactions)
        
        logger.info(f"Starte Batch-Evaluierung für {total} Interaktionen")
        
        for i, interaction in enumerate(interactions):
            if show_progress and i % 10 == 0:
                logger.info(f"Fortschritt: {i}/{total} ({i/total*100:.1f}%)")
            
            try:
                result = self.evaluate_interaction(**interaction)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Fehler bei Interaktion {i}: {e}")
                results.append(self._create_fallback_result(str(e)))
        
        logger.info(f"Batch-Evaluierung abgeschlossen: {len(results)} Ergebnisse")
        return results

def create_ollama_evaluator_wrapper(
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434",
    **kwargs
) -> MCPEvaluatorWrapper:
    """
    Convenience-Funktion zum Erstellen eines Ollama-Evaluator-Wrappers
    
    Args:
        model: Ollama-Modell-Name
        base_url: Ollama-Server URL
        **kwargs: Zusätzliche Parameter für den Wrapper
        
    Returns:
        MCPEvaluatorWrapper-Instanz
    """
    return MCPEvaluatorWrapper(
        model=model,
        base_url=base_url,
        **kwargs
    )

def create_multi_model_evaluator_wrapper(
    models: List[Dict[str, Any]],
    base_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> MCPEvaluatorWrapper:
    """
    Convenience-Funktion zum Erstellen eines Multi-Model-Evaluator-Wrappers
    
    Args:
        models: Liste von Modell-Konfigurationen
               Beispiel: [
                   {"name": "llama3.2", "provider": "ollama", "base_url": "http://localhost:11434"},
                   {"name": "llama3.1", "provider": "ollama", "base_url": "http://localhost:11434"}
               ]
        base_config: Basis-Konfiguration für alle Evaluatoren
        **kwargs: Zusätzliche Parameter für den Wrapper
        
    Returns:
        MCPEvaluatorWrapper-Instanz im Multi-Model-Modus
    """
    merged_kwargs = {**base_config} if base_config else {}
    merged_kwargs.update(kwargs)
    
    return MCPEvaluatorWrapper(
        models=models,
        multi_model_mode=True,
        **merged_kwargs
    )

# Beispiel für die Verwendung
def example_usage():
    """Zeigt die Verwendung des Wrappers"""
    
    print("=== MCPEvaluatorWrapper Beispiel ===\n")
    
    # 1. Standard Single-Model-Wrapper
    print("1. Single-Model-Evaluator:")
    evaluator_wrapper = create_ollama_evaluator_wrapper(
        model="llama3.2",
        max_retries=3
    )
    
    # 2. Multi-Model-Wrapper
    print("\n2. Multi-Model-Evaluator:")
    test_models = [
        {"name": "llama3.2", "provider": "ollama", "base_url": "http://localhost:11434"},
        {"name": "llama3.1", "provider": "ollama", "base_url": "http://localhost:11434"}
    ]
    
    multi_evaluator_wrapper = create_multi_model_evaluator_wrapper(
        models=test_models,
        base_config={"max_retries": 2, "temperature": 0.0}
    )
    
    # 3. Beispiel-Interaktion
    example_interaction = {
        "original_prompt": "Wie ist das Wetter in Hamburg?",
        "model_initial": "Ich rufe das Wetter-Tool für Hamburg auf.",
        "tool_call_json": {
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "hamburg", "country": "deutschland"}'
            }
        },
        "tool_response": {
            "city": "Hamburg",
            "temperature": 15,
            "condition": "regnerisch",
            "humidity": 85
        },
        "model_final": "Das Wetter in Hamburg ist heute regnerisch bei 15°C und 85% Luftfeuchtigkeit.",
        "expected_tool_call": "get_weather",
        "expected_parameters": {"city": "hamburg"}
    }
    
    # 4. Single-Model-Evaluierung
    print("\n=== Single-Model-Evaluierung ===")
    try:
        result = evaluator_wrapper.evaluate_interaction(**example_interaction)
        
        print("✓ Single-Model-Evaluierung erfolgreich:")
        print(f"   Tool Usage Correctness: {result.tool_usage_correctness:.2f}")
        print(f"   Final Answer Correctness: {result.final_answer_correctness:.2f}")
        print(f"   Final Answer Completeness: {result.final_answer_completeness:.2f}")
        print(f"   Overall Score: {result.overall_score:.1f}/100")
        print(f"   Explanation: {result.short_explanation[:100]}...")
        
        if result.evaluation_error:
            print(f"   ⚠ Error: {result.evaluation_error}")
            
    except Exception as e:
        print(f"❌ Fehler bei Single-Model-Evaluierung: {e}")
    
    # 5. Multi-Model-Evaluierung
    print(f"\n=== Multi-Model-Evaluierung ===")
    try:
        multi_results = multi_evaluator_wrapper.evaluate_interaction(**example_interaction)
        
        print(f"✓ Multi-Model-Evaluierung erfolgreich ({len(multi_results)} Modelle):")
        
        for i, result in enumerate(multi_results, 1):
            # Modell-Name aus der Explanation extrahieren
            model_name = result.short_explanation.split(']')[0].replace('[', '') if ']' in result.short_explanation else f"Model_{i}"
            
            print(f"\n   📋 {model_name}:")
            print(f"      Tool Usage: {result.tool_usage_correctness:.2f}")
            print(f"      Answer Correctness: {result.final_answer_correctness:.2f}")
            print(f"      Overall Score: {result.overall_score:.1f}/100")
            print(f"      Time: {result.evaluation_time:.2f}s")
            
            if result.evaluation_error:
                print(f"      ❌ Error: {result.evaluation_error}")
        
        # Durchschnittswerte berechnen
        successful_results = [r for r in multi_results if r.evaluation_error is None]
        if successful_results:
            avg_overall = sum(r.overall_score for r in successful_results) / len(successful_results)
            avg_tool = sum(r.tool_usage_correctness for r in successful_results) / len(successful_results)
            avg_answer = sum(r.final_answer_correctness for r in successful_results) / len(successful_results)
            
            print(f"\n   📈 Durchschnittswerte (erfolgreiche Evaluierungen):")
            print(f"      Overall Score: {avg_overall:.1f}/100")
            print(f"      Tool Usage: {avg_tool:.2f}")
            print(f"      Answer Correctness: {avg_answer:.2f}")
            
    except Exception as e:
        print(f"❌ Multi-Model-Evaluierung fehlgeschlagen: {e}")
    
    # 6. Batch-Evaluierung Beispiel
    print(f"\n=== Batch-Evaluierung (Single-Model) ===")
    
    batch_interactions = [example_interaction] * 3  # 3 identische für Demo
    
    try:
        batch_results = evaluator_wrapper.batch_evaluate_interactions(
            interactions=batch_interactions,
            show_progress=True
        )
        
        avg_score = sum(r.overall_score for r in batch_results) / len(batch_results)
        print(f"✓ Batch-Evaluierung abgeschlossen:")
        print(f"   {len(batch_results)} Interaktionen evaluiert") 
        print(f"   Durchschnittlicher Score: {avg_score:.1f}/100")
        
    except Exception as e:
        print(f"❌ Batch-Evaluierung fehlgeschlagen: {e}")
    
    print(f"\n=== Multi-Model-Wrapper Vorteile ===")
    print("✓ Evaluierung mit mehreren Modellen gleichzeitig")
    print("✓ Array von Evaluierungsergebnissen pro TestCase") 
    print("✓ Sequentielle Ausführung mit detailliertem Progress")
    print("✓ Automatische Fehlerbehandlung pro Modell")
    print("✓ Rückwärts-kompatibel mit bestehender EvaluationResult-API")
    print("✓ Unterstützung verschiedener Provider (Ollama, OpenAI, etc.)")
    print("✓ Aggregierte Statistiken können aus Ergebnis-Arrays berechnet werden")

if __name__ == "__main__":
    example_usage()
