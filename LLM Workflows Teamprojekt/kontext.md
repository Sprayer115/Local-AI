# 📋 Umfassendes Kontextdokument: Teamprojekt "LLM Workflows"

## 🎯 Projektübersicht

### Grunddaten
- **Projektname**: LLM Workflows (früher "Local AI")
- **Institution**: HTWG Konstanz (Hochschule Konstanz)
- **Zeitraum**: 2 Semester (WS 2024/25 + SS 2025)
- **Team**: 
  - **Führung**: Prof. Matthias Franz, Prof. Oliver Dürr
  - **Ausführung**: Benedikt Scheffel, Simon Driescher, David Layer-Reiss (+ Robert Breuer)
- **Meeting-Rhythmus**: Alle 2 Wochen
- **Projektansatz**: Dynamisch-adaptiv (Ziele werden iterativ angepasst)

### Projektziel
Erforschung, Entwicklung und Implementierung von Workflows zur effizienten Nutzung von Large Language Models (LLMs) in verschiedenen Anwendungsbereichen. Fokus auf:
- **MCP (Model Context Protocol)** - Protokoll zur Tool-Integration für LLMs
- **n8n** - Workflow-Automation mit Fair-Use Lizenzmodell
- **Lokale LLM-Kompetenz** - On-Premise Lösungen

## 🔧 Technische Infrastruktur

### Hardware-Setup (3 Server)

1. **Ollama Workstation**
   - Intel Xeon E5-2667v4, 32GB RAM
   - NVIDIA Quadro P6000 (24GB VRAM)
   - SSH: ollama.ios.htwg-konstanz.de

2. **Atlas Server**
   - 2× Intel Xeon E5-2630v3, 128GB RAM
   - NVIDIA Tesla P100 + Tesla M10
   - SSH: atlas.ios.htwg-konstanz.de

3. **GPU-PC01**
   - AMD Ryzen 9 5900X, 32GB RAM
   - NVIDIA GeForce RTX 3080 Ti
   - SSH: gpu-pc01.ios.htwg-konstanz.de

### Software-Stack
- **LLM Frameworks**: Ollama (primär), vLLM (Testing, performanter aber Speicherprobleme)
- **Web-UIs**: OpenWebUI (Single-User), LibreChat (Multi-User Fokus)
- **Workflow**: n8n (http://n8n.ios.htwg-konstanz.de:5678)
- **Bridge**: MCPO (MCP-to-OpenAPI proxy server) - universeller Übersetzer von MCP zu REST/OpenAPI

## 🛠️ MCP (Model Context Protocol) Details

### Was ist MCP?
Ein Protokoll, das LLMs ermöglicht, externe Tools/Services zu nutzen. Erweitert die Fähigkeiten von Sprachmodellen durch strukturierte Tool-Definitionen.

### Transport-Typen
1. **stdio**: Subprocess-Kommunikation (JSON-RPC über stdin/stdout)
2. **HTTP**: Streamable HTTP mit POST/GET, optional SSE

### Tool-Definition Beispiel
```python
Tool(
    name="get_weather",
    description="Ruft aktuelle Wetterdaten für eine Stadt ab",
    inputSchema={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "Name der Stadt"},
            "use_real_api": {"type": "boolean", "default": False}
        },
        "required": ["city"]
    }
)
```

## 📊 Bisheriger Projektverlauf

### Phase 1: Setup & Exploration
- Aufsetzen der Ollama Workstation mit OpenWebUI
- Erste Experimente mit lokalen LLMs
- Verständnis der MCP-Technologie

### Phase 2: Tool-Entwicklung
- **Demo-Tools implementiert**: Wetterdaten-MCP, HTWG Mensa-MCP
- **Repository**: https://github.com/Sprayer115/Local-AI
- **Learning**: Tool-Definition simpel, robuste Integration komplex
- Debugging der Kommunikation zwischen LLM und Tools

### Phase 3: Benchmarking-Framework
- Entwicklung eines eigenen Evaluierungs-Tools
- **Zwei-Stufen Ansatz**:
  1. Standard LLM-Benchmarking (Antwortqualität)
  2. MCP-spezifische Tests (Tool-Nutzung, Result-Interpretation)
- **Multi-Model Cross-Validation**: Modelle bewerten sich gegenseitig
- **Herausforderung**: Qualitative Evaluation ohne klare Metriken

### Phase 4: Integration & Erweiterung
- n8n als Workflow-Automation Tool evaluiert (Fair-Use Lizenzmodell, Alternative zu Zapier)
- LibreChat und vLLM als Alternativen getestet
- MCPO für cloud-taugliche MCP-Integration verstanden

### Phase 5: Aktuell - Weather-Forecast MCP
- Kooperation mit anderem Teamprojekt (Wetterstation/Wettermodelle)
- Entwicklung eines praxisnahen MCP-Tools
- Vorbereitung Live-Demo für Zwischenpräsentation

## 📈 Erkenntnisse & Learnings

### Technische Erkenntnisse
- **MCP-Kompatibilität** variiert stark zwischen Modellen
- Nicht alle LLMs sind gleich gut für Tool-Usage geeignet
- **MCP ist keine Magie** - erfordert sorgfältige Implementierung
- Qualitative Evaluation von LLM-Outputs ist inhärent komplex

### Benchmark-Ergebnisse (vorläufig)
- Tool-Usage Fähigkeiten: GPT-4 > Claude-3 > Llama-3 > Mixtral
- Result-Interpretation ebenfalls modellabhängig
- Multi-Model Evaluation verbessert Robustheit der Bewertung

## 🎤 Zwischenpräsentation Details

### Kontext
- **Zeitpunkt**: Semesterbeginn WS 2024/25
- **Dauer**: 15 Minuten + Diskussion
- **Zielgruppe**: Technisch versierte Professoren/Studenten (nicht alle MCP/n8n-vertraut)
- **Format**: Slidev Markdown-Präsentation
- **Sprache**: Deutsch (Fachbegriffe Englisch)

### Präsentationsstruktur
1. Titelfolie - LLM Workflows: MCP Integration & Benchmarking
2. Agenda - Überblick
3. Projektkontext - Team, Timeline, Ziele
4. Was ist MCP? - Erklärung und Transport-Typen
5. Infrastruktur - Evolution von Workstation zu 3-Server-Setup
6. Demo-Tools - Wetter & Mensa als Lernprojekte
7. Benchmarking-Framework - Multi-Model Cross-Validation
8. Benchmark-Ergebnisse - Scores und Kompatibilitäts-Matrix
9. n8n Integration - Workflow-Automation meets LLM
10. Weather-Forecast MCP - Aktuelles Hauptprojekt
11. Learnings - Evaluation komplex, MCP keine Magie
12. Nächste Schritte - Alternativen, Use-Cases
13. Diskussion - Offene Fragen

### Design-Entscheidungen
- **Theme**: Dunkles, technisch-professionelles Design
- **Visuals**: Mermaid-Diagramme, Progress-Bars, Icons
- **Animationen**: v-clicks für schrittweise Enthüllung
- **Fokus**: Technische Details über praktische Demos

## 🚀 Ausblick & Nächste Schritte

### Kurzfristig (bis Semesterende)
- Weather-Forecast MCP fertigstellen
- Live-Demo vorbereiten und durchführen
- Benchmark-Framework dokumentieren
- Code aufräumen für GitHub-Veröffentlichung

### Mittelfristig (2. Semester)
- MCP-Alternativen evaluieren (Function Calling APIs, Custom Integrations)
- Weitere Use-Cases explorieren:
  - IoT-Integration (Sensoren/Aktoren)
  - Hochschul-Services (LSF, Moodle-Integration)
  - Paperless-Integration (Dokumenten-KI)

### Optional
- Brown Bag Seminar über MCP halten
- 2-seitiger Bericht verfassen
- Finale Präsentation am Projektende

## 💬 Offene Diskussionspunkte

1. **Größtes Potential für MCP?** - Wo macht Tool-Integration am meisten Sinn?
2. **Gewünschte Tool-Integrationen?** - Was würde den größten Mehrwert bieten?
3. **Alternative Pfade zu MCP?** - Function Calling, API-Integration, andere Standards?

## 📝 Wichtige Links & Ressourcen

- **Discord**: https://discord.gg/3kSYMfgMNF
- **Mattermost**: Teamprojekt Local-AI auf chat.ios.htwg-konstanz.de
- **GitHub**: https://github.com/Sprayer115/Local-AI
- **MCP Dokumentation**: https://modelcontextprotocol.io
- **n8n Instanz**: http://n8n.ios.htwg-konstanz.de:5678

## ⚠️ Besondere Hinweise

- Präsentation soll **technischen Fokus** haben, nicht auf praktischen Anwendungen
- Weather-MCP ist **Hauptdemonstrator**, nicht die kleinen Demo-Tools
- Benchmark-Ergebnisse sind **noch nicht final** - Key Learnings bleiben zu füllen
- **MCP-Kompatibilität** ist noch offene Forschungsfrage
- Projekt hat **offenes Ende** - Ziele werden dynamisch angepasst

---