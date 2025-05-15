# LLM-Projekt-Dokumentation

Dieses Repository dient zur Dokumentation und Automatisierung unserer LLM-Infrastruktur auf dem Linux-Server.

## Struktur

Wir verwenden eine Kombination aus Markdown-Dateien und Ansible-Playbooks, um unsere Infrastruktur zu dokumentieren und zu automatisieren:

```
/docs              # Allgemeine Dokumentation in Markdown
  /services        # Dokumentation für einzelne Dienste
  /infrastructure  # Dokumentation für Infrastrukturkomponenten
/ansible           # Infrastructure-as-Code
  /inventory       # Server-Inventar
  /playbooks       # Ansible-Playbooks
  /roles           # Wiederverwendbare Ansible-Rollen
/docker            # Docker-Compose-Dateien und verwandte Konfigurationen
  /service-name    # Ordner für jeden Docker-Service
README.md          # Übersicht über das Projekt
```

## Philosophie

Unser Dokumentationsansatz folgt dem Prinzip der "lebendigen Dokumentation":

1. **Markdown-Dateien** dienen als menschenlesbare Dokumentation:
   - Enthalten Erklärungen, Kontext und das "Warum" hinter Entscheidungen
   - Bieten Anweisungen zur Nutzung und Konfiguration
   - Dokumentieren Probleme und Lösungen

  *OPTIONAL: Wäre interessant für reproduzierbare Ergebnisse, aber nicht zwingend notwendig*
2. **Ansible-Playbooks** dienen als ausführbare Dokumentation:
   - Automatisieren die Installation und Konfiguration
   - Gewährleisten Konsistenz zwischen Umgebungen
   - Ermöglichen schnelles Onboarding neuer Teammitglieder

3. **Docker-Compose-Dateien** für containerisierte Dienste:
   - Definieren die genaue Konfiguration jedes Dienstes
   - Werden durch Ansible-Playbooks bereitgestellt und verwaltet

## Komponenten

Aktuell dokumentieren und automatisieren wir folgende Komponenten:

- **Ollama**: LLM-Engine für lokale Ausführung von Modellen
- **Docker**: Container-Infrastruktur
- **OpenWebUI**: Weboberfläche für die Interaktion mit Ollama
