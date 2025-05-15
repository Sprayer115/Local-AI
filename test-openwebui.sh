#!/bin/bash
# Skript zum Testen der OpenWebUI-Installation
# Funktioniert sowohl in normaler Umgebung als auch im Docker-in-Docker Test-Container

echo "Teste OpenWebUI-Verfügbarkeit..."

# Prüfen ob wir uns in einem Docker-Container befinden
if [ -f /.dockerenv ]; then
  echo "Ausführung in Docker-Container erkannt"
  # Prüfen ob Docker-Socket verfügbar ist
  if [ -S /var/run/docker.sock ]; then
    echo "Docker-Socket verfügbar, lese Port aus Container-Konfiguration aus"
    PORT=$(docker port open-webui 2>/dev/null | grep '8080/tcp' | awk -F':' '{print $NF}')
  else
    echo "Docker-Socket nicht verfügbar, verwende konfigurierten Port"
    # Wert aus der .env-Datei lesen, falls vorhanden
    if [ -f "/opt/openwebui/.env" ]; then
      PORT=$(grep "OPEN_WEBUI_PORT" /opt/openwebui/.env | cut -d= -f2)
    else
      PORT=3000 # Standardwert, falls .env nicht existiert
    fi
  fi
else
  # Normale Umgebung, Port aus Docker-Container auslesen
  PORT=$(docker port open-webui 2>/dev/null | grep '8080/tcp' | awk -F':' '{print $NF}')
fi

# Wenn kein Port ermittelt werden konnte, prüfen wir beide Ports, wobei Port 4123 vorrangig ist
if [ -z "$PORT" ]; then
  echo "Kein Port ermittelt, prüfe Port 4123 und 3000"
  # Prüfe zuerst Port 4123, da dieser auf Ihrem System verwendet wird
  if curl -s --head --max-time 2 "http://localhost:4123" > /dev/null; then
    PORT=4123
    echo "OpenWebUI auf Port 4123 gefunden (Host-System-Port)"
  elif curl -s --head --max-time 2 "http://localhost:3000" > /dev/null; then
    PORT=3000
    echo "OpenWebUI auf Standard-Port 3000 gefunden"
  else
    PORT=4123 # Fallback auf 4123, da dieser auf Ihrem System verwendet wird
    echo "Keine Antwort auf beiden Ports, verwende Fallback-Port 4123"
  fi
fi
if [ -z "$PORT" ]; then
  echo "Fehler: OpenWebUI-Container wurde nicht gefunden oder der Port konnte nicht ermittelt werden!"
  docker ps --filter "name=open-webui"
  exit 1
fi

echo "OpenWebUI-Port erkannt: $PORT"

# Prüfen, ob der Service erreichbar ist
echo "Prüfe Verbindung zu http://localhost:$PORT..."

# Container-Status ausgeben, falls Docker verfügbar
if [ -S /var/run/docker.sock ]; then
  echo "Docker-Container-Status:"
  docker ps --filter "name=open-webui" --no-trunc
fi

# curl mit Timeout und maximal 3 Wiederholungen
for i in {1..3}; do
  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://localhost:$PORT")
  if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "302" ] || [ "$HTTP_CODE" = "401" ]; then
    echo "Erfolg: OpenWebUI ist erreichbar und antwortet mit Status $HTTP_CODE."
    echo "Sie können die WebUI unter http://localhost:$PORT aufrufen."
    
    # Zusätzliche Informationen ausgeben
    if [ -f "/opt/openwebui/docker-compose.yml" ]; then
      echo "Docker-Compose-Konfiguration:"
      grep -A2 "ports:" /opt/openwebui/docker-compose.yml || true
      echo "Umgebungsvariablen:"
      cat /opt/openwebui/.env 2>/dev/null || echo "Keine .env-Datei gefunden"
    fi
    
    exit 0
  else
    echo "Versuch $i: OpenWebUI antwortet mit Status $HTTP_CODE oder ist noch nicht bereit. Warte 5 Sekunden..."
    sleep 5
  fi
done

echo "Fehler: OpenWebUI konnte nicht erreicht werden."

# Erweiterte Fehlersuche
echo "Durchführe erweiterte Fehlersuche..."

# Prüfen ob docker verfügbar ist
if command -v docker &> /dev/null && [ -S /var/run/docker.sock ]; then
  echo "Docker-Container-Status:"
  docker ps -a --filter "name=open-webui"
  
  echo "Container-Logs:"
  docker logs open-webui --tail 20 2>/dev/null || echo "Keine Logs verfügbar oder Container nicht gefunden"
fi

# Prüfen ob die Docker-Compose-Datei existiert
if [ -f "/opt/openwebui/docker-compose.yml" ]; then
  echo "Docker-Compose-Konfiguration ist vorhanden:"
  cat /opt/openwebui/docker-compose.yml || echo "Datei kann nicht gelesen werden"
  
  if [ -f "/opt/openwebui/.env" ]; then
    echo "Umgebungsvariablen (.env):"
    cat /opt/openwebui/.env
  fi
fi

# Netzwerk-Ports prüfen
echo "Aktive Ports (falls netstat verfügbar):"
netstat -tulpn 2>/dev/null | grep -E "(3000|4123|8080)" || echo "Netstat nicht verfügbar oder keine passenden Ports gefunden"

echo "Test fehlgeschlagen. Bitte überprüfen Sie die Konfiguration und die Docker-Logs."
exit 1