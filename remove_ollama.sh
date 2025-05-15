#!/bin/bash
# Skript zum Entfernen aller Ollama-Komponenten, die durch Ansible installiert wurden
echo "Entferne Ollama und alle zugehörigen Komponenten..."
# 1. Stoppe und deaktiviere den Ollama-Service (wenn systemd verwendet wird)
if systemctl list-unit-files | grep -q ollama.service; then
  echo "Stoppe und deaktiviere Ollama-Systemdienst..."
  sudo systemctl stop ollama
  sudo systemctl disable ollama
  sudo rm -f /etc/systemd/system/ollama.service
  sudo systemctl daemon-reload
fi
# 2. Stoppe Ollama-Prozess in Docker-Umgebung
if [ -f /usr/local/bin/stop-ollama.sh ]; then
  echo "Stoppe Ollama-Prozess in Docker-Umgebung..."
  sudo /usr/local/bin/stop-ollama.sh
fi
# 3. Entferne Ollama-Binärdatei
if [ -f /usr/local/bin/ollama ]; then
  echo "Entferne Ollama-Binärdatei..."
  sudo rm -f /usr/local/bin/ollama
fi
# 4. Entferne Hilfsskripte
echo "Entferne Hilfsskripte..."
sudo rm -f /usr/local/bin/start-ollama.sh
sudo rm -f /usr/local/bin/stop-ollama.sh
# 5. Entferne Datenverzeichnis
echo "Entferne Ollama-Datenverzeichnis..."
sudo rm -rf /var/lib/ollama
# 6. Entferne Ollama-Benutzer und -Gruppe
if id ollama &>/dev/null; then
  echo "Entferne Ollama-Benutzer und -Gruppe..."
  sudo userdel -r ollama 2>/dev/null
  sudo groupdel ollama 2>/dev/null
fi
# 7. Entferne PID-Datei
sudo rm -f /var/run/ollama.pid
# 8. Entferne Log-Datei
sudo rm -f /var/log/ollama.log
echo "Ollama wurde vollständig entfernt."
