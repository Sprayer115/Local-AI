#!/bin/bash
# Script: test-ansible-docker.sh
# Description: A simple menu-driven script to spin up a Docker container, run an Ansible playbook against it,
# and reset the environment. Tailored for your LLM project documentation.
# Assumes Ansible playbooks are in /ansible/playbooks/.
# Default playbook: /ansible/playbooks/test-playbook.yml (change as needed).

# Configuration variables
DOCKER_IMAGE="ubuntu:24.04"  # Base image for the container
CONTAINER_NAME="llm-test-container-$(date +%s)"  # Name of the container with timestamp to ensure uniqueness
NETWORK_NAME="llm-test-network-$(date +%s)"  # Custom Docker network with timestamp
STATIC_IP="192.168.200.2"  # Static IP für den Container - ungewöhnlicher IP-Bereich
SSH_USER="ansible_user"  # User for SSH in the container
SSH_PASSWORD="ansible_test"  # Password for SSH (change for security)
# Array mit Playbooks, die nacheinander ausgeführt werden sollen
PLAYBOOKS=("./../ansible/playbooks/docker.yml" "./../ansible/playbooks/services/ollama.yml" "./../ansible/playbooks/services/apache.yml")
# Fallback für einzelnes Playbook zur Kompatibilität
PLAYBOOK_PATH="./../ansible/playbooks/docker.yml"  # Path to your Ansible playbook (Wird nur verwendet, wenn Array nicht genutzt wird)
CUSTOM_INVENTORY=""  # Benutzerdefinierte Inventory-Datei (leer = generierte Datei verwenden)
USE_CACHE="true"  # Container-Image zwischenspeichern und vorbereitetes Image verwenden

# Function to create the custom network if it doesn't exist
create_network() {
    # Remove network if it already exists to ensure clean configuration
    if docker network ls | grep -q "$NETWORK_NAME"; then
        echo "Removing existing network '$NETWORK_NAME'..."
        docker network rm "$NETWORK_NAME" || true
    fi
    
    echo "Creating custom network '$NETWORK_NAME' with subnet..."
    # Create network with explicit gateway to ensure proper subnet configuration
    docker network create --driver bridge --subnet=192.168.200.0/24 --gateway=192.168.200.1 "$NETWORK_NAME"
    
    # Verify network was created successfully
    if ! docker network ls | grep -q "$NETWORK_NAME"; then
        echo "Fehler: Netzwerk konnte nicht erstellt werden!"
        return 1
    fi
    echo "Netzwerk '$NETWORK_NAME' erfolgreich erstellt."
}

# Function to start and prepare the container
start_container() {
    create_network  # Ensure the network exists
    
    # Entferne mögliche alte SSH-Host-Schlüssel für die Ziel-IP, um SSH-Warnungen zu vermeiden
    echo "Entferne alte SSH-Host-Schlüssel für $STATIC_IP..."
    ssh-keygen -f "$HOME/.ssh/known_hosts" -R "$STATIC_IP" 2>/dev/null || true
    
    # Prüfen, ob der Container bereits läuft
    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        echo "Container '$CONTAINER_NAME' is already running. Skipping creation."
        return
    fi
    
    # Check if we should use a cached image
    USING_CACHED_IMAGE=false
    CACHED_IMAGE="ansible-ready-ubuntu:latest"
    if [ "$USE_CACHE" = "true" ] && docker image inspect "$CACHED_IMAGE" &>/dev/null; then
        echo "Verwende vorkonfiguriertes Container-Image mit SSH und Python..."
        DOCKER_IMAGE="$CACHED_IMAGE"
        USING_CACHED_IMAGE=true
    fi
    
    echo "Starting container '$CONTAINER_NAME' with static IP '$STATIC_IP'..."
    docker run -d --name "$CONTAINER_NAME" \
        --net "$NETWORK_NAME" \
        --ip "$STATIC_IP" \
        -v /var/run/docker.sock:/var/run/docker.sock \
        "$DOCKER_IMAGE" \
        tail -f /dev/null  # Keep the container running
    
    # Überprüfen, ob der Container erfolgreich gestartet wurde
    if [ $? -ne 0 ]; then
        echo "Fehler beim Starten des Containers. Versuche es ohne statische IP..."
        # Alternative ohne statische IP
        docker run -d --name "$CONTAINER_NAME" \
            --net "$NETWORK_NAME" \
            "$DOCKER_IMAGE" \
            tail -f /dev/null
        
        # Container-IP für das spezifische Netzwerk abrufen
        STATIC_IP=$(docker inspect -f "{{range .NetworkSettings.Networks}}{{if eq \"$NETWORK_NAME\" \"$.NetworkID\"}}{{.IPAddress}}{{end}}{{end}}" "$CONTAINER_NAME")
        
        # Wenn die IP immer noch leer ist, versuchen wir es mit einer allgemeineren Methode
        if [ -z "$STATIC_IP" ]; then
            STATIC_IP=$(docker inspect -f "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}" "$CONTAINER_NAME")
        fi
        
        echo "Container wurde mit dynamischer IP gestartet: $STATIC_IP"
    fi
    
    # Wait for the container to start
    sleep 5
    
    if [ "$USING_CACHED_IMAGE" = "true" ]; then
        # Bei Verwendung des gecachten Images nur den SSH-Dienst starten
        echo "Container verwendet fertiges Image. Starte SSH-Dienst..."
        docker exec "$CONTAINER_NAME" mkdir -p /run/sshd
        docker exec "$CONTAINER_NAME" /usr/sbin/sshd || { echo "Fehler beim Starten des SSH-Dienstes"; return 1; }
    else
        # Bei neuem Image komplett einrichten
        echo "Preparing container: Installing SSH and setting up user..."
        
        # Aktualisieren und Installieren mit Fehlerbehandlung
        docker exec "$CONTAINER_NAME" apt update || { echo "Fehler beim APT-Update"; return 1; }
        docker exec "$CONTAINER_NAME" apt install -y openssh-server sudo net-tools procps python3 python3-pip || { echo "Fehler bei der Installation von SSH und Netzwerk-Tools"; return 1; }
        
        # Python-Module für Ansible Docker-Module installieren
        docker exec "$CONTAINER_NAME" pip3 install requests docker || { echo "Fehler bei der Installation der Python-Module"; return 1; }
        
        # Benutzer anlegen oder aktualisieren
        if docker exec "$CONTAINER_NAME" id -u "$SSH_USER" &>/dev/null; then
            echo "Benutzer '$SSH_USER' existiert bereits, aktualisiere Passwort..."
            echo "$SSH_USER:$SSH_PASSWORD" | docker exec -i "$CONTAINER_NAME" chpasswd
            docker exec "$CONTAINER_NAME" usermod -aG sudo "$SSH_USER" || true
        else
            echo "Erstelle neuen Benutzer '$SSH_USER'..."
            docker exec "$CONTAINER_NAME" useradd -m -s /bin/bash "$SSH_USER" || { echo "Fehler beim Anlegen des Benutzers"; return 1; }
            echo "$SSH_USER:$SSH_PASSWORD" | docker exec -i "$CONTAINER_NAME" chpasswd
            docker exec "$CONTAINER_NAME" usermod -aG sudo "$SSH_USER"
        fi
        
        # SSH-Konfiguration
        docker exec "$CONTAINER_NAME" mkdir -p /home/$SSH_USER/.ssh
        docker exec "$CONTAINER_NAME" chown -R $SSH_USER:$SSH_USER /home/$SSH_USER/.ssh
        docker exec "$CONTAINER_NAME" ssh-keygen -A  # Generate host keys
        
        # SSH-Konfiguration für PasswordAuthentication sicherstellen
        docker exec "$CONTAINER_NAME" bash -c "echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config"
        docker exec "$CONTAINER_NAME" bash -c "echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config"
        
        # SSH-Dienst starten
        echo "Starte SSH-Dienst..."
        docker exec "$CONTAINER_NAME" mkdir -p /run/sshd
        docker exec "$CONTAINER_NAME" /usr/sbin/sshd
    fi
    
    # Kurz warten und prüfen, ob SSH-Dienst läuft
    sleep 3
    if docker exec "$CONTAINER_NAME" pgrep sshd > /dev/null; then
        echo "SSH-Dienst wurde erfolgreich gestartet."
    else
        echo "SSH-Dienst konnte nicht gestartet werden. Versuche erneut..."
        docker exec "$CONTAINER_NAME" /usr/sbin/sshd -D &
    fi
    
    echo "Container is ready at IP: $STATIC_IP"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Function to run the Ansible playbook
run_ansible_playbook() {
    # Check if ansible-playbook is installed
    if ! command_exists ansible-playbook; then
        echo "Error: ansible-playbook is not installed. Please install it with:"
        echo "sudo apt install ansible"
        return 1
    fi
    
    # Check if sshpass is installed (needed for password authentication)
    if ! command_exists sshpass; then
        echo "Error: sshpass is not installed but required for password-based SSH connections."
        echo "Please install it with: sudo apt install sshpass"
        return 1
    fi
    
    if ! start_container; then
        echo "Fehler beim Starten des Containers. Abbruch."
        return 1
    fi
    
    # Test SSH-Verbindung mit verbesserter Fehlerbehandlung
    echo "Teste SSH-Verbindung zum Container..."
    MAX_RETRIES=10  # Erhöht auf 10 für mehr Geduld
    RETRY_COUNT=0
    
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        # Prüfe, ob SSH läuft - verschiedene Methoden, da nicht alle Container die gleichen Tools haben
        if docker exec "$CONTAINER_NAME" bash -c "command -v ss &>/dev/null && ss -tnl | grep -q ':22' || netstat -tnl 2>/dev/null | grep -q ':22' || ps aux | grep -v grep | grep -q sshd"; then
            echo "SSH-Port ist offen oder SSH-Dienst läuft. Fahre fort..."
            break
        fi
        echo "Warte auf SSH-Service... ($((RETRY_COUNT+1))/$MAX_RETRIES)"
        RETRY_COUNT=$((RETRY_COUNT+1))

        # Versuche SSH-Dienst neu zu starten, wenn er nicht läuft
        if [ $RETRY_COUNT -ge 2 ]; then
            echo "Starte SSH-Dienst erneut..."
            docker exec "$CONTAINER_NAME" pkill -9 sshd || true
            sleep 1
            docker exec "$CONTAINER_NAME" mkdir -p /run/sshd
            docker exec "$CONTAINER_NAME" /usr/sbin/sshd
        fi
        
        sleep 3
    done
    
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        echo "SSH-Service konnte nicht gestartet werden. Führe finale Diagnose durch..."
        docker exec "$CONTAINER_NAME" bash -c "ps aux | grep sshd"
        docker exec "$CONTAINER_NAME" bash -c "cat /etc/ssh/sshd_config | tail -10"
        
        echo "Versuche letzten SSH-Neustart mit Debug-Modus..."
        docker exec "$CONTAINER_NAME" pkill -9 sshd || true
        docker exec "$CONTAINER_NAME" /usr/sbin/sshd -d &
        sleep 3
    fi
    


    # Create or use inventory file based on settings
    if [ -z "$CUSTOM_INVENTORY" ]; then
        # Erstelle temporäre Inventory-Datei
        INVENTORY_FILE="./inventory_$(date +%s).txt"
        echo "[llm_servers]" > "$INVENTORY_FILE"
        echo "$STATIC_IP ansible_user=$SSH_USER ansible_ssh_pass=$SSH_PASSWORD ansible_sudo_pass=$SSH_PASSWORD ansible_connection=ssh ansible_ssh_common_args='-o StrictHostKeyChecking=no'" >> "$INVENTORY_FILE"
        CLEANUP_INVENTORY=true
    else
        # Verwende benutzerdefinierte Inventory-Datei
        INVENTORY_FILE="$CUSTOM_INVENTORY"
        CLEANUP_INVENTORY=false
        echo "Verwende benutzerdefinierte Inventory-Datei: $INVENTORY_FILE"
    fi
    
    # Überprüfen, ob wir das Array oder den einzelnen Playbook-Pfad verwenden
    if [ ${#PLAYBOOKS[@]} -gt 0 ]; then
        echo "Es werden ${#PLAYBOOKS[@]} Playbooks nacheinander ausgeführt:"
        for playbook in "${PLAYBOOKS[@]}"; do
            echo " - $playbook"
        done
        
        # Über alle Playbooks im Array iterieren
        OVERALL_RESULT=0
        
        for (( i=0; i<${#PLAYBOOKS[@]}; i++ )); do
            CURRENT_PLAYBOOK="${PLAYBOOKS[$i]}"
            echo -e "\n\033[1;34m=== Führe Playbook $((i+1))/${#PLAYBOOKS[@]} aus: $CURRENT_PLAYBOOK ===\033[0m"
            
            ansible-playbook -i "$INVENTORY_FILE" "$CURRENT_PLAYBOOK" -v
            ANSIBLE_RESULT=$?
            
            if [ $ANSIBLE_RESULT -ne 0 ]; then
                echo -e "\033[1;31mAnsible-Playbook '$CURRENT_PLAYBOOK' fehlgeschlagen!\033[0m"
                OVERALL_RESULT=1
                break  # Abbruch bei Fehler
            else
                echo -e "\033[1;32mAnsible-Playbook '$CURRENT_PLAYBOOK' erfolgreich ausgeführt.\033[0m"
            fi
        done
    else
        # Fallback für den Fall, dass das Array leer ist
        echo "Führe einzelnes Playbook '$PLAYBOOK_PATH' aus..."
        ansible-playbook -i "$INVENTORY_FILE" "$PLAYBOOK_PATH" -v
        OVERALL_RESULT=$?
    fi
    
    # Clean up the inventory file if needed
    if [ "$CLEANUP_INVENTORY" = "true" ]; then
        rm -f "$INVENTORY_FILE"
    fi
    
    if [ $OVERALL_RESULT -ne 0 ]; then
        echo "Ansible-Playbook-Ausführung fehlgeschlagen. Container bleibt für Fehlersuche erhalten."
        echo "SSH-Verbindung möglich mit: ssh $SSH_USER@$STATIC_IP (Passwort: $SSH_PASSWORD)"
    else
        echo "Alle Ansible-Playbooks wurden erfolgreich ausgeführt."
    fi
}

# Function to create a cached image
create_cached_image() {
    echo "Erstelle ein gecachtes Container-Image mit vorinstalliertem SSH und Python..."
    
    # Temporäre Container- und Netzwerknamen
    TEMP_CONTAINER="temp-ansible-cache-container"
    TEMP_NETWORK="temp-ansible-network"
    
    # Erstelle temporäres Netzwerk
    docker network create --driver bridge "$TEMP_NETWORK"
    
    # Starte Container
    docker run -d --name "$TEMP_CONTAINER" \
        --net "$TEMP_NETWORK" \
        "$DOCKER_IMAGE" \
        tail -f /dev/null
    
    # Container vorbereiten
    docker exec "$TEMP_CONTAINER" apt update
    docker exec "$TEMP_CONTAINER" apt install -y openssh-server sudo net-tools procps python3 python3-pip
    
    
    # Benutzer anlegen oder aktualisieren
    if docker exec "$TEMP_CONTAINER" id -u "$SSH_USER" &>/dev/null; then
        echo "Benutzer '$SSH_USER' existiert bereits, aktualisiere Passwort..."
        echo "$SSH_USER:$SSH_PASSWORD" | docker exec -i "$TEMP_CONTAINER" chpasswd
        docker exec "$TEMP_CONTAINER" usermod -aG sudo "$SSH_USER" || true
    else
        echo "Erstelle neuen Benutzer '$SSH_USER'..."
        docker exec "$TEMP_CONTAINER" useradd -m -s /bin/bash "$SSH_USER"
        echo "$SSH_USER:$SSH_PASSWORD" | docker exec -i "$TEMP_CONTAINER" chpasswd
        docker exec "$TEMP_CONTAINER" usermod -aG sudo "$SSH_USER"
    fi
    
    # SSH konfigurieren
    docker exec "$TEMP_CONTAINER" mkdir -p /home/$SSH_USER/.ssh
    docker exec "$TEMP_CONTAINER" chown -R $SSH_USER:$SSH_USER /home/$SSH_USER/.ssh
    docker exec "$TEMP_CONTAINER" ssh-keygen -A
    docker exec "$TEMP_CONTAINER" bash -c "echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config"
    docker exec "$TEMP_CONTAINER" bash -c "echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config"
    docker exec "$TEMP_CONTAINER" mkdir -p /run/sshd
    
    # Container als Image speichern
    docker commit "$TEMP_CONTAINER" "ansible-ready-ubuntu:latest"
    
    # Aufräumen
    docker stop "$TEMP_CONTAINER"
    docker rm "$TEMP_CONTAINER"
    docker network rm "$TEMP_NETWORK"
    
    echo "Gecachtes Image 'ansible-ready-ubuntu:latest' erfolgreich erstellt!"
}

# Function to reset (stop and remove) the container and network
reset_container() {
    # Entferne Container, die mit dem Präfix beginnen
    CONTAINERS=$(docker ps -a --filter "name=llm-test-container-" -q)
    if [ -n "$CONTAINERS" ]; then
        echo "Stopping and removing containers that match 'llm-test-container-'..."
        docker stop $CONTAINERS 2>/dev/null || true
        docker rm $CONTAINERS 2>/dev/null || true
    fi
    
    # Auch den alten Container ohne Zeitstempel entfernen, falls vorhanden
    if docker ps -a --filter "name=llm-test-container" -q | grep -q .; then
        echo "Removing old container 'llm-test-container'..."
        docker stop llm-test-container 2>/dev/null || true
        docker rm llm-test-container 2>/dev/null || true
    fi
    
    # Entferne Netzwerke, die mit dem Präfix beginnen
    NETWORKS=$(docker network ls --filter "name=llm-test-network-" -q)
    if [ -n "$NETWORKS" ]; then
        echo "Removing networks that match 'llm-test-network-'..."
        for NET in $NETWORKS; do
            docker network rm $NET 2>/dev/null || true
        done
    fi
    
    # Auch das alte Netzwerk ohne Zeitstempel entfernen, falls vorhanden
    if docker network ls --filter "name=llm-test-network" -q | grep -q .; then
        echo "Removing old network 'llm-test-network'..."
        docker network rm llm-test-network 2>/dev/null || true
    fi
    
    echo "Cleanup abgeschlossen!"
}

# Funktion zur Verwaltung der Playbooks
manage_playbooks() {
    clear
    echo "=== Playbook-Verwaltung ==="
    echo "Aktuell konfigurierte Playbooks:"
    
    if [ ${#PLAYBOOKS[@]} -eq 0 ]; then
        echo "Keine Playbooks im Array konfiguriert. Es wird das Standard-Playbook verwendet:"
        echo "- $PLAYBOOK_PATH"
    else
        for (( i=0; i<${#PLAYBOOKS[@]}; i++ )); do
            echo "$((i+1)). ${PLAYBOOKS[$i]}"
        done
    fi
    
    echo ""
    echo "1. Playbooks zurücksetzen und neues hinzufügen"
    echo "2. Weiteres Playbook hinzufügen"
    echo "3. Playbook entfernen"
    echo "4. Zurück zum Konfigurationsmenü"
    
    read -p "Wähle eine Option: " playbook_choice
    
    case $playbook_choice in
        1)
            # Zurücksetzen und neues hinzufügen
            PLAYBOOKS=()
            read -p "Pfad zum Playbook (./../ansible/playbooks/...): " new_playbook
            if [ -n "$new_playbook" ]; then
                PLAYBOOKS+=("$new_playbook")
                PLAYBOOK_PATH="$new_playbook"  # Auch für Einzelkompatibilität setzen
                echo "Playbook-Liste zurückgesetzt und neues Playbook hinzugefügt: $new_playbook"
            fi
            ;;
        2)
            # Weiteres hinzufügen
            read -p "Pfad zum weiteren Playbook: " new_playbook
            if [ -n "$new_playbook" ]; then
                PLAYBOOKS+=("$new_playbook")
                echo "Playbook hinzugefügt: $new_playbook"
            fi
            ;;
        3)
            # Entfernen
            if [ ${#PLAYBOOKS[@]} -eq 0 ]; then
                echo "Keine Playbooks zum Entfernen vorhanden."
            else
                read -p "Nummer des zu entfernenden Playbooks (1-${#PLAYBOOKS[@]}): " remove_num
                if [[ "$remove_num" =~ ^[0-9]+$ ]] && [ "$remove_num" -ge 1 ] && [ "$remove_num" -le "${#PLAYBOOKS[@]}" ]; then
                    removed_playbook="${PLAYBOOKS[$((remove_num-1))]}"
                    # Array manipulieren: Element entfernen
                    unset "PLAYBOOKS[$((remove_num-1))]"
                    # Array neu indizieren
                    PLAYBOOKS=("${PLAYBOOKS[@]}")
                    echo "Playbook entfernt: $removed_playbook"
                    
                    # Falls Array leer, setze Standard-Playbook
                    if [ ${#PLAYBOOKS[@]} -eq 0 ]; then
                        PLAYBOOK_PATH="./../ansible/playbooks/docker.yml"
                        echo "Playbook-Array ist leer. Standard-Playbook gesetzt: $PLAYBOOK_PATH"
                    fi
                else
                    echo "Ungültige Auswahl."
                fi
            fi
            ;;
        4)
            return
            ;;
        *)
            echo "Ungültige Option."
            ;;
    esac
    
    read -p "Drücke Enter, um fortzufahren..."
    manage_playbooks
}

# Funktion zum Anzeigen des Konfigurationsmenüs
show_config_menu() {
    clear
    echo "=== Konfigurationsmenü ==="
    echo "1. Ansible Playbooks verwalten"
    echo "2. Benutzerdefiniertes Inventory festlegen (aktuell: ${CUSTOM_INVENTORY:-Automatisch generiert})"
    echo "3. SSH Benutzer ändern (aktuell: $SSH_USER)"
    echo "4. SSH Passwort ändern (aktuell: $SSH_PASSWORD)"
    echo "5. Container-Caching aktivieren/deaktivieren (aktuell: $USE_CACHE)"
    echo "6. Zurück zum Hauptmenü"
    
    read -p "Wähle eine Option: " config_choice
    
    case $config_choice in
        1)
            manage_playbooks
            ;;
        2)
            echo "Verfügbare Inventory-Dateien:"
            echo "1. Automatisch generieren"
            echo "2. ./../ansible/inventory/hosts (Remote Server)"
            echo "3. ./../ansible/inventory/local (Localhost)"
            echo "4. Eigenen Pfad eingeben"
            
            read -p "Wähle eine Option: " inv_choice
            
            case $inv_choice in
                1) CUSTOM_INVENTORY=""; echo "Verwende automatisch generiertes Inventory." ;;
                2) CUSTOM_INVENTORY="./../ansible/inventory/hosts"; echo "Verwende hosts Inventory." ;;
                3) CUSTOM_INVENTORY="./../ansible/inventory/local"; echo "Verwende local Inventory." ;;
                4) 
                    read -p "Eigener Inventory-Pfad: " custom_path
                    if [ -n "$custom_path" ]; then
                        CUSTOM_INVENTORY="$custom_path"
                        echo "Verwende benutzerdefiniertes Inventory: $CUSTOM_INVENTORY"
                    fi
                    ;;
            esac
            ;;
        3)
            read -p "Neuer SSH Benutzer: " new_user
            if [ -n "$new_user" ]; then
                SSH_USER="$new_user"
                echo "SSH-Benutzer aktualisiert auf: $SSH_USER"
            fi
            ;;
        4)
            read -p "Neues SSH Passwort: " new_pass
            if [ -n "$new_pass" ]; then
                SSH_PASSWORD="$new_pass"
                echo "SSH-Passwort aktualisiert."
            fi
            ;;
        5)
            if [ "$USE_CACHE" = "true" ]; then
                USE_CACHE="false"
                echo "Container-Caching deaktiviert."
            else
                USE_CACHE="true"
                echo "Container-Caching aktiviert."
            fi
            ;;
        6)
            return
            ;;
        *)
            echo "Ungültige Option. Bitte erneut versuchen."
            ;;
    esac
    
    read -p "Drücke Enter, um fortzufahren..."
    show_config_menu
}

# Main menu loop
while true; do
    clear
    echo "=== Ansible Testing Menu ==="
    
    # Zeige konfigurierte Playbooks an
    if [ ${#PLAYBOOKS[@]} -gt 0 ]; then
        echo -e "\033[1;36mKonfigurierte Playbooks:\033[0m"
        for playbook in "${PLAYBOOKS[@]}"; do
            echo " - $playbook"
        done
        echo ""
    fi
    
    echo "1. Start and test (Container starten und Playbooks ausführen)"
    echo "2. Reset (Container und Netzwerk entfernen)"
    echo "3. Konfiguration anpassen"
    echo "4. Container-Image für Caching erstellen"
    echo "5. Exit"
    read -p "Wähle eine Option: " choice
    
    case $choice in
        1)
            run_ansible_playbook
            read -p "Drücke Enter, um fortzufahren..."
            ;;
        2)
            reset_container
            read -p "Drücke Enter, um fortzufahren..."
            ;;
        3)
            show_config_menu
            ;;
        4)
            create_cached_image
            read -p "Drücke Enter, um fortzufahren..."
            ;;
        5)
            echo "Beende..."
            exit 0
            ;;
        *)
            echo "Ungültige Option. Bitte erneut versuchen."
            read -p "Drücke Enter, um fortzufahren..."
            ;;
    esac
done