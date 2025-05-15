#!/bin/bash

# Sicherstellen, dass Ansible installiert ist
if ! command -v ansible &> /dev/null; then
    echo "Ansible ist nicht installiert. Installation wird gestartet..."
    apt update
    apt install -y ansible
fi

# Playbooks in der richtigen Reihenfolge ausführen
ansible-playbook -i ansible/inventory/local ansible/playbooks/docker.yml
ansible-playbook -i ansible/inventory/local ansible/playbooks/services/ollama.yml
ansible-playbook -i ansible/inventory/local ansible/playbooks/services/openwebui.yml

echo "Installation abgeschlossen!"
