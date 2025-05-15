#!/bin/bash
# Script: test-ansible-docker.sh
# Description: A simple menu-driven script to spin up a Docker container, run an Ansible playbook against it,
# and reset the environment. Tailored for your LLM project documentation.
# Assumes Ansible playbooks are in /ansible/playbooks/.
# Default playbook: /ansible/playbooks/test-playbook.yml (change as needed).

# Configuration variables
DOCKER_IMAGE="ubuntu:24.04"  # Base image for the container
CONTAINER_NAME="llm-test-container"  # Name of the container
NETWORK_NAME="llm-test-network"  # Custom Docker network
STATIC_IP="172.18.0.2"  # Static IP for the container
SSH_USER="ansible_user"  # User for SSH in the container
SSH_PASSWORD="ansible_test"  # Password for SSH (change for security)
PLAYBOOK_PATH="./../ansible/playbooks/test-playbook.yml"  # Path to your Ansible playbook

# Function to create the custom network if it doesn't exist
create_network() {
    if ! docker network ls | grep -q "$NETWORK_NAME"; then
        echo "Creating custom network '$NETWORK_NAME'..."
        docker network create --driver bridge "$NETWORK_NAME"
    fi
}

# Function to start and prepare the container
start_container() {
    create_network  # Ensure the network exists
    
    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        echo "Container '$CONTAINER_NAME' is already running. Skipping creation."
        return
    fi
    
    echo "Starting container '$CONTAINER_NAME' with static IP '$STATIC_IP'..."
    docker run -d --name "$CONTAINER_NAME" \
        --net "$NETWORK_NAME" \
        --ip "$STATIC_IP" \
        "$DOCKER_IMAGE" \
        tail -f /dev/null  # Keep the container running
    
    # Wait for the container to start
    sleep 5
    
    # Install and configure SSH in the container
    echo "Preparing container: Installing SSH and setting up user..."
    docker exec "$CONTAINER_NAME" apt update
    docker exec "$CONTAINER_NAME" apt install -y openssh-server
    docker exec "$CONTAINER_NAME" useradd -m -s /bin/bash "$SSH_USER"
    echo "$SSH_USER:$SSH_PASSWORD" | docker exec -i "$CONTAINER_NAME" chpasswd
    docker exec "$CONTAINER_NAME" mkdir -p /home/$SSH_USER/.ssh
    docker exec "$CONTAINER_NAME" ssh-keygen -A  # Generate host keys
    docker exec "$CONTAINER_NAME" service ssh start  # Start SSH service
    
    echo "Container is ready at IP: $STATIC_IP"
}

# Function to run the Ansible playbook
run_ansible_playbook() {
    start_container  # Ensure the container is running
    
    # Create a temporary Ansible inventory file
    INVENTORY_FILE="./inventory.txt"
    echo "[test_group]" > "$INVENTORY_FILE"
    echo "$STATIC_IP ansible_user=$SSH_USER ansible_ssh_pass=$SSH_PASSWORD ansible_connection=ssh" >> "$INVENTORY_FILE"
    
    echo "Running Ansible playbook '$PLAYBOOK_PATH' against the container..."
    ansible-playbook -i "$INVENTORY_FILE" "$PLAYBOOK_PATH"
    
    # Clean up the inventory file
    rm "$INVENTORY_FILE"
}

# Function to reset (stop and remove) the container
reset_container() {
    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        echo "Stopping and removing container '$CONTAINER_NAME'..."
        docker stop "$CONTAINER_NAME"
        docker rm "$CONTAINER_NAME"
    elif [ "$(docker ps -a -q -f name=$CONTAINER_NAME)" ]; then
        echo "Removing stopped container '$CONTAINER_NAME'..."
        docker rm "$CONTAINER_NAME"
    else
        echo "No container to remove."
    fi
}

# Main menu loop
while true; do
    echo "=== Ansible Testing Menu ==="
    echo "1. Start and test (Spin up container and run playbook)"
    echo "2. Reset (Stop and remove container)"
    echo "3. Exit"
    read -p "Choose an option: " choice
    
    case $choice in
        1)
            run_ansible_playbook
            ;;
        2)
            reset_container
            ;;
        3)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid option. Please try again."
            ;;
    esac
done