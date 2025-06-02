#!/bin/bash
# MCP Docker Services Management Script

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"
PROJECT_NAME="mcp-services"

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Prüfe Voraussetzungen..."
    
    if ! command -v docker &> /dev/null; then
        error "Docker ist nicht installiert"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose ist nicht installiert"
        exit 1
    fi
    
    if [[ ! -f $COMPOSE_FILE ]]; then
        error "docker-compose.yml nicht gefunden"
        exit 1
    fi
    
    if [[ ! -f $ENV_FILE ]]; then
        warning ".env Datei nicht gefunden - erstelle Template..."
        create_env_template
    fi
    
    success "Voraussetzungen erfüllt"
}

# Create .env template
create_env_template() {
    cat > $ENV_FILE << 'EOF'
# MCP Services Environment Configuration
LOG_LEVEL=INFO
EOF
    
    warning "Bitte fülle die API Keys in der .env Datei aus!"
}

# Build all services
build() {
    log "Baue alle Services..."
    docker-compose build --no-cache
    success "Build abgeschlossen"
}

# Start all services
start() {
    log "Starte MCP Services..."
    
    # Clean up any orphaned networks first
    log "Räume verwaiste Netzwerke auf..."
    docker network prune -f > /dev/null 2>&1 || true
    
    # Try to remove existing network if it conflicts
    docker network rm ${PROJECT_NAME}_mcp-network > /dev/null 2>&1 || true
    
    docker-compose up -d
    
    # Wait for services to be healthy
    log "Warte auf Service-Bereitschaft..."
    sleep 10
    
    show_status
}

# Stop all services
stop() {
    log "Stoppe MCP Services..."
    docker-compose down
    success "Services gestoppt"
}

# Restart all services
restart() {
    log "Starte Services neu..."
    docker-compose down
    sleep 2
    docker-compose up -d
    sleep 10
    show_status
}

# Reload Python code without rebuild
reload() {
    local service=$1
    
    if [[ -n $service ]]; then
        log "Lade $service Service neu..."
        case $service in
            weather|weather-service)
                docker-compose restart weather-service
                ;;
            mensa|mensa-service)
                docker-compose restart mensa-service
                ;;
            *)
                warning "Unbekannter Service: $service"
                log "Verfügbare Services: weather, mensa"
                return 1
                ;;
        esac
        sleep 5
        show_status
    else
        log "Lade alle Services neu..."
        docker-compose restart
        sleep 10
        show_status
    fi
}

# Hot reload for development
dev_reload() {
    log "Development Hot-Reload..."
    
    # Nur Container neustarten, nicht rebuilden
    docker-compose restart
    
    # Kurz warten
    sleep 5
    
    log "Services neu geladen - teste APIs..."
    test_api
}

# Show service status
show_status() {
    log "Service Status:"
    echo
    
    docker-compose ps
    echo
    
    # Check individual service health
    services=("weather-proxy" "mensa-proxy")
    ports=(8007 8008)
    
    for i in "${!services[@]}"; do
        service=${services[$i]}
        port=${ports[$i]}
        
        if curl -s "http://localhost:$port/docs" > /dev/null 2>&1; then
            echo -e "  🟢 ${service} - Port $port - ${GREEN}HEALTHY (OpenAPI)${NC}"
        elif docker-compose ps -q $service | xargs docker inspect --format='{{.State.Status}}' 2>/dev/null | grep -q running; then
            echo -e "  🟡 ${service} - Port $port - ${YELLOW}RUNNING${NC}"
        else
            echo -e "  🔴 ${service} - Port $port - ${RED}STOPPED${NC}"
        fi
    done
    
    echo
    echo -e "📊 OpenAPI Services verfügbar unter:"
    echo -e "  Weather: http://localhost:8007/docs"
    echo -e "  Mensa: http://localhost:8008/docs"
    echo -e "  Weather API: http://localhost:8007"
    echo -e "  Mensa API: http://localhost:8008"
}

# Show logs
logs() {
    local service=$1
    
    if [[ -n $service ]]; then
        log "Logs für Service: $service"
        case $service in
            weather|weather-service)
                log "Docker Container Logs:"
                docker-compose logs -f weather-service
                ;;
            mensa|mensa-service)
                log "Docker Container Logs:"
                docker-compose logs -f mensa-service
                ;;
            *)
                warning "Unbekannter Service: $service"
                log "Verfügbare Services: weather, mensa"
                ;;
        esac
    else
        log "Logs für alle Services:"
        docker-compose logs -f
    fi
}

# Clean up everything
cleanup() {
    log "Räume alles auf..."
    docker-compose down -v --remove-orphans
    
    # Clean up networks specifically
    docker network rm ${PROJECT_NAME}_mcp-network > /dev/null 2>&1 || true
    docker network prune -f > /dev/null 2>&1 || true
    
    # Clean up the old logs directory if it exists and is owned by root
    if [[ -d logs && ! -w logs ]]; then
        log "Entferne altes root-logs Verzeichnis..."
        sudo rm -rf logs 2>/dev/null || true
    fi
    
    docker system prune -f
    success "Cleanup abgeschlossen"
}

# Update services
update() {
    log "Aktualisiere Services..."
    docker-compose pull
    docker-compose build --no-cache
    docker-compose down
    docker-compose up -d
    success "Update abgeschlossen"
}

# Backup configuration
backup() {
    local backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p $backup_dir
    
    log "Erstelle Backup in $backup_dir..."
    
    # Copy configuration files
    cp $COMPOSE_FILE $backup_dir/
    cp $ENV_FILE $backup_dir/
    cp nginx.conf $backup_dir/ 2>/dev/null || true
    
    # Export container configurations
    docker-compose config > $backup_dir/resolved-compose.yml
    
    success "Backup erstellt in $backup_dir"
}

# Health check
health() {
    log "Führe Gesundheitsprüfung durch..."
    
    # Check Docker daemon
    if ! docker info > /dev/null 2>&1; then
        error "Docker Daemon nicht erreichbar"
        return 1
    fi
    
    # Check compose file
    if ! docker-compose config > /dev/null 2>&1; then
        error "Docker Compose Konfiguration ungültig"
        return 1
    fi
    
    # Check services
    local healthy=0
    local total=3
    
    for port in 8007 8008 8009; do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            ((healthy++))
        fi
    done
    
    echo "Gesunde Services: $healthy/$total"
    
    if [[ $healthy -eq $total ]]; then
        success "Alle Services sind gesund"
        return 0
    else
        warning "Nicht alle Services sind gesund"
        return 1
    fi
}

# Fix network issues
fix_network() {
    log "Behebe Netzwerk-Probleme..."
    
    # Stop all services first
    docker-compose down > /dev/null 2>&1 || true
    
    # Remove conflicting networks
    log "Entferne konfliktende Netzwerke..."
    docker network rm ${PROJECT_NAME}_mcp-network > /dev/null 2>&1 || true
    
    # Prune all unused networks
    docker network prune -f > /dev/null 2>&1 || true
    
    # List remaining networks for debugging
    log "Verfügbare Netzwerke:"
    docker network ls | grep -E "(bridge|mcp)" || true
    
    success "Netzwerk-Probleme behoben"
}

# Fix permissions issues
fix_permissions() {
    log "Behebe Berechtigungsprobleme..."
    
    # Stop services first
    docker-compose down > /dev/null 2>&1 || true
    
    # Remove old problematic logs directory
    log "Entferne problematisches logs-Verzeichnis..."
    if [[ -d logs ]]; then
        if [[ ! -w logs ]]; then
            warning "Logs-Verzeichnis gehört root - verwende sudo zum Entfernen"
            sudo rm -rf logs || {
                error "Konnte logs-Verzeichnis nicht entfernen"
                return 1
            }
            success "Root-logs Verzeichnis entfernt"
        else
            rm -rf logs
            success "Logs-Verzeichnis entfernt"
        fi
    fi
    
    # Rebuild containers to ensure correct user setup
    log "Baue Container neu..."
    docker-compose build --no-cache
    
    success "Berechtigungsprobleme behoben - verwende jetzt Docker Volumes für Logs"
}

# Test API functionality
test_api() {
    log "Teste API-Funktionalität..."
    echo
    
    echo "🧪 Weather API Test:"
    if curl -s -H "Content-Type: application/json" \
            -X POST http://localhost:8007/get_weather \
            -d '{"city": "berlin"}' | jq . > /dev/null 2>&1; then
        echo -e "  ✅ Weather API funktioniert"
    else
        echo -e "  ❌ Weather API Error"
    fi
    
    echo "🧪 Mensa API Test (Live-Daten):"
    if curl -s "http://localhost:8008/mensa?days_ahead=0" | jq . > /dev/null 2>&1; then
        echo -e "  ✅ Mensa API funktioniert"
    else
        echo -e "  ❌ Mensa API Error"
    fi
    
    echo "🧪 Mensa MCP Tool Test (Live-Daten):"
    if curl -s -H "Content-Type: application/json" \
            -X POST http://localhost:8008/mcp/call \
            -d '{"name": "get_daily_menu", "arguments": {"days_ahead": 0}}' | jq . > /dev/null 2>&1; then
        echo -e "  ✅ Mensa MCP Tool funktioniert"
    else
        echo -e "  ❌ Mensa MCP Tool Error"
    fi
    
    echo "🧪 Live-Daten Test:"
    echo "  Teste echte Seezeit-Website..."
    if curl -s "https://seezeit.com/essen/speiseplaene/mensa-htwg/" > /dev/null 2>&1; then
        echo -e "  ✅ Seezeit-Website erreichbar"
    else
        echo -e "  ⚠️ Seezeit-Website nicht erreichbar (Fallback-Daten)"
    fi
}

# Container-Diagnose hinzufügen
diagnose() {
    log "Führe Container-Diagnose durch..."
    
    echo "🔍 Container Status:"
    docker-compose ps
    echo
    
    echo "🔍 Container Logs (letzte 20 Zeilen):"
    for service in weather-service mensa-service; do
        echo "--- $service ---"
        docker-compose logs --tail=20 $service 2>/dev/null || echo "Keine Logs verfügbar"
        echo
    done
    
    echo "🔍 Restart-Anzahl:"
    docker stats --no-stream --format "table {{.Container}}\t{{.Name}}" | grep mcp || echo "Keine Container gefunden"
    
    echo "🔍 Port-Status:"
    for port in 8007 8008; do
        if netstat -tuln 2>/dev/null | grep -q ":$port "; then
            echo "  ✅ Port $port: Belegt"
        else
            echo "  ❌ Port $port: Frei"
        fi
    done
    
    echo
    echo "🔍 Debug-Routen Test:"
    echo "  Teste verfügbare Endpunkte..."
    curl -s "http://localhost:8008/debug/routes" | jq . 2>/dev/null || echo "Debug-Route nicht verfügbar"
}

# Debug-Funktion hinzufügen
debug() {
    log "Debug-Informationen sammeln..."
    
    echo "🐛 Service Debug-Informationen:"
    echo
    
    # Test alle bekannten Endpunkte
    endpoints=(
        "http://localhost:8008/health"
        "http://localhost:8008/"
        "http://localhost:8008/debug/routes"
        "http://localhost:8008/mcp/resources"
        "http://localhost:8008/mcp/tools"
        "http://localhost:8008/mensa?days_ahead=0"
        "http://localhost:8008/get_daily_menu?days_ahead=0"
    )
    
    for endpoint in "${endpoints[@]}"; do
        echo "Testing: $endpoint"
        response_code=$(curl -s -o /dev/null -w "%{http_code}" "$endpoint" 2>/dev/null || echo "CONNECTION_FAILED")
        if [[ "$response_code" == "200" ]]; then
            echo -e "  ✅ $endpoint - HTTP $response_code"
        else
            echo -e "  ❌ $endpoint - HTTP $response_code"
        fi
    done
    
    echo
    echo "🔍 Container Environment:"
    docker-compose exec mensa-service env | grep -E "(MCP|DOCKER)" 2>/dev/null || echo "Kann Container-Environment nicht abrufen"
    
    echo
    echo "🔍 Live Container Logs (letzte 10 Zeilen):"
    docker-compose logs --tail=10 mensa-service 2>/dev/null || echo "Keine Container-Logs verfügbar"
    
    echo
    echo "🔍 Container Prozesse:"
    docker-compose exec mensa-service ps aux 2>/dev/null || echo "Kann Container-Prozesse nicht abrufen"
    
    echo
    echo "🔍 Python-Module Test:"
    docker-compose exec mensa-service python3 -c "import aiohttp; print('aiohttp OK')" 2>/dev/null || echo "aiohttp nicht verfügbar"
    docker-compose exec mensa-service python3 -c "import httpx; print('httpx OK')" 2>/dev/null || echo "httpx nicht verfügbar"
    docker-compose exec mensa-service python3 -c "import bs4; print('beautifulsoup4 OK')" 2>/dev/null || echo "beautifulsoup4 nicht verfügbar"
}

# Container-Neustart für problematische Services
force_restart() {
    local service=$1
    
    if [[ -z $service ]]; then
        log "Force restart für alle Services..."
        docker-compose down --remove-orphans
        docker-compose build --no-cache
        docker-compose up -d
    else
        log "Force restart für Service: $service"
        case $service in
            mensa|mensa-service)
                docker-compose stop mensa-service
                docker-compose rm -f mensa-service
                docker-compose build --no-cache mensa-service
                docker-compose up -d mensa-service
                ;;
            weather|weather-service)
                docker-compose stop weather-service
                docker-compose rm -f weather-service
                docker-compose build --no-cache weather-service
                docker-compose up -d weather-service
                ;;
            *)
                warning "Unbekannter Service: $service"
                log "Verfügbare Services: weather, mensa"
                return 1
                ;;
        esac
    fi
    
    sleep 10
    show_status
}

# Show help
show_help() {
    echo "MCP Docker Services Management"
    echo
    echo "Verwendung: $0 {command} [options]"
    echo
    echo "Befehle:"
    echo "  build              - Baue alle Services"
    echo "  start              - Starte alle Services"
    echo "  stop               - Stoppe alle Services"
    echo "  restart            - Starte Services neu"
    echo "  reload [service]   - Lade Service-Code neu (ohne Rebuild)"
    echo "  dev-reload         - Schneller Development-Reload"
    echo "  force-restart [service] - Kompletter Neustart mit Rebuild"
    echo "  status             - Zeige Service-Status"
    echo "  logs [service]     - Zeige Logs (optional: spezifischer Service)"
    echo "  cleanup            - Entferne alle Container und Volumes"
    echo "  update             - Aktualisiere Services"
    echo "  backup             - Erstelle Konfigurations-Backup"
    echo "  health             - Führe Gesundheitsprüfung durch"
    echo "  fix-network        - Behebe Netzwerk-Probleme"
    echo "  fix-permissions    - Behebe Berechtigungsprobleme"
    echo "  test               - Teste API-Funktionalität"
    echo "  diagnose           - Führe Container-Diagnose durch"
    echo "  debug              - Sammle Debug-Informationen"
    echo "  show-log-volumes   - Zeige Log-Volume Informationen"
    echo "  help               - Zeige diese Hilfe"
    echo
    echo "Beispiele:"
    echo "  $0 force-restart mensa  # Mensa Service komplett neu starten"
    echo "  $0 debug               # Debug-Informationen sammeln"
    echo "  $0 logs mensa          # Mensa Service Logs live verfolgen"
    echo
    echo "Debug-Commands:"
    echo "  curl 'http://localhost:8008/debug/routes'        # Verfügbare Routen"
    echo "  curl 'http://localhost:8008/get_daily_menu?days_ahead=0' # MCPO Endpoint"
}

# Main logic
case "${1:-help}" in
    build)
        check_prerequisites
        build
        ;;
    start)
        check_prerequisites
        start
        ;;
    stop)
        stop
        ;;
    restart)
        check_prerequisites
        restart
        ;;
    reload)
        reload $2
        ;;
    dev-reload)
        dev_reload
        ;;
    force-restart)
        force_restart $2
        ;;
    status)
        show_status
        ;;
    logs)
        logs $2
        ;;
    cleanup)
        cleanup
        ;;
    update)
        check_prerequisites
        update
        ;;
    backup)
        backup
        ;;
    health)
        health
        ;;
    fix-network)
        fix_network
        ;;
    fix-permissions)
        fix_permissions
        ;;
    test)
        test_api
        ;;
    diagnose)
        diagnose
        ;;
    debug)
        debug
        ;;
    help|*)
        show_help
        ;;
esac