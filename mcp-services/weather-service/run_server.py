#!/usr/bin/env python3
"""
Docker-optimized Starter Script für MCP Weather Server
"""
import sys
import os
import logging
import signal
import asyncio
from pathlib import Path

# Logging Setup für Docker
def setup_logging():
    """Setup logging mit Fehlerbehandlung für Docker-Umgebungen"""
    handlers = [logging.StreamHandler(sys.stdout)]
    
    # Versuche Log-File zu erstellen, falls möglich
    log_dir = '/app/logs'
    log_file = os.path.join(log_dir, 'weather-service.log')
    
    try:
        if os.path.exists(log_dir):
            # Teste ob wir schreiben können
            test_file = os.path.join(log_dir, '.write_test')
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                # Wenn wir hierhin kommen, können wir schreiben
                handlers.append(logging.FileHandler(log_file))
                print(f"✅ Log-File wird erstellt: {log_file}")
            except (PermissionError, OSError) as e:
                print(f"⚠️ Kann nicht in Log-File schreiben: {e}")
                print("📝 Logs werden nur auf stdout ausgegeben")
        else:
            print(f"📁 Log-Verzeichnis existiert nicht: {log_dir}")
    except Exception as e:
        print(f"⚠️ Fehler beim Setup des Log-Files: {e}")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

# Setup logging
setup_logging()

logger = logging.getLogger(__name__)

# Füge src zum Python Path hinzu
src_path = Path(__file__).parent / 'src'
if src_path.exists():
    sys.path.insert(0, str(src_path))

try:
    from weather_server import main
except ImportError as e:
    logger.error(f"Fehler beim Importieren des Weather Servers: {e}")
    logger.error(f"Python Path: {sys.path}")
    logger.error(f"Current Directory: {os.getcwd()}")
    logger.error(f"Files in current directory: {os.listdir('.')}")
    if os.path.exists('src'):
        logger.error(f"Files in src directory: {os.listdir('src')}")
    sys.exit(1)

# Signal Handler für graceful shutdown
def signal_handler(signum, frame):
    logger.info(f"Signal {signum} empfangen. Beende Service...")
    sys.exit(0)

def setup_signal_handlers():
    """Setup signal handlers für Docker"""
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

async def start_server():
    """Startet den MCP Server mit Docker-spezifischen Einstellungen"""
    port = 8007
    
    logger.info(f"🌤️ Weather MCP Server wird gestartet...")
    logger.info(f"📊 Port: {port}")
    logger.info(f"🐳 Docker Container: {'Ja' if os.path.exists('/.dockerenv') else 'Nein'}")
    
    try:
        await main()
    except Exception as e:
        logger.error(f"Fehler beim Starten des Servers: {e}")
        raise

if __name__ == "__main__":
    setup_signal_handlers()
    
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        logger.info("Server durch Benutzer beendet")
    except Exception as e:
        logger.error(f"Unerwarteter Fehler: {e}")
        sys.exit(1)