#!/usr/bin/env python3
"""
MCP Mensa Server - Echter Service für HTWG Mensapläne
"""

import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
)
from pydantic import BaseModel
import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Server initialisieren
app = Server("mensa-server")

# Pydantic Models
class MensaDish(BaseModel):
    name: str
    price: str
    category: str
    dietary_icons: List[str] = []
    allergens: List[str] = []

class MensaMenu(BaseModel):
    date: str
    day_name: str
    dishes: List[MensaDish]

class MensaData(BaseModel):
    name: str
    location: str
    menus: List[MensaMenu]

def parse_dietary_icons(icon_elements):
    """Parse dietary restriction icons"""
    icons = []
    if icon_elements:
        for icon in icon_elements:
            class_name = icon.get('class', [])
            if 'speiseplanTagKatIcon' in class_name:
                for cls in class_name:
                    if cls in ['Veg', 'Vegan', 'B', 'R', 'G', 'F', 'Sch', 'L', 'W']:
                        icons.append(cls)
    return icons

def parse_allergens(allergen_text):
    """Extract allergen codes from text"""
    if not allergen_text:
        return []
    
    # Find allergen codes in format (25a,26) or (25a) etc.
    allergen_pattern = r'\(([^)]+)\)'
    matches = re.findall(allergen_pattern, allergen_text)
    
    allergens = []
    for match in matches:
        # Split by comma and clean up
        codes = [code.strip() for code in match.split(',')]
        allergens.extend(codes)
    
    return allergens

def clean_dish_name(dish_name):
    """Clean dish name by removing allergen codes"""
    if not dish_name:
        return ""
    
    # Remove allergen codes in parentheses
    cleaned = re.sub(r'\s*\([^)]*\)\s*', ' ', dish_name)
    # Remove extra whitespace
    cleaned = ' '.join(cleaned.split())
    return cleaned.strip()

def parse_price(price_text):
    """Extract student price from price text"""
    if not price_text:
        return "Preis nicht verfügbar"
    
    # Look for student price pattern: "X,XX € Studierende"
    student_price_pattern = r'(\d+,\d+)\s*€\s*Studierende'
    match = re.search(student_price_pattern, price_text)
    
    if match:
        return f"{match.group(1)}€"
    
    # Fallback: look for any price
    price_pattern = r'(\d+,\d+)\s*€'
    match = re.search(price_pattern, price_text)
    
    if match:
        return f"{match.group(1)}€"
    
    return "Preis nicht verfügbar"

async def fetch_mensa_data() -> Dict[str, Any]:
    """Fetch real mensa data from Seezeit website"""
    url = "https://seezeit.com/essen/speiseplaene/mensa-htwg/"
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            logger.info(f"Lade Mensa-Daten von {url}...")
            response = await client.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Debug: Log the structure we find
            tabs = soup.find_all('a', class_='tab')
            logger.info(f"Found {len(tabs)} tabs on the website")
            
            menus = []
            
            for i, tab in enumerate(tabs):
                if i >= 5:  # Nur die ersten 5 Werktage
                    break
                    
                # Extract day name and date from tab
                tab_text = tab.get_text(strip=True)
                logger.info(f"Verarbeite Tab {i}: {tab_text}")
                
                # Find corresponding content - handle array rel attributes
                tab_rel_raw = tab.get('rel')
                logger.info(f"Tab rel attribute (raw): {tab_rel_raw}")
                
                # Extract actual rel value (handle list format)
                if isinstance(tab_rel_raw, list) and tab_rel_raw:
                    tab_rel = tab_rel_raw[0]
                else:
                    tab_rel = str(tab_rel_raw) if tab_rel_raw else ""
                
                logger.info(f"Tab rel attribute (cleaned): {tab_rel}")
                
                if tab_rel:
                    # Try different content selectors with corrected rel value
                    content_div = None
                    
                    # Try various selector patterns
                    selectors = [
                        f'div.contents.contents_{tab_rel}',
                        f'div#contents_{tab_rel}',
                        f'div.contents_{tab_rel}',
                        f'div[class*="contents_{tab_rel}"]',
                        f'div[id*="contents_{tab_rel}"]'
                    ]
                    
                    for selector in selectors:
                        try:
                            content_divs = soup.select(selector)
                            if content_divs:
                                content_div = content_divs[0]
                                logger.info(f"Found content div with selector: {selector}")
                                break
                        except Exception as e:
                            logger.warning(f"Selector '{selector}' failed: {e}")
                            continue
                    
                    # If still no content div found, try fallback approach
                    if not content_div:
                        # Look for any div with "contents" class
                        all_content_divs = soup.select('div[class*="contents"]')
                        if len(all_content_divs) > i:
                            content_div = all_content_divs[i]
                            logger.info(f"Using fallback content div {i}")
                    
                    logger.info(f"Found content div: {content_div is not None}")
                    
                    if content_div:
                        dishes = []
                        
                        # Find all dish entries in this day - try multiple selectors
                        dish_selectors = [
                            'div.speiseplanTagKat',
                            'div.speiseplan',
                            'div[class*="speiseplan"]',
                            'div[class*="menu"]',
                            'div[class*="dish"]',
                            '.menu-item',
                            '.dish-item',
                            '.food-item'
                        ]
                        
                        dish_divs = []
                        for dish_selector in dish_selectors:
                            try:
                                found_dishes = content_div.select(dish_selector)
                                if found_dishes:
                                    dish_divs = found_dishes
                                    logger.info(f"Found {len(dish_divs)} dishes with selector: {dish_selector}")
                                    break
                            except Exception as e:
                                logger.warning(f"Dish selector '{dish_selector}' failed: {e}")
                                continue
                        
                        if not dish_divs:
                            # Log the structure we actually find
                            logger.warning(f"No dish divs found. Content structure:")
                            logger.warning(f"Content div classes: {content_div.get('class', [])}")
                            logger.warning(f"Content div id: {content_div.get('id', 'none')}")
                            child_divs = content_div.find_all('div', recursive=False)
                            logger.warning(f"Direct child divs: {len(child_divs)}")
                            for j, div in enumerate(child_divs[:5]):  # Show first 5
                                logger.warning(f"  Child {j}: classes={div.get('class', [])} id={div.get('id', 'none')} text='{div.get_text(strip=True)[:50]}...'")
                        
                        for dish_div in dish_divs:
                            try:
                                # Extract category - try multiple selectors
                                category_elem = None
                                for cat_selector in ['div.category', '.category', '.type', '.section', 'h3', 'h4']:
                                    category_elem = dish_div.select_one(cat_selector)
                                    if category_elem:
                                        break
                                
                                category = category_elem.get_text(strip=True) if category_elem else "Unbekannt"
                                
                                # Extract dish name - try multiple selectors
                                title_elem = None
                                for title_selector in ['div.title', '.title', '.name', '.dish-name', 'h3', 'h4', '.food-name']:
                                    title_elem = dish_div.select_one(title_selector)
                                    if title_elem:
                                        break
                                
                                raw_title = title_elem.get_text(strip=True) if title_elem else ""
                                
                                if not raw_title:
                                    # Try to get any text from the dish div
                                    raw_title = dish_div.get_text(strip=True)
                                    if len(raw_title) > 100:  # Too long, probably not a dish name
                                        continue
                                
                                if not raw_title:
                                    continue
                                
                                # Clean dish name and extract allergens
                                dish_name = clean_dish_name(raw_title)
                                allergens = parse_allergens(raw_title)
                                
                                # Extract price - try multiple selectors
                                price_elem = None
                                for price_selector in ['div.preise', '.preise', '.price', '.cost', '.preis']:
                                    price_elem = dish_div.select_one(price_selector)
                                    if price_elem:
                                        break
                                
                                price = parse_price(price_elem.get_text(strip=True) if price_elem else "")
                                
                                # Extract dietary icons
                                icon_container = dish_div.select_one('div.title_preise_2, .icons, .dietary-icons')
                                icon_elements = icon_container.select('div.speiseplanTagKatIcon, .icon, .dietary-icon') if icon_container else []
                                dietary_icons = parse_dietary_icons(icon_elements)
                                
                                if dish_name:  # Only add if we have a valid dish name
                                    dishes.append(MensaDish(
                                        name=dish_name,
                                        price=price,
                                        category=category,
                                        dietary_icons=dietary_icons,
                                        allergens=allergens
                                    ))
                                    logger.info(f"Added dish: {dish_name} ({category}) - {price}")
                                    
                            except Exception as e:
                                logger.warning(f"Fehler beim Parsen eines Gerichts: {e}")
                                continue
                        
                        if dishes:  # Only add menu if we have dishes
                            # Parse date from tab text
                            date_match = re.search(r'(\d{2}\.\d{2}\.)', tab_text)
                            if date_match:
                                date_str = date_match.group(1)
                                # Add current year
                                current_year = datetime.now().year
                                full_date = f"{date_str}{current_year}"
                                
                                # Parse day name
                                day_parts = tab_text.split('.')
                                day_name = day_parts[0].strip() if day_parts else "Unbekannt"
                                
                                menus.append(MensaMenu(
                                    date=full_date,
                                    day_name=day_name,
                                    dishes=dishes
                                ))
                                
                                logger.info(f"Parsed {len(dishes)} Gerichte für {day_name}")
                        else:
                            logger.warning(f"No dishes found for tab {i} ({tab_text})")
            
            logger.info(f"Erfolgreich {len(menus)} Menüs geparst")
            
            # If no menus found, fall back to mock data
            if not menus:
                logger.warning("No real menus found, falling back to mock data")
                return generate_fallback_data()
            
            return {
                "name": "HTWG Mensa",
                "location": "Konstanz",
                "menus": [menu.dict() for menu in menus],
                "last_updated": datetime.now().isoformat(),
                "source": "seezeit.com"
            }
            
    except Exception as e:
        logger.error(f"Fehler beim Laden der Mensa-Daten: {e}")
        # Fallback zu Mock-Daten
        return generate_fallback_data()

def generate_fallback_data():
    """Generate fallback mock data when website is unavailable"""
    logger.warning("Verwende Fallback Mock-Daten")
    
    base_date = datetime.now()
    menus = []
    
    dishes_rotation = [
        # Montag
        [
            {"name": "Schnitzel mit Pommes", "price": "4.50€", "category": "Seezeit-Teller", "dietary_icons": ["B"], "allergens": ["25a", "31"]},
            {"name": "Vegetarische Lasagne", "price": "3.80€", "category": "hin&weg", "dietary_icons": ["Veg"], "allergens": ["25a", "31"]},
            {"name": "Salat der Saison", "price": "2.90€", "category": "Salatbeilage", "dietary_icons": ["Vegan"], "allergens": []}
        ],
        # Dienstag  
        [
            {"name": "Currywurst mit Reis", "price": "4.20€", "category": "kœriwerk®", "dietary_icons": ["R"], "allergens": ["33", "34"]},
            {"name": "Vegane Buddha Bowl", "price": "4.10€", "category": "hin&weg", "dietary_icons": ["Vegan"], "allergens": []},
            {"name": "Caesar Salad", "price": "3.40€", "category": "Salatbeilage", "dietary_icons": ["Veg"], "allergens": ["31"]}
        ],
        # Mittwoch
        [
            {"name": "Hähnchen-Geschnetzeltes", "price": "4.80€", "category": "KombinierBar", "dietary_icons": ["G"], "allergens": ["31"]},
            {"name": "Vegetarisches Chili", "price": "3.60€", "category": "hin&weg", "dietary_icons": ["Vegan"], "allergens": ["33", "34"]},
            {"name": "Griechischer Salat", "price": "3.20€", "category": "Salatbeilage", "dietary_icons": ["Veg"], "allergens": ["31"]}
        ],
        # Donnerstag
        [
            {"name": "Fischstäbchen mit Kartoffeln", "price": "4.40€", "category": "Seezeit-Teller", "dietary_icons": ["F"], "allergens": ["25a", "26"]},
            {"name": "Quinoa-Gemüse-Pfanne", "price": "4.00€", "category": "hin&weg", "dietary_icons": ["Vegan"], "allergens": []},
            {"name": "Rucola-Salat mit Pinienkernen", "price": "3.60€", "category": "Salatbeilage", "dietary_icons": ["Vegan"], "allergens": ["32a"]}
        ],
        # Freitag
        [
            {"name": "Rindergulasch mit Nudeln", "price": "5.20€", "category": "Seezeit-Teller", "dietary_icons": ["B", "R"], "allergens": ["25a"]},
            {"name": "Gemüse-Curry mit Reis", "price": "3.90€", "category": "hin&weg", "dietary_icons": ["Vegan"], "allergens": []},
            {"name": "Bunter Blattsalat", "price": "2.80€", "category": "Salatbeilage", "dietary_icons": ["Vegan"], "allergens": []}
        ]
    ]
    
    # Finde die nächsten 5 Werktage ab heute
    current_date = base_date
    weekdays_found = 0
    
    while weekdays_found < 5:
        if current_date.weekday() < 5:  # 0-4 = Montag bis Freitag
            date_str = current_date.strftime("%d.%m.%Y")
            day_name_de = {
                0: "Mo", 1: "Di", 2: "Mi", 
                3: "Do", 4: "Fr"
            }
            
            menu_index = current_date.weekday()
            
            menus.append({
                "date": date_str,
                "day_name": day_name_de[current_date.weekday()],
                "dishes": dishes_rotation[menu_index]
            })
            weekdays_found += 1
        
        current_date += timedelta(days=1)
    
    return {
        "name": "HTWG Mensa",
        "location": "Konstanz",
        "menus": menus,
        "last_updated": datetime.now().isoformat(),
        "source": "fallback_mock_data"
    }

# Cache für Mensa-Daten (1 Stunde gültig)
_mensa_cache = None
_cache_timestamp = None
CACHE_DURATION = 3600  # 1 Stunde

async def get_mensa_data():
    """Get mensa data with caching"""
    global _mensa_cache, _cache_timestamp
    
    now = datetime.now().timestamp()
    
    # Check if cache is still valid
    if _mensa_cache and _cache_timestamp and (now - _cache_timestamp) < CACHE_DURATION:
        logger.info("Verwende gecachte Mensa-Daten")
        return _mensa_cache
    
    # Fetch fresh data
    logger.info("Lade neue Mensa-Daten...")
    _mensa_cache = await fetch_mensa_data()
    _cache_timestamp = now
    
    return _mensa_cache

@app.list_resources()
async def list_resources() -> List[Resource]:
    """Liste verfügbare Mensa-Ressourcen"""
    return [
        Resource(
            uri="mensa://menu",
            name="HTWG Mensa Menu",
            description="Aktueller Mensaplan für HTWG Konstanz (von seezeit.com)",
            mimeType="application/json"
        )
    ]

@app.read_resource()
async def read_resource(uri: str) -> str:
    """Lese spezifische Mensa-Ressource"""
    if uri == "mensa://menu":
        mensa_data = await get_mensa_data()
        return json.dumps(mensa_data, indent=2, ensure_ascii=False)
    
    return json.dumps({"error": "Ressource nicht gefunden"})

@app.list_tools()
async def list_tools() -> List[Tool]:
    """Liste verfügbare Mensa Tools"""
    return [
        Tool(
            name="get_daily_menu",
            description="Ruft das Tagesmenü für einen bestimmten Tag ab (heute bis in 4 Tagen)",
            inputSchema={
                "type": "object",
                "properties": {
                    "days_ahead": {
                        "type": "integer",
                        "description": "Tage in die Zukunft (0=heute, 1=morgen, 2=übermorgen, 3=in 3 Tagen, 4=in 4 Tagen)",
                        "minimum": 0,
                        "maximum": 4,
                        "default": 0
                    }
                },
                "required": []
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Führt Mensa Tool-Aufrufe aus"""
    
    if name == "get_daily_menu":
        days_ahead = arguments.get("days_ahead", 0)
        
        # Validierung der Tage
        if days_ahead < 0 or days_ahead > 4:
            return [TextContent(
                type="text",
                text="❌ days_ahead muss zwischen 0 (heute) und 4 (in 4 Tagen) liegen."
            )]
        
        mensa_data = await get_mensa_data()
        
        # Validiere verfügbare Menüs
        if days_ahead >= len(mensa_data['menus']):
            return [TextContent(
                type="text",
                text=f"❌ Kein Menü für Tag +{days_ahead} verfügbar. Verfügbare Tage: 0-{len(mensa_data['menus'])-1}"
            )]
        
        menu = mensa_data['menus'][days_ahead]
        
        # Beschreibung je nach Tag
        day_descriptions = {
            0: "Heute",
            1: "Morgen", 
            2: "Übermorgen",
            3: "In 3 Tagen",
            4: "In 4 Tagen"
        }
        
        day_description = day_descriptions.get(days_ahead, f"In {days_ahead} Tagen")
        
        result = f"""🍽️ **{mensa_data['name']} - {mensa_data['location']}**

📅 **{day_description} ({menu['day_name']}, {menu['date']})**

**Tagesmenü:**
"""
        
        # Group dishes by category for better display
        categories = {}
        for dish in menu['dishes']:
            category = dish['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(dish)
        
        for category, dishes in categories.items():
            result += f"\n**{category}:**\n"
            for dish in dishes:
                result += f"\n🍽️ **{dish['name']}**\n"
                result += f"   💰 {dish['price']}"
                
                # Add dietary icons
                if dish.get('dietary_icons'):
                    icon_meanings = {
                        'Veg': '🥕 Vegetarisch',
                        'Vegan': '🌱 Vegan',
                        'B': '🐄 Bessere Tierhaltung',
                        'R': '🐂 Rind/Kalb',
                        'G': '🐔 Geflügel',
                        'F': '🐟 Fisch',
                        'Sch': '🐷 Schwein',
                        'L': '🐑 Lamm',
                        'W': '🦌 Wild'
                    }
                    icons = [icon_meanings.get(icon, icon) for icon in dish['dietary_icons']]
                    result += f" | {', '.join(icons)}"
                
                result += "\n"
        
        # Add data source info
        source_info = {
            'seezeit.com': '📡 Live-Daten von seezeit.com',
            'fallback_mock_data': '⚠️ Mock-Daten (Website nicht erreichbar)'
        }
        result += f"\n{source_info.get(mensa_data.get('source'), '')}"
        result += f"\n🕒 Stand: {mensa_data.get('last_updated', 'Unbekannt')[:16]}"
        
        return [TextContent(type="text", text=result)]
    
    else:
        return [TextContent(type="text", text=f"❌ Unbekanntes Tool: {name}")]

async def main():
    """Hauptfunktion - startet den MCP Server"""
    logger.info("🍽️ Mensa MCP Server wird gestartet...")
    
    # Debug startup environment
    logger.info("🔍 Startup Debug Information:")
    logger.info(f"   Current working directory: {os.getcwd()}")
    logger.info(f"   Python executable: {sys.executable}")
    logger.info(f"   Python version: {sys.version}")
    
    # Test if we can load mensa data
    try:
        logger.info("🧪 Testing mensa data loading...")
        test_data = await get_mensa_data()
        logger.info(f"✅ Mensa data loaded successfully: {len(test_data.get('menus', []))} menus")
        logger.info(f"   Source: {test_data.get('source', 'unknown')}")
        
        # Debug: Show first menu if available
        if test_data.get('menus'):
            first_menu = test_data['menus'][0]
            logger.info(f"   First menu: {first_menu.get('day_name', 'unknown')} with {len(first_menu.get('dishes', []))} dishes")
        else:
            logger.warning("   No menus found - falling back to mock data")
            
    except Exception as e:
        logger.error(f"❌ Error loading mensa data: {e}")
        import traceback
        traceback.print_exc()
    
    # Standard MCP stdio Modus (wie beim Weather-Service)
    logger.info("📡 MCP stdio Modus (MCPO-kompatibel)")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream, 
            app.create_initialization_options()
        )

if __name__ == "__main__":
    import sys
    import asyncio
    
    # Ensure we have sys imported for debug info
    logger.info(f"Starting mensa_server.py directly...")
    asyncio.run(main())
