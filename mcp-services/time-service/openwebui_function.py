""
title: Time Service Functions
author: Time MCP Service
version: 1.0.0
description: Get current time and convert between timezones using the Time MCP Service
"""

import requests
from typing import Optional
from datetime import datetime


class Tools:
    def __init__(self):
        self.base_url = "http://192.168.50.17:8009"
        self.timeout = 5
    
    def get_current_time(
        self, 
        timezone: str = "UTC",
        __user__: dict = {}
    ) -> str:
        """
        Get the current time and date in a specific timezone.
        IMPORTANT: Use this tool FIRST to get today's date before calling weather APIs!
        
        :param timezone: IANA timezone name (e.g., 'Europe/Berlin', 'America/New_York', 'Asia/Tokyo')
        :return: Current time, date, and day of week information
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/current-time",
                params={"timezone": timezone},
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            # Format the response nicely
            return f"""
**Aktuelle Zeit in {data['timezone']}:**
- 📅 Datum: {data['datetime'].split('T')[0]}
- 🕐 Zeit: {data['datetime'].split('T')[1]}
- 📆 Wochentag: {data['day_of_week']}
- ☀️ Sommerzeit: {'Ja' if data['is_dst'] else 'Nein'}
""".strip()
            
        except requests.RequestException as e:
            return f"❌ Fehler beim Abrufen der Zeit: {str(e)}"
        except Exception as e:
            return f"❌ Unerwarteter Fehler: {str(e)}"
    
    def get_future_date(
        self,
        days: int,
        timezone: str = "UTC",
        __user__: dict = {}
    ) -> str:
        """
        Calculate a future date by adding days to today.
        Perfect for weather forecasts! Use this to get the date for "in 4 days", "next week", etc.
        Returns the exact date in YYYY-MM-DD format that can be used with weather APIs.
        
        :param days: Number of days to add (e.g., 1 for tomorrow, 4 for in 4 days, 7 for next week)
        :param timezone: IANA timezone name (e.g., 'Europe/Berlin')
        :return: Future date information including the exact date, day of week, and whether it's a weekend
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/future-date",
                params={"days": days, "timezone": timezone},
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            weekend_emoji = "🌴" if data['is_weekend'] else "💼"
            
            return f"""
**Datum in {days} Tag(en):**
- 📅 Datum: **{data['date']}**
- 📆 Wochentag: {data['day_of_week']} {weekend_emoji}
- 📊 Kalenderwoche: {data['week_number']}
- 🌍 Zeitzone: {data['timezone']}
- ⏰ Vollständige Uhrzeit: {data['full_datetime']}

💡 *Verwende dieses Datum `{data['date']}` für Wettervorhersagen!*
""".strip()
            
        except requests.RequestException as e:
            return f"❌ Fehler bei der Datumsberechnung: {str(e)}"
        except Exception as e:
            return f"❌ Unerwarteter Fehler: {str(e)}"
    
    def get_past_date(
        self,
        days: int,
        timezone: str = "UTC",
        __user__: dict = {}
    ) -> str:
        """
        Calculate a past date by subtracting days from today.
        
        :param days: Number of days to subtract (e.g., 1 for yesterday, 7 for last week)
        :param timezone: IANA timezone name (e.g., 'Europe/Berlin')
        :return: Past date information including the exact date and day of week
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/past-date",
                params={"days": days, "timezone": timezone},
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            weekend_emoji = "🌴" if data['is_weekend'] else "💼"
            
            return f"""
**Datum vor {days} Tag(en):**
- 📅 Datum: **{data['date']}**
- 📆 Wochentag: {data['day_of_week']} {weekend_emoji}
- 📊 Kalenderwoche: {data['week_number']}
- 🌍 Zeitzone: {data['timezone']}
""".strip()
            
        except requests.RequestException as e:
            return f"❌ Fehler bei der Datumsberechnung: {str(e)}"
        except Exception as e:
            return f"❌ Unerwarteter Fehler: {str(e)}"
    
    def get_date_info(
        self,
        date: str,
        timezone: str = "UTC",
        __user__: dict = {}
    ) -> str:
        """
        Get detailed information about a specific date.
        
        :param date: Date in YYYY-MM-DD format (e.g., '2025-12-31')
        :param timezone: IANA timezone name (e.g., 'Europe/Berlin')
        :return: Detailed information about the date
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/date-info",
                params={"date": date, "timezone": timezone},
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            weekend_emoji = "🌴" if data['is_weekend'] else "💼"
            
            return f"""
**Informationen über {data['date']}:**
- 📆 Wochentag: {data['day_of_week']} {weekend_emoji}
- 📊 Kalenderwoche: {data['week_number']}
- 🌍 Zeitzone: {data['timezone']}
- ⏰ Vollständige Uhrzeit: {data['full_datetime']}
""".strip()
            
        except requests.RequestException as e:
            return f"❌ Fehler beim Abrufen der Datumsinformationen: {str(e)}"
        except Exception as e:
            return f"❌ Unerwarteter Fehler: {str(e)}"
    
    def convert_time(
        self,
        source_timezone: str,
        time: str,
        target_timezone: str,
        __user__: dict = {}
    ) -> str:
        """
        Convert a time from one timezone to another.
        
        :param source_timezone: Source IANA timezone name (e.g., 'Europe/Berlin')
        :param time: Time in HH:MM format (24-hour, e.g., '15:30')
        :param target_timezone: Target IANA timezone name (e.g., 'America/New_York')
        :return: Conversion result with both times and the time difference
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/convert-time",
                params={
                    "source": source_timezone,
                    "time": time,
                    "target": target_timezone
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            # Format the response nicely
            return f"""
**Zeitumrechnung:**

🌍 **{data['source']['timezone']}:**
- Zeit: {data['source']['datetime']}
- Wochentag: {data['source']['day_of_week']}
- Sommerzeit: {'Ja' if data['source']['is_dst'] else 'Nein'}

🌎 **{data['target']['timezone']}:**
- Zeit: {data['target']['datetime']}
- Wochentag: {data['target']['day_of_week']}
- Sommerzeit: {'Ja' if data['target']['is_dst'] else 'Nein'}

⏰ **Zeitdifferenz:** {data['time_difference']}
""".strip()
            
        except requests.RequestException as e:
            return f"❌ Fehler bei der Zeitumrechnung: {str(e)}"
        except Exception as e:
            return f"❌ Unerwarteter Fehler: {str(e)}"
    
    def get_time_in_multiple_timezones(
        self,
        timezones: list[str],
        __user__: dict = {}
    ) -> str:
        """
        Get the current time in multiple timezones at once.
        
        :param timezones: List of IANA timezone names (e.g., ['Europe/Berlin', 'America/New_York', 'Asia/Tokyo'])
        :return: Current times for all specified timezones
        """
        results = []
        
        for tz in timezones:
            try:
                response = requests.get(
                    f"{self.base_url}/api/current-time",
                    params={"timezone": tz},
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                
                results.append(
                    f"🌍 **{data['timezone']}:** {data['datetime']} ({data['day_of_week']})"
                )
            except Exception as e:
                results.append(f"❌ {tz}: Fehler - {str(e)}")
        
        return "\n".join(results)


# Häufig verwendete Zeitzonen als Referenz
COMMON_TIMEZONES = {
    "Europa": [
        "Europe/Berlin",
        "Europe/London", 
        "Europe/Paris",
        "Europe/Rome",
        "Europe/Moscow"
    ],
    "Amerika": [
        "America/New_York",
        "America/Chicago",
        "America/Denver",
        "America/Los_Angeles",
        "America/Toronto"
    ],
    "Asien": [
        "Asia/Tokyo",
        "Asia/Shanghai",
        "Asia/Dubai",
        "Asia/Kolkata",
        "Asia/Singapore"
    ],
    "Ozeanien": [
        "Australia/Sydney",
        "Pacific/Auckland"
    ]
}
