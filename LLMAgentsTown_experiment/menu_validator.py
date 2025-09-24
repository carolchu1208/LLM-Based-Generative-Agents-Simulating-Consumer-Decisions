#!/usr/bin/env python
# coding: utf-8

"""
Menu Validation System - Prevents Agent Death from Invalid Food Requests

This module validates agent food purchases against actual restaurant menus,
preventing agents from starving due to LLM hallucination of non-existent items.

Key Features:
1. Validates menu items against configuration
2. Suggests closest valid alternatives
3. Prevents agent death from failed food purchases
4. Logs invalid requests for debugging
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from difflib import get_close_matches
import logging

class MenuValidator:
    """Validates agent food requests against available restaurant menus."""

    def __init__(self, config_data: Dict):
        """Initialize with restaurant configuration data."""
        self.config_data = config_data
        self.restaurant_menus = self._build_menu_database()
        self.logger = logging.getLogger(__name__)

    def _build_menu_database(self) -> Dict[str, Dict]:
        """Build a searchable database of all available menu items."""
        menu_db = {}

        dining_locations = self.config_data.get('town_areas', {}).get('dining', {})

        for location_name, location_data in dining_locations.items():
            menu = location_data.get('menu', {})
            menu_db[location_name] = {
                'hours': location_data.get('hours', {}),
                'items': {}
            }

            # Build item database with hour restrictions
            for meal_type, meal_info in menu.items():
                item_name = meal_info.get('item', '')
                available_hours = meal_info.get('available_hours', [])
                base_price = meal_info.get('base_price', 0)

                menu_db[location_name]['items'][item_name.lower()] = {
                    'original_name': item_name,
                    'meal_type': meal_type,
                    'available_hours': available_hours,
                    'base_price': base_price,
                    'description': meal_info.get('description', '')
                }

                # Also add meal type as searchable
                menu_db[location_name]['items'][meal_type.lower()] = {
                    'original_name': item_name,
                    'meal_type': meal_type,
                    'available_hours': available_hours,
                    'base_price': base_price,
                    'description': meal_info.get('description', '')
                }

        return menu_db

    def validate_food_request(self, agent_name: str, location_name: str,
                            requested_item: str, current_hour: int) -> Tuple[bool, Dict]:
        """
        Validate if an agent can purchase the requested food item.

        Args:
            agent_name: Name of the agent
            location_name: Restaurant location
            requested_item: Food item or meal type requested
            current_hour: Current simulation hour

        Returns:
            Tuple[bool, dict]: (is_valid, result_info)
            result_info contains: valid_item, price, error_message, suggestion
        """

        # Check if location exists
        if location_name not in self.restaurant_menus:
            return False, {
                'error': f'Location "{location_name}" does not exist',
                'suggestion': self._suggest_valid_location(location_name),
                'valid_item': None,
                'price': 0
            }

        location_menu = self.restaurant_menus[location_name]

        # Check if restaurant is open
        hours = location_menu.get('hours', {})
        open_hour = hours.get('open', 0)
        close_hour = hours.get('close', 24)

        if not (open_hour <= current_hour <= close_hour):
            return False, {
                'error': f'{location_name} is closed (opens at {open_hour}:00, closes at {close_hour}:00)',
                'suggestion': self._suggest_open_restaurant(current_hour),
                'valid_item': None,
                'price': 0
            }

        # Search for the requested item (case-insensitive)
        requested_lower = requested_item.lower()
        items = location_menu.get('items', {})

        # Direct match
        if requested_lower in items:
            item_info = items[requested_lower]

            # Check if available at this hour
            if current_hour in item_info['available_hours']:
                # Apply discount if applicable
                base_price = item_info['base_price']
                discount_info = self.config_data.get('town_areas', {}).get('dining', {}).get(location_name, {}).get('discount', {})

                final_price = self._calculate_price(base_price, discount_info)

                return True, {
                    'valid_item': item_info['original_name'],
                    'meal_type': item_info['meal_type'],
                    'price': final_price,
                    'description': item_info['description'],
                    'error': None
                }
            else:
                return False, {
                    'error': f'{item_info["original_name"]} not available at hour {current_hour}',
                    'suggestion': self._suggest_available_item(location_name, current_hour),
                    'valid_item': None,
                    'price': 0
                }

        # Fuzzy matching for similar items
        all_item_names = [info['original_name'] for info in items.values()]
        close_matches = get_close_matches(requested_item, all_item_names, n=1, cutoff=0.6)

        if close_matches:
            suggested_item = close_matches[0]
            return False, {
                'error': f'Item "{requested_item}" not found at {location_name}',
                'suggestion': f'Did you mean "{suggested_item}"?',
                'valid_item': None,
                'price': 0
            }

        # No match found, suggest what's available
        return False, {
            'error': f'Item "{requested_item}" not available at {location_name}',
            'suggestion': self._suggest_available_item(location_name, current_hour),
            'valid_item': None,
            'price': 0
        }

    def _calculate_price(self, base_price: float, discount_info: Dict) -> float:
        """Calculate final price with discount if applicable."""
        if not discount_info:
            return base_price

        # TODO: Add day-based discount logic from simulation_types.py TimeManager
        # For now, return base price
        return base_price

    def _suggest_valid_location(self, invalid_location: str) -> str:
        """Suggest a valid restaurant location."""
        valid_locations = list(self.restaurant_menus.keys())
        close_matches = get_close_matches(invalid_location, valid_locations, n=1, cutoff=0.6)

        if close_matches:
            return f'Did you mean "{close_matches[0]}"?'

        return f'Available restaurants: {", ".join(valid_locations)}'

    def _suggest_open_restaurant(self, current_hour: int) -> str:
        """Suggest restaurants that are open at the current hour."""
        open_restaurants = []

        for location_name, location_info in self.restaurant_menus.items():
            hours = location_info.get('hours', {})
            open_hour = hours.get('open', 0)
            close_hour = hours.get('close', 24)

            if open_hour <= current_hour <= close_hour:
                open_restaurants.append(location_name)

        if open_restaurants:
            return f'Try these open restaurants: {", ".join(open_restaurants)}'

        return 'No restaurants currently open'

    def _suggest_available_item(self, location_name: str, current_hour: int) -> str:
        """Suggest items available at the current hour."""
        if location_name not in self.restaurant_menus:
            return ''

        available_items = []
        items = self.restaurant_menus[location_name].get('items', {})

        for item_name, item_info in items.items():
            if current_hour in item_info.get('available_hours', []):
                available_items.append(item_info['original_name'])

        # Remove duplicates (since meal types and item names both exist)
        unique_items = list(set(available_items))

        if unique_items:
            return f'Available now: {", ".join(unique_items)}'

        return 'No items available at this hour'

    def get_emergency_food_options(self, current_hour: int, max_price: float = 50.0) -> List[Dict]:
        """Get emergency food options for starving agents."""
        emergency_options = []

        for location_name, location_info in self.restaurant_menus.items():
            # Check if restaurant is open
            hours = location_info.get('hours', {})
            open_hour = hours.get('open', 0)
            close_hour = hours.get('close', 24)

            if not (open_hour <= current_hour <= close_hour):
                continue

            # Find available items within budget
            items = location_info.get('items', {})
            for item_key, item_info in items.items():
                if (current_hour in item_info.get('available_hours', []) and
                    item_info.get('base_price', 999) <= max_price):

                    emergency_options.append({
                        'location': location_name,
                        'item': item_info['original_name'],
                        'meal_type': item_info['meal_type'],
                        'price': item_info['base_price'],
                        'description': item_info.get('description', '')
                    })

        # Sort by price (cheapest first) for emergency situations
        return sorted(emergency_options, key=lambda x: x['price'])

    def log_invalid_request(self, agent_name: str, location_name: str,
                          requested_item: str, current_hour: int, error_info: Dict):
        """Log invalid food requests for debugging."""
        self.logger.warning(
            f"INVALID FOOD REQUEST - Agent: {agent_name}, "
            f"Location: {location_name}, Item: {requested_item}, "
            f"Hour: {current_hour}, Error: {error_info.get('error', 'Unknown')}"
        )