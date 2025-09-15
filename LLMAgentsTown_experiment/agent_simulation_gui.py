"""
agent_simulation_gui.py

A professional GUI for visualizing the town simulation data.
Designed to work with the existing TownSimulation class without modifications.

Features:
- Real-time updates of agent positions and stats
- Visualization of locations (houses, businesses, etc.)
- Clean, modern interface with proper error handling
"""

import random
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import traceback
from PIL import Image, ImageTk, ImageDraw, ImageFont
import json
import time
import threading
import os
from datetime import datetime
import math
from typing import Dict, Optional, Any

class AgentSimulationGUI:
    def __init__(self, master, simulation=None):
        """
        Initialize the main GUI window with optional simulation reference.
        
        Args:
            master: The root tkinter window
            simulation: Optional reference to TownSimulation instance
        """
        self.master = master
        self.simulation = simulation  # Reference to the simulation
        
        # Modern color scheme
        self.colors = {
            'primary': '#2C3E50',      # Dark blue
            'secondary': '#3498DB',    # Bright blue
            'accent': '#E74C3C',       # Red
            'background': '#ECF0F1',   # Light gray
            'surface': '#FFFFFF',      # White
            'text': '#2C3E50',         # Dark text
            'text_light': '#7F8C8D',   # Light text
            'grid': '#BDC3C7',         # Grid lines
            'success': '#27AE60',      # Green
            'warning': '#F39C12',      # Orange
        }
        
        master.title("Town Simulation Dashboard")
        master.geometry("1400x900")
        master.minsize(1200, 800)
        master.configure(bg=self.colors['background'])
        
        # Configure window icon
        try:
            master.iconbitmap(default='agent_icon.ico')
        except:
            pass
        
        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self._configure_styles()
        
        # Data storage
        self.agents = {}
        self.locations = {}
        self.sprites = {}
        
        # Zoom state management
        self.zoom_level = 1.0
        self.zoom_center_x = 0
        self.zoom_center_y = 0
        
        # GUI update control
        self.update_interval = 1  # Update every 1 second for better real-time updates
        self.update_thread = None
        self.running = True
        
        # Create GUI components
        self._create_widgets()
        
        # Load default sprites
        self._load_default_sprites()
        
        # Start the update loop
        self.start_update_loop()
        
        # Handle window close properly
        master.protocol("WM_DELETE_WINDOW", self.on_close)

    def _configure_styles(self):
        """Configure custom styles for a modern look."""
        # Configure main styles
        self.style.configure('TFrame', background=self.colors['background'])
        self.style.configure('TLabel', background=self.colors['background'], 
                            font=('Segoe UI', 10), foreground=self.colors['text'])
        self.style.configure('TButton', font=('Segoe UI', 10, 'bold'),
                            background=self.colors['secondary'], foreground='white')
        self.style.configure('Header.TLabel', font=('Segoe UI', 16, 'bold'),
                            foreground=self.colors['primary'])
        self.style.configure('Subheader.TLabel', font=('Segoe UI', 12, 'bold'),
                            foreground=self.colors['secondary'])
        self.style.configure('Map.TFrame', background=self.colors['surface'],
                            relief='flat', borderwidth=0)
        
        # Map button styles
        self.style.configure('MapControl.TButton', font=('Segoe UI', 9),
                            background=self.colors['surface'], 
                            foreground=self.colors['text'],
                            padding=(10, 5))
        
        # Configure hover effects
        self.style.map('TButton',
                      background=[('active', self.colors['primary'])],
                      foreground=[('active', 'white')])
        
        self.style.map('MapControl.TButton',
                      background=[('active', self.colors['secondary'])],
                      foreground=[('active', 'white')])

    def _create_widgets(self):
        """Create and arrange all GUI components with modern design."""
        # Main container with modern styling
        self.main_frame = ttk.Frame(self.master, style='TFrame')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header with modern design
        self.header_frame = ttk.Frame(self.main_frame, style='TFrame')
        self.header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # App title with modern typography
        title_frame = ttk.Frame(self.header_frame, style='TFrame')
        title_frame.pack(side=tk.LEFT)
        
        ttk.Label(title_frame, text="TOWN SIMULATION", style='Header.TLabel').pack()
        ttk.Label(title_frame, text="Real-time Agent Monitoring Dashboard", 
                 style='Subheader.TLabel').pack()
        
        # Stats overview panel
        self.stats_frame = ttk.Frame(self.header_frame, style='TFrame')
        self.stats_frame.pack(side=tk.RIGHT)
        
        # Stats labels will be updated dynamically
        self.agent_count_label = ttk.Label(self.stats_frame, 
                                          text="Agents: 0",
                                          style='Subheader.TLabel')
        self.agent_count_label.pack(anchor=tk.E)
        
        self.location_count_label = ttk.Label(self.stats_frame,
                                             text="Locations: 0",
                                             style='Subheader.TLabel')
        self.location_count_label.pack(anchor=tk.E)
        
        # Main content area with modern card design
        content_frame = ttk.Frame(self.main_frame, style='TFrame')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Map container with modern card styling
        map_card = ttk.Frame(content_frame, style='Map.TFrame')
        map_card.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add subtle shadow effect (simulated with border)
        map_card.configure(relief='raised', borderwidth=1)
        
        # Map controls toolbar
        controls_frame = ttk.Frame(map_card, style='Map.TFrame')
        controls_frame.pack(fill=tk.X, padx=15, pady=10)
        
        ttk.Label(controls_frame, text="Simulation Map", 
                 style='Subheader.TLabel').pack(side=tk.LEFT)
        
        # Zoom controls
        zoom_frame = ttk.Frame(controls_frame, style='Map.TFrame')
        zoom_frame.pack(side=tk.RIGHT)
        
        ttk.Button(zoom_frame, text="➖", style='MapControl.TButton',
                  command=lambda: self._zoom_map(0.9)).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="➕", style='MapControl.TButton',
                  command=lambda: self._zoom_map(1.1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="⟳ Reset", style='MapControl.TButton',
                  command=self._reset_zoom).pack(side=tk.LEFT, padx=5)
        
        # Map canvas with modern styling
        self.map_canvas = tk.Canvas(
            map_card, 
            bg=self.colors['surface'],
            highlightthickness=0,
            relief='flat'
        )
        self.map_canvas.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        # Add mouse wheel scrolling for zoom
        self.map_canvas.bind("<MouseWheel>", self._on_mousewheel)
        
        # Footer with subtle branding
        footer_frame = ttk.Frame(self.main_frame, style='TFrame')
        footer_frame.pack(fill=tk.X, pady=(20, 0))
        
        ttk.Label(footer_frame, text="Town Simulation v1.0 • Real-time Monitoring",
                 style='Status.TLabel', foreground=self.colors['text_light']).pack(side=tk.RIGHT)

    def _zoom_map(self, factor):
        """Zoom the map by the given factor."""
        # Update zoom level
        self.zoom_level *= factor
        
        # Get current canvas center
        canvas_width = self.map_canvas.winfo_width()
        canvas_height = self.map_canvas.winfo_height()
        
        # Apply zoom transformation
        self.map_canvas.scale("all", 
                             canvas_width // 2,
                             canvas_height // 2,
                             factor, factor)
        
        # Store zoom center for later restoration
        self.zoom_center_x = canvas_width // 2
        self.zoom_center_y = canvas_height // 2

    def _reset_zoom(self):
        """Reset the map zoom to default."""
        self.zoom_level = 1.0
        self._update_map()

    def _on_mousewheel(self, event):
        """Handle zooming with mouse wheel."""
        # Scale all items on the canvas
        scale_factor = 1.1
        if event.delta < 0:
            scale_factor = 0.9
        
        # Update zoom level
        self.zoom_level *= scale_factor
        
        # Apply zoom with mouse position as center
        self.map_canvas.scale("all", event.x, event.y, scale_factor, scale_factor)
        
        # Store zoom center for later restoration
        self.zoom_center_x = event.x
        self.zoom_center_y = event.y

    def _load_default_sprites(self):
        """Load default sprites with modern design."""
        # Modern color palette for locations
        location_colors = {
            'house': '#E74C3C',      # Red
            'business': '#3498DB',   # Blue
            'restaurant': '#F39C12', # Orange
            'park': '#27AE60',       # Green
            'default': '#95A5A6'     # Gray
        }
        
        default_sprites = {
            'house_default': self._create_modern_sprite(location_colors['house'], '🏠'),
            'business_default': self._create_modern_sprite(location_colors['business'], '🏢'),
            'restaurant_default': self._create_modern_sprite(location_colors['restaurant'], '🍽'),
            'park_default': self._create_modern_sprite(location_colors['park'], '🌳'),
            'default': self._create_modern_sprite(location_colors['default'], '📍')
        }
        
        for sprite_id, sprite_img in default_sprites.items():
            self.sprites[sprite_id] = sprite_img

    def _create_modern_sprite(self, color, emoji=None):
        """Create a modern sprite with the given color and optional emoji."""
        size = 40  # Larger size for better visibility
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw modern circular background
        draw.ellipse([2, 2, size-2, size-2], fill=color, outline=self.colors['primary'], width=2)
        
        # Add emoji if provided
        if emoji:
            try:
                # Try to use a font that supports emoji
                font = ImageFont.truetype("seguiemj.ttf", 20)
                bbox = draw.textbbox((0, 0), emoji, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                draw.text(((size - text_width) // 2, (size - text_height) // 2), 
                         emoji, font=font, fill='white')
            except:
                # Fallback: just use the emoji as text
                draw.text((size//2-5, size//2-10), emoji, fill='white')
        
        return ImageTk.PhotoImage(img)
    
    def _create_agent_sprite(self, agent_name):
        """Create a modern agent sprite with consistent color."""
        size = 36
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Generate consistent color from agent name
        name_hash = hash(agent_name)
        r = (name_hash & 0xFF0000) >> 16
        g = (name_hash & 0x00FF00) >> 8
        b = name_hash & 0x0000FF
        
        # Ensure good contrast
        color = (max(50, r), max(50, g), max(50, b), 255)
        
        # Draw modern circular agent
        draw.ellipse([2, 2, size-2, size-2], fill=color, outline='white', width=2)
        
        # Add initial letter
        initial = agent_name[0].upper() if agent_name else 'A'
        try:
            font = ImageFont.truetype("arialbd.ttf", 14)
            bbox = draw.textbbox((0, 0), initial, font=font)
            text_width = bbox[2] - bbox[0]
            draw.text(((size - text_width) // 2, size//2 - 7), initial, font=font, fill='white')
        except:
            draw.text((size//2-3, size//2-7), initial, fill='white')
        
        return ImageTk.PhotoImage(img)

    def start_update_loop(self):
        """Start the background thread for periodic updates."""
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()

    def _update_loop(self):
        """Background thread for periodic updates."""
        while self.running:
            try:
                # Fetch new data from simulation
                self._fetch_simulation_data()
                
                # Update GUI
                self.master.after(0, self._update_gui)
                
                # Wait for next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                time.sleep(5)  # Wait before retrying

    def _fetch_simulation_data(self):
        """Fetch simulation data from the TownSimulation instance."""
        if not self.simulation:
            return
            
        try:
            # Get agents data
            self.agents = {}
            for agent_name, agent in self.simulation.agents.items():
                # Get current position from location tracker
                position_data = self.simulation.location_tracker.get_agent_position(agent_name)
                
                # Initialize default values
                current_location = "Traveling"
                coordinates = (0, 0)  # Default to origin

                # Handle position data - use coordinates directly
                if isinstance(position_data, (list, tuple)):
                    if len(position_data) == 2 and all(isinstance(x, (int, float)) for x in position_data):
                        # This is a direct coordinate pair (x,y)
                        coordinates = position_data
                        # Display the actual coordinates as the location
                        current_location = f"Position ({coordinates[0]:.1f}, {coordinates[1]:.1f})"
                    elif len(position_data) == 3:
                        # Working format: (agent_name, location_name, timestamp)
                        _, loc_data, _ = position_data
                        
                        # Check if the location data is actually coordinates
                        if isinstance(loc_data, str) and loc_data.startswith('(') and loc_data.endswith(')'):
                            try:
                                # Parse coordinate string like "(x, y)"
                                coordinates = eval(loc_data)
                                current_location = f"Position ({coordinates[0]:.1f}, {coordinates[1]:.1f})"
                            except:
                                coordinates = (0, 0)
                                current_location = "Unknown Position"
                        elif isinstance(loc_data, (list, tuple)) and len(loc_data) == 2:
                            # Direct coordinate pair
                            coordinates = loc_data
                            current_location = f"Position ({coordinates[0]:.1f}, {coordinates[1]:.1f})"
                        else:
                            # Regular location name
                            current_location = loc_data
                            coordinates = self.simulation.town_map.get_coordinates_for_location(loc_data)
                            if coordinates is None:
                                coordinates = (0, 0)
                                current_location = "Unknown Location"

                # If we still don't have valid coordinates, use a default
                if coordinates is None:
                    coordinates = (0, 0)
                    current_location = "Unknown Position"
                
                # Get agent stats
                try:
                    energy_result = agent.energy_system.get_energy(agent_name)
                    energy = energy_result.value if hasattr(energy_result, 'success') and energy_result.success else 0
                except Exception as e:
                    print(f"Error getting energy for {agent_name}: {str(e)}")
                    energy = 0
                    
                try:
                    grocery_result = agent.grocery_system.get_level(agent_name)
                    grocery = grocery_result.value if hasattr(grocery_result, 'success') and grocery_result.success else 0
                except Exception as e:
                    print(f"Error getting grocery level for {agent_name}: {str(e)}")
                    grocery = 0
                
                self.agents[agent_name] = {
                    'id': agent_name,
                    'name': agent_name,
                    'age': getattr(agent, 'age', 0),
                    'occupation': getattr(agent, 'occupation', 'Unknown'),
                    'residence': getattr(agent, 'residence', 'Unknown'),
                    'current_location': current_location,
                    'coordinates': coordinates,
                    'stats': {
                        'energy': energy,
                        'grocery': grocery,
                        'money': getattr(agent, 'money', 0)
                    },
                    'daily_plan': getattr(agent, 'daily_plan', {}),
                    'history': getattr(agent, 'history', [])
                }
            
            # Get NPCs data if available
            if hasattr(self.simulation, 'npcs'):
                for npc_name, npc in self.simulation.npcs.items():
                    # Get current position from location tracker
                    position_data = self.simulation.location_tracker.get_agent_position(npc_name)
                    
                    # Initialize default values
                    current_location = "Traveling"
                    coordinates = (0, 0)  # Default to origin

                    # Handle position data - use coordinates directly
                    if isinstance(position_data, (list, tuple)):
                        if len(position_data) == 2 and all(isinstance(x, (int, float)) for x in position_data):
                            # This is a direct coordinate pair (x,y)
                            coordinates = position_data
                            # Display the actual coordinates as the location
                            current_location = f"Position ({coordinates[0]:.1f}, {coordinates[1]:.1f})"
                        elif len(position_data) == 3:
                            # Working format: (agent_name, location_name, timestamp)
                            _, loc_data, _ = position_data
                            
                            # Check if the location data is actually coordinates
                            if isinstance(loc_data, str) and loc_data.startswith('(') and loc_data.endswith(')'):
                                try:
                                    # Parse coordinate string like "(x, y)"
                                    coordinates = eval(loc_data)
                                    current_location = f"Position ({coordinates[0]:.1f}, {coordinates[1]:.1f})"
                                except:
                                    coordinates = (0, 0)
                                    current_location = "Unknown Position"
                            elif isinstance(loc_data, (list, tuple)) and len(loc_data) == 2:
                                # Direct coordinate pair
                                coordinates = loc_data
                                current_location = f"Position ({coordinates[0]:.1f}, {coordinates[1]:.1f})"
                            else:
                                # Regular location name
                                current_location = loc_data
                                coordinates = self.simulation.town_map.get_coordinates_for_location(loc_data)
                                if coordinates is None:
                                    coordinates = (0, 0)
                                    current_location = "Unknown Location"

                    # If we still don't have valid coordinates, use a default
                    if coordinates is None:
                        coordinates = (0, 0)
                        current_location = "Unknown Position"
                    
                    # Add NPC to agents dictionary with NPC prefix
                    npc_id = f"NPC_{npc_name}"
                    self.agents[npc_id] = {
                        'id': npc_id,
                        'name': npc_name,
                        'age': getattr(npc, 'age', 0),
                        'occupation': getattr(npc, 'occupation', 'NPC'),
                        'residence': getattr(npc, 'residence', 'Unknown'),
                        'current_location': current_location,
                        'coordinates': coordinates,
                        'stats': {
                            'energy': 100,  # Default values for NPCs
                            'grocery': 100,
                            'money': getattr(npc, 'money', 0)
                        },
                        'daily_plan': getattr(npc, 'daily_plan', {}),
                        'history': getattr(npc, 'history', [])
                    }
            
            # Get locations data
            self.locations = {}
            for loc_name, location in self.simulation.locations.items():
                coords = self.simulation.town_map.get_coordinates_for_location(loc_name)
                if coords is None:
                    print(f"Warning: No coordinates found for location {loc_name}")
                    coords = (0, 0)
                    
                self.locations[loc_name] = {
                    'id': loc_name,
                    'name': loc_name,
                    'type': getattr(location, 'location_type', 'unknown'),
                    'position': coords,
                    'hours': getattr(location, 'hours', {}),
                    'prices': getattr(location, 'prices', {})
                }
                    
        except Exception as e:
            print(f"Error fetching simulation data: {str(e)}")
            traceback.print_exc()

    def _update_gui(self):
        """Update all GUI elements with current data."""
        try:
            # Update stats counters
            self.agent_count_label.config(text=f"Agents: {len(self.agents)}")
            self.location_count_label.config(text=f"Locations: {len(self.locations)}")
            
            # Update map
            self._update_map()
                
        except Exception as e:
            print(f"Error updating GUI: {str(e)}")
            traceback.print_exc()

    def _update_map(self):
        """Update the simulation map with current data in grid format."""
        # Store current view position before clearing
        if self.map_canvas.find_all():
            # Get current canvas center if we have items
            canvas_width = self.map_canvas.winfo_width()
            canvas_height = self.map_canvas.winfo_height()
            if self.zoom_center_x == 0 and self.zoom_center_y == 0:
                self.zoom_center_x = canvas_width // 2
                self.zoom_center_y = canvas_height // 2
        
        # Clear the canvas
        self.map_canvas.delete('all')
        
        # Grid configuration
        grid_size = 60  # Larger grid for better visibility
        x_offset = 60   # Left margin
        y_offset = 60   # Top margin
        
        # Calculate grid dimensions based on locations
        if self.locations:
            all_x = [loc['position'][0] for loc in self.locations.values()]
            all_y = [loc['position'][1] for loc in self.locations.values()]
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
        else:
            min_x, max_x, min_y, max_y = 0, 10, 0, 10
        
        # Calculate grid dimensions with padding
        grid_width = max_x - min_x + 3
        grid_height = max_y - min_y + 3
        
        # Draw grid lines with modern styling
        for x in range(grid_width):
            self.map_canvas.create_line(
                x_offset + x * grid_size, y_offset,
                x_offset + x * grid_size, y_offset + grid_height * grid_size,
                fill=self.colors['grid'], dash=(2,2)
            )
        for y in range(grid_height):
            self.map_canvas.create_line(
                x_offset, y_offset + y * grid_size,
                x_offset + grid_width * grid_size, y_offset + y * grid_size,
                fill=self.colors['grid'], dash=(2,2)
            )
        
        # Draw locations on grid with position adjustment
        for loc_id, location in self.locations.items():
            if 'position' in location and len(location['position']) == 2:
                x, y = location['position']
                # Adjust positions to account for negative coordinates
                adjusted_x = x - min_x + 1
                adjusted_y = y - min_y + 1
                center_x = x_offset + adjusted_x * grid_size + grid_size//2
                center_y = y_offset + adjusted_y * grid_size + grid_size//2
                
                # Determine sprite based on location type
                loc_type = location.get('type', 'unknown').lower()
                if 'house' in loc_type or 'residence' in loc_type:
                    sprite_key = 'house_default'
                elif 'restaurant' in loc_type or 'cafe' in loc_type:
                    sprite_key = 'restaurant_default'
                elif 'park' in loc_type:
                    sprite_key = 'park_default'
                elif 'business' in loc_type or 'office' in loc_type:
                    sprite_key = 'business_default'
                else:
                    sprite_key = 'default'
                
                if sprite_key in self.sprites:
                    # Draw location
                    img_id = self.map_canvas.create_image(
                        center_x, center_y,
                        image=self.sprites[sprite_key],
                        tags=('location', loc_id))
                    
                    # Always show location name with modern styling
                    label_id = self.map_canvas.create_text(
                        center_x, center_y + grid_size//2 + 10,
                        text=location.get('name', loc_id),
                        font=('Segoe UI', 9, 'bold'),
                        fill=self.colors['text'],
                        tags=('location_label', loc_id))
        
        # Draw agents and NPCs
        for agent_id, agent in self.agents.items():
            # Get agent coordinates directly
            if 'coordinates' in agent and agent['coordinates'] is not None:
                x, y = agent['coordinates']
                
                # Add name-based offset to prevent overlapping
                name_offset = self._calculate_name_offset(agent_id)
                x += name_offset[0]
                y += name_offset[1]
                
                # Adjust positions to account for negative coordinates
                adjusted_x = x - min_x + 1
                adjusted_y = y - min_y + 1
                center_x = x_offset + adjusted_x * grid_size + grid_size//2
                center_y = y_offset + adjusted_y * grid_size + grid_size//2
                
                # Create or get unique sprite for this agent
                sprite_key = f'agent_{agent_id}'
                if sprite_key not in self.sprites:
                    self.sprites[sprite_key] = self._create_agent_sprite(agent_id)
                
                # Draw agent
                img_id = self.map_canvas.create_image(
                    center_x, center_y,
                    image=self.sprites[sprite_key],
                    tags=('agent', agent_id))
                
                # Draw label (hidden by default for agents, always shown for NPCs)
                label_text = f"{agent.get('name', agent_id)}"
                if agent_id.startswith('NPC_'):
                    # Always show NPC names with distinct styling
                    label_id = self.map_canvas.create_text(
                        center_x, center_y + grid_size//2 + 10,
                        text=label_text,
                        font=('Segoe UI', 9, 'bold'),
                        fill=self.colors['accent'],
                        tags=('agent_label', agent_id))
                else:
                    # For regular agents, create hover label
                    label_id = self.map_canvas.create_text(
                        center_x, center_y + grid_size//2 + 10,
                        text=label_text,
                        font=('Segoe UI', 9),
                        fill=self.colors['text_light'],
                        tags=('agent_label', agent_id, 'hover_label'),
                        state='hidden')
                    
                    # Bind hover events for regular agents
                    self.map_canvas.tag_bind(img_id, '<Enter>',
                        lambda e, lid=label_id: self._show_label(lid))
                    self.map_canvas.tag_bind(img_id, '<Leave>',
                        lambda e, lid=label_id: self._hide_label(lid))
        
        # Set scroll region to include all content
        self.map_canvas.configure(scrollregion=(
            0, 0,
            x_offset + grid_width * grid_size,
            y_offset + grid_height * grid_size
        ))
        
        # Apply the stored zoom level after drawing
        if self.zoom_level != 1.0:
            self.map_canvas.scale("all", 
                                 self.zoom_center_x,
                                 self.zoom_center_y,
                                 self.zoom_level, self.zoom_level)

    def _calculate_name_offset(self, agent_name):
        """Calculate a small offset based on agent name to prevent overlapping."""
        if not agent_name:
            return (0, 0)
        
        # Use the first few characters of the name to generate consistent offsets
        hash_val = 0
        for i, char in enumerate(agent_name[:4]):
            hash_val = (hash_val * 31 + ord(char)) % 1000
        
        # Convert hash to angle and then to x,y offsets
        angle = (hash_val / 1000.0) * 2 * math.pi
        radius = 0.3
        
        x_offset = math.cos(angle) * radius
        y_offset = math.sin(angle) * radius
        
        return (x_offset, y_offset)
    
    def _show_label(self, label_id):
        """Show a label when hovered over."""
        self.map_canvas.itemconfigure(label_id, state='normal')

    def _hide_label(self, label_id):
        """Hide a label when no longer hovered over."""
        self.map_canvas.itemconfigure(label_id, state='hidden')

    def on_close(self):
        """Handle window close event."""
        self.running = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=1)
        self.master.destroy()

def launch_gui(simulation=None):
    """Launch the GUI with optional simulation reference."""
    root = tk.Tk()
    gui = AgentSimulationGUI(root, simulation)
    root.mainloop()