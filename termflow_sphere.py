#!/usr/bin/env python3
"""
TermFlow Sphere - A minimal, sophisticated sphere visualization for system and Claude CLI metrics
"""

import os
import sys
import time
import math
import random
import signal
import psutil
import json
import threading
import termios
import tty
import select
from collections import deque, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import colorsys

class SphereVisualizer:
    def __init__(self):
        self.width = 80
        self.height = 24
        self.running = True
        
        # Sphere parameters
        self.sphere_radius = min(self.width // 3, int(self.height * 0.7) - 2)
        self.base_radius = self.sphere_radius
        
        # Multi-axis rotation
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0
        self.rotation_speed_x = 0.005
        self.rotation_speed_y = 0.01
        self.rotation_speed_z = 0.003
        
        # Breathing effect
        self.breath_phase = 0.0
        self.breath_amplitude = 0.0
        
        # Ripple system
        self.ripples = []  # List of active ripples
        self.max_ripples = 12
        self.spiral_ripples = []  # Spiral patterns
        
        # Particle trails
        self.particle_trails = {}  # Dict of point_id -> trail positions
        self.max_trail_length = 8
        
        # Energy arcs
        self.energy_arcs = []
        self.max_arcs = 5
        
        # Aurora bands
        self.aurora_bands = []
        self.aurora_phase = 0.0
        
        # Turbulence
        self.turbulence_field = {}
        self.turbulence_time = 0.0
        
        # Activity metrics
        self.activity_level = 0.0  # 0.0 to 1.0
        self.smoothed_activity = 0.0
        
        # Metrics
        self.cpu_percent = 0.0
        self.memory_percent = 0.0
        self.network_activity = 0.0
        self.claude_activity = 0.0
        
        # History for smoothing
        self.metric_history = defaultdict(lambda: deque(maxlen=10))
        
        # Colors - Monokai Pro inspired palette
        self.colors = {
            'void': '\033[38;2;25;25;28m',      # Deep background
            'dim': '\033[38;2;45;45;50m',       # Dark grey
            'outline': '\033[38;2;65;65;70m',   # Medium grey
            
            # CPU colors (cyan/blue spectrum - Monokai blue)
            'cpu_low': '\033[38;2;78;201;176m',    # Soft cyan
            'cpu_mid': '\033[38;2;102;217;239m',   # Sky blue  
            'cpu_high': '\033[38;2;130;229;255m',  # Bright cyan
            
            # Memory colors (magenta/pink spectrum - Monokai pink)
            'mem_low': '\033[38;2;255;97;136m',    # Soft pink
            'mem_mid': '\033[38;2;255;121;198m',   # Hot pink
            'mem_high': '\033[38;2;255;154;230m',  # Bright pink
            
            # Network colors (green spectrum - Monokai green)
            'net_low': '\033[38;2;166;226;46m',   # Lime green
            'net_mid': '\033[38;2;190;232;70m',   # Bright green
            'net_high': '\033[38;2;214;238;94m',  # Electric green
            
            # Claude colors (orange spectrum - Claude brand color)  
            'claude_low': '\033[38;2;255;216;102m',  # Soft yellow-orange
            'claude_mid': '\033[38;2;255;203;107m',  # Golden orange
            'claude_high': '\033[38;2;255;184;108m', # Bright orange
            
            'accent': '\033[38;2;189;147;249m',  # Monokai purple
            'white': '\033[38;2;248;248;242m',   # Monokai foreground
            'reset': '\033[0m'
        }
        
        # Grid points on sphere (latitude/longitude)
        self.grid_resolution = 24  # Number of latitude lines
        self.grid_points = self._generate_sphere_grid()
        
        # Background thread for metrics
        self.metrics_lock = threading.Lock()
        self.metrics_thread = threading.Thread(target=self.update_metrics_background, daemon=True)
        
        # Terminal setup
        self.setup_terminal()
        
        # Store original terminal settings for restoration
        self.old_settings = None
        
    def _generate_sphere_grid(self):
        """Generate a grid of points on the sphere surface"""
        points = []
        
        # Create latitude/longitude grid
        for lat_idx in range(self.grid_resolution):
            lat = (lat_idx / (self.grid_resolution - 1) - 0.5) * math.pi  # -pi/2 to pi/2
            
            # Fewer points near poles
            lon_points = max(4, int(self.grid_resolution * math.cos(lat)))
            
            for lon_idx in range(lon_points):
                lon = (lon_idx / lon_points) * 2 * math.pi  # 0 to 2pi
                
                # Convert spherical to cartesian
                x = math.cos(lat) * math.cos(lon)
                y = math.sin(lat)
                z = math.cos(lat) * math.sin(lon)
                
                points.append((x, y, z, lat_idx, lon_idx))
                
        return points
    
    def setup_terminal(self):
        """Setup terminal for visualization"""
        # Save terminal settings
        try:
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except:
            self.old_settings = None
            
        # Clear screen and hide cursor
        sys.stdout.write('\033[2J\033[H\033[?25l')
        sys.stdout.flush()
        
        # Get terminal size
        try:
            size = os.get_terminal_size()
            self.width = size.columns
            self.height = size.lines - 1
            self.sphere_radius = min(self.width // 3, int(self.height * 0.7) - 2)
        except:
            pass
            
    def cleanup(self):
        """Restore terminal state"""
        sys.stdout.write('\033[?25h\033[0m')
        sys.stdout.flush()
        
        # Restore terminal settings
        if self.old_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except:
                pass
        
    def project_3d_to_2d(self, x, y, z):
        """Project 3D point to 2D screen coordinates with multi-axis rotation"""
        # Apply rotation around X axis
        cos_x = math.cos(self.rotation_x)
        sin_x = math.sin(self.rotation_x)
        y_rot = y * cos_x - z * sin_x
        z_rot = y * sin_x + z * cos_x
        
        # Apply rotation around Y axis
        cos_y = math.cos(self.rotation_y)
        sin_y = math.sin(self.rotation_y)
        x_rot = x * cos_y - z_rot * sin_y
        z_rot2 = x * sin_y + z_rot * cos_y
        
        # Apply rotation around Z axis
        cos_z = math.cos(self.rotation_z)
        sin_z = math.sin(self.rotation_z)
        x_final = x_rot * cos_z - y_rot * sin_z
        y_final = x_rot * sin_z + y_rot * cos_z
        z_final = z_rot2
        
        # Simple perspective projection
        perspective = 2.5
        scale = perspective / (perspective + z_final)
        
        # Apply breathing effect
        breath_scale = 1.0 + math.sin(self.breath_phase) * self.breath_amplitude
        
        # Convert to screen coordinates
        # Move sphere up by 10% of height
        center_y_offset = int(self.height * 0.1)
        screen_x = int(self.width // 2 + x_final * self.sphere_radius * scale * breath_scale)
        screen_y = int(self.height // 2 - center_y_offset - y_final * self.sphere_radius * scale * breath_scale * 0.5)  # Aspect ratio correction
        
        return screen_x, screen_y, z_final
        
    def add_ripple(self, lat_origin, lon_origin, intensity, metric_type='mixed'):
        """Add a new ripple at the specified location"""
        if len(self.ripples) < self.max_ripples:
            # Sometimes create spiral ripples
            is_spiral = random.random() < 0.3 and intensity > 0.5
            
            ripple = {
                'lat': lat_origin,
                'lon': lon_origin,
                'radius': 0.0,
                'max_radius': 0.6 + intensity * 1.2,  # Larger ripples
                'intensity': intensity,
                'age': 0.0,
                'speed': 0.25 + intensity * 0.2,  # Faster ripples
                'type': metric_type,  # Track which metric caused this ripple
                'spiral': is_spiral,
                'spiral_angle': random.uniform(0, 2 * math.pi),
                'spin_speed': random.uniform(0.5, 2.0) * (1 if random.random() > 0.5 else -1)
            }
            
            if is_spiral:
                self.spiral_ripples.append(ripple)
            else:
                self.ripples.append(ripple)
            
    def update_ripples(self, dt):
        """Update ripple animations"""
        # Add new ripples based on activity - increased frequency
        if self.smoothed_activity > 0.05 and random.random() < self.smoothed_activity * 0.3:
            # Choose metric type based on which is most active
            metrics = {
                'cpu': self.cpu_percent,
                'memory': self.memory_percent,
                'network': self.network_activity,
                'claude': self.claude_activity
            }
            
            # Weight the choice by metric values
            total_weight = sum(metrics.values())
            if total_weight > 0:
                rand_val = random.random() * total_weight
                cumulative = 0
                metric_type = 'mixed'
                
                for m_type, m_value in metrics.items():
                    cumulative += m_value
                    if rand_val <= cumulative:
                        metric_type = m_type
                        break
            else:
                metric_type = 'mixed'
            
            # Random location weighted towards equator
            lat = (random.random() - 0.5) * math.pi * 0.8
            lon = random.random() * 2 * math.pi
            self.add_ripple(lat, lon, self.smoothed_activity, metric_type)
            
        # Update existing ripples
        self.ripples = [r for r in self.ripples if r['age'] < 1.0]
        self.spiral_ripples = [r for r in self.spiral_ripples if r['age'] < 1.0]
        
        for ripple in self.ripples + self.spiral_ripples:
            ripple['radius'] += ripple['speed'] * dt
            ripple['age'] = ripple['radius'] / ripple['max_radius']
            
            # Update spiral angle
            if ripple.get('spiral', False):
                ripple['spiral_angle'] += ripple['spin_speed'] * dt
            
    def get_ripple_influence(self, lat, lon):
        """Calculate ripple intensity and dominant type at a given point"""
        intensities = {'mixed': 0.0, 'cpu': 0.0, 'memory': 0.0, 'network': 0.0, 'claude': 0.0}
        
        for ripple in self.ripples + self.spiral_ripples:
            # Calculate angular distance
            d_lat = lat - ripple['lat']
            d_lon = lon - ripple['lon']
            
            # Apply spiral distortion if it's a spiral ripple
            if ripple.get('spiral', False):
                spiral_offset = math.sin(ripple['spiral_angle'] + d_lon * 4) * 0.1
                d_lat += spiral_offset
            
            # Handle longitude wrap
            if d_lon > math.pi:
                d_lon -= 2 * math.pi
            elif d_lon < -math.pi:
                d_lon += 2 * math.pi
                
            # Angular distance on sphere
            angular_dist = math.sqrt(d_lat**2 + (d_lon * math.cos(lat))**2)
            
            # Check if point is within ripple
            if angular_dist < ripple['radius']:
                # Calculate intensity with smooth falloff
                dist_ratio = angular_dist / ripple['radius']
                fade = 1.0 - ripple['age']
                
                # Sine wave pattern for ripple - more pronounced
                if ripple.get('spiral', False):
                    # Spiral ripples have a different wave pattern
                    wave = math.sin((1.0 - dist_ratio) * math.pi * 4 + ripple['spiral_angle']) * 0.7 + 0.3
                else:
                    wave = math.sin((1.0 - dist_ratio) * math.pi * 3) * 0.6 + 0.4
                intensity = wave * fade * ripple['intensity'] * 1.5  # Amplified
                
                intensities[ripple['type']] += intensity
        
        # Find dominant type and total intensity
        total_intensity = sum(intensities.values())
        dominant_type = max(intensities.items(), key=lambda x: x[1])[0] if total_intensity > 0 else 'mixed'
        
        return min(total_intensity, 1.0), dominant_type
    
    def update_energy_arcs(self, dt):
        """Update energy arc connections between high activity zones"""
        # Remove expired arcs
        self.energy_arcs = [arc for arc in self.energy_arcs if arc['life'] > 0]
        
        # Update existing arcs
        for arc in self.energy_arcs:
            arc['life'] -= dt * 2.0  # Arcs are short-lived
            arc['phase'] += dt * 10.0  # Fast pulsing
        
        # Add new arcs based on activity
        if len(self.energy_arcs) < self.max_arcs and self.smoothed_activity > 0.4 and random.random() < 0.1:
            # Find two high-activity points
            lat1 = random.uniform(-math.pi/3, math.pi/3)
            lon1 = random.uniform(0, 2 * math.pi)
            lat2 = lat1 + random.uniform(-0.5, 0.5)
            lon2 = lon1 + random.uniform(-1.0, 1.0)
            
            self.energy_arcs.append({
                'lat1': lat1, 'lon1': lon1,
                'lat2': lat2, 'lon2': lon2,
                'life': 1.0,
                'phase': 0.0,
                'intensity': self.smoothed_activity
            })
    
    def update_aurora_bands(self, dt):
        """Update aurora-like flowing bands"""
        self.aurora_phase += dt * 0.5
        
        # Ensure we have aurora bands
        if len(self.aurora_bands) < 3:
            for i in range(3 - len(self.aurora_bands)):
                self.aurora_bands.append({
                    'base_lat': random.uniform(-math.pi/2, math.pi/2),
                    'phase_offset': random.uniform(0, 2 * math.pi),
                    'amplitude': random.uniform(0.1, 0.3),
                    'frequency': random.uniform(2, 4)
                })
    
    def get_aurora_influence(self, lat, lon):
        """Calculate aurora band influence at a point"""
        influence = 0.0
        
        for band in self.aurora_bands:
            # Calculate distance from band
            band_lat = band['base_lat'] + math.sin(lon * band['frequency'] + self.aurora_phase + band['phase_offset']) * band['amplitude']
            dist = abs(lat - band_lat)
            
            if dist < 0.3:  # Band width
                # Smooth falloff
                band_influence = (1.0 - dist / 0.3) * 0.5
                # Add shimmer
                shimmer = math.sin(self.aurora_phase * 3 + lon * 5) * 0.3 + 0.7
                influence += band_influence * shimmer * self.smoothed_activity
        
        return min(influence, 1.0)
    
    def get_turbulence(self, lat, lon, time):
        """Calculate turbulence distortion at a point"""
        # Create turbulence using multiple sine waves
        turb_x = math.sin(lat * 4 + time * 2) * math.cos(lon * 3 + time * 1.5) * 0.05
        turb_y = math.cos(lat * 3 + time * 1.8) * math.sin(lon * 4 + time * 2.2) * 0.05
        turb_z = math.sin(lat * 5 + lon * 5 + time * 2.5) * 0.03
        
        # Scale by activity
        scale = self.smoothed_activity * 0.5
        return turb_x * scale, turb_y * scale, turb_z * scale
        
    def update_metrics_background(self):
        """Background thread to update system metrics"""
        last_net_io = psutil.net_io_counters()
        last_time = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time
                
                # CPU and Memory
                cpu = psutil.cpu_percent(interval=0.1)
                mem = psutil.virtual_memory().percent
                
                # Network activity
                net_io = psutil.net_io_counters()
                bytes_sent = net_io.bytes_sent - last_net_io.bytes_sent
                bytes_recv = net_io.bytes_recv - last_net_io.bytes_recv
                last_net_io = net_io
                
                # Normalize network activity (MB/s)
                net_rate = (bytes_sent + bytes_recv) / (1024 * 1024) / max(dt, 0.1)
                net_activity = min(net_rate / 10.0, 1.0)  # Normalize to 0-1 (10 MB/s = 1.0)
                
                # Update Claude metrics
                claude_activity = self.update_claude_metrics()
                
                # Store in history for smoothing
                with self.metrics_lock:
                    self.metric_history['cpu'].append(cpu / 100.0)
                    self.metric_history['memory'].append(mem / 100.0)
                    self.metric_history['network'].append(net_activity)
                    self.metric_history['claude'].append(claude_activity)
                    
                    # Calculate smoothed values
                    self.cpu_percent = sum(self.metric_history['cpu']) / len(self.metric_history['cpu'])
                    self.memory_percent = sum(self.metric_history['memory']) / len(self.metric_history['memory'])
                    self.network_activity = sum(self.metric_history['network']) / len(self.metric_history['network'])
                    self.claude_activity = sum(self.metric_history['claude']) / len(self.metric_history['claude'])
                    
                    # Overall activity level - increased weights for more impact
                    self.activity_level = (
                        self.cpu_percent * 0.35 +
                        self.memory_percent * 0.25 +
                        self.network_activity * 0.35 +
                        self.claude_activity * 0.25
                    ) * 1.2  # Amplify overall impact
                    
                time.sleep(0.5)
                
            except Exception as e:
                time.sleep(1)
                
    def update_claude_metrics(self):
        """Check Claude CLI activity"""
        activity = 0.0
        
        # Check both possible Claude project directories
        claude_dirs = [
            Path.home() / '.claude' / 'projects',
            Path.home() / '.config' / 'claude' / 'projects'
        ]
        
        for claude_dir in claude_dirs:
            if not claude_dir.exists():
                continue
                
            try:
                # Look for recent activity (files modified in last 5 minutes)
                recent_files = 0
                now = datetime.now()
                
                for project_dir in claude_dir.iterdir():
                    if project_dir.is_dir():
                        for file in project_dir.rglob('*'):
                            if file.is_file():
                                try:
                                    mtime = datetime.fromtimestamp(file.stat().st_mtime)
                                    if now - mtime < timedelta(minutes=5):
                                        recent_files += 1
                                except:
                                    pass
                                    
                # Normalize activity (0-1 range)
                activity = max(activity, min(recent_files / 10.0, 1.0))
                
            except:
                pass
                
        return activity
        
    def render_frame(self):
        """Render a single frame"""
        # Create screen buffer
        buffer = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        z_buffer = [[-float('inf') for _ in range(self.width)] for _ in range(self.height)]
        
        # Update multi-axis rotation with activity-based speed variations
        speed_multiplier = 1.0 + self.smoothed_activity * 2.0  # Speed up with activity
        
        # Add wobble based on individual metrics
        wobble_x = math.sin(time.time() * 0.7) * self.cpu_percent * 0.02
        wobble_y = math.cos(time.time() * 0.5) * self.memory_percent * 0.02
        wobble_z = math.sin(time.time() * 0.9) * self.network_activity * 0.01
        
        self.rotation_x += (self.rotation_speed_x + wobble_x) * speed_multiplier
        self.rotation_y += (self.rotation_speed_y + wobble_y) * speed_multiplier
        self.rotation_z += (self.rotation_speed_z + wobble_z) * speed_multiplier
        
        # Update breathing effect
        self.breath_phase += 0.05
        self.breath_amplitude = self.smoothed_activity * 0.1  # Breathe more with activity
        
        # Smooth activity level
        self.smoothed_activity = self.smoothed_activity * 0.9 + self.activity_level * 0.1
        
        # Update all animation systems
        dt = 1.0 / 30.0  # 30 FPS
        self.update_ripples(dt)
        self.update_energy_arcs(dt)
        self.update_aurora_bands(dt)
        self.turbulence_time += dt
        
        # Render sphere points
        visible_points = []
        
        for point_idx, (x, y, z, lat_idx, lon_idx) in enumerate(self.grid_points):
            # Apply turbulence
            lat = (lat_idx / (self.grid_resolution - 1) - 0.5) * math.pi
            lon = math.atan2(z, x)
            turb_x, turb_y, turb_z = self.get_turbulence(lat, lon, self.turbulence_time)
            
            # Apply turbulence to position
            x_turb = x + turb_x
            y_turb = y + turb_y
            z_turb = z + turb_z
            
            screen_x, screen_y, z_rot = self.project_3d_to_2d(x_turb, y_turb, z_turb)
            
            # Update particle trail for this point
            if point_idx not in self.particle_trails:
                self.particle_trails[point_idx] = deque(maxlen=self.max_trail_length)
            
            # Only track trails for active points
            if self.smoothed_activity > 0.3 and point_idx % 20 == 0:  # Sample points
                self.particle_trails[point_idx].append((screen_x, screen_y, z_rot))
            
            # Only render front-facing points (adjusted for better depth)
            if z_rot > -0.5 and 0 <= screen_x < self.width and 0 <= screen_y < self.height:
                # Calculate latitude/longitude for ripple lookup
                lat = (lat_idx / (self.grid_resolution - 1) - 0.5) * math.pi
                lon = math.atan2(z, x) + self.rotation_y
                
                # Get ripple influence at this point
                ripple_intensity, ripple_type = self.get_ripple_influence(lat, lon)
                
                # Get aurora influence
                aurora_intensity = self.get_aurora_influence(lat, lon)
                
                # Base visibility based on activity and position
                edge_factor = abs(z_rot)  # Stronger at edges when rotating
                # Enhanced depth perception - points further back are dimmer
                depth_factor = (z_rot + 1.0) / 2.0  # 0 to 1 (back to front)
                base_visibility = 0.15 + edge_factor * 0.1 + depth_factor * 0.1
                
                # Combine with activity, ripples, and aurora - increased impact
                point_intensity = base_visibility + self.smoothed_activity * 0.5 + ripple_intensity * 0.9 + aurora_intensity * 0.7
                
                # Add pulsing based on local metrics
                pulse = math.sin(time.time() * 3 + lat * 2 + lon) * 0.1 + 0.9
                point_intensity *= pulse
                point_intensity = min(point_intensity, 1.0)
                
                # Scale factor based on latitude (smaller near poles)
                lat_scale = math.cos(lat) * 0.85 + 0.15  # 0.15 to 1.0 - more dramatic scaling
                
                # Only render if above threshold or on edge - higher contrast
                if point_intensity > 0.25 or edge_factor > 0.85:
                    visible_points.append((screen_x, screen_y, z_rot, point_intensity, ripple_type, lat_scale, aurora_intensity > 0.1))
                    
        # Sort by z-depth and render
        visible_points.sort(key=lambda p: p[2])
        
        # Calculate border radius for clipping
        center_x = self.width // 2
        center_y = self.height // 2 - int(self.height * 0.1)  # Move up by 10%
        border_radius = self.sphere_radius + 3  # Same as border
        
        # Render particle trails first (behind main points)
        for point_idx, trail in self.particle_trails.items():
            if len(trail) > 1:
                for i in range(len(trail) - 1):
                    x1, y1, z1 = trail[i]
                    x2, y2, z2 = trail[i + 1]
                    
                    # Only render if both points are visible
                    if 0 <= x1 < self.width and 0 <= y1 < self.height and 0 <= x2 < self.width and 0 <= y2 < self.height:
                        # Fade based on age
                        fade = (i + 1) / len(trail)
                        if fade > 0.3:
                            # Simple line drawing - just use the midpoint
                            mid_x = (x1 + x2) // 2
                            mid_y = (y1 + y2) // 2
                            if 0 <= mid_x < self.width and 0 <= mid_y < self.height:
                                trail_char = '·' if fade < 0.7 else '∘'
                                buffer[mid_y][mid_x] = f"{self.colors['accent']}{trail_char}{self.colors['reset']}"
        
        # Render energy arcs
        for arc in self.energy_arcs:
            # Convert arc endpoints to 3D
            x1 = math.cos(arc['lat1']) * math.cos(arc['lon1'])
            y1 = math.sin(arc['lat1'])
            z1 = math.cos(arc['lat1']) * math.sin(arc['lon1'])
            
            x2 = math.cos(arc['lat2']) * math.cos(arc['lon2'])
            y2 = math.sin(arc['lat2'])
            z2 = math.cos(arc['lat2']) * math.sin(arc['lon2'])
            
            # Project to 2D
            sx1, sy1, sz1 = self.project_3d_to_2d(x1, y1, z1)
            sx2, sy2, sz2 = self.project_3d_to_2d(x2, y2, z2)
            
            # Only render if both endpoints are visible
            if sz1 > -0.5 and sz2 > -0.5:
                # Draw arc segments
                steps = 10
                for i in range(steps):
                    t = i / float(steps - 1)
                    # Interpolate along arc
                    x = x1 * (1 - t) + x2 * t
                    y = y1 * (1 - t) + y2 * t
                    z = z1 * (1 - t) + z2 * t
                    
                    # Normalize
                    length = math.sqrt(x*x + y*y + z*z)
                    x /= length
                    y /= length
                    z /= length
                    
                    sx, sy, sz = self.project_3d_to_2d(x, y, z)
                    
                    if 0 <= sx < self.width and 0 <= sy < self.height and sz > -0.5:
                        # Pulsing intensity
                        pulse = math.sin(arc['phase'] + t * math.pi) * 0.5 + 0.5
                        arc_char = '*' if pulse > 0.7 else '∘' if pulse > 0.4 else '·'
                        color = self.colors['claude_high'] if arc['life'] > 0.5 else self.colors['claude_mid']
                        buffer[sy][sx] = f"{color}{arc_char}{self.colors['reset']}"
        
        for screen_x, screen_y, z_rot, intensity, ripple_type, lat_scale, has_aurora in visible_points:
            # Check if point is within the circular border
            dx = screen_x - center_x
            dy = (screen_y - center_y) * 2  # Account for aspect ratio
            distance = math.sqrt(dx * dx + dy * dy)
            
            if distance <= border_radius and z_buffer[screen_y][screen_x] < z_rot:
                z_buffer[screen_y][screen_x] = z_rot
                
                # Choose character based on intensity and scale
                # Near poles (lat_scale < 0.3) use smallest characters
                if lat_scale < 0.3:
                    if intensity > 0.6:
                        char = '∘'
                    else:
                        char = '·'
                # Mid-latitudes (0.3 - 0.7)
                elif lat_scale < 0.7:
                    if intensity > 0.7:
                        char = '○'
                    elif intensity > 0.4:
                        char = '∘'
                    else:
                        char = '·'
                # Near equator (lat_scale >= 0.7) use largest characters
                else:
                    if intensity > 0.8:
                        char = '●'
                    elif intensity > 0.6:
                        char = '◉'
                    elif intensity > 0.4:
                        char = '○'
                    elif intensity > 0.2:
                        char = '∘'
                    else:
                        char = '·'
                
                # Choose color based on ripple type, intensity, and aurora
                if has_aurora:
                    # Aurora colors - shifting between green, blue, and purple
                    aurora_hue = (math.sin(self.aurora_phase + screen_x * 0.02) + 1) * 0.5
                    if aurora_hue < 0.33:
                        color = self.colors['net_high']  # Green
                    elif aurora_hue < 0.66:
                        color = self.colors['cpu_high']  # Blue
                    else:
                        color = self.colors['accent']  # Purple
                elif ripple_type == 'cpu':
                    if intensity > 0.6:
                        color = self.colors['cpu_high']
                    elif intensity > 0.3:
                        color = self.colors['cpu_mid']
                    else:
                        color = self.colors['cpu_low']
                elif ripple_type == 'memory':
                    if intensity > 0.6:
                        color = self.colors['mem_high']
                    elif intensity > 0.3:
                        color = self.colors['mem_mid']
                    else:
                        color = self.colors['mem_low']
                elif ripple_type == 'network':
                    if intensity > 0.6:
                        color = self.colors['net_high']
                    elif intensity > 0.3:
                        color = self.colors['net_mid']
                    else:
                        color = self.colors['net_low']
                elif ripple_type == 'claude':
                    if intensity > 0.6:
                        color = self.colors['claude_high']
                    elif intensity > 0.3:
                        color = self.colors['claude_mid']
                    else:
                        color = self.colors['claude_low']
                else:  # mixed or default
                    if intensity > 0.6:
                        color = self.colors['accent']
                    elif intensity > 0.3:
                        color = self.colors['cpu_mid']
                    else:
                        color = self.colors['outline']
                    
                buffer[screen_y][screen_x] = f"{color}{char}{self.colors['reset']}"
                
        # Draw bright yellow border around sphere
        self._draw_sphere_border(buffer, z_buffer)
                
        # Add metrics display (minimal, bottom corners)
        if self.height > 10:
            # Left corner - system metrics
            cpu_bar = self._create_mini_bar(self.cpu_percent, 8)
            mem_bar = self._create_mini_bar(self.memory_percent, 8)
            
            self._draw_text(buffer, 2, self.height - 4, f"{self.colors['white']}CPU {self.colors['cpu_mid']}{cpu_bar}")
            self._draw_text(buffer, 2, self.height - 3, f"{self.colors['white']}MEM {self.colors['mem_mid']}{mem_bar}")
            
            # Right corner - network and claude
            net_bar = self._create_mini_bar(self.network_activity, 8)
            claude_bar = self._create_mini_bar(self.claude_activity, 8)
            
            self._draw_text(buffer, self.width - 20, self.height - 4, 
                          f"{self.colors['white']}NET {self.colors['net_mid']}{net_bar}")
            self._draw_text(buffer, self.width - 20, self.height - 3, 
                          f"{self.colors['white']}CLD {self.colors['claude_mid']}{claude_bar}")
            
            # Center - overall activity
            activity_text = f"◈ {int(self.smoothed_activity * 100)}%"
            # Calculate proper centering based on actual text length
            text_length = len(activity_text)
            x_position = self.width // 2 - text_length // 2
            self._draw_text(buffer, x_position, self.height - 2, 
                          f"{self.colors['accent']}{activity_text}")
            
        # Render buffer to screen
        output = '\033[H'  # Home cursor
        for row in buffer:
            output += ''.join(row) + '\n'
            
        sys.stdout.write(output)
        sys.stdout.flush()
        
    def _create_mini_bar(self, value, width):
        """Create a minimal progress bar"""
        filled = int(value * width)
        bar = '▪' * filled + '·' * (width - filled)
        return bar
        
    def _draw_text(self, buffer, x, y, text):
        """Draw text to buffer (handles ANSI codes)"""
        if 0 <= y < len(buffer) and 0 <= x < self.width:
            # Split text into segments by ANSI codes
            segments = []
            current_segment = ""
            current_color = ""
            i = 0
            
            while i < len(text):
                if text[i] == '\033' and i + 1 < len(text) and text[i + 1] == '[':
                    # Found ANSI escape sequence
                    if current_segment:
                        segments.append((current_segment, current_color))
                        current_segment = ""
                    
                    # Extract the full ANSI sequence
                    end = text.find('m', i)
                    if end != -1:
                        current_color = text[i:end + 1]
                        i = end + 1
                    else:
                        i += 1
                else:
                    current_segment += text[i]
                    i += 1
                    
            if current_segment:
                segments.append((current_segment, current_color))
                
            # Now draw the segments
            col = x
            for segment_text, color in segments:
                for char in segment_text:
                    if 0 <= col < self.width:
                        buffer[y][col] = color + char + self.colors['reset']
                    col += 1
                
    def _draw_sphere_border(self, buffer, z_buffer):
        """Draw a bright animated border around the sphere"""
        center_x = self.width // 2
        center_y = self.height // 2 - int(self.height * 0.1)  # Move up by 10%
        radius = self.sphere_radius + 3  # Slightly larger than sphere for clear border
        
        # Number of border dots for a perfect circle
        num_dots = int(2 * math.pi * radius * 0.8)  # Adjust density as needed
        
        # Get combined activity level
        activity = self.smoothed_activity
        
        # Always use the smallest dot character
        char = '·'
        
        # Animated rainbow border based on activity
        base_hue = time.time() * 0.1  # Slowly rotating hue
        
        # Calculate color based on activity and position
        if activity > 0.8:
            # Full rainbow when very active
            use_rainbow = True
        elif activity > 0.6:
            color = '\033[93m'  # Bright yellow
            use_rainbow = False
        elif activity > 0.4:
            color = '\033[38;5;178m'  # Pale yellow
            use_rainbow = False
        elif activity > 0.2:
            color = '\033[38;5;136m'  # Dark yellow
            use_rainbow = False
        else:
            color = '\033[38;5;58m'   # Very dark yellow/brown
            use_rainbow = False
        
        for i in range(num_dots):
            angle = (i / num_dots) * 2 * math.pi
            
            # Calculate position for perfect circle
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle) * 0.5  # Aspect ratio correction
            
            # Round to nearest integer for terminal positioning
            x_int = int(round(x))
            y_int = int(round(y))
            
            # Check bounds
            if 0 <= x_int < self.width and 0 <= y_int < self.height:
                # Choose color
                if 'use_rainbow' in locals() and use_rainbow:
                    # Rainbow effect
                    hue = (base_hue + angle / (2 * math.pi)) % 1.0
                    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
                    current_color = f'\033[38;2;{int(r*255)};{int(g*255)};{int(b*255)}m'
                else:
                    # Add subtle pulsing to brightness
                    pulse = math.sin(time.time() * 2.5 + angle * 3) * 0.3 + 0.7
                    if pulse > 0.85 and activity > 0.5:
                        current_color = '\033[93m'  # Flash to bright yellow
                    else:
                        current_color = color
                
                # Vary character based on activity
                if activity > 0.7 and random.random() < 0.1:
                    char = '⚬'  # Occasional different character
                elif activity > 0.5 and i % 3 == 0:
                    char = '∘'
                
                # Draw the border dot
                buffer[y_int][x_int] = f"{current_color}{char}{self.colors['reset']}"
                
    def run(self):
        """Main run loop"""
        self.metrics_thread.start()
        
        # Allow metrics to initialize
        time.sleep(0.5)
        
        frame_time = 1.0 / 30.0  # 30 FPS
        last_frame = time.time()
        
        try:
            while self.running:
                current_time = time.time()
                
                # Check for keyboard input
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1)
                    if key.lower() == 'q':
                        self.running = False
                        break
                
                if current_time - last_frame >= frame_time:
                    self.render_frame()
                    last_frame = current_time
                    
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()
            
def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    sys.exit(0)
    
def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    visualizer = SphereVisualizer()
    visualizer.run()
    
if __name__ == "__main__":
    main()