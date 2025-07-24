#!/usr/bin/env python3
import os
import sys
import time
import psutil
import random
import math
from collections import deque
import termios
import tty
import select
import signal
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import threading
from io import StringIO

class UnifiedVisualizer:
    def __init__(self):
        self.running = True
        self.width, self.height = self.get_terminal_size()
        self.time = 0
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.target_fps = 30
        self.frame_time = 1.0 / self.target_fps
        
        # Performance optimization: use a screen buffer
        self.screen_buffer = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        self.color_buffer = [['' for _ in range(self.width)] for _ in range(self.height)]
        
        self.cpu_percent = 0
        self.mem_percent = 0
        self.network_activity = 0
        self.disk_activity = 0
        self.bytes_sent = 0
        self.bytes_recv = 0
        self.last_bytes_sent = 0
        self.last_bytes_recv = 0
        self.send_rate = 0
        self.recv_rate = 0
        
        self.smoothed_cpu = 0
        self.smoothed_mem = 0
        self.smoothed_send = 0
        self.smoothed_recv = 0
        self.smoothing_factor = 0.15
        
        # Use fixed-size arrays for better performance
        self.cpu_history = deque(maxlen=100)
        self.mem_history = deque(maxlen=100)
        self.network_history = deque(maxlen=100)
        
        # Claude CLI metrics
        self.claude_data_paths = [
            Path.home() / '.claude' / 'projects',
            Path.home() / '.config' / 'claude' / 'projects'
        ]
        self.claude_metrics = {
            'total_cost': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_cache_creation_tokens': 0,
            'total_cache_read_tokens': 0,
            'session_count': 0,
            'active_sessions': 0,
            'tokens_per_minute': 0,
            'cost_per_hour': 0,
            'last_update': 0
        }
        self.previous_metrics = self.claude_metrics.copy()
        self.recent_activity = 0
        self.smoothed_claude_cost = 0
        self.smoothed_claude_activity = 0
        self.claude_particles = []
        self.token_bursts = []
        self.processed_messages = set()
        self.last_update_time = time.time()
        self.last_metrics_time = time.time()
        self.session_data = {}
        self.first_load_complete = False
        self.startup_metrics = None
        
        # Reduce update frequency for Claude metrics
        self.claude_update_interval = 2.0  # Update every 2 seconds instead of every frame
        self.last_claude_update = 0
        
        # Threading for background updates
        self.metrics_lock = threading.Lock()
        self.metrics_thread = None
        self.metrics_thread_running = True
        
        self.structure_points = []
        self.connections = []
        self.energy_waves = []
        self.flow_particles = []
        
        # Pre-calculate color codes for efficiency
        self.colors = {
            'reset': '\033[0m',
            'dim': '\033[2m',
            'bold': '\033[1m',
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'magenta': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'orange': '\033[38;5;208m',
        }
        
        # Pre-calculate energy color mapping
        self.energy_colors = [
            (0.3, self.colors['blue']),
            (0.6, self.colors['cyan']),
            (0.8, self.colors['yellow']),
            (1.0, self.colors['magenta'])
        ]
        
        self.init_structure()
        signal.signal(signal.SIGWINCH, self.handle_resize)
        signal.signal(signal.SIGINT, self.handle_exit)
        
        # Pre-load Claude metrics
        self.update_claude_metrics(show_progress=False)
        self.startup_metrics = self.claude_metrics.copy()
        
        # Start metrics thread
        self.start_metrics_thread()
    
    def start_metrics_thread(self):
        """Start background thread for system metrics updates"""
        self.metrics_thread = threading.Thread(target=self.metrics_update_loop, daemon=True)
        self.metrics_thread.start()
    
    def metrics_update_loop(self):
        """Background thread for updating system metrics"""
        while self.metrics_thread_running and self.running:
            try:
                # Update system metrics more frequently
                self.update_system_metrics()
                
                # Update Claude metrics less frequently
                current_time = time.time()
                if current_time - self.last_claude_update > self.claude_update_interval:
                    self.update_claude_metrics()
                    self.last_claude_update = current_time
                
                time.sleep(0.1)  # 10Hz update rate for system metrics
            except Exception:
                pass
    
    def handle_resize(self, signum, frame):
        self.width, self.height = self.get_terminal_size()
        self.screen_buffer = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        self.color_buffer = [['' for _ in range(self.width)] for _ in range(self.height)]
        self.init_structure()
    
    def handle_exit(self, signum, frame):
        self.cleanup()
        sys.exit(0)
    
    def get_terminal_size(self):
        rows, cols = os.popen('stty size', 'r').read().split()
        return int(cols), int(rows)
    
    def init_structure(self):
        center_x = self.width // 2
        center_y = self.height // 2
        
        self.structure_points = []
        num_nodes = 12
        radius = min(self.width, self.height) // 3
        
        for i in range(num_nodes):
            angle = (i / num_nodes) * 2 * math.pi
            x = center_x + int(radius * math.cos(angle))
            y = center_y + int(radius * math.sin(angle) * 0.5)
            
            self.structure_points.append({
                'x': x,
                'y': y,
                'base_x': x,
                'base_y': y,
                'energy': 0.5,
                'phase': random.random() * 2 * math.pi,
                'connections': []
            })
        
        # Pre-calculate connections
        for i, point in enumerate(self.structure_points):
            next_i = (i + 1) % len(self.structure_points)
            opposite_i = (i + len(self.structure_points) // 2) % len(self.structure_points)
            adjacent_i = (i + 2) % len(self.structure_points)
            
            point['connections'] = [next_i, opposite_i, adjacent_i]
    
    def update_system_metrics(self):
        """Update system metrics with smoothing"""
        with self.metrics_lock:
            raw_cpu = psutil.cpu_percent(interval=0.01)  # Reduced interval
            raw_mem = psutil.virtual_memory().percent
            
            self.smoothed_cpu += (raw_cpu - self.smoothed_cpu) * self.smoothing_factor
            self.smoothed_mem += (raw_mem - self.smoothed_mem) * self.smoothing_factor
            
            self.cpu_percent = self.smoothed_cpu
            self.mem_percent = self.smoothed_mem
            
            try:
                net_io = psutil.net_io_counters()
                current_sent = net_io.bytes_sent
                current_recv = net_io.bytes_recv
                
                if self.last_bytes_sent > 0:
                    raw_send = (current_sent - self.last_bytes_sent) / 1024.0
                    raw_recv = (current_recv - self.last_bytes_recv) / 1024.0
                    
                    self.smoothed_send += (raw_send - self.smoothed_send) * self.smoothing_factor
                    self.smoothed_recv += (raw_recv - self.smoothed_recv) * self.smoothing_factor
                    
                    self.send_rate = self.smoothed_send
                    self.recv_rate = self.smoothed_recv
                
                self.last_bytes_sent = current_sent
                self.last_bytes_recv = current_recv
                self.network_activity = (self.send_rate + self.recv_rate) / 1000.0
                
            except Exception:
                self.send_rate = random.random() * 100
                self.recv_rate = random.random() * 100
                self.network_activity = random.random() * 0.2
            
            self.cpu_history.append(self.cpu_percent)
            self.mem_history.append(self.mem_percent)
            self.network_history.append(self.network_activity)
    
    def update_claude_metrics(self, show_progress=False):
        """Update Claude metrics - optimized version"""
        try:
            with self.metrics_lock:
                # Store previous values
                self.previous_metrics = self.claude_metrics.copy()
                
                # Reuse existing totals if already loaded
                if self.first_load_complete:
                    total_cost = self.claude_metrics['total_cost']
                    total_input_tokens = self.claude_metrics['total_input_tokens']
                    total_output_tokens = self.claude_metrics['total_output_tokens']
                    total_cache_creation_tokens = self.claude_metrics['total_cache_creation_tokens']
                    total_cache_read_tokens = self.claude_metrics['total_cache_read_tokens']
                else:
                    total_cost = 0
                    total_input_tokens = 0
                    total_output_tokens = 0
                    total_cache_creation_tokens = 0
                    total_cache_read_tokens = 0
                
                active_sessions = 0
                session_count = 0
                new_messages_count = 0
                
                # Find and read JSONL files
                for base_path in self.claude_data_paths:
                    if base_path.exists():
                        jsonl_files = list(base_path.rglob('*.jsonl'))
                        if show_progress and jsonl_files:
                            print(f"\rLoading Claude data from {len(jsonl_files)} files...", end='', flush=True)
                        
                        for jsonl_file in jsonl_files:
                            try:
                                # Skip if file hasn't been modified recently (optimization)
                                if self.first_load_complete:
                                    file_mtime = jsonl_file.stat().st_mtime
                                    if file_mtime < self.last_update_time.timestamp():
                                        continue
                                
                                with open(jsonl_file, 'r') as f:
                                    for line in f:
                                        if line.strip():
                                            data = json.loads(line)
                                            
                                            # Extract unique ID
                                            message_id = data.get('message', {}).get('id', '') or data.get('message_id', '')
                                            request_id = data.get('requestId', '') or data.get('request_id', '')
                                            uuid = data.get('uuid', '')
                                            unique_id = f"{message_id}:{request_id}:{uuid}"
                                            
                                            # Process assistant messages
                                            if data.get('type') == 'assistant':
                                                is_new_message = unique_id not in self.processed_messages
                                                
                                                if is_new_message or not self.first_load_complete:
                                                    if is_new_message:
                                                        self.processed_messages.add(unique_id)
                                                        new_messages_count += 1
                                                    
                                                    # Extract usage data
                                                    usage = data.get('usage', {}) or data.get('message', {}).get('usage', {}) or {}
                                                    
                                                    input_tokens = usage.get('input_tokens', 0) or usage.get('inputTokens', 0) or 0
                                                    output_tokens = usage.get('output_tokens', 0) or usage.get('outputTokens', 0) or 0
                                                    cache_creation = usage.get('cache_creation_input_tokens', 0) or 0
                                                    cache_read = usage.get('cache_read_input_tokens', 0) or 0
                                                    
                                                    if input_tokens > 0 or output_tokens > 0:
                                                        # Calculate cost
                                                        model = data.get('model', '') or data.get('message', {}).get('model', '')
                                                        cost = self.calculate_cost(model, input_tokens, output_tokens)
                                                        
                                                        # Accumulate totals
                                                        total_input_tokens += input_tokens
                                                        total_output_tokens += output_tokens
                                                        total_cache_creation_tokens += cache_creation
                                                        total_cache_read_tokens += cache_read
                                                        total_cost += cost
                                                        
                                                        active_sessions = 1
                                            
                            except Exception:
                                continue
                
                self.first_load_complete = True
                
                # Calculate rates
                current_time = time.time()
                time_elapsed = current_time - self.last_metrics_time
                
                delta_cost = total_cost - self.previous_metrics['total_cost']
                delta_tokens = (total_input_tokens + total_output_tokens) - \
                               (self.previous_metrics['total_input_tokens'] + self.previous_metrics['total_output_tokens'])
                
                if time_elapsed > 0 and (delta_tokens > 0 or new_messages_count > 0):
                    tokens_per_minute = (delta_tokens / time_elapsed) * 60
                    cost_per_hour = (delta_cost / time_elapsed) * 3600 if delta_cost > 0 else 0
                    self.last_metrics_time = current_time
                else:
                    tokens_per_minute = self.claude_metrics.get('tokens_per_minute', 0) * 0.95
                    cost_per_hour = self.claude_metrics.get('cost_per_hour', 0) * 0.95
                
                self.claude_metrics = {
                    'total_cost': total_cost,
                    'total_input_tokens': total_input_tokens,
                    'total_output_tokens': total_output_tokens,
                    'total_cache_creation_tokens': total_cache_creation_tokens,
                    'total_cache_read_tokens': total_cache_read_tokens,
                    'session_count': session_count,
                    'active_sessions': active_sessions,
                    'tokens_per_minute': tokens_per_minute,
                    'cost_per_hour': cost_per_hour,
                    'last_update': time.time()
                }
                
                # Update activity
                if delta_cost > 0 or delta_tokens > 0:
                    self.recent_activity = 1.0
                    self.last_update_time = datetime.now()
                    self.smoothed_claude_cost = total_cost
                    self.smoothed_claude_activity = min(1.0, abs(delta_tokens) / 5000.0)
                else:
                    self.recent_activity *= 0.98
                    self.smoothed_claude_activity *= 0.98
                
        except Exception:
            self.recent_activity *= 0.95
        
        if show_progress:
            print("\r" + " " * 60 + "\r", end='', flush=True)
    
    def calculate_cost(self, model, input_tokens, output_tokens):
        """Calculate cost based on model and tokens"""
        if 'opus' in model:
            return (input_tokens * 0.000015) + (output_tokens * 0.000075)
        elif 'sonnet' in model:
            return (input_tokens * 0.000003) + (output_tokens * 0.000015)
        elif 'haiku' in model:
            return (input_tokens * 0.00000025) + (output_tokens * 0.00000125)
        else:
            return (input_tokens * 0.000003) + (output_tokens * 0.000015)
    
    def update_structure(self, dt):
        """Update structure with frame-independent timing"""
        self.time += dt * 10  # Scale time for animation speed
        
        with self.metrics_lock:
            cpu_influence = self.cpu_percent / 100.0
            mem_influence = self.mem_percent / 100.0
            network_influence = min(1.0, self.network_activity)
            claude_influence = self.recent_activity
            cost_influence = min(1.0, self.claude_metrics['tokens_per_minute'] / 500.0)
            activity_level = (cpu_influence + mem_influence + network_influence + claude_influence) / 4.0
        
        # Update structure points with interpolation
        for i, point in enumerate(self.structure_points):
            # Use smoother sine functions with dt-based animation
            wave_offset = math.sin(self.time * 0.8 + point['phase']) * 3 * (1 + network_influence * 0.5)
            cpu_pulse = math.sin(self.time * 2 + i * 0.3) * cpu_influence * 6
            network_pulse = math.cos(self.time * 3 + i * 0.5) * network_influence * 3
            
            # Smooth interpolation for position
            target_x = point['base_x'] + wave_offset + cpu_pulse + network_pulse
            target_y = point['base_y'] + math.cos(self.time * 0.7 + point['phase']) * (mem_influence * 3 + network_influence * 1.5)
            
            # Lerp for smoother movement
            lerp_factor = 0.15
            point['x'] += (target_x - point['x']) * lerp_factor
            point['y'] += (target_y - point['y']) * lerp_factor
            
            # Update energy with smoothing
            target_energy = 0.2 + activity_level * 0.8
            target_energy += math.sin(self.time * 1.5 + i * 0.5) * 0.3
            target_energy = max(0.1, min(1.0, target_energy))
            point['energy'] += (target_energy - point['energy']) * 0.1
        
        # Optimize particle generation
        self.update_particles(activity_level, network_influence, claude_influence, cost_influence, dt)
    
    def update_particles(self, activity_level, network_influence, claude_influence, cost_influence, dt):
        """Update particles with optimized generation"""
        # Energy waves - limit total count
        if len(self.energy_waves) < 10 and random.random() < activity_level * 0.4 * dt * 20:
            source = random.choice(self.structure_points)
            self.energy_waves.append({
                'x': source['x'],
                'y': source['y'],
                'radius': 0,
                'max_radius': 20 + activity_level * 20 + network_influence * 8,
                'intensity': activity_level,
                'color': self.colors['cyan'] if network_influence > 0.5 else self.colors['blue']
            })
        
        # Claude cost waves
        if len(self.energy_waves) < 15 and claude_influence > 0.1 and random.random() < claude_influence * 0.3 * dt * 20:
            center_x = self.width // 2
            center_y = self.height // 2
            self.energy_waves.append({
                'x': center_x,
                'y': center_y,
                'radius': 0,
                'max_radius': 15 + cost_influence * 25,
                'intensity': cost_influence,
                'color': self.colors['orange']
            })
        
        # Update energy waves
        self.energy_waves = [w for w in self.energy_waves if w['radius'] < w['max_radius']]
        for wave in self.energy_waves:
            wave['radius'] += (0.7 + wave['intensity'] * 0.8) * dt * 20
        
        # Flow particles - limit count
        if len(self.flow_particles) < 30 and random.random() < activity_level * 0.6 * dt * 20:
            if len(self.structure_points) > 1:
                source = random.choice(self.structure_points)
                target = random.choice([self.structure_points[i] for i in source['connections']])
                
                color = self.colors['yellow'] if self.send_rate > self.recv_rate else \
                        self.colors['cyan'] if self.recv_rate > self.send_rate else \
                        self.colors['white']
                
                self.flow_particles.append({
                    'x': float(source['x']),
                    'y': float(source['y']),
                    'target_x': target['x'],
                    'target_y': target['y'],
                    'progress': 0.0,
                    'speed': 0.03 + activity_level * 0.05 + network_influence * 0.02,
                    'color': color
                })
        
        # Update flow particles
        self.flow_particles = [p for p in self.flow_particles if p['progress'] < 1.0]
        for particle in self.flow_particles:
            particle['progress'] += particle['speed'] * dt * 20
            particle['x'] += (particle['target_x'] - particle['x']) * particle['speed'] * 1.5 * dt * 20
            particle['y'] += (particle['target_y'] - particle['y']) * particle['speed'] * 1.5 * dt * 20
        
        # Claude particles - limit count
        if len(self.claude_particles) < 20 and claude_influence > 0.05:
            self.update_claude_particles(claude_influence, dt)
        
        # Cache usage indicators
        cache_activity = (self.claude_metrics['total_cache_creation_tokens'] + self.claude_metrics['total_cache_read_tokens']) / 50000.0
        if cache_activity > 0.01 and random.random() < cache_activity * 0.4 * dt * 20:
            # Cache creation - expanding circles
            if self.claude_metrics['total_cache_creation_tokens'] > 0:
                node = random.choice(self.structure_points)
                self.energy_waves.append({
                    'x': node['x'],
                    'y': node['y'],
                    'radius': 0,
                    'max_radius': 10 + cache_activity * 15,
                    'intensity': cache_activity,
                    'color': self.colors['yellow'],
                    'type': 'cache_creation'
                })
            
            # Cache reads - converging circles
            if self.claude_metrics['total_cache_read_tokens'] > 0 and random.random() < 0.5:
                node = random.choice(self.structure_points)
                self.token_bursts.append({
                    'x': node['x'],
                    'y': node['y'],
                    'radius': 20,
                    'min_radius': 2,
                    'color': self.colors['cyan'],
                    'type': 'cache_read'
                })
        
        # Update token bursts (converging circles for cache reads)
        self.token_bursts = [b for b in self.token_bursts if b['radius'] > b['min_radius']]
        for burst in self.token_bursts:
            burst['radius'] -= 0.5 * dt * 20
    
    def update_claude_particles(self, claude_influence, dt):
        """Update Claude-specific particles"""
        center_x = self.width // 2
        center_y = self.height // 2
        
        # Input particles
        if random.random() < claude_influence * 0.15 * dt * 20:
            angle = random.random() * 2 * math.pi
            start_radius = min(self.width, self.height) // 2
            
            self.claude_particles.append({
                'x': center_x + start_radius * math.cos(angle),
                'y': center_y + start_radius * math.sin(angle) * 0.5,
                'target_x': center_x,
                'target_y': center_y,
                'progress': 0.0,
                'speed': 0.02 + claude_influence * 0.03,
                'type': 'input',
                'color': self.colors['green']
            })
        
        # Output particles
        if random.random() < claude_influence * 0.1 * dt * 20:
            angle = random.random() * 2 * math.pi
            end_radius = min(self.width, self.height) // 2.5
            
            self.claude_particles.append({
                'x': center_x,
                'y': center_y,
                'target_x': center_x + end_radius * math.cos(angle),
                'target_y': center_y + end_radius * math.sin(angle) * 0.5,
                'progress': 0.0,
                'speed': 0.015 + claude_influence * 0.02,
                'type': 'output',
                'color': self.colors['magenta']
            })
        
        # Active session indicators - pulsing particles around active nodes
        if self.claude_metrics['active_sessions'] > 0 and random.random() < 0.3 * dt * 20:
            # Create particles around active structure points
            for i in range(min(self.claude_metrics['active_sessions'], 3)):
                if self.structure_points:
                    node = self.structure_points[i % len(self.structure_points)]
                    for _ in range(2):
                        angle = random.random() * 2 * math.pi
                        distance = random.randint(5, 15)
                        self.claude_particles.append({
                            'x': node['x'] + distance * math.cos(angle),
                            'y': node['y'] + distance * math.sin(angle) * 0.5,
                            'target_x': node['x'],
                            'target_y': node['y'],
                            'progress': 0.0,
                            'speed': 0.02,
                            'type': 'session_active',
                            'color': self.colors['yellow']
                        })
        
        # Update Claude particles
        self.claude_particles = [p for p in self.claude_particles if p['progress'] < 1.0]
        for particle in self.claude_particles:
            particle['progress'] += particle['speed'] * dt * 20
            particle['x'] += (particle['target_x'] - particle['x']) * particle['speed'] * 2 * dt * 20
            particle['y'] += (particle['target_y'] - particle['y']) * particle['speed'] * 2 * dt * 20
    
    def clear_screen(self):
        print('\033[2J\033[H', end='')
    
    def get_energy_color(self, energy):
        """Optimized energy color lookup"""
        for threshold, color in self.energy_colors:
            if energy < threshold:
                return color
        return self.colors['magenta']
    
    def draw_to_buffer(self, x, y, char, color=''):
        """Draw character to buffer instead of directly to screen"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.screen_buffer[int(y)][int(x)] = char
            self.color_buffer[int(y)][int(x)] = color
    
    def draw_connection_buffered(self, x1, y1, x2, y2, intensity):
        """Draw connection to buffer"""
        steps = int(math.sqrt((x2-x1)**2 + (y2-y1)**2) / 2)
        if steps == 0:
            return
        
        for i in range(steps):
            t = i / float(steps)
            x = x1 + (x2 - x1) * t
            y = y1 + (y2 - y1) * t
            
            if 0 <= x < self.width and 0 <= y < self.height:
                wave = math.sin(self.time * 2 + i * 0.2) * 0.5 + 0.5
                if wave * intensity > 0.25:
                    if intensity > 0.8:
                        char = '═' if abs(x2-x1) > abs(y2-y1) else '║'
                    elif intensity > 0.5:
                        char = '─' if abs(x2-x1) > abs(y2-y1) else '│'
                    else:
                        char = '·'
                    
                    if self.network_activity > 0.7:
                        color = self.colors['cyan']
                    elif intensity < 0.5:
                        color = self.colors['dim']
                    else:
                        color = ''
                    
                    self.draw_to_buffer(x, y, char, color)
    
    def draw_structure_buffered(self):
        """Draw entire structure to buffer"""
        # Clear buffers
        for y in range(self.height):
            for x in range(self.width):
                self.screen_buffer[y][x] = ' '
                self.color_buffer[y][x] = ''
        
        # Draw connections
        for point in self.structure_points:
            for conn_idx in point['connections']:
                conn_point = self.structure_points[conn_idx]
                intensity = (point['energy'] + conn_point['energy']) / 2
                self.draw_connection_buffered(
                    point['x'], point['y'],
                    conn_point['x'], conn_point['y'],
                    intensity
                )
        
        # Draw energy waves
        for wave in self.energy_waves:
            radius = int(wave['radius'])
            for angle in range(0, 360, 15):
                rad = math.radians(angle)
                x = wave['x'] + radius * math.cos(rad)
                y = wave['y'] + radius * math.sin(rad) * 0.5
                
                opacity = 1 - (wave['radius'] / wave['max_radius'])
                if opacity > 0.3:
                    char = '◦' if opacity < 0.6 else '○'
                    color = wave.get('color', self.colors['cyan'])
                    self.draw_to_buffer(x, y, char, color)
        
        # Draw particles
        for particle in self.flow_particles:
            char = '◈' if particle.get('color') == self.colors['yellow'] else '◊'
            self.draw_to_buffer(particle['x'], particle['y'], char, particle.get('color', self.colors['white']))
        
        # Draw Claude particles
        for particle in self.claude_particles:
            if particle['type'] == 'input':
                char = '▸'
            elif particle['type'] == 'output':
                char = '▪'
            elif particle['type'] == 'session_active':
                char = '★'
            else:
                char = '•'
            self.draw_to_buffer(particle['x'], particle['y'], char, particle['color'])
        
        # Draw token bursts (cache reads)
        for burst in self.token_bursts:
            radius = int(burst['radius'])
            if radius > 0:
                for angle in range(0, 360, 30):
                    rad = math.radians(angle)
                    x = burst['x'] + radius * math.cos(rad)
                    y = burst['y'] + radius * math.sin(rad) * 0.5
                    
                    opacity = (burst['radius'] - burst['min_radius']) / 18.0
                    if opacity > 0.3:
                        char = '·' if opacity < 0.6 else '•'
                        self.draw_to_buffer(x, y, char, burst['color'])
        
        # Draw structure points
        for point in self.structure_points:
            color = self.get_energy_color(point['energy'])
            if self.network_activity > 0.7:
                size = '◉' if point['energy'] > 0.6 else '●' if point['energy'] > 0.3 else '○'
            else:
                size = '◉' if point['energy'] > 0.7 else '●' if point['energy'] > 0.4 else '○'
            self.draw_to_buffer(point['x'], point['y'], size, color + self.colors['bold'])
    
    def render_buffer(self):
        """Render buffer to screen in one pass"""
        output = StringIO()
        
        # Draw the buffer
        for y in range(self.height):
            output.write(f'\033[{y+1};1H')
            current_color = ''
            for x in range(self.width):
                char = self.screen_buffer[y][x]
                color = self.color_buffer[y][x]
                
                if color != current_color:
                    if current_color:
                        output.write(self.colors['reset'])
                    if color:
                        output.write(color)
                    current_color = color
                
                output.write(char)
            
            if current_color:
                output.write(self.colors['reset'])
        
        # Draw info overlay
        self.draw_info_to_buffer(output)
        
        # Write everything at once
        print(output.getvalue(), end='', flush=True)
    
    def draw_info_to_buffer(self, output):
        """Draw info panel to output buffer"""
        info_y = self.height - 6
        
        with self.metrics_lock:
            # Title
            output.write(f'\033[{info_y};2H')
            output.write(f"{self.colors['dim']}TermFlow{self.colors['reset']}")
            
            # Network rates
            output.write(f'\033[{info_y+1};2H')
            send_color = self.colors['yellow'] if self.send_rate > 100 else self.colors['dim']
            recv_color = self.colors['cyan'] if self.recv_rate > 100 else self.colors['dim']
            output.write(f"{send_color}↑{self.send_rate:5.1f} KB/s{self.colors['reset']}  ")
            output.write(f"{recv_color}↓{self.recv_rate:5.1f} KB/s{self.colors['reset']}")
            
            # System metrics
            output.write(f'\033[{info_y+2};2H')
            output.write(f"{self.colors['dim']}CPU: {self.cpu_percent:4.1f}% | RAM: {self.mem_percent:4.1f}%{self.colors['reset']}")
            
            # Claude metrics
            output.write(f'\033[{info_y+3};2H')
            cost_color = self.colors['orange'] if self.claude_metrics['total_cost'] > 0.01 else self.colors['dim']
            token_color = self.colors['green'] if self.claude_metrics['total_input_tokens'] + self.claude_metrics['total_output_tokens'] > 0 else self.colors['dim']
            rate_color = self.colors['cyan'] if self.claude_metrics['tokens_per_minute'] > 0 else self.colors['dim']
            
            # Calculate session deltas
            if self.startup_metrics:
                session_cost = self.claude_metrics['total_cost'] - self.startup_metrics['total_cost']
                session_tokens = (self.claude_metrics['total_input_tokens'] + self.claude_metrics['total_output_tokens']) - \
                               (self.startup_metrics['total_input_tokens'] + self.startup_metrics['total_output_tokens'])
            else:
                session_cost = 0
                session_tokens = 0
            
            output.write(f"{cost_color}Claude: ${self.claude_metrics['total_cost']:.2f}{self.colors['reset']}")
            if session_cost > 0:
                output.write(f" {self.colors['green']}(+${session_cost:.2f}){self.colors['reset']}")
            output.write(f" | ")
            
            output.write(f"{token_color}{self.claude_metrics['total_input_tokens'] + self.claude_metrics['total_output_tokens']:,} tokens{self.colors['reset']}")
            if session_tokens > 0:
                output.write(f" {self.colors['green']}(+{session_tokens:,}){self.colors['reset']}")
            output.write(f" | ")
            
            output.write(f"{rate_color}{int(self.claude_metrics['tokens_per_minute'])} tok/min{self.colors['reset']}")
            
            # Activity indicator
            if self.recent_activity > 0.1:
                activity_char = '●' if self.recent_activity > 0.5 else '○'
                output.write(f" {self.colors['green']}{activity_char}{self.colors['reset']}")
            
            # Instructions
            output.write(f'\033[{info_y+4};2H')
            output.write(f"{self.colors['dim']}Press 'q' to quit | FPS: {int(1.0/max(0.001, time.time() - self.last_frame_time))}{self.colors['reset']}")
    
    def check_input(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            if key.lower() == 'q':
                self.running = False
                return True
        return False
    
    def setup_terminal(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
    
    def cleanup(self):
        self.metrics_thread_running = False
        if self.metrics_thread:
            self.metrics_thread.join(timeout=1.0)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        self.show_cursor()
        self.clear_screen()
        print('\033[1;1H', end='')
    
    def hide_cursor(self):
        print('\033[?25l', end='')
    
    def show_cursor(self):
        print('\033[?25h', end='')
    
    def run(self):
        self.setup_terminal()
        self.hide_cursor()
        self.clear_screen()
        
        try:
            while self.running:
                current_time = time.time()
                dt = current_time - self.last_frame_time
                
                # Frame rate limiting
                if dt < self.frame_time:
                    time.sleep(self.frame_time - dt)
                    current_time = time.time()
                    dt = current_time - self.last_frame_time
                
                self.last_frame_time = current_time
                
                # Update animations with delta time
                self.update_structure(dt)
                
                # Draw to buffer
                self.draw_structure_buffered()
                
                # Render buffer to screen
                self.render_buffer()
                
                # Check for input
                if self.check_input():
                    break
                
                self.frame_count += 1
        
        finally:
            self.cleanup()

if __name__ == "__main__":
    visualizer = UnifiedVisualizer()
    visualizer.run()