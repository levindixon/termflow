#!/usr/bin/env python3
"""
TermFlow Game of Life - Conway's Game of Life visualization influenced by system and Claude CLI metrics
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

class GameOfLifeVisualizer:
    def __init__(self):
        self.width = 80
        self.height = 24
        self.running = True
        
        # Game of Life grid
        self.grid_width = self.width - 2  # Leave borders (║ on each side)
        self.grid_height = self.height - 6  # Leave space for metrics
        self.grid = [[False for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        self.next_grid = [[False for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        
        # Game parameters
        self.base_survival = [2, 3]  # Standard Conway rules
        self.base_birth = [3]
        self.current_survival = self.base_survival.copy()
        self.current_birth = self.base_birth.copy()
        
        # Pattern injection system
        self.pattern_cooldown = 0
        self.pattern_injection_rate = 0.1  # Base rate
        
        # Calibration phase
        self.calibrating = True
        self.calibration_start = time.time()
        self.calibration_duration = 20.0  # 20 seconds
        self.calibration_samples = defaultdict(list)
        
        # Scaling factors (will be set after calibration)
        self.metric_scales = {
            'cpu': 1.0,
            'memory': 1.0,
            'network': 1.0,
            'disk': 1.0,
            'claude': 1.0
        }
        
        # Organism definitions - each type has unique characteristics
        self.organisms = {
            'cpu': {
                'color': '\033[96m',  # Cyan
                'patterns': ['blinker', 'pulsar', 'beacon'],  # Oscillators
                'behavior': 'oscillator',  # Periodic patterns
                'spread_rate': 0.3,
                'death_resistance': 0.2
            },
            'memory': {
                'color': '\033[95m',  # Magenta
                'patterns': ['block', 'beehive', 'loaf'],  # Still lifes
                'behavior': 'stable',  # Tends to form stable structures
                'spread_rate': 0.1,
                'death_resistance': 0.4
            },
            'network': {
                'color': '\033[92m',  # Green
                'patterns': ['glider', 'lwss'],  # Spaceships
                'behavior': 'mobile',  # Moving patterns
                'spread_rate': 0.5,
                'death_resistance': 0.1
            },
            'disk': {
                'color': '\033[93m',  # Yellow
                'patterns': ['toad', 'beacon', 'blinker'],  # Slow oscillators
                'behavior': 'slow_oscillator',
                'spread_rate': 0.2,
                'death_resistance': 0.3
            },
            'claude': {
                'color': '\033[38;5;208m',  # Orange
                'patterns': ['random', 'rpentomino', 'glider'],  # Chaotic (glider_gun removed as fallback)
                'behavior': 'chaotic',  # Unpredictable growth
                'spread_rate': 0.4,
                'death_resistance': 0.15
            }
        }
        
        # Metrics
        self.cpu_percent = 0.0
        self.memory_percent = 0.0
        self.network_activity = 0.0
        self.claude_activity = 0.0
        self.disk_activity = 0.0
        
        # Smoothed metrics
        self.smoothed_cpu = 0.0
        self.smoothed_memory = 0.0
        self.smoothed_network = 0.0
        self.smoothed_disk = 0.0
        self.smoothed_claude = 0.0
        self.smoothed_activity = 0.0
        
        # History for smoothing
        self.metric_history = defaultdict(lambda: deque(maxlen=10))
        
        # Colors - Terminal-friendly palette
        self.colors = {
            'reset': '\033[0m',
            'dim': '\033[2m',
            'bold': '\033[1m',
            
            # Cell colors based on metrics (using organism colors)
            'cpu_cell': self.organisms['cpu']['color'],
            'memory_cell': self.organisms['memory']['color'],
            'network_cell': self.organisms['network']['color'],
            'disk_cell': self.organisms['disk']['color'],
            'claude_cell': self.organisms['claude']['color'],
            'mixed_cell': '\033[97m',    # White
            
            # UI colors
            'border': '\033[90m',        # Dark grey
            'text': '\033[37m',          # Light grey
            'highlight': '\033[91m',     # Red
            'accent': '\033[94m',        # Blue
        }
        
        # Cell types based on what spawned them
        self.cell_types = [[None for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        
        # Effects
        self.gliders = []  # Track glider positions
        self.explosions = []  # Track explosion effects
        self.waves = []  # Energy waves
        
        # Generation tracking
        self.generation = 0
        self.cells_born = 0
        self.cells_died = 0
        self.population_history = deque(maxlen=50)
        
        # Background thread for metrics
        self.metrics_lock = threading.Lock()
        self.metrics_thread = threading.Thread(target=self.update_metrics_background, daemon=True)
        
        # Terminal setup
        self.setup_terminal()
        self.old_settings = None
        
        # Initialize with some patterns
        self.initialize_patterns()
        
    def initialize_patterns(self):
        """Initialize the grid with some interesting patterns"""
        # Start with a minimal seed during calibration
        if self.calibrating:
            # Just add a small central pattern
            center_x = self.grid_width // 2
            center_y = self.grid_height // 2
            
            # Add a small oscillator in the center
            self.add_blinker(center_x - 2, center_y, 'mixed')
            return
        
        # After calibration, add patterns based on detected activity
        # Only add patterns for metrics that show activity
        active_metrics = []
        if self.smoothed_cpu > 0.1:
            active_metrics.append('cpu')
        if self.smoothed_memory > 0.1:
            active_metrics.append('memory')
        if self.smoothed_network > 0.1:
            active_metrics.append('network')
        if self.smoothed_disk > 0.1:
            active_metrics.append('disk')
        if self.smoothed_claude > 0.1:
            active_metrics.append('claude')
        
        # Add a few starter patterns for active metrics
        for metric in active_metrics[:3]:  # Limit initial patterns
            organism = self.organisms[metric]
            pattern_choice = random.choice(organism['patterns'])
            
            if pattern_choice == 'glider':
                self.add_glider(
                    random.randint(5, self.grid_width - 10),
                    random.randint(5, self.grid_height - 10),
                    metric
                )
            elif pattern_choice == 'blinker':
                self.add_blinker(
                    random.randint(5, self.grid_width - 5),
                    random.randint(5, self.grid_height - 5),
                    metric
                )
    
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
            # Recalculate grid size
            self.grid_width = self.width - 2
            self.grid_height = self.height - 6
            # Reinitialize grids with new size
            self.grid = [[False for _ in range(self.grid_width)] for _ in range(self.grid_height)]
            self.next_grid = [[False for _ in range(self.grid_width)] for _ in range(self.grid_height)]
            self.cell_types = [[None for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        except:
            pass
    
    def cleanup(self):
        """Restore terminal state"""
        sys.stdout.write('\033[?25h\033[0m\033[2J\033[H')
        sys.stdout.flush()
        
        if self.old_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except:
                pass
    
    def update_metrics_background(self):
        """Background thread to update system metrics"""
        last_net_io = psutil.net_io_counters()
        last_disk_io = psutil.disk_io_counters()
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
                
                # Disk activity
                disk_io = psutil.disk_io_counters()
                disk_read = disk_io.read_bytes - last_disk_io.read_bytes
                disk_write = disk_io.write_bytes - last_disk_io.write_bytes
                last_disk_io = disk_io
                
                # Raw activities
                net_rate = (bytes_sent + bytes_recv) / (1024 * 1024) / max(dt, 0.1)
                disk_rate = (disk_read + disk_write) / (1024 * 1024) / max(dt, 0.1)
                
                # Update Claude metrics
                claude_activity = self.update_claude_metrics()
                
                # During calibration, collect raw samples
                if self.calibrating:
                    self.calibration_samples['cpu'].append(cpu)
                    self.calibration_samples['memory'].append(mem)
                    self.calibration_samples['network'].append(net_rate)
                    self.calibration_samples['disk'].append(disk_rate)
                    self.calibration_samples['claude'].append(claude_activity)
                    
                    # Check if calibration is complete
                    if time.time() - self.calibration_start >= self.calibration_duration:
                        self.complete_calibration()
                
                # Apply scaling factors
                scaled_cpu = min((cpu / 100.0) / self.metric_scales['cpu'], 1.0)
                scaled_mem = min((mem / 100.0) / self.metric_scales['memory'], 1.0)
                scaled_net = min(net_rate / self.metric_scales['network'], 1.0)
                scaled_disk = min(disk_rate / self.metric_scales['disk'], 1.0)
                scaled_claude = min(claude_activity / self.metric_scales['claude'], 1.0) if self.metric_scales['claude'] > 0 else claude_activity
                
                # Store in history for smoothing
                with self.metrics_lock:
                    self.metric_history['cpu'].append(scaled_cpu)
                    self.metric_history['memory'].append(scaled_mem)
                    self.metric_history['network'].append(scaled_net)
                    self.metric_history['disk'].append(scaled_disk)
                    self.metric_history['claude'].append(scaled_claude)
                    
                    # Calculate smoothed values
                    self.cpu_percent = sum(self.metric_history['cpu']) / len(self.metric_history['cpu'])
                    self.memory_percent = sum(self.metric_history['memory']) / len(self.metric_history['memory'])
                    self.network_activity = sum(self.metric_history['network']) / len(self.metric_history['network'])
                    self.disk_activity = sum(self.metric_history['disk']) / len(self.metric_history['disk'])
                    self.claude_activity = sum(self.metric_history['claude']) / len(self.metric_history['claude'])
                    
                    # Smooth individual metrics
                    self.smoothed_cpu = self.smoothed_cpu * 0.9 + self.cpu_percent * 0.1
                    self.smoothed_memory = self.smoothed_memory * 0.9 + self.memory_percent * 0.1
                    self.smoothed_network = self.smoothed_network * 0.9 + self.network_activity * 0.1
                    self.smoothed_claude = self.smoothed_claude * 0.9 + self.claude_activity * 0.1
                    self.smoothed_disk = self.smoothed_disk * 0.9 + self.disk_activity * 0.1 if hasattr(self, 'smoothed_disk') else self.disk_activity
                    
                    # Overall activity level - more balanced
                    self.smoothed_activity = (
                        self.smoothed_cpu * 0.2 +
                        self.smoothed_memory * 0.2 +
                        self.smoothed_network * 0.2 +
                        self.smoothed_disk * 0.2 +
                        self.smoothed_claude * 0.2
                    )
                    
                time.sleep(0.5)
                
            except Exception:
                time.sleep(1)
    
    def complete_calibration(self):
        """Complete calibration and set scaling factors"""
        self.calibrating = False
        
        # Calculate scaling based on 95th percentile to handle spikes
        for metric in ['cpu', 'memory', 'network', 'disk', 'claude']:
            samples = sorted(self.calibration_samples[metric])
            if samples:
                # Use 95th percentile as the scale reference
                idx = int(len(samples) * 0.95)
                percentile_95 = samples[idx] if idx < len(samples) else samples[-1]
                
                # Set scale with minimum thresholds
                if metric == 'cpu' or metric == 'memory':
                    # CPU and memory are already in percentage
                    self.metric_scales[metric] = max(percentile_95 / 100.0, 0.3)  # At least 30%
                elif metric == 'network':
                    # Network in MB/s
                    self.metric_scales[metric] = max(percentile_95, 0.5)  # At least 0.5 MB/s
                elif metric == 'disk':
                    # Disk in MB/s
                    self.metric_scales[metric] = max(percentile_95, 2.0)  # At least 2 MB/s
                elif metric == 'claude':
                    # Claude activity
                    self.metric_scales[metric] = max(percentile_95, 0.1) if percentile_95 > 0 else 1.0
    
    def update_claude_metrics(self):
        """Check Claude CLI activity"""
        activity = 0.0
        
        claude_dirs = [
            Path.home() / '.claude' / 'projects',
            Path.home() / '.config' / 'claude' / 'projects'
        ]
        
        for claude_dir in claude_dirs:
            if not claude_dir.exists():
                continue
                
            try:
                # Look for recent activity
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
                
                activity = max(activity, min(recent_files / 10.0, 1.0))
                
            except:
                pass
                
        return activity
    
    def count_neighbors(self, x, y):
        """Count live neighbors for a cell"""
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                    if self.grid[ny][nx]:
                        count += 1
        return count
    
    def get_dominant_neighbor_type(self, x, y):
        """Get the most common type among neighbors"""
        type_counts = defaultdict(int)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                    if self.grid[ny][nx] and self.cell_types[ny][nx]:
                        type_counts[self.cell_types[ny][nx]] += 1
        
        if type_counts:
            return max(type_counts.items(), key=lambda x: x[1])[0]
        return 'mixed'
    
    def update_game_rules(self):
        """Update game rules based on metrics"""
        # Base rules
        self.current_survival = self.base_survival.copy()
        self.current_birth = self.base_birth.copy()
        
        # Organism-specific rule modifications
        # CPU (oscillators): More flexible survival rules
        if self.smoothed_cpu > 0.5:
            if 1 not in self.current_survival:
                self.current_survival.append(1)  # Can survive with 1 neighbor
        
        # Memory (stable): Harder to kill
        if self.smoothed_memory > 0.5:
            if 4 not in self.current_survival:
                self.current_survival.append(4)  # Can survive with 4 neighbors
        
        # Network (mobile): Easier birth for movement
        if self.smoothed_network > 0.5:
            if 2 not in self.current_birth:
                self.current_birth.insert(0, 2)  # Can be born with 2 neighbors
        
        # Disk (slow oscillators): Moderate changes
        if self.smoothed_disk > 0.5:
            if 4 not in self.current_birth:
                self.current_birth.append(4)
        
        # Claude (chaotic): Random rule modifications
        if self.smoothed_claude > 0.3:
            # Add chaos to the rules
            if random.random() < self.smoothed_claude * 0.3:
                # Randomly modify rules
                if random.random() < 0.5:
                    new_survive = random.randint(0, 8)
                    if new_survive not in self.current_survival:
                        self.current_survival.append(new_survive)
                else:
                    new_birth = random.randint(1, 7)
                    if new_birth not in self.current_birth:
                        self.current_birth.append(new_birth)
        
        # Sort rules for display
        self.current_survival.sort()
        self.current_birth.sort()
    
    def update_grid(self):
        """Update the game grid for one generation"""
        self.cells_born = 0
        self.cells_died = 0
        
        # Update rules based on metrics
        self.update_game_rules()
        
        # Apply Game of Life rules
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                neighbors = self.count_neighbors(x, y)
                
                if self.grid[y][x]:
                    # Cell is alive
                    cell_type = self.cell_types[y][x] or 'mixed'
                    
                    # Check survival with organism-specific death resistance
                    if neighbors in self.current_survival:
                        self.next_grid[y][x] = True
                    else:
                        # Apply death resistance based on cell type
                        if cell_type in self.organisms:
                            death_resistance = self.organisms[cell_type]['death_resistance']
                            if random.random() < death_resistance:
                                # Cell resists death
                                self.next_grid[y][x] = True
                                continue
                        
                        self.next_grid[y][x] = False
                        self.cells_died += 1
                else:
                    # Cell is dead
                    if neighbors in self.current_birth:
                        self.next_grid[y][x] = True
                        self.cells_born += 1
                        # Assign type based on neighbors or current metrics
                        self.cell_types[y][x] = self.get_cell_type_from_metrics()
                    else:
                        self.next_grid[y][x] = False
        
        # Swap grids
        self.grid, self.next_grid = self.next_grid, self.grid
        self.next_grid = [[False for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        
        # Inject patterns based on activity
        self.inject_patterns()
        
        # Update generation counter
        self.generation += 1
        
        # Track population
        population = sum(sum(row) for row in self.grid)
        self.population_history.append(population)
    
    def get_cell_type_from_metrics(self):
        """Determine cell type based on current metrics"""
        # Weight by current metric values
        weights = {
            'cpu': self.cpu_percent,
            'memory': self.memory_percent,
            'network': self.network_activity,
            'disk': self.disk_activity,
            'claude': self.claude_activity
        }
        
        # If one metric is dominant
        max_metric = max(weights.items(), key=lambda x: x[1])
        if max_metric[1] > 0.6:
            return max_metric[0]
        
        # Otherwise return mixed
        return 'mixed'
    
    def inject_patterns(self):
        """Inject new patterns based on system activity"""
        if self.pattern_cooldown > 0:
            self.pattern_cooldown -= 1
            return
        
        # Don't inject patterns during calibration
        if self.calibrating:
            return
        
        # Inject patterns based on metrics with organism-specific behavior
        metrics = [
            ('cpu', self.smoothed_cpu),
            ('memory', self.smoothed_memory),
            ('network', self.smoothed_network),
            ('disk', self.smoothed_disk),
            ('claude', self.smoothed_claude)
        ]
        
        # Sort by activity level
        metrics.sort(key=lambda x: x[1], reverse=True)
        
        # Inject patterns for active metrics
        for metric_type, activity in metrics:
            if activity > 0.3 and random.random() < activity * self.organisms[metric_type]['spread_rate']:
                # Get organism-specific patterns
                patterns = self.organisms[metric_type]['patterns']
                
                # Filter patterns based on available space
                available_patterns = []
                for pattern in patterns:
                    if pattern == 'pulsar' and (self.grid_width < 22 or self.grid_height < 22):
                        continue
                    elif pattern == 'glider_gun' and (self.grid_width < 45 or self.grid_height < 20):
                        continue
                    elif pattern == 'lwss' and (self.grid_width < 15 or self.grid_height < 13):
                        continue
                    else:
                        available_patterns.append(pattern)
                
                if not available_patterns:
                    available_patterns = ['blinker', 'block', 'glider']  # Fallback patterns
                
                pattern_choice = random.choice(available_patterns)
                
                if pattern_choice == 'blinker':
                    self.add_blinker(
                        random.randint(2, self.grid_width - 5),
                        random.randint(2, self.grid_height - 5),
                        metric_type
                    )
                elif pattern_choice == 'block':
                    self.add_block(
                        random.randint(2, self.grid_width - 4),
                        random.randint(2, self.grid_height - 4),
                        metric_type
                    )
                elif pattern_choice == 'glider':
                    self.add_glider(
                        random.randint(5, self.grid_width - 10),
                        random.randint(5, self.grid_height - 10),
                        metric_type
                    )
                elif pattern_choice == 'beehive':
                    self.add_beehive(
                        random.randint(2, self.grid_width - 5),
                        random.randint(2, self.grid_height - 4),
                        metric_type
                    )
                elif pattern_choice == 'loaf':
                    self.add_loaf(
                        random.randint(2, self.grid_width - 6),
                        random.randint(2, self.grid_height - 6),
                        metric_type
                    )
                elif pattern_choice == 'toad':
                    self.add_toad(
                        random.randint(2, self.grid_width - 6),
                        random.randint(2, self.grid_height - 5),
                        metric_type
                    )
                elif pattern_choice == 'beacon':
                    self.add_beacon(
                        random.randint(2, self.grid_width - 6),
                        random.randint(2, self.grid_height - 6),
                        metric_type
                    )
                elif pattern_choice == 'pulsar':
                    # Pulsar needs 13x13 space
                    if self.grid_width >= 22 and self.grid_height >= 22:
                        self.add_pulsar(
                            random.randint(7, max(7, self.grid_width - 15)),
                            random.randint(7, max(7, self.grid_height - 15)),
                            metric_type
                        )
                elif pattern_choice == 'glider_gun':
                    # Glider gun needs ~36x9 space
                    if self.grid_width >= 45 and self.grid_height >= 20:
                        self.add_glider_gun(
                            random.randint(5, self.grid_width - 40),
                            random.randint(5, self.grid_height - 15),
                            metric_type
                        )
                elif pattern_choice == 'lwss':  # Lightweight spaceship
                    self.add_lwss(
                        random.randint(5, self.grid_width - 10),
                        random.randint(5, self.grid_height - 8),
                        metric_type
                    )
                elif pattern_choice == 'rpentomino':
                    self.add_rpentomino(
                        random.randint(5, self.grid_width - 8),
                        random.randint(5, self.grid_height - 8),
                        metric_type
                    )
                elif pattern_choice == 'random':
                    self.add_random_pattern(
                        random.randint(5, self.grid_width - 10),
                        random.randint(5, self.grid_height - 10),
                        metric_type
                    )
                
                # Set cooldown based on activity
                self.pattern_cooldown = int(5 / max(0.1, activity))
                break  # Only inject one pattern per update
    
    def add_glider(self, x, y, cell_type='mixed'):
        """Add a glider pattern"""
        pattern = [
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1]
        ]
        self.add_pattern(x, y, pattern, cell_type)
    
    def add_blinker(self, x, y, cell_type='cpu'):
        """Add a blinker oscillator"""
        pattern = [[1, 1, 1]]
        self.add_pattern(x, y, pattern, cell_type)
    
    def add_block(self, x, y, cell_type='memory'):
        """Add a block still life"""
        pattern = [
            [1, 1],
            [1, 1]
        ]
        self.add_pattern(x, y, pattern, cell_type)
    
    def add_glider_gun(self, x, y, cell_type='claude'):
        """Add a Gosper glider gun (simplified version)"""
        # This is a simplified pattern that produces gliders
        pattern = [
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
            [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
            [1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        ]
        
        # Only add if there's space
        if x + len(pattern[0]) < self.grid_width and y + len(pattern) < self.grid_height:
            self.add_pattern(x, y, pattern, cell_type)
    
    def add_random_pattern(self, x, y, cell_type='mixed'):
        """Add a random pattern"""
        size = random.randint(3, 7)
        density = 0.3 + self.smoothed_activity * 0.4
        pattern = []
        for i in range(size):
            row = []
            for j in range(size):
                row.append(1 if random.random() < density else 0)
            pattern.append(row)
        self.add_pattern(x, y, pattern, cell_type)
    
    def add_pattern(self, x, y, pattern, cell_type):
        """Add a pattern to the grid"""
        for py, row in enumerate(pattern):
            for px, cell in enumerate(row):
                if cell and 0 <= x + px < self.grid_width and 0 <= y + py < self.grid_height:
                    self.grid[y + py][x + px] = True
                    self.cell_types[y + py][x + px] = cell_type
    
    def add_beehive(self, x, y, cell_type='memory'):
        """Add a beehive still life"""
        pattern = [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0]
        ]
        self.add_pattern(x, y, pattern, cell_type)
    
    def add_loaf(self, x, y, cell_type='memory'):
        """Add a loaf still life"""
        pattern = [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 0]
        ]
        self.add_pattern(x, y, pattern, cell_type)
    
    def add_toad(self, x, y, cell_type='disk'):
        """Add a toad oscillator"""
        pattern = [
            [0, 1, 1, 1],
            [1, 1, 1, 0]
        ]
        self.add_pattern(x, y, pattern, cell_type)
    
    def add_beacon(self, x, y, cell_type='disk'):
        """Add a beacon oscillator"""
        pattern = [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1]
        ]
        self.add_pattern(x, y, pattern, cell_type)
    
    def add_pulsar(self, x, y, cell_type='cpu'):
        """Add a pulsar oscillator"""
        pattern = [
            [0,0,1,1,1,0,0,0,1,1,1,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,0,0,0,0,1,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,1,0,0,0,0,1],
            [0,0,1,1,1,0,0,0,1,1,1,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,1,1,1,0,0,0,1,1,1,0,0],
            [1,0,0,0,0,1,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,1,0,0,0,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,1,1,1,0,0,0,1,1,1,0,0]
        ]
        self.add_pattern(x, y, pattern, cell_type)
    
    def add_lwss(self, x, y, cell_type='network'):
        """Add a lightweight spaceship"""
        pattern = [
            [0,1,0,0,1],
            [1,0,0,0,0],
            [1,0,0,0,1],
            [1,1,1,1,0]
        ]
        self.add_pattern(x, y, pattern, cell_type)
    
    def add_rpentomino(self, x, y, cell_type='claude'):
        """Add an R-pentomino (chaotic growth)"""
        pattern = [
            [0, 1, 1],
            [1, 1, 0],
            [0, 1, 0]
        ]
        self.add_pattern(x, y, pattern, cell_type)
    
    def render_frame(self):
        """Render a single frame"""
        # Create screen buffer
        output = '\033[H'  # Home cursor
        
        # Show calibration progress if calibrating
        if self.calibrating:
            elapsed = time.time() - self.calibration_start
            progress = min(elapsed / self.calibration_duration, 1.0)
            bar_width = self.width - 20
            filled = int(progress * bar_width)
            
            output += f"\n\n{self.colors['accent']}Calibrating system metrics...{self.colors['reset']}\n"
            output += f"{self.colors['border']}[{'█' * filled}{'░' * (bar_width - filled)}] {int(progress * 100)}%{self.colors['reset']}\n"
            output += f"\n{self.colors['text']}Collecting baseline measurements for adaptive scaling...{self.colors['reset']}\n"
            output += f"{self.colors['dim']}This will take {int(self.calibration_duration - elapsed)} more seconds{self.colors['reset']}\n"
            
            # Show current readings
            output += f"\n{self.colors['cpu_cell']}CPU: {self.cpu_percent*100:.1f}%{self.colors['reset']}  "
            output += f"{self.colors['memory_cell']}Memory: {self.memory_percent*100:.1f}%{self.colors['reset']}  "
            output += f"{self.colors['network_cell']}Network: {self.network_activity*100:.1f}%{self.colors['reset']}  "
            output += f"{self.colors['disk_cell']}Disk: {self.disk_activity*100:.1f}%{self.colors['reset']}  "
            output += f"{self.colors['claude_cell']}Claude: {self.claude_activity*100:.1f}%{self.colors['reset']}\n"
            
            sys.stdout.write(output)
            sys.stdout.flush()
            return
        
        # Top border
        output += f"{self.colors['border']}╔{'═' * (self.width - 2)}╗{self.colors['reset']}\n"
        
        # Game grid
        for y in range(self.grid_height):
            output += f"{self.colors['border']}║{self.colors['reset']}"
            
            for x in range(self.grid_width):
                if self.grid[y][x]:
                    # Choose character and color based on cell type
                    cell_type = self.cell_types[y][x] or 'mixed'
                    
                    # Choose character based on density
                    neighbors = self.count_neighbors(x, y)
                    if neighbors >= 6:
                        char = '●'
                    elif neighbors >= 4:
                        char = '◉'
                    elif neighbors >= 2:
                        char = '○'
                    else:
                        char = '∘'
                    
                    # Get color for cell type
                    color = self.colors.get(f'{cell_type}_cell', self.colors['mixed_cell'])
                    output += f"{color}{char}{self.colors['reset']}"
                else:
                    # Empty space - show faint grid
                    if (x + y) % 2 == 0:
                        output += f"{self.colors['dim']}·{self.colors['reset']}"
                    else:
                        output += ' '
            
            output += f"{self.colors['border']}║{self.colors['reset']}\n"
        
        # Bottom border
        output += f"{self.colors['border']}╠{'═' * (self.width - 2)}╣{self.colors['reset']}\n"
        
        # Metrics display
        output = self._render_metrics(output)
        
        # Status bar
        population = sum(sum(row) for row in self.grid)
        status = f" Gen: {self.generation} │ Pop: {population} │ Born: {self.cells_born} │ Died: {self.cells_died} "
        
        # Center the status
        padding = (self.width - len(status) - 2) // 2
        output += f"{self.colors['border']}║{self.colors['text']}{' ' * padding}{status}{' ' * (self.width - len(status) - padding - 2)}{self.colors['border']}║{self.colors['reset']}\n"
        
        # Final border
        output += f"{self.colors['border']}╚{'═' * (self.width - 2)}╝{self.colors['reset']}"
        
        sys.stdout.write(output)
        sys.stdout.flush()
    
    def _render_metrics(self, output):
        """Render metrics bars"""
        bar_width = max((self.width - 40) // 5, 5)  # Adjust for 5 metrics
        
        # Create metric bars
        cpu_bar = self._create_bar(self.smoothed_cpu, bar_width, self.colors['cpu_cell'])
        mem_bar = self._create_bar(self.smoothed_memory, bar_width, self.colors['memory_cell'])
        net_bar = self._create_bar(self.smoothed_network, bar_width, self.colors['network_cell'])
        disk_bar = self._create_bar(self.smoothed_disk, bar_width, self.colors['disk_cell'])
        claude_bar = self._create_bar(self.smoothed_claude, bar_width, self.colors['claude_cell'])
        
        # Build metrics line - need to calculate visible length without ANSI codes
        prefix = "CPU:"
        metrics_text = f"{prefix}{cpu_bar} MEM:{mem_bar} NET:{net_bar} DSK:{disk_bar} CLD:{claude_bar}"
        
        # Calculate visible length (bar_width * 5 metrics + labels)
        visible_length = 4 + bar_width + 5 + bar_width + 5 + bar_width + 5 + bar_width + 5 + bar_width
        padding_needed = self.width - visible_length - 2  # -2 for borders
        
        # Construct the line with proper padding
        output += f"{self.colors['border']}║{self.colors['reset']}{metrics_text}{' ' * max(0, padding_needed)}{self.colors['border']}║{self.colors['reset']}\n"
        
        # Rules line
        rules_text = f" Survive:{self.current_survival} Birth:{self.current_birth} Activity:{int(self.smoothed_activity * 100)}% "
        rules_padding = self.width - len(rules_text) - 2  # -2 for borders
        output += f"{self.colors['border']}║{self.colors['text']}{rules_text}{' ' * max(0, rules_padding)}{self.colors['border']}║{self.colors['reset']}\n"
        
        return output
    
    def _create_bar(self, value, width, color):
        """Create a progress bar"""
        filled = int(value * width)
        bar = f"{color}{'█' * filled}{self.colors['dim']}{'░' * (width - filled)}{self.colors['reset']}"
        return bar
    
    def run(self):
        """Main run loop"""
        self.metrics_thread.start()
        
        # Allow metrics to initialize
        time.sleep(0.5)
        
        # Variable frame rate based on activity
        base_frame_time = 2.0  # 0.5 FPS base (10x slower)
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
                    elif key == ' ':
                        # Pause/unpause
                        input()  # Wait for another key
                    elif key == 'r':
                        # Reset grid
                        self.grid = [[False for _ in range(self.grid_width)] for _ in range(self.grid_height)]
                        self.cell_types = [[None for _ in range(self.grid_width)] for _ in range(self.grid_height)]
                        self.initialize_patterns()
                        self.generation = 0
                
                # Adjust frame rate based on activity (less aggressive scaling with slower base)
                frame_time = base_frame_time / (1 + self.smoothed_activity * 0.5)
                
                if current_time - last_frame >= frame_time:
                    self.update_grid()
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
    
    visualizer = GameOfLifeVisualizer()
    visualizer.run()

if __name__ == "__main__":
    main()