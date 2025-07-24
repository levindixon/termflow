# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TermFlow is a terminal-based real-time visualization application that displays system metrics and Claude CLI usage with animated graphics. It's a standalone Python application that creates beautiful, dynamic visualizations directly in the terminal.

## Commands

### Running the Application
```bash
python3 termflow.py
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

The only external dependency is `psutil==5.9.8` for system monitoring.

## Architecture

### Core Components

1. **UnifiedVisualizer Class** (`termflow.py`)
   - Main application class that orchestrates all visualization components
   - Manages terminal rendering, animation loops, and metric collection
   - Implements a 30 FPS frame-limited render loop with screen buffering

2. **Key Subsystems**:
   - **System Metrics Collection**: CPU, memory, network, and disk monitoring via psutil
   - **Claude CLI Integration**: Monitors `~/.claude/projects` and `~/.config/claude/projects` for usage metrics
   - **Particle System**: Dynamic particles flow between structure points based on activity
   - **Wave Animation**: Energy waves radiate from active nodes
   - **Terminal Management**: Direct ANSI escape code manipulation for graphics

3. **Performance Optimizations**:
   - Screen buffering to minimize terminal writes
   - Background thread for metric updates (reduces main thread blocking)
   - Smoothing algorithms for metric values
   - Fixed-size deques for history tracking
   - Pre-calculated color codes and positions

### Key Methods to Know

- `run()`: Main application loop at line 1143
- `update_metrics_background()`: Background thread for system/Claude metrics
- `render_frame()`: Core rendering logic that draws all visual elements
- `update_claude_metrics()`: Parses Claude CLI project files for usage data
- `setup_terminal()`/`cleanup()`: Terminal state management

### Data Flow

1. Background thread continuously updates system and Claude metrics
2. Main thread renders frames at 30 FPS using buffered metrics
3. Particle systems and animations update based on metric changes
4. Terminal output is optimized through screen buffering

## Development Notes

- No test framework is currently set up
- No linting configuration exists
- Application exits cleanly on Ctrl+C via signal handling
- Terminal size changes are handled dynamically
- All rendering uses ANSI escape codes - no external UI libraries