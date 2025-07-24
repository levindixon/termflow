# TermFlow

<p align="center">
  <strong>🌊 A mesmerizing terminal-based real-time visualization of system metrics and Claude CLI activity</strong>
</p>



https://github.com/user-attachments/assets/817c82be-fa9d-47fb-a6b2-3a98bcad49a8



<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#what-it-shows">What It Shows</a> •
  <a href="#requirements">Requirements</a> •
  <a href="#contributing">Contributing</a>
</p>

---

TermFlow transforms your terminal into a dynamic canvas displaying beautiful, real-time visualizations of your system's performance and Claude CLI usage. Watch as particles flow between nodes, waves ripple across your screen, and metrics dance in response to system activity.

## ✨ Features

- **Real-time System Monitoring**: CPU, memory, network, and disk usage visualized with smooth animations
- **Claude CLI Integration**: Tracks and displays your Claude CLI project activity
- **Dynamic Particle System**: Particles flow between structure points based on system activity
- **Wave Animations**: Energy waves radiate from active nodes
- **Performance Optimized**: 30 FPS frame-limited rendering with intelligent buffering
- **Responsive Design**: Adapts to terminal size changes automatically
- **Clean Exit**: Press 'q' to quit or handle Ctrl+C gracefully with proper terminal cleanup

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/termflow.git
cd termflow
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

That's it! TermFlow only requires Python 3.6+ and two lightweight dependencies.

## 🚀 Usage

Simply run:
```bash
python3 termflow.py
```

To exit, press `q` to quit gracefully, or use `Ctrl+C` for immediate termination. TermFlow will clean up and restore your terminal properly in both cases.

### 🌐 Sphere Visualization

For an alternative stunning 3D visualization, try the sphere mode:
```bash
python3 termflow_sphere.py
```

This renders a rotating 3D sphere with:
- Real-time wireframe rendering with depth-based shading
- System metrics mapped to sphere rotation speed and activity
- Dynamic particle effects flowing across the sphere surface
- Smooth 30 FPS animation with terminal-optimized rendering

## 📊 What It Shows

### System Metrics
- **CPU Usage**: Real-time processor utilization across all cores
- **Memory**: Current RAM usage and availability
- **Network**: Upload/download speeds with directional indicators
- **Disk I/O**: Read/write activity visualization

### Claude CLI Metrics
- **Project Activity**: Monitors your Claude CLI projects
- **Token Usage**: Tracks input/output tokens
- **File Interactions**: Shows recent file operations
- **Session History**: Displays your Claude interaction patterns

### Visual Elements
- **Particle Flow**: Dynamic particles that intensify with system activity
- **Wave Propagation**: Energy waves that emanate from active nodes
- **Connection Lines**: Shows relationships between different metrics
- **Color Coding**: Intuitive color schemes for different metric types
  - 🟡 Yellow: Network upload activity
  - 🔵 Cyan: Network download activity
  - 🟢 Green: Normal activity levels
  - 🔴 Red: High activity or warnings

## 💻 Requirements

- **Python**: 3.6 or higher
- **Operating System**: macOS, Linux (Unix-like systems with terminal support)
- **Terminal**: Any modern terminal emulator with ANSI escape code support
- **Dependencies**:
  - `psutil==5.9.8` - System and process monitoring
  - `numpy==1.26.4` - Numerical operations for animations

## 🏗️ Architecture

TermFlow is built with performance and elegance in mind:

- **Unified Visualizer**: Single class orchestrating all visual components
- **Background Threading**: Metrics collection runs separately from rendering
- **Buffer Management**: Intelligent screen buffering minimizes terminal writes
- **ANSI Graphics**: Direct escape code manipulation for maximum compatibility

## 🤝 Contributing

Contributions are welcome! Whether it's bug fixes, new visualizations, or performance improvements, feel free to:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Built with love for the terminal enthusiast community
- Inspired by system monitoring tools like htop and btop
- Special thanks to the Claude CLI team for their amazing tool

---

<p align="center">
  Made with ❤️ by developers who appreciate beautiful terminal applications
</p>
