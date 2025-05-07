<img src="../../assets/meowmotion_logo.png" alt="MeowMotion Logo" width="250"/><br>
# 📦 Installation Guide



Welcome to **MeowMotion**, a Python package for detecting trips and transport modes from GPS data, built with ❤️ and structured using [Poetry](https://python-poetry.org/).

---

## 📌 Prerequisites

- **Python 3.11** is recommended for best compatibility.
- **Poetry** (optional, for source installs):  
  
  ```bash
  pip install poetry
  ```
## 🎉 Install via PyPI (Recommended)
The easiest way to get started is to install directly from PyPI:

```bash
poetry new project_name
cd project_name
poetry add meowmotion
```

## 🚀 Install from Source

If you prefer to run the latest code or contribute:

```bash
git clone https://github.com/faraz-m-awan/meowmotion.git
cd meowmotion
poetry install
```

### ⚠️ Facing Issues with Other Python Versions?
If you're not using Python 3.11 and encounter errors while installing dependencies (especially related to compiled packages or lock file constraints):
1. **Delete the existing lock file:**
```bash
rm poetry.lock
```
2. **Install dependencies using [uv](https://github.com/astral-sh/uv) (a fast alternative to pip):**
```bash
uv pip install -r pyproject.toml
```
This can help resolve compatibility issues with newer or older Python versions.

## 🛠️ Troubleshooting Build Errors (e.g., GDAL, igraph, C extensions)

Some dependencies in the geospatial stack require native libraries. If you hit errors related to GDAL, PROJ, or other C extensions, follow the platform-specific steps below.

### 🪟 For Windows:

1. **Install GDAL** (if not already installed by QGIS/OSGeo4W):
   ```bash
   choco install gdal -y
   refreshenv
   ```
2. Install CMake and build tools:
 - CMake (via [cmake.org](cmake.org) or Chocolatey):
   ```bash
   choco install cmake -y
   ```
 - Microsoft C++ Build Tools:
   - Download from: [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Include C++ build tools in setup

This ensures packages like python-igraph, fiona, or rasterio can compile or find pre-built wheels.

### 🐧 For Linux:
Install system dependencies before installing Python packages:
```bash
   sudo apt update
   sudo apt install build-essential cmake gdal-bin libgdal-dev libxml2-dev libglpk-dev libigraph-dev
```

## ✅ You're Ready!
You're all set to use MeowMotion! 🎉
You can now import MeowMotion and dive into the [Quick Start Guide](https://urbanbigdatacentre.github.io/meowmotion/getting-started/quick-start/) or explore the modules in the meowmotion/ directory. 
Happy analyzing! 🐾
