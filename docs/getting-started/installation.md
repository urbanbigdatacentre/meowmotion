<img src="../../assets/meowmotion_logo.png" alt="MeowMotion Logo" width="250"/><br>
# 📦 Installation Guide



Welcome to **MeowMotion**, a Python package for detecting trips and transport modes from GPS data, built with ❤️ and structured using [Poetry](https://python-poetry.org/).

---

## 🚀 Clone the Repository

First, clone the repository locally:

```bash
git clone https://github.com/faraz-m-awan/meowmotion.git
cd meowmotion
```

## 📌 Prerequisites
Python 3.11 is recommended for best compatibility.

Poetry should be installed. If not, install it via:
```bash
pip install poetry
```

## 📥 Install Dependencies (Preferred Method)
To install the dependencies using Poetry:
```bash
poetry install
```

## ⚠️ Facing Issues with Other Python Versions?
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

## 🛠️ Troubleshooting Build Errors (e.g., igraph, C extensions)
If you encounter installation issues related to igraph or other native extensions, make sure system build tools are installed:

### 🪟 For Windows:

1. Install cmake:
   1. You can install it via [cmake.org](cmake.org) or
   2. Using Chocolatey:
   ```bash
   choco install cmake
   ```
2. Install Microsoft C++ Build Tools:
   - Download and install from: [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - During setup, make sure to include C++ build tools

This step is critical for packages like python-igraph that need native compilation.

## 🐧 For Linux:
Install system dependencies before installing Python packages:
```bash
sudo apt update
sudo apt install build-essential cmake
```
You may also need:

```bash
sudo apt install libxml2-dev libglpk-dev libigraph-dev
```

## ✅ You're Ready!
Now you're all set to use MeowMotion! 🎉
Check out the Usage Guide or explore the modules in the meowmotion/ directory.

## 📦 PyPI Release Coming Soon
MeowMotion will soon be available on PyPI. Once published, you'll be able to install it with:
```bash
pip install meowmotion
```
Stay tuned! 🐾
