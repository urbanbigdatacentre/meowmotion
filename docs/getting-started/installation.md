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

## ✅ You're Ready!
Now you're all set to use MeowMotion! 🎉
Check out the Usage Guide or explore the modules in the meowmotion/ directory.

## 📦 PyPI Release Coming Soon
MeowMotion will soon be available on PyPI. Once published, you'll be able to install it with:
```bash
pip install meowmotion
```
Stay tuned! 🐾


