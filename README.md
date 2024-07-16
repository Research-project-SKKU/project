## Installation

### Prerequisites

- Python 3.10 or higher
- Poetry

### Install Poetry

```sh
curl -sSL https://install.python-poetry.org | python3 -
brew install poetry
```

### Install dependencies & commit-lint

```sh
git clone https://github.com/Research-project-SKKU/project.git
poetry install
poetry run pre-commit install
```

## Run Project

```sh
poetry run python app/server.py
poetry run python app/server.py --host 0.0.0.0 --port 8000
