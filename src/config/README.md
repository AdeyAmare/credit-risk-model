# Config Module

Centralized project configuration.

- **PROJECT_ROOT**: Root directory of the project.
- **DATA_DIR**: Directory for raw data (default: `data/raw`). Can be overridden via the `DATA_DIR` environment variable or `.env` file.

## Usage

```python
from src.config.config import Config

print(Config.PROJECT_ROOT)
print(Config.DATA_DIR)
```

## Environment Variables

You can override the default `DATA_DIR` by creating a `.env` file in the project root:
`DATA_DIR=/path/to/custom/data`