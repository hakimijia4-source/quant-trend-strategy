from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StrategyConfig:
    path: Path
    root: Path
    raw: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.raw.get(name, {}))

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.raw.get(section, {}).get(key, default)

    @property
    def data_dir(self) -> Path:
        value = self.get("project", "data_dir", "data")
        return (self.root / value).resolve()

    @property
    def reports_dir(self) -> Path:
        value = self.get("project", "reports_dir", "reports")
        return (self.root / value).resolve()

    @property
    def target_symbol(self) -> str:
        return str(self.get("market", "target_symbol", "AAPL")).upper()

    @property
    def context_symbols(self) -> list[str]:
        return [str(x).upper() for x in self.get("market", "context_symbols", [])]

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir / "raw" / "market",
            self.data_dir / "raw" / "events",
            self.data_dir / "processed",
            self.reports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path) -> StrategyConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    root = config_path.parent.parent
    config = StrategyConfig(path=config_path, root=root, raw=raw)
    config.ensure_directories()
    return config
