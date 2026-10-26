"""Запуск: python scripts/make_audio.py --country Italy --mode melody --out italy.wav (см. fourier_atlas.cli_audio)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fourier_atlas.cli_audio import main  # noqa: E402

sys.exit(main())
