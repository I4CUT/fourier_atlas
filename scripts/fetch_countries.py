"""Скачивает контуры стран Natural Earth (общественное достояние) в data/countries.

    python scripts/fetch_countries.py            # 110m (лёгкий, ~0.8 МБ)
    python scripts/fetch_countries.py --scale 50m  # подробнее (~4 МБ)

Файл проверяется (валидный GeoJSON, > 100 объектов) перед сохранением.
Если автоматическая загрузка недоступна, скачайте ne_<scale>_admin_0_countries.geojson
вручную с https://www.naturalearthdata.com и положите в src/fourier_atlas/data/countries/
(или укажите путь в переменной окружения FOURIER_ATLAS_COUNTRIES).
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

URLS = ["https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_{s}_admin_0_countries.geojson"]
DEST = Path(__file__).resolve().parents[1] / "src" / "fourier_atlas" / "data" / "countries"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=["110m", "50m"], default="110m")
    scale = ap.parse_args().scale
    DEST.mkdir(parents=True, exist_ok=True)
    out = DEST / f"ne_{scale}_admin_0_countries.geojson"
    for tpl in URLS:
        url = tpl.format(s=scale)
        try:
            print("Загрузка:", url)
            data = urllib.request.urlopen(url, timeout=60).read()
            obj = json.loads(data.decode("utf-8"))
            if obj.get("type") != "FeatureCollection" or len(obj.get("features", [])) < 100:
                raise ValueError("неожиданное содержимое файла")
            out.write_bytes(data)
            print(f"Готово: {out} ({len(obj['features'])} объектов)")
            return 0
        except (urllib.error.URLError, ValueError, OSError) as e:
            print("Не удалось:", e, file=sys.stderr)
    print("Скачайте файл вручную (см. docstring этого скрипта).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
