"""Звук по контуру страны из командной строки (без окна).

    python -m fourier_atlas.cli_audio --country Italy --mode melody --out italy_melody.wav --midi italy.mid
    python -m fourier_atlas.cli_audio --file my_country.geojson --mode timbre --seconds 6 --out timbre.wav

Страна берётся из базы (см. scripts/fetch_countries.py) по имени (поиск по подстроке: Italy, Италия, ITA)
либо из любого поддерживаемого файла (--file: GeoJSON, Shapefile, SVG, CSV).
"""
from __future__ import annotations
import argparse
import sys

from .audio import (AudioError, AudioParams, MelodyParams, SCALES, audible_chain, buildup_buffer,
                    buildup_schedule, render_melody, select_voices, timbre_buffer, write_midi, write_wav)
from .data.catalog import Catalog
from .data.samples import contour_to_spectrum
from .geo import GeoLoadError, PrepareOptions, load_file, prepare_region


def _region(args):
    if args.file:
        regions = load_file(args.file)
        return regions[0] if len(regions) == 1 and not args.country else _pick(regions, args.country, args.file)
    cat = Catalog.from_defaults()
    if cat.warning:
        print(cat.warning, file=sys.stderr)
    if not args.country:
        raise GeoLoadError("Укажите --country или --file")
    hits = cat.search(args.country)
    exact = [h for h in hits if h.casefold() == args.country.casefold()]
    hits = exact or hits
    if not hits:
        raise GeoLoadError(f"Страна «{args.country}» не найдена в базе")
    if len(hits) > 1:
        print(f"Найдено несколько: {', '.join(hits[:8])}; беру «{hits[0]}»", file=sys.stderr)
    return cat.get(hits[0])


def _pick(regions, name, path):
    if name:
        for r in regions:
            if name.casefold() in {r.name.casefold(), r.iso.casefold(), *(a.casefold() for a in r.aliases)}:
                return r
        raise GeoLoadError(f"В файле {path} нет объекта «{name}»")
    raise GeoLoadError(f"В файле {len(regions)} объектов; укажите --country: "
                       + ", ".join(r.name for r in regions[:10]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="WAV/MIDI по контуру страны")
    ap.add_argument("--country")
    ap.add_argument("--file")
    ap.add_argument("--part", type=int, default=1, help="часть контура по площади (1 — самая большая)")
    ap.add_argument("--mode", choices=["timbre", "buildup", "melody"], default="melody")
    ap.add_argument("--out", required=True, help="выходной .wav")
    ap.add_argument("--midi", help="дополнительно сохранить MIDI (только для --mode melody)")
    ap.add_argument("--f0", type=float, default=110.0)
    ap.add_argument("--mono", action="store_true")
    ap.add_argument("--epicycles", type=int, default=None, help="число эпициклов для тембра (по умолчанию все)")
    ap.add_argument("--seconds", type=float, default=8.0, help="длительность (тембр) или всей мелодии")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--step-seconds", type=float, default=0.6)
    ap.add_argument("--voices", type=int, default=8)
    ap.add_argument("--bar-seconds", type=float, default=4.0)
    ap.add_argument("--scale", choices=list(SCALES), default="minor_pentatonic")
    ap.add_argument("--root", type=int, default=57, help="тоника, MIDI-номер (57 = A3)")
    ap.add_argument("--quantize", type=int, default=0, choices=[0, 4, 8, 16, 32])
    a = ap.parse_args(argv)
    try:
        region = _region(a)
        prep = prepare_region(region, PrepareOptions(part=a.part))
        _, spec = contour_to_spectrum(prep.xy)
        p = AudioParams(f0=a.f0, stereo=not a.mono)
        if a.mode == "timbre":
            w = timbre_buffer(spec, p, a.seconds, a.epicycles)
        elif a.mode == "buildup":
            w = buildup_buffer(spec, p, buildup_schedule(len(audible_chain(spec, p)[0]), a.steps), a.step_seconds)
        else:
            mp = MelodyParams(voices=a.voices, bar_seconds=a.bar_seconds, scale=a.scale,
                              root_midi=a.root, quantize=a.quantize)
            bars = max(1, round(a.seconds / a.bar_seconds))
            w = render_melody(spec, mp, p, bars)
            if a.midi:
                write_midi(a.midi, select_voices(spec, mp), mp, bars)
                print("MIDI:", a.midi)
        write_wav(a.out, w, p.sample_rate)
    except (GeoLoadError, AudioError, ValueError) as e:
        print("Ошибка:", e, file=sys.stderr)
        return 1
    print(f"{region.name}: часть {prep.part_rank} из {prep.n_parts}; {a.mode}; {len(w) / p.sample_rate:.1f} с → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
