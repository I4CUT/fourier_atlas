"""Звук по спектру контура. Не зависит от Qt."""
from .synth import (AudioParams, AudioError, Cycle, synth_cycle, max_harmonic, audible_chain,
                    loop_buffer, timbre_buffer, buildup_schedule, buildup_buffer, note_name,
                    to_int16, write_wav)
from .melody import (MelodyParams, Voice, SCALES, QUANTIZE, select_voices, event_times, render_melody,
                     render_loop, loop_melody, event_strength, write_midi, midi_for_degree, midi_to_freq)

__all__ = [n for n in dir() if not n.startswith("_")]
