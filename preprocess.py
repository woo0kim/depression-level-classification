"""Extract F3/F4, middle 300 s, resample to 128 Hz and cache all pre-treatment recordings.

    python preprocess.py --data-dir ../Baseline_pre-RECT
"""
import argparse

from deplevel.data import build_cache, load_cache


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="../Baseline_pre-RECT")
    p.add_argument("--cache", default="cache/f3f4_mid300s_128hz.npz")
    p.add_argument("--pattern", default="*_pre_cleaned.mat")
    args = p.parse_args()

    build_cache(args.data_dir, args.cache, args.pattern)
    c = load_cache(args.cache)
    print(f"saved {args.cache}: signals {c['signals'].shape} @ {int(c['sfreq'])} Hz, channels {list(c['channels'])}")


if __name__ == "__main__":
    main()
