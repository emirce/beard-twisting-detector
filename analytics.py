"""
Beard Detector — Stats Viewer
Run this to see your detection history and patterns.
Usage: python stats.py
"""

import json
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

LOG_PATH = Path(__file__).parent / "detections.json"

def load_logs():
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH) as f:
        return json.load(f)

def show_stats():
    logs = load_logs()
    if not logs:
        print("No detections logged yet. Run detector.py first.")
        return

    print("\n" + "─" * 50)
    print("  🧔 BEARD DETECTOR — STATS")
    print("─" * 50)
    print(f"  Total detections: {len(logs)}")

    # By hour
    by_hour = defaultdict(int)
    by_day = defaultdict(int)
    for entry in logs:
        dt = datetime.fromisoformat(entry["timestamp"])
        by_hour[dt.hour] += 1
        by_day[dt.strftime("%Y-%m-%d")] += 1

    print(f"\n  Peak hour: {max(by_hour, key=by_hour.get):02d}:00 ({max(by_hour.values())} times)")
    print(f"\n  By day:")
    for day, count in sorted(by_day.items())[-7:]:
        bar = "█" * count
        print(f"    {day}  {bar} ({count})")

    print(f"\n  By hour of day:")
    for h in sorted(by_hour.keys()):
        bar = "█" * by_hour[h]
        print(f"    {h:02d}:00  {bar} ({by_hour[h]})")

    print("\n  Last 5 detections:")
    for entry in logs[-5:]:
        dt = datetime.fromisoformat(entry["timestamp"])
        print(f"    {dt.strftime('%Y-%m-%d %H:%M:%S')}")

    print("─" * 50 + "\n")

if __name__ == "__main__":
    show_stats()