import json
import serial
import time
from datetime import datetime
import subprocess
PORT       = "COM13"
BAUD       = 1000000

GESTURES = [
    "Gigem", "HornsDown", "PointForward", "FingerGun", "Shaka",
    "Fist", "2FingerPoint", "PointLeft", "PointRight", "ThumbDown",
    "ThumbSide", "PalmUp", "PalmDown", "Five", "Four", "Three",
    "Two", "One", "MiddleFinger", "MiddleFingerThumb", "Pinky",
]

POSITIONS      = ["early", "middle", "late"]
TRIALS_PER     = 3
TRIAL_DURATION = 5
REST_DURATION  = 5


def countdown(seconds, message):
    for i in range(seconds, 0, -1):
        print(f"\r  {message} in {i}s...", end='', flush=True)
        time.sleep(1)
    print()


def read_frames(ser, duration):
    frames = []
    end = time.time() + duration
    buf = b''
    while time.time() < end:
        buf += ser.read(ser.in_waiting or 1)
        while b'\n' in buf:
            line, buf = buf.split(b'\n', 1)
            line = line.decode(errors="ignore").strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 12:
                continue
            try:
                frames.append({
                    "ts":   int(parts[0]),
                    "acc":  list(map(float, parts[1:4])),
                    "gyr":  list(map(float, parts[4:7])),
                    "flex": list(map(float, parts[7:12]))
                })
            except:
                continue
    return frames


def main():
    print("=== GESTURE DATA COLLECTOR ===")
    print(f"Gestures:    {len(GESTURES)}")
    print(f"Positions:   {POSITIONS}")
    print(f"Trials each: {TRIALS_PER}")
    total = len(GESTURES) * len(POSITIONS) * TRIALS_PER
    print(f"Total trials: {total}")
    est_min = total * (TRIAL_DURATION + REST_DURATION) / 60
    print(f"Est. time:   ~{est_min:.0f} minutes")
    print()

    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    print(f"Connected to {PORT} — waiting for Teensy to boot...")
    time.sleep(2)
    ser.reset_input_buffer()
    print("Ready.\n")

    fname   = f"raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    records = []
    total_trials = 0
    session_start = time.time()

    try:
        for gesture in GESTURES:
            for position in POSITIONS:
                for trial in range(1, TRIALS_PER + 1):

                    print(f"{'─'*50}")
                    print(f"  Gesture:  {gesture.upper()}")
                    print(f"  Position: {position}")
                    print(f"  Trial:    {trial}/{TRIALS_PER}")
                    print()

                    if position == "early":
                        print("  START gesture immediately when recording begins")
                        print("  Hold for ~3s then relax")
                    elif position == "middle":
                        print("  WAIT ~3s after recording starts")
                        print("  Then hold gesture for ~3s then relax")
                    elif position == "late":
                        print("  WAIT ~7s after recording starts")
                        print("  Then hold gesture for ~3s")

                    print()
                    input("  Press Enter when ready...")
                    print()

                    countdown(REST_DURATION, "Recording starts")
                    ser.reset_input_buffer()
                    print(f"  RECORDING — {TRIAL_DURATION} seconds")
                    frames = read_frames(ser, TRIAL_DURATION)
                    print(f"  Captured {len(frames)} frames")

                    records.append({
                        'label':    gesture,
                        'position': position,
                        'trial':    trial,
                        'frames':   frames,
                    })

                    total_trials += 1
                    remaining = total - total_trials
                    est_left  = remaining * (TRIAL_DURATION + REST_DURATION) / 60
                    elapsed   = (time.time() - session_start) / 60
                    print(f"  Saved. [{total_trials}/{total} trials | "
                          f"{elapsed:.0f}min elapsed | ~{est_left:.0f}min left]")
                    print()

    except KeyboardInterrupt:
        print("\n[Collector] Interrupted — saving progress...")

    finally:
        with open(fname, 'w') as f:
            json.dump(records, f)
        print(f"Saved {total_trials} trials to {fname}")
    import subprocess

    fname = f"raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # save recording
    with open(fname, 'w') as f:
        json.dump(records, f)

    print(f"Saved {total_trials} trials to {fname}")

    # preprocess
    preprocessed = fname.replace("raw_", "pre_")
    subprocess.run(['py', 'quicktest', fname, preprocessed])





if __name__ == "__main__":
    main()