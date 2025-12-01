#!/usr/bin/env python3
import argparse, csv, json, re

FIELDS = [
    "t_console","stat","line","posx","posy","posz",
    "mpox","mpoy","mpoz","vel","feed","unit","dist",
    "plane","coor","momo","raw_json"
]

TIME_RE = re.compile(r'^[\+\-\~]\s*(\d{2}:\d{2}:\d{2}\.\d{3})\s+(.*)$')

def extract_sr(obj):
    if isinstance(obj, dict):
        if "sr" in obj and isinstance(obj["sr"], dict):
            return obj["sr"]
        r = obj.get("r")
        if isinstance(r, dict) and "sr" in r and isinstance(r["sr"], dict):
            return r["sr"]
    return None

def main():
    ap = argparse.ArgumentParser(
        description="Extract TinyG/g2core SR objects from Bantam console log"
    )
    ap.add_argument("logfile")
    ap.add_argument("--out_csv", default="sr_clean.csv")
    args = ap.parse_args()

    with open(args.out_csv, "w", newline="") as f_out:
        w_csv = csv.DictWriter(f_out, fieldnames=FIELDS)
        w_csv.writeheader()

        with open(args.logfile, "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                tstamp = ""
                payload = line

                m = TIME_RE.match(line)
                if m:
                    tstamp, payload = m.group(1), m.group(2)

                i = payload.find("{")
                if i < 0:
                    continue

                payload = payload[i:]
                raw_str = payload.strip()  # raw JSON text from the log line

                try:
                    obj = json.loads(payload)
                except Exception:
                    continue

                sr = extract_sr(obj)
                if not sr:
                    continue

                row = {k: "" for k in FIELDS}
                row["t_console"] = tstamp
                for k, v in sr.items():
                    if k in row:
                        row[k] = v

                row["raw_json"] = raw_str
                w_csv.writerow(row)

if __name__ == "__main__":
    main()
