import cv2
import subprocess
import csv
import time
import boto3
import os
from datetime import datetime
from ultralytics import YOLO

YOUTUBE_URL = "https://www.youtube.com/watch?v=DoUOrTJbIu4"
MODEL_PATH = "yolov8n.pt"
CSV_FILE = "/tmp/dwell_log.csv"
S3_BUCKET = os.environ.get("S3_BUCKET", "mlops-course-ehb-datastore-jr-justus-dev")
S3_KEY = "iot-tracker/dwell_log.csv"

# Hardcoded zone coordinates (Jackson Hole Town Square crosswalk)
ZONE = [(530, 580), (950, 720)]

# Tracks active persons in zone: {track_id: entry_timestamp}
active_in_zone = {}

def in_zone(cx, cy):
    x1, y1 = min(ZONE[0][0], ZONE[1][0]), min(ZONE[0][1], ZONE[1][1])
    x2, y2 = max(ZONE[0][0], ZONE[1][0]), max(ZONE[0][1], ZONE[1][1])
    return x1 <= cx <= x2 and y1 <= cy <= y2

def get_stream_url(youtube_url: str) -> str:
    cmd = [
        "yt-dlp", "--cookies", "cookies.txt",
        "--extractor-args", "youtube:player_client=web",
        "-g", "--format", "best[ext=mp4]/best", youtube_url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp error:\n{result.stderr}")
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("http"):
            return line
    raise RuntimeError("No valid stream URL found.")

def upload_to_s3():
    s3 = boto3.client("s3")
    s3.upload_file(CSV_FILE, S3_BUCKET, S3_KEY)
    print(f"Uploaded to s3://{S3_BUCKET}/{S3_KEY}")

def main():
    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    print("Getting stream URL...")
    stream_url = get_stream_url(YOUTUBE_URL)
    print("Stream found.")

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise RuntimeError("Could not open stream.")

    with open(CSV_FILE, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestamp_entry", "timestamp_exit", "dwell_seconds", "track_id"])

        frame_count = 0
        last_upload = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("No frame received. Stream may have ended.")
                break

            results = model.track(frame, persist=True, verbose=False, classes=[0])
            current_ids_in_zone = set()

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)

                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) // 2
                    cy_ground = y2

                    if in_zone(cx, cy_ground):
                        current_ids_in_zone.add(track_id)
                        if track_id not in active_in_zone:
                            active_in_zone[track_id] = time.time()
                            print(f"Person {track_id} entered zone at {datetime.now().strftime('%H:%M:%S')}")

            # Check for people who left the zone
            exited_ids = set(active_in_zone.keys()) - current_ids_in_zone
            for track_id in exited_ids:
                entry_time = active_in_zone.pop(track_id)
                exit_time = time.time()
                dwell_seconds = round(exit_time - entry_time, 2)
                entry_str = datetime.fromtimestamp(entry_time).strftime('%Y-%m-%d %H:%M:%S')
                exit_str = datetime.fromtimestamp(exit_time).strftime('%Y-%m-%d %H:%M:%S')
                csv_writer.writerow([entry_str, exit_str, dwell_seconds, track_id])
                csv_file.flush()
                print(f"Person {track_id} left zone. Dwell time: {dwell_seconds}s")

            frame_count += 1

            # Upload to S3 every 60 seconds
            if time.time() - last_upload > 60:
                upload_to_s3()
                last_upload = time.time()

    cap.release()
    upload_to_s3()
    print("Done.")

if __name__ == "__main__":
    main()