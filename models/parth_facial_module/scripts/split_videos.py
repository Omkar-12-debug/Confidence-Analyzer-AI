import os
import subprocess
import sys

# Define tool paths
# User requested C:\ffmpeg\bin\ paths, but we installed locally due to permissions
LOCAL_FFMPEG_PATH = r"c:\Users\parth\sy_aiml\edai4\confidence_ai_project\ffmpeg\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe"
LOCAL_FFPROBE_PATH = r"c:\Users\parth\sy_aiml\edai4\confidence_ai_project\ffmpeg\ffmpeg-8.1-essentials_build\bin\ffprobe.exe"
GLOBAL_FFMPEG_PATH = "ffmpeg"
GLOBAL_FFPROBE_PATH = "ffprobe"

def get_ffmpeg_path():
    """Checks if ffmpeg is in PATH, else uses local absolute path."""
    try:
        subprocess.run([GLOBAL_FFMPEG_PATH, "-version"], capture_output=True, check=True)
        return GLOBAL_FFMPEG_PATH
    except (subprocess.CalledProcessError, FileNotFoundError):
        return LOCAL_FFMPEG_PATH

def get_ffprobe_path():
    """Checks if ffprobe is in PATH, else uses local absolute path."""
    try:
        subprocess.run([GLOBAL_FFPROBE_PATH, "-version"], capture_output=True, check=True)
        return GLOBAL_FFPROBE_PATH
    except (subprocess.CalledProcessError, FileNotFoundError):
        return LOCAL_FFPROBE_PATH

FFMPEG_EXE = get_ffmpeg_path()
FFPROBE_EXE = get_ffprobe_path()

# Directories
INPUT_OUTPUT_DIR = r"c:\Users\parth\sy_aiml\edai4\confidence_ai_project\video_data"

def get_duration(file_path):
    """Returns the duration of a video file in seconds."""
    cmd = [
        FFPROBE_EXE, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Error processing file: {os.path.basename(file_path)} - {e}")
        return None

def split_video(file_path, filename):
    """Splits a video into 10-20 second segments."""
    duration = get_duration(file_path)
    if duration is None:
        return

    if duration <= 15:
        print(f"Skipping short video: {filename} ({duration:.2f}s)")
        return

    print(f"Splitting file: {filename} ({duration:.2f}s)")

    start = 0
    part = 1
    created_parts_count = 0

    while start < duration:
        remaining = duration - start
        
        # Segment logic to maintain 10-20s target
        if remaining <= 20:
            # If current chunk is 10-20, just use it all
            chunk_duration = remaining
        elif remaining < 30:
            # Avoid a <10s remaining clip by splitting in half
            chunk_duration = remaining / 2
        else:
            # Normal target length ≈ 15 sec
            chunk_duration = 15

        output_name = f"{os.path.splitext(filename)[0]}_part{part}.mp4"
        output_path = os.path.join(INPUT_OUTPUT_DIR, output_name)

        # Safety: Do NOT overwrite files
        if os.path.exists(output_path):
            print(f"Part already exists, skipping: {output_name}")
            start += chunk_duration
            part += 1
            continue

        cmd = [
            FFMPEG_EXE,
            "-y",
            "-ss", str(start),
            "-t", str(chunk_duration),
            "-i", file_path,
            "-c", "copy",
            output_path
        ]

        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            created_parts_count += 1
        except Exception as e:
            print(f"Error processing part {part} for {filename}: {e}")
            break

        start += chunk_duration
        part += 1
        
        # End loop if we reached the duration
        if start >= duration - 0.1: # Small epsilon
            break

    if created_parts_count > 0:
        print(f"Created {created_parts_count} parts for {filename}")

def main():
    if not os.path.exists(INPUT_OUTPUT_DIR):
        print(f"Error: Input directory {INPUT_OUTPUT_DIR} does not exist.")
        return

    files = os.listdir(INPUT_OUTPUT_DIR)
    # Filter only .mp4 files and ignore already split files
    mp4_files = sorted([f for f in files if f.lower().endswith(".mp4") and "_part" not in f])

    if not mp4_files:
        print("No .mp4 files found for processing.")
        return

    for filename in mp4_files:
        full_path = os.path.join(INPUT_OUTPUT_DIR, filename)
        split_video(full_path, filename)

if __name__ == "__main__":
    main()
