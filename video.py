import os
import cv2
import argparse
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

def parse_hour_from_filename(filename):
    # Example filename format assumed: prefix_YYYYMMDDHHMMSS.jpg
    # Extract hour from position [8:10]
    timestamp_part = filename.split("_")[-1].split(".")[0]
    hour = timestamp_part[8:10]
    return hour

def load_and_resize_image(img_path, frame_size):
    frame = cv2.imread(img_path)
    if frame is not None:
        frame = cv2.resize(frame, frame_size)
    return frame

def generate_video(input_dir, output_file, image_prefixes, start_date, end_date, max_images_per_hour, frame_rate, output_resolution):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_file_name = f"videos/{output_file}-{timestamp}.mp4"

    print("Generating video...")
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file_name}")
    print(f"Image prefixes: {image_prefixes}")
    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")
    print(f"Max images per hour: {max_images_per_hour}")
    print(f"Frame rate: {frame_rate}")
    print(f"Output resolution: {output_resolution}")

    # Parse start/end dates into tuples for easy comparison
    sy, sm, sd = map(int, start_date.split('-'))
    ey, em, ed = map(int, end_date.split('-'))
    start_tuple = (sy, sm, sd)
    end_tuple = (ey, em, ed)

    # Check directory structure is year/month/day
    # We'll prune directories by date before looking at images.
    filtered_images = []
    print("Scanning and filtering images...")
    with tqdm(desc="Scanning files", unit="image") as pbar:
        # Scan years
        for year_entry in os.scandir(input_dir):
            if not year_entry.is_dir():
                continue
            try:
                y = int(year_entry.name)
            except ValueError:
                continue
            # Skip entire year if out of range
            if (y, 12, 31) < start_tuple or (y, 1, 1) > end_tuple:
                # If the entire year is before start date or after end date, skip
                continue

            # Scan months
            for month_entry in os.scandir(year_entry.path):
                if not month_entry.is_dir():
                    continue
                try:
                    m = int(month_entry.name)
                except ValueError:
                    continue
                # Construct tuple for max day of month 31 as upper bound
                # For simplicity, we don't validate month-day correctness strictly (just a broad skip)
                if (y, m, 31) < start_tuple or (y, m, 1) > end_tuple:
                    # Month completely out of range
                    continue

                # Scan days
                for day_entry in os.scandir(month_entry.path):
                    if not day_entry.is_dir():
                        continue
                    try:
                        d = int(day_entry.name)
                    except ValueError:
                        continue
                    date_tuple = (y, m, d)
                    if date_tuple < start_tuple or date_tuple > end_tuple:
                        # Day out of range
                        continue

                    # If we reach here, (y,m,d) is within date range. Check the images
                    for file_entry in os.scandir(day_entry.path):
                        if file_entry.is_file() and file_entry.name.endswith(".jpg"):
                            filename = file_entry.name
                            # Check prefixes
                            if not any(filename.startswith(prefix) for prefix in image_prefixes):
                                continue
                            # This image passes all filters
                            # Store (y,m,d,path) for sorting and processing
                            filtered_images.append((y, m, d, file_entry.path))
                            pbar.update(1)

    # Sort images by date and path
    filtered_images.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    total_images = len(filtered_images)

    if total_images == 0:
        print("No images found matching the specified criteria.")
        return

    width, height = map(int, output_resolution.split('x'))
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_file_name, fourcc, frame_rate, frame_size)

    day_hour_count = {}
    current_day = None
    processed_count = 0

    print("Processing and writing frames... (Using parallel loading)")

    # Parallel load and resize images
    frames = [None] * total_images
    with ThreadPoolExecutor() as executor:
        futures = {}
        for i, (y, m, d, img_path) in enumerate(filtered_images):
            futures[executor.submit(load_and_resize_image, img_path, frame_size)] = i

        with tqdm(total=total_images, desc="Loading images", unit="image") as pbar:
            for future in as_completed(futures):
                i = futures[future]
                frames[i] = future.result()
                pbar.update(1)

    # Write frames to video, respecting max_images_per_hour
    with tqdm(total=total_images, desc="Writing video", unit="image") as pbar:
        for i, (y, m, d, img_path) in enumerate(filtered_images):
            date_str = f"{y:04d}-{m:02d}-{d:02d}"
            if current_day != date_str:
                day_hour_count = {}
                current_day = date_str

            filename = os.path.basename(img_path)
            hour = parse_hour_from_filename(filename)
            if hour not in day_hour_count:
                day_hour_count[hour] = 0
            if day_hour_count[hour] >= max_images_per_hour:
                # Skip this image, reached max per hour
                pbar.update(1)
                continue

            frame = frames[i]
            if frame is None:
                # Failed to load or invalid image
                pbar.update(1)
                continue

            video_writer.write(frame)
            day_hour_count[hour] += 1
            processed_count += 1
            pbar.update(1)

    video_writer.release()

    if processed_count == 0:
        print("No images processed after filtering.")
    else:
        video_duration = processed_count / frame_rate
        print(f"Number of frames: {processed_count}")
        print(f"Frame size: {frame_size}")
        print(f"Video duration: {video_duration:.2f} seconds")
        print(f"Video generated successfully: {output_file_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from JPG files")
    parser.add_argument("input_dir", help="Path to the input directory")
    parser.add_argument("--output_file", default="video", help="Base name for the output video file (no extension)")
    parser.add_argument("--image-prefixes", nargs="+", default=["Garten", "Einfahrt"],
                        help="Prefixes of the image files (default: Garten Einfahrt)")
    parser.add_argument("--start-date", default="2000-01-01", help="Start date in YYYY-MM-DD (default: 2000-01-01)")
    parser.add_argument("--end-date", default="9999-12-31", help="End date in YYYY-MM-DD (default: 9999-12-31)")
    parser.add_argument("--max-images-per-hour", type=int, default=10,
                        help="Maximum number of images per hour (default: 10)")
    parser.add_argument("--frame-rate", type=int, default=30,
                        help="Frame rate of the output video (default: 30)")
    parser.add_argument("--output-resolution", default="1280x720",
                        help="Output video resolution WIDTHxHEIGHT (default: 1280x720)")
    args = parser.parse_args()

    generate_video(
        args.input_dir,
        args.output_file,
        args.image_prefixes,
        args.start_date,
        args.end_date,
        args.max_images_per_hour,
        args.frame_rate,
        args.output_resolution
    )
