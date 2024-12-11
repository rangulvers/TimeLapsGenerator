# README

Hey there! This script helps you generate a video from a bunch of images organized by date in a folder structure like `root/year/month/day/`. The idea is that you give it a directory filled with dated images, and it’ll scan through them, pick the ones that fall within a certain date range, optionally filter by filename prefixes, and then compile them into an MP4 video. It’ll even respect a maximum number of images per hour, so you don’t blow up your video with a flood of frames.

We’ve also worked on making it faster. Here’s the scoop on what it does and how to use it:

## What It Does

- **Scans directories by year/month/day:**  
  Instead of trudging through every single file blindly, this script zeroes in on the dates you actually care about. It skips entire years or months if they’re outside the range you give.

- **Filters by prefix and date:**  
  If you only care about images that start with certain words (like “Garten” or “Einfahrt”), you can specify those prefixes. The script will only pick images that match both the prefix and your time window.

- **Respects a start and end date:**  
  You can tell the script: “Only consider images from 2021-05-01 to 2021-05-07,” and it’ll stick to that, ignoring everything else.

- **Picks a limited number of frames per hour:**  
  If you’ve got hours where a camera took tons of images, you can cap how many from that hour appear in the final video. For example, if you say `--max-images-per-hour 10`, you’ll never see more than 10 images from any one hour.

- **Generates an MP4 video:**  
  It takes all the chosen frames, resizes them to your desired resolution, and writes them into a neat MP4 file at the frame rate you specify.

- **Parallel image loading:**  
  To speed things up, it uses multiple threads to load and resize images at the same time, so you’re not waiting forever.

## How To Use It

1. **Organize Your Images:**
   Put your images in a structure like this:
   ```
   root/
     2021/
       05/
         01/
           Garten_20210501093000.jpg
           Einfahrt_20210501094500.jpg
         02/
           Garten_20210502120000.jpg
           ...
   ```
   Make sure your images follow a naming convention that includes the date and time in the filename. The code expects something like `prefix_YYYYMMDDHHMMSS.jpg`.

2. **Run the Script:**
   In your terminal:
   ```
   python generate_video.py path_to_root_directory --start-date 2021-05-01 --end-date 2021-05-07 --image-prefixes Garten Einfahrt --max-images-per-hour 10 --frame-rate 30 --output-resolution 1280x720
   ```

   Replace `path_to_root_directory` with the path where your images live. Adjust the parameters as you like:
   - `--start-date` and `--end-date` define the date range.
   - `--image-prefixes` lists any prefixes you want to include. If your images don’t all start with these prefixes, they’ll be skipped.
   - `--max-images-per-hour` controls how many images per hour end up in the video.
   - `--frame-rate` sets how many frames per second the final video will have.
   - `--output-resolution` sets the width and height of the video (e.g. `1920x1080`).

   The script will create a `videos/` directory (if not there) and drop your final MP4 in there with a timestamped name.

3. **Check The Output:**
   When it finishes, you’ll see something like:
   ```
   Video generated successfully: videos/video-2024-12-11-13-37-00.mp4
   ```
   Just open that file with your favorite video player and you’ll see a time-lapse or collection of images stitched together.

## Tips

- If you have tons of images, consider placing them on a fast SSD. That’ll help speed up reading and processing.
- The directory pruning logic relies on a strict `year/month/day` hierarchy. If your structure is different, you’ll need to tweak the code.
- The prefix and datetime parsing is pretty simple. Make sure your filenames and directory structure follow a predictable pattern.

## Dependencies

- Python 3.x
- OpenCV (`pip install opencv-python`)
- tqdm for the progress bar (`pip install tqdm`)

## That’s It!

Once set up, just run the script, tweak your parameters, and enjoy having a neat video compilation of your images without the hassle of doing it all manually. Have fun!