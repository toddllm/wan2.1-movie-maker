# HDMY 5 Movie Generation System

This system helps generate and display videos for "A HDMY 5 Movie: Extended Universe Cut" based on the provided prompts.

## Files Overview

- `hdmy5movie_prompts.txt`: Contains all the prompts for video generation, one per line.
- `hdmy5movie.html`: The HTML page that displays the generated videos.
- `generate_hdmy5movie.py`: Python script that generates videos from the prompts.
- `hdmy5movie_updater.js`: Node.js script that updates the HTML page with new videos.
- `start_hdmy5movie_generation.sh`: Shell script to start both the generation and updater processes.

## Prerequisites

- Python 3.6 or higher
- Node.js (for the HTML updater)
- A video generation model (e.g., Wan2.1)

## Setup

1. Ensure all the files are in the `/home/tdeshane/movie_maker/` directory.
2. Make sure the shell script is executable:
   ```
   chmod +x start_hdmy5movie_generation.sh
   ```
3. Edit the `generate_hdmy5movie.py` file to point to your actual video generation model.
4. Ensure the `hdmy5movie_prompts.txt` file contains all the prompts, one per line.

## Usage

### Starting the Generation Process

Run the shell script to start both the video generation and HTML updater:

```bash
./start_hdmy5movie_generation.sh
```

This will:
- Create necessary directories
- Start the HTML updater in the background
- Start the video generation process in the background
- Save log files to the `logs` directory

### Monitoring Progress

You can monitor the progress of the generation process:

```bash
tail -f logs/generation.log
```

And monitor the HTML updater:

```bash
tail -f logs/updater.log
```

### Viewing the Results

Open the `hdmy5movie.html` file in a web browser to view the generated videos. The page will automatically update as new videos are generated.

### Stopping the Processes

To stop both the generator and updater processes:

```bash
kill $(cat logs/generator.pid) $(cat logs/updater.pid)
```

## Customization

### Modifying the HTML Template

You can edit the `hdmy5movie.html` file to change the appearance of the video gallery. The updater script will preserve your changes while adding the new videos.

### Changing the Output Directory

If you want to change where the videos are saved, edit the `OUTPUT_DIR` variable in the shell script and update the corresponding paths in the Python and JavaScript files.

### Adjusting the Update Interval

The HTML updater checks for new videos every minute by default. You can change this by editing the `updateInterval` value in the `hdmy5movie_updater.js` file.

## Troubleshooting

### Videos Not Appearing in the HTML Page

- Check that the video files are being generated correctly in the output directory.
- Ensure the HTML updater is running (check the logs).
- Verify that the file paths in the JavaScript updater match your actual directory structure.

### Generation Process Failing

- Check the generation logs for error messages.
- Ensure your video generation model is properly configured.
- Verify that the prompts file is formatted correctly (one prompt per line, no empty lines).

## Advanced Usage

### Resuming Generation

If the generation process was interrupted, you can resume from a specific prompt:

```bash
python3 generate_hdmy5movie.py --start 50 --prompts hdmy5movie_prompts.txt
```

This will start generation from the 50th prompt.

### Generating a Subset of Videos

To generate only a specific range of videos:

```bash
python3 generate_hdmy5movie.py --start 100 --end 200 --prompts hdmy5movie_prompts.txt
```

This will generate videos for prompts 100 through 199. 