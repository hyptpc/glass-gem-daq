# glass-gem-daq

FastAPI web server for capturing oscilloscope data, displaying waveforms and histograms in the browser, and saving triggered captures to disk.

## Run

1. Install dependencies (Python 3.9+)

   ```bash
   cd /home/sks/software/glass-gem
   pip install -r requirements.txt
   ```

2. Configure environment variables

   - Edit `.env` (this project is started with `start.sh`, which automatically sources `.env`).
   - Minimum required:
     - `OSC_IP`
   - Optional:
     - `OUTPUT_DIR`
     - `CAMERA_STREAM_URL` (MJPEG URL; if unset, the Camera section is hidden)

3. Start the server

   ```bash
   ./start.sh
   ```

4. Open the web UI

   - Visit: `http://localhost:8000/`

## Notes

- The server uses `uvicorn --reload` by default (see `start.sh`).