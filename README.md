# Shoemaker

> A CLI that makes thumbnails and proxies from images, RAW files, and videos.

**Version:** 0.1.8 | **License:** MIT | **Node.js:** 20+

## Features

- **Fast thumbnail generation** - Extract embedded camera previews for instant thumbnails
- **RAW file support** - Sony ARW, Canon CR2/CR3, Nikon NEF, Fuji RAF, and more
- **Video thumbnail support** - Extract poster frames, preview images, and timeline strips
- **Video proxy generation** - Create lower-resolution video copies for editing workflows
- **Hardware acceleration** - Auto-detect VideoToolbox (macOS), NVENC, VAAPI, QSV
- **LUT support** - Apply .cube LUT files for color grading during proxy encoding
- **Smart fallback** - Automatically decode RAW when previews are insufficient
- **XMP integration** - Track thumbnail and proxy state in XMP sidecars
- **Multiple presets** - Fast import, high quality, portable (CI/CD friendly)
- **Library + CLI** - Use programmatically or from the command line

## Installation

```bash
npm install -g shoemaker
```

Or use locally in a project:

```bash
npm install shoemaker
```

### Requirements

- Node.js 20+
- libvips (Sharp will try to install automatically)
- **For video support:** FFmpeg and FFprobe

## Quick Start

```bash
# Generate thumbnails for a directory (images, RAW, and video)
shoemaker thumb /photos/import/

# Use high-quality preset (RAW decode)
shoemaker thumb /photos/import/ --preset quality

# Check what's embedded in a RAW file or video
shoemaker info /photos/image.arw
shoemaker info /videos/clip.mp4

# See what needs processing
shoemaker status /photos/import/

# Check system dependencies (including video tools)
shoemaker doctor
```

## CLI Commands

### `shoemaker thumb <path>`

Generate thumbnails for files or directories. Automatically detects images, RAW files, and videos.

```bash
shoemaker thumb /photos/import/           # Process directory
shoemaker thumb /photos/import/ -r        # Process recursively
shoemaker thumb /photos/image.arw         # Process single file
shoemaker thumb /videos/clip.mp4          # Process video file
shoemaker thumb /photos/ --preset quality # Use high-quality preset
shoemaker thumb /photos/ --force          # Regenerate existing
shoemaker thumb /photos/ --resume         # Resume interrupted batch
shoemaker thumb /photos/ --dry-run        # Show what would be done
shoemaker thumb /photos/ --json           # Output as JSON
shoemaker thumb /photos/ -c 8             # Use 8 concurrent workers
shoemaker thumb /photos/ --error-log ./errors.json  # Custom error log path

# Video proxy generation
shoemaker thumb /videos/ --proxy                   # Generate proxies (default: 720p)
shoemaker thumb /videos/ --proxy --proxy-size small,medium,large  # All sizes
shoemaker thumb /videos/ --proxy --lut /path/to/grade.cube  # Apply LUT
```

**Options:**
- `-r, --recursive` - Process subdirectories
- `-p, --preset <name>` - Preset to use (fast, quality, portable)
- `-f, --force` - Regenerate even if thumbnails exist
- `--resume` - Skip files that already have thumbnails (resume interrupted batch)
- `--dry-run` - Show what would be done without writing files
- `-c, --concurrency <n>` - Number of files to process in parallel
- `-q, --quiet` - Suppress progress output
- `--json` - Output results as JSON
- `--error-log <path>` - Write error log to specified JSON file (default: `.shoemaker-errors.json`)
- `--proxy` - Generate video proxy files (lower-res copies for editing)
- `--proxy-size <sizes>` - Proxy sizes to generate: small (540p), medium (720p), large (1080p)
- `--lut <path>` - Apply .cube LUT file to proxies for color grading

### `shoemaker info <file>`

Show embedded preview and thumbnail information. Works with images, RAW files, and videos.

```bash
shoemaker info /photos/image.arw
shoemaker info /videos/clip.mp4
shoemaker info /path/to/file --json
```

**Image/RAW Output:**
```
image.arw
------------------------------------------------------

Embedded Previews:
  JpgFromRaw: 4240x2832 (3.2MB)
  PreviewImage: 1616x1080 (245KB)
  OtherImage: Not present
  ThumbnailImage: 160x120 (8KB)

  Best: JpgFromRaw (4240x2832)
  Needs RAW decode: No

Generated Thumbnails:
  Generated at: 2025-12-23T10:30:00Z
  Method: extracted
  thumb: 300px webp (24KB)
  preview: 1600px webp (164KB)
  ml: 2560px jpeg (410KB)
```

**Video Output:**
```
clip.mp4
------------------------------------------------------

Video Info:
  Duration: 2:34.5 (154.5s)
  Resolution: 1920x1080
  Frame Rate: 29.97 fps
  Codec: h264
  Bitrate: 12.5 Mbps
  Interlaced: No
  HDR: No
  Rotation: None

Audio:
  Codec: aac
  Channels: 2
  Sample Rate: 48000 Hz

Generated Thumbnails:
  Generated at: 2025-12-23T10:30:00Z
  Method: video
  poster: 300px webp (18KB)
  preview: 1600px webp (145KB)
  timeline: 720x90px jpeg (52KB)
```

### `shoemaker clean <path>`

Remove generated thumbnails and clear XMP metadata.

```bash
shoemaker clean /photos/import/           # Clean directory
shoemaker clean /photos/import/ -r        # Clean recursively
shoemaker clean /photos/image.arw         # Clean single file
shoemaker clean /photos/ --dry-run        # Show what would be removed
```

### `shoemaker status <path>`

Show which files need thumbnail generation.

```bash
shoemaker status /photos/import/
shoemaker status /photos/import/ -r
shoemaker status /photos/import/ --json
```

### `shoemaker doctor`

Check system dependencies and capabilities.

```bash
shoemaker doctor
shoemaker doctor --json
```

**Output:**
```
Shoemaker v0.2.0 - System Check

RAW Decoders:
  + embedded           Built-in
  + libraw             WASM (bundled)
  + rawtherapee-cli    v5.10 (/opt/homebrew/bin/rawtherapee-cli)
  + darktable-cli      v4.6.0 (/opt/homebrew/bin/darktable-cli)
  - dcraw              Not found

Video Tools:
  + ffmpeg             v6.0 (/opt/homebrew/bin/ffmpeg)
  + ffprobe            v6.0 (/opt/homebrew/bin/ffprobe)

Metadata Tools:
  + exiftool           Bundled via exiftool-vendored
  + exiv2              v0.28.1 (/opt/homebrew/bin/exiv2)

Image Processing:
  + sharp              v8.15.1 (libvips)
  + Formats: jpeg, png, webp, avif, tiff

Recommended Presets:
  - fast-import    -> embedded + libraw fallback
  - high-quality   -> rawtherapee-cli (best available)
  - portable       -> libraw only (CI/CD safe)

+ All core systems operational.
```

## Presets

### `fast` (default)

Extract embedded camera previews. Fastest option.

- Uses JpgFromRaw, PreviewImage, or OtherImage tags
- Falls back to libraw if no preview available
- Best for: Quick imports, initial triage

### `quality`

Decode RAW files for best quality output.

- Uses RawTherapee or darktable for decoding
- Proper demosaicing and color science
- Best for: Final archives, important images

### `portable`

Uses only bundled dependencies (no system tools).

- Uses libraw WASM for RAW decoding
- No RawTherapee/darktable required
- Best for: CI/CD, Docker, environments without system tools

## Video Thumbnails

When processing video files, Shoemaker generates three outputs:

| Type | Description | Use Case |
|------|-------------|----------|
| **poster** | Single frame at 25% position | Gallery grid thumbnails |
| **preview** | Single frame at 50% position | Lightbox/detail view |
| **timeline** | Horizontal strip of 8 frames | Video scrubbing preview |

### Video Configuration

Video processing can be customized in your config:

```toml
[video]
concurrency = 2          # Parallel video processing (1-8)
posterPosition = 25      # Position for poster frame (0-100%)
previewPosition = 50     # Position for preview frame (0-100%)
timelineFrames = 8       # Number of frames in timeline (4-20)
timelineHeight = 90      # Height of timeline strip in pixels (40-200)
skipBlackFrames = true   # Skip black/blank frames
autoDeinterlace = true   # Auto-detect and deinterlace
autoRotate = true        # Apply rotation metadata
hdrToneMap = true        # Tone map HDR content to SDR
```

### Supported Video Formats

- **Common:** MP4, MOV, AVI, MKV, WebM, WMV, FLV, M4V
- **Professional:** MXF, MTS, M2TS, MPG, MPEG, VOB, DV
- **Camera-specific:** TOD, MOD, 3GP, R3D (RED), BRAW (Blackmagic)

### Smart Frame Extraction

Shoemaker uses intelligent frame selection:

- **Safe zone:** Skips first/last 5% of video to avoid credits/intros
- **Black frame detection:** Automatically skips black/blank frames
- **Deinterlacing:** Detects and deinterlaces interlaced content (yadif)
- **HDR tone mapping:** Converts HDR10/HLG to SDR for compatibility
- **Rotation handling:** Applies rotation metadata automatically

## Video Proxies

Generate lower-resolution video copies for editing workflows. Proxies are essential for editing high-bitrate footage (4K, ProRes, RAW) on less powerful hardware.

### Quick Start

```bash
# Generate 720p proxy (default)
shoemaker thumb /videos/footage.mp4 --proxy

# Generate multiple sizes
shoemaker thumb /videos/ --proxy --proxy-size small,medium,large -r

# Apply a LUT during encoding
shoemaker thumb /videos/ --proxy --lut /path/to/rec709.cube
```

### Proxy Sizes

| Size | Resolution | CRF | Use Case |
|------|------------|-----|----------|
| **small** | 540p | 28 | Offline edit, rough cuts |
| **medium** | 720p | 23 | Standard editing (default) |
| **large** | 1080p | 20 | High-quality preview |

### Hardware Acceleration

Shoemaker auto-detects and uses hardware encoders when available:

| Platform | Encoder | How to Enable |
|----------|---------|---------------|
| macOS | VideoToolbox | Automatic (Apple Silicon/Intel) |
| NVIDIA | NVENC | Install CUDA toolkit |
| AMD/Intel Linux | VAAPI | Install vaapi drivers |
| Intel | QuickSync (QSV) | Install Intel Media SDK |

Falls back to software encoding (libx264) if no hardware encoder is available.

### Proxy Configuration

Configure proxy generation in your config file:

```toml
[video.proxy]
enabled = false              # Enable via CLI with --proxy
codec = "h264"               # h264, h265, or prores
format = "mp4"               # mp4 or mov
preset = "fast"              # ultrafast to slow
audioCodec = "aac"           # aac, copy, or none
audioBitrate = "128k"
hwAccel = "auto"             # auto, none, videotoolbox, nvenc, vaapi, qsv
deinterlace = true           # Auto-deinterlace interlaced content
fastStart = true             # Enable streaming (movflags +faststart)
lutPath = ""                 # Path to .cube LUT file

[video.proxy.sizes.small]
height = 540
crf = 28

[video.proxy.sizes.medium]
height = 720
crf = 23

[video.proxy.sizes.large]
height = 1080
crf = 20
```

### Output Structure

Proxies are stored in a `proxies` subdirectory within the thumbnail folder:

```
/videos/
|-- DJI_0003.MOV
|-- DJI_0003.MOV.xmp           # XMP with proxy metadata
+-- DJI_0003_thumbs/
    |-- DJI_0003_poster_300.webp
    |-- DJI_0003_preview_1600.webp
    |-- DJI_0003_timeline.jpg
    +-- proxies/
        |-- DJI_0003_proxy_540p.mp4   # 1.9 MB (vs 247 MB original)
        |-- DJI_0003_proxy_720p.mp4   # 2.1 MB
        +-- DJI_0003_proxy_1080p.mp4  # 4.2 MB
```

### LUT Support

Apply color grading during proxy encoding with .cube LUT files:

```bash
# Apply Rec.709 conversion for LOG footage
shoemaker thumb /videos/ --proxy --lut /luts/slog3_to_rec709.cube

# Apply creative grade
shoemaker thumb /videos/ --proxy --lut /luts/cinematic.cube
```

The LUT is applied after scaling, so you get color-correct proxies ready for editing.

## Library Usage

Shoemaker is designed to be used as a library in your own applications.

### Simple Usage

```typescript
import { generateThumbnails } from 'shoemaker';

// One-liner: generate thumbnails with default settings
const result = await generateThumbnails('/path/to/image.arw');
console.log(result.thumbnails);
// [{ path: '.../thumb_300.webp', width: 300, ... }, ...]

// Works with videos too
const videoResult = await generateThumbnails('/path/to/video.mp4');
console.log(videoResult.method); // 'video'
```

### Video-Specific Functions

```typescript
import {
  probeVideo,
  isVideoFormat,
  extractPosterFrame,
  extractPreviewFrame,
  generateTimelineStrip,
  findVideoFiles,
} from 'shoemaker';

// Check if file is a video
if (isVideoFormat('/path/to/file.mp4')) {
  // Get video metadata
  const info = await probeVideo('/path/to/file.mp4');
  console.log(info);
  // { duration: 120.5, width: 1920, height: 1080, frameRate: 29.97, codec: 'h264', ... }

  // Extract specific frames
  const posterFrame = await extractPosterFrame('/path/to/file.mp4', config.video);
  const previewFrame = await extractPreviewFrame('/path/to/file.mp4', config.video);

  // Generate timeline strip
  const timelineBuffer = await generateTimelineStrip(
    '/path/to/file.mp4',
    8,    // number of frames
    90,   // height in pixels
    { skipBlackFrames: true, deinterlace: true }
  );
}

// Find all videos in a directory
const videos = await findVideoFiles('/videos/', true); // recursive
```

### Full Control

```typescript
import {
  generateForFile,
  generateForBatch,
  findImageFiles,
  loadConfig,
  loadPreset,
  applyPreset,
} from 'shoemaker';

// Load config and preset
const config = await loadConfig();
const preset = await loadPreset('fast', config);
const finalConfig = applyPreset(config, preset);

// Process single file (image or video)
const result = await generateForFile('/path/to/file', {
  config: finalConfig,
  preset,
  force: true,
  onProgress: (info) => console.log(info),
});

// Process batch (includes images, RAW, and video)
const files = await findImageFiles('/media/', finalConfig, true);
const batchResult = await generateForBatch(files, {
  config: finalConfig,
  preset,
  onProgress: (info) => {
    console.log(`[${info.completed}/${info.total}] ${info.current} (${info.method})`);
  },
});
```

### TypeScript Types

All types are exported for TypeScript users:

```typescript
import type {
  Config,
  Preset,
  VideoInfo,
  VideoConfig,
  GenerationResult,
  BatchResult,
  ProgressInfo,
  PreviewAnalysis,
  ThumbnailResult,
  DecoderType,
  DecodeOptions,
} from 'shoemaker';
```

## Generated Thumbnails

By default, thumbnails are created in a sidecar folder next to the source:

**Image/RAW Files:**
```
/photos/
|-- IMG_1234.ARW
|-- IMG_1234.ARW.xmp           # XMP sidecar (updated by shoemaker)
+-- IMG_1234_thumbs/           # Thumbnail folder
    |-- IMG_1234_thumb_300.webp
    |-- IMG_1234_preview_1600.webp
    +-- IMG_1234_ml_2560.jpg
```

**Video Files:**
```
/videos/
|-- clip.mp4
|-- clip.mp4.xmp               # XMP sidecar (updated by shoemaker)
+-- clip_thumbs/               # Thumbnail folder
    |-- clip_poster_300.webp   # Frame at 25%
    |-- clip_preview_1600.webp # Frame at 50%
    +-- clip_timeline.jpg      # 8-frame strip
```

## Configuration

Create `~/.config/shoemaker/config.toml` for user-wide settings:

```toml
default_preset = "fast"

[output]
location = "sidecar"
sidecar_folder = "{stem}_thumbs"

[processing]
concurrency = 4
min_preview_size = 2560
skip_existing = true

[sizes.thumb]
width = 300
format = "webp"
quality = 80

[sizes.preview]
width = 1600
format = "webp"
quality = 85

[sizes.ml]
width = 2560
format = "jpeg"
quality = 90

[video]
concurrency = 2
posterPosition = 25
previewPosition = 50
timelineFrames = 8
timelineHeight = 90
skipBlackFrames = true
autoDeinterlace = true
autoRotate = true
hdrToneMap = true
```

Or create `.shoemaker.toml` in your project directory for project-specific settings.

## Requirements

- **Node.js 20+**
- **libvips** (Sharp will try to install automatically, but manual install may be needed)
- **FFmpeg + FFprobe** (for video support)
- **Optional:** RawTherapee or darktable for high-quality RAW decoding

### Installing Dependencies

**macOS:**
```bash
brew install vips ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libvips-dev ffmpeg
```

**Windows:**
Sharp includes prebuilt binaries. For FFmpeg, download from https://ffmpeg.org/download.html and add to PATH.

## Error Handling

Shoemaker provides detailed error messages for common issues:

- **Path not found** - The specified file or directory doesn't exist
- **Permission denied** - Cannot read/write to the specified location
- **No embedded preview** - RAW file doesn't have extractable previews
- **File is empty** - Source file has zero bytes
- **Corrupt file** - Cannot decode the image/video data
- **FFprobe not found** - FFmpeg not installed (for video files)
- **Decode failed** - Video frame extraction failed

Use `--json` flag for machine-readable error output.

## Troubleshooting

### "ExifTool process not responding"
Kill hanging ExifTool processes:
```bash
pkill -f exiftool
```

### "ENOMEM" errors
Reduce concurrency to lower memory usage:
```bash
shoemaker thumb /photos/ -c 1
```

### "No embedded preview found"
Try decoding the RAW file instead:
```bash
shoemaker thumb /photos/ --preset quality
```

### "FFprobe not found"
Install FFmpeg (includes FFprobe):
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg
```

### "Failed to extract frame"
Check if the video file is playable:
```bash
ffprobe /path/to/video.mp4
```

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for contribution guidelines and architecture details.

## License

MIT
