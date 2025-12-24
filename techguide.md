# Shoemaker Tech Guide

Project-specific development details for Shoemaker.

## Commands

```bash
# Development
npm run dev          # Watch mode TypeScript compilation
npm run build        # Build for production
npm run typecheck    # Type check without emitting

# Testing
npm test             # Run unit tests
npm run test:watch   # Watch mode tests
npm run test:integration  # Run integration tests

# Linting
npm run lint         # Check for lint errors
npm run lint:fix     # Auto-fix lint errors

# CLI (after build)
npm start            # Run CLI
node dist/bin/shoemaker.js thumb /path/to/images
node dist/bin/shoemaker.js info /path/to/video.mp4
```

## Project Structure

```
shoemaker/
|-- src/
|   |-- index.ts              # Library entry point (exports)
|   |-- bin/
|   |   +-- shoemaker.ts      # CLI entry point
|   |-- cli/
|   |   |-- index.ts          # CLI setup (Commander.js)
|   |   +-- commands/         # CLI command implementations
|   |       |-- thumb.ts      # Thumbnail generation command
|   |       |-- info.ts       # File info command (images + video)
|   |       |-- status.ts     # Status check command
|   |       |-- clean.ts      # Cleanup command
|   |       +-- doctor.ts     # System check command
|   |-- core/
|   |   |-- config.ts         # Configuration loading (TOML)
|   |   |-- errors.ts         # Error classes and handling
|   |   |-- extractor.ts      # ExifTool preview extraction
|   |   |-- resizer.ts        # Sharp resize + color management
|   |   |-- decoder.ts        # RAW decoding (libraw, rawtherapee, etc.)
|   |   |-- ffprobe.ts        # Video metadata extraction (FFprobe)
|   |   +-- frame-extractor.ts # Video frame extraction (FFmpeg)
|   |-- services/
|   |   |-- thumbnail-generator.ts  # Main pipeline (images + video)
|   |   +-- xmp-updater.ts    # XMP sidecar updates
|   +-- schemas/
|       +-- index.ts          # Zod schemas (Config, VideoInfo, etc.)
|-- presets/                  # TOML preset files
|-- tests/
|   |-- unit/                 # Unit tests
|   |   |-- ffprobe.test.ts   # Video format & schema tests
|   |   +-- ...
|   +-- integration/          # Integration tests
|       |-- video.test.ts     # Video pipeline tests
|       +-- ...
|-- sme/                      # SME documentation
|-- package.json
|-- tsconfig.json
+-- vitest.config.ts
```

## Architecture

### Pipeline Flow

```
Input File
    |
    +-- isVideoFormat? -----> Video Pipeline
    |                              |
    |                              +-- probeVideo()
    |                              |       |
    |                              |       v
    |                              +-- extractPosterFrame()
    |                              +-- extractPreviewFrame()
    |                              +-- generateTimelineStrip()
    |                              |       |
    |                              |       v
    |                              +-- resizeThumbnails()
    |                              |       |
    |                              |       v
    |                              +-- updateXmpSidecar()
    |
    +-- isDecodedFormat? ---> Read directly
    |
    +-- isRawFormat? -------> analyzeEmbeddedPreviews()
                                   |
                                   +-- bestPreview >= minSize? --> extractBestPreview()
                                   |
                                   +-- fallbackToRaw? -----------> decodeRawFile()
                                                                        |
                                                                        v
                                                              resizeThumbnails()
                                                                        |
                                                                        v
                                                              updateXmpSidecar()
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `extractor.ts` | ExifTool wrapper for preview extraction + isVideoFormat() |
| `resizer.ts` | Sharp wrapper for resize + sRGB conversion |
| `config.ts` | TOML config loading and preset management |
| `errors.ts` | Error classes with recovery strategies |
| `decoder.ts` | RAW decoding with multiple backend support |
| `ffprobe.ts` | Video metadata extraction (duration, codec, etc.) |
| `frame-extractor.ts` | FFmpeg frame extraction (poster, preview, timeline) |
| `thumbnail-generator.ts` | Main orchestration service (images + video) |
| `xmp-updater.ts` | XMP sidecar read/write |

### Video Pipeline Details

The video pipeline uses FFmpeg for frame extraction:

1. **Probe**: `ffprobe` extracts video metadata (duration, resolution, codec, etc.)
2. **Filter Chain**: Builds FFmpeg filter chain based on video properties:
   - `yadif` for deinterlacing (if interlaced)
   - `zscale + tonemap` for HDR to SDR conversion
   - Auto-rotation based on metadata
3. **Frame Extraction**: Extracts frames at calculated positions
   - Positions calculated as percentage of safe zone (5%-95%)
   - Black frames detected and skipped automatically
4. **Timeline Strip**: Multiple frames composited horizontally using Sharp

```typescript
// Frame extraction with filter chain
const filterChain = buildFilterChain(videoInfo, options);
// -> 'yadif=mode=0,zscale=t=linear:npl=100,format=gbrpf32le,...'

// Safe zone calculation
const safeStart = duration * 0.05;
const safeEnd = duration * 0.95;
const seekTime = safeStart + (safeRange * percentage);
```

## Gotchas

### ExifTool Cleanup

**CRITICAL**: Always call `shutdownExiftool()` when the app exits, or ExifTool processes will leak:

```typescript
import { shutdownExiftool } from './core/extractor.js';

process.on('SIGINT', async () => {
  await shutdownExiftool();
  process.exit(0);
});
```

### Sharp Memory

Sharp streams images and doesn't load full files into memory, but be careful with:
- Very large files (100MP+)
- High concurrency
- Peak memory ~200-300MB per 50MP image

Reduce concurrency if you hit OOM errors.

### FFmpeg Timeout

FFmpeg operations have a 30-second timeout. For very long videos, frame extraction may need longer:

```typescript
await execFileAsync('ffmpeg', args, {
  timeout: 30000,  // 30 seconds
  maxBuffer: 50 * 1024 * 1024,  // 50MB buffer
});
```

### Video Concurrency

Video processing is CPU-intensive. Default concurrency for videos is 2 (vs 4 for images):

```toml
[video]
concurrency = 2  # Lower than image processing
```

### HEIC on Linux

Sharp's HEIC support requires libheif. Install it before `npm install`:

```bash
apt install libheif-dev
npm rebuild sharp
```

### ExifTool Binary Tags

Binary tags like `JpgFromRaw` are not returned by `exiftool.read()` directly. Use `extractBinaryTag()`:

```typescript
// Wrong - returns metadata only
const tags = await exiftool.read(file);
tags.JpgFromRaw; // Just metadata, not the actual JPEG

// Correct - returns buffer
const buffer = await exiftool.extractBinaryTag('JpgFromRaw', file);
```

### Color Space

All outputs are converted to sRGB with embedded ICC profile. This ensures consistency but may lose wide-gamut colors from Adobe RGB or ProPhoto RGB sources.

### HDR Video

HDR videos (HDR10, HLG) are automatically tone-mapped to SDR for compatibility. The tone mapping uses the Hable algorithm which preserves highlight detail.

### Interlaced Video

Interlaced content is automatically detected via `field_order` metadata and deinterlaced using the `yadif` filter. Field order values `tt`, `bb`, `tb`, `bt` indicate interlacing.

## Testing

### Unit Tests

Test individual functions in isolation:

```typescript
// tests/unit/ffprobe.test.ts
import { isVideoFormat } from '../../src/core/extractor';

describe('video detection', () => {
  it('identifies video formats', () => {
    expect(isVideoFormat('video.mp4')).toBe(true);
    expect(isVideoFormat('image.arw')).toBe(false);
  });
});
```

### Integration Tests

Test with real files (download fixtures first):

```bash
npm run test:download-fixtures
npm run test:integration
```

Video integration tests require FFmpeg and test video fixtures.

### Test Fixtures

Place test files in:
- `tests/fixtures/images/` - JPEG, PNG test images
- `tests/fixtures/raw/` - RAW file samples
- `tests/fixtures/video/` - MP4, MOV test videos

## Dependencies

### Runtime

| Package | Purpose |
|---------|---------|
| sharp | Image processing (libvips) |
| exiftool-vendored | Metadata extraction (bundled ExifTool) |
| commander | CLI framework |
| zod | Schema validation |
| smol-toml | TOML parsing |
| ora | Terminal spinners |
| p-queue | Concurrency control |

### External (System)

| Tool | Purpose | Required For |
|------|---------|--------------|
| ffmpeg | Video frame extraction | Video support |
| ffprobe | Video metadata | Video support |
| rawtherapee-cli | RAW decoding | Quality preset |
| darktable-cli | RAW decoding | Quality preset |

### Development

| Package | Purpose |
|---------|---------|
| typescript | Type checking |
| vitest | Test runner |
| eslint | Linting |
| tsx | TypeScript execution |

## Video Format Support

Shoemaker supports these video formats:

```typescript
export const VIDEO_EXTENSIONS = [
  // Common formats
  'mp4', 'mov', 'avi', 'mkv', 'webm', 'wmv', 'flv', 'm4v',
  // Professional formats
  'mxf', 'mts', 'm2ts', 'mpg', 'mpeg', 'vob', 'dv',
  // Camera-specific formats
  'tod', 'mod', '3gp', 'r3d', 'braw',
] as const;
```

Detection is case-insensitive and based on file extension.

## Troubleshooting

### "ExifTool process not responding"

ExifTool maintains a persistent process. If it hangs:
1. Kill all ExifTool processes: `pkill -f exiftool`
2. Check for file locks on the image
3. Try with a different image to isolate the issue

### "Sharp: Input file is missing"

Sharp requires the input to be a valid file path or buffer. Common causes:
- File was deleted during processing
- Path contains special characters (use absolute paths)
- File is locked by another process

### "ENOMEM" errors

Reduce concurrency:
```bash
shoemaker thumb /photos/ -c 1
```

Or set in config:
```toml
[processing]
concurrency = 2

[video]
concurrency = 1
```

### "No embedded preview found"

The RAW file doesn't have embedded previews, or they're corrupted. Options:
1. Use `--preset quality` to decode the RAW file
2. Check if the file opens in other software
3. Try `exiftool -preview:all image.arw` to see what's embedded

### "FFprobe not found"

Install FFmpeg (includes FFprobe):
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Verify installation
ffprobe -version
```

### "Failed to extract frame"

Common causes:
1. Corrupt video file - try `ffprobe /path/to/video.mp4`
2. Unsupported codec - check if FFmpeg was built with required codecs
3. File permissions - ensure read access
4. Timeout - very long videos may need extended timeout

### "Black frame detection not working"

Black frame detection uses Sharp to analyze frame luminance. If it's not working:
1. Check Sharp is properly installed
2. Adjust threshold in frame-extractor.ts (default: avgLuminance < 10)

---

## XMP Pipeline Integration

shoemaker is the **second stage** in the media processing pipeline, running after wake-n-blake and before visual-buffet.

### Pipeline Order (CRITICAL)

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  wake-n-blake   │ ──► │    shoemaker    │ ──► │  visual-buffet  │
│   (import)      │     │  (thumbnails)   │     │   (ML tags)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

**wake-n-blake MUST run first** to create the base XMP sidecar with provenance data.

### Current XMP Implementation

shoemaker writes thumbnail metadata to XMP sidecars using:
- `XMP-dc:Source` - JSON payload prefixed with `shoemaker:`
- `XMP-dc:Description` - Human-readable thumbnail list
- `XMP-xmp:Label` - `shoemaker-managed` marker

### Integration with wake-n-blake

When processing files that already have wake-n-blake XMP sidecars:

1. **ExifTool preserves namespaces** - shoemaker uses ExifTool which preserves unknown namespaces (like `wnb:`), so wake-n-blake data is NOT lost
2. **Sidecar hash invalidation** - wake-n-blake's `wnb:SidecarHash` will become invalid after shoemaker modifies the XMP (this is expected behavior indicating modification)

### TODO: Add Custody Events

shoemaker should add a custody event to the chain when modifying XMP:

```typescript
// Future implementation in xmp-updater.ts
await exiftool.write(xmpPath, {}, [
  '-overwrite_original',
  // Existing shoemaker fields...
  `-XMP-wnb:EventCount+=1`,  // Increment event count
  // Would need to append to CustodyChain (complex with ExifTool)
]);
```

### TODO: Respect Related Files

Check `wnb:RelationType` before processing to avoid duplicate work:
- If `wnb:IsPrimaryFile=false`, consider skipping (thumbnail already generated for primary)
- For Live Photos, generate thumbnail from the image, not the video

### Namespace Migration (Future)

Consider migrating from `dc:Source` hack to proper `shoemaker:` namespace:

```
Proposed Namespace URI: http://shoemaker.dev/xmp/1.0/
Proposed Prefix: shoemaker
```

This would be cleaner and avoid overloading the Dublin Core `Source` field.
