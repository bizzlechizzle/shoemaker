# Shoemaker Development Guide

This guide is for developers who want to contribute to Shoemaker or understand its internals.

## Prerequisites

- Node.js 20+
- npm 10+
- libvips (for Sharp)
- Git

### Installing libvips

**macOS:**
```bash
brew install vips
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libvips-dev
```

**Windows:**
Sharp includes prebuilt binaries, no manual installation needed.

### Optional Dependencies

For full functionality:

**macOS:**
```bash
brew install ffmpeg rawtherapee darktable
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg rawtherapee darktable
```

## Getting Started

```bash
# Clone the repository
git clone https://github.com/youruser/shoemaker.git
cd shoemaker

# Install dependencies
npm install

# Build
npm run build

# Run tests
npm test

# Run the CLI
npm start -- doctor
```

## Project Architecture

### Overview

```
+-----------------------------------------------------------+
|                         CLI                                |
|  (Commander.js: thumb, info, clean, status, doctor)       |
+--------------------------+--------------------------------+
                           |
+--------------------------v--------------------------------+
|                      Services                              |
|  (thumbnail-generator.ts, xmp-updater.ts)                 |
+--------------------------+--------------------------------+
                           |
+--------------------------v--------------------------------+
|                        Core                                |
| Image: extractor.ts, resizer.ts, decoder.ts               |
| Video: ffprobe.ts, frame-extractor.ts                     |
| Config: config.ts, errors.ts                              |
+--------------------------+--------------------------------+
                           |
+--------------------------v--------------------------------+
|                    External Tools                          |
| Image: ExifTool, Sharp/libvips, RawTherapee, darktable    |
| Video: FFmpeg, FFprobe                                     |
+-----------------------------------------------------------+
```

### Module Responsibilities

| Module | File | Responsibility |
|--------|------|----------------|
| **Extractor** | `src/core/extractor.ts` | Extract embedded previews via ExifTool, format detection |
| **Decoder** | `src/core/decoder.ts` | RAW file decoding (embedded, sharp, RawTherapee, darktable, dcraw) |
| **Resizer** | `src/core/resizer.ts` | Resize images via Sharp, convert to sRGB |
| **FFprobe** | `src/core/ffprobe.ts` | Video metadata extraction |
| **Frame Extractor** | `src/core/frame-extractor.ts` | Video frame extraction, timeline strips |
| **Proxy Generator** | `src/core/proxy-generator.ts` | Video proxy encoding with HW acceleration |
| **Config** | `src/core/config.ts` | Load TOML config and presets |
| **Errors** | `src/core/errors.ts` | Error classes and recovery logic |
| **Generator** | `src/services/thumbnail-generator.ts` | Main pipeline orchestration (images + video) |
| **XMP** | `src/services/xmp-updater.ts` | XMP sidecar read/write |
| **Schemas** | `src/schemas/index.ts` | Zod validation schemas |

## Adding a New Feature

### 1. Add a New CLI Command

1. Create `src/cli/commands/mycommand.ts`:

```typescript
import { Command } from 'commander';

export const myCommand = new Command('mycommand')
  .description('Description of my command')
  .argument('<arg>', 'Argument description')
  .option('-o, --option <value>', 'Option description')
  .action(async (arg, options) => {
    // Implementation
  });
```

2. Register in `src/cli/index.ts`:

```typescript
import { myCommand } from './commands/mycommand.js';
program.addCommand(myCommand);
```

### 2. Add a New Core Module

1. Create `src/core/mymodule.ts`
2. Export from `src/index.ts` for library access
3. Add unit tests in `tests/unit/mymodule.test.ts`

### 3. Add a New Preset

Create `presets/mypreset.toml`:

```toml
[sizes.thumb]
width = 300
format = "webp"
quality = 80

[behavior]
fallback_to_raw = false
decoder = "embedded"
```

### 4. Add a New Video Format

1. Add extension to `VIDEO_EXTENSIONS` in `src/schemas/index.ts`:

```typescript
export const VIDEO_EXTENSIONS = [
  // ... existing
  'newformat',
] as const;
```

2. Verify FFprobe can read the format:

```bash
ffprobe /path/to/file.newformat
```

### 5. Add a New RAW Format

1. Add extension to `RAW_EXTENSIONS` in `src/schemas/index.ts`
2. Test with real files to ensure ExifTool can extract previews

## Testing

### Unit Tests

```bash
npm test                 # Run all unit tests
npm run test:watch       # Watch mode
npm test -- --coverage   # With coverage
```

### Integration Tests

Integration tests require real RAW/video files:

```bash
npm run test:download-fixtures  # Download test files
npm run test:integration        # Run integration tests
```

### Writing Tests

```typescript
// tests/unit/mymodule.test.ts
import { describe, it, expect } from 'vitest';
import { myFunction } from '../../src/core/mymodule.js';

describe('myFunction', () => {
  it('should do something', () => {
    const result = myFunction('input');
    expect(result).toBe('expected');
  });
});
```

### Test Fixtures

Place test files in:
- `tests/fixtures/images/` - JPEG, PNG samples
- `tests/fixtures/raw/` - RAW file samples
- `tests/fixtures/video/` - MP4, MOV samples

## Code Style

- TypeScript strict mode
- ESM modules (no CommonJS)
- Explicit types (no `any`)
- Early returns over deep nesting
- Descriptive variable names

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Files | kebab-case | `thumbnail-generator.ts` |
| Functions | camelCase | `generateThumbnails()` |
| Classes | PascalCase | `ShoemakerError` |
| Constants | SCREAMING_SNAKE | `DEFAULT_CONFIG` |
| Types/Interfaces | PascalCase | `ThumbnailResult` |

## Error Handling

Use `ShoemakerError` for all errors:

```typescript
import { ShoemakerError, ErrorCode, wrapError } from '../core/errors.js';

// Create specific error
throw new ShoemakerError(
  'File not found',
  ErrorCode.FILE_NOT_FOUND,
  filePath,
  true  // recoverable
);

// Wrap unknown errors
try {
  await riskyOperation();
} catch (err) {
  throw wrapError(err, filePath);
}
```

### Error Codes

| Code | Meaning | Recoverable |
|------|---------|-------------|
| `FILE_NOT_FOUND` | File doesn't exist | No |
| `PERMISSION_DENIED` | Can't read/write | No |
| `CORRUPT_FILE` | File is damaged | Yes |
| `NO_PREVIEW` | No embedded preview | Yes |
| `DECODE_FAILED` | Decoding error | Yes |
| `DECODER_NOT_AVAILABLE` | Tool not installed | No |
| `INVALID_PATH` | Path validation failed | No |

## Building and Releasing

### Build

```bash
npm run build
```

Output goes to `dist/`.

### Version Bump

1. Update `VERSION` file
2. Update `package.json` version
3. Commit both in the same commit

### Release

```bash
git tag v0.2.0
git push --tags
```

GitHub Actions will publish to npm automatically.

## Debugging

### Debug ExifTool

```bash
exiftool -a -G1 "*Preview*" "*Jpg*" image.arw
```

### Debug FFprobe

```bash
ffprobe -v error -show_format -show_streams video.mp4
```

### Debug Sharp

```javascript
const sharp = require('sharp');
console.log(sharp.versions);  // Shows libvips version
```

### Debug with Verbose Output

```bash
DEBUG=* npm start -- thumb /path/to/image
```

## Module Deep Dive

### schemas/index.ts

Centralized location for:
- **Constants**: `DEFAULT_MIN_PREVIEW_SIZE`, `RAW_EXTENSIONS`, `VIDEO_EXTENSIONS`, `MAX_CONCURRENCY`, etc.
- **Zod Schemas**: Validation for config, presets, results, and video info
- **Type Exports**: All TypeScript types used across the codebase

When adding new constants or types, add them here to maintain single source of truth.

### core/errors.ts

Error handling strategy:
- `ShoemakerError` class with `ErrorCode` enum
- `wrapError()` for consistent error wrapping
- `isRecoverable()` for batch processing decisions
- `shouldStopBatch()` and `shouldReduceConcurrency()` for flow control

### core/extractor.ts

ExifTool integration:
- `analyzeEmbeddedPreviews()` - Read metadata about available previews
- `extractBestPreview()` - Get the largest preview as a buffer
- `isRawFormat()` / `isDecodedFormat()` / `isVideoFormat()` - Format detection

Key considerations:
- ExifTool runs as a persistent process - always call `shutdownExiftool()` on exit
- Preview extraction uses temp files - cleaned up in `finally` block
- Type guards used for safe value extraction from ExifTool output

### core/ffprobe.ts

Video metadata extraction:
- `probeVideo()` - Get full video info (duration, resolution, codec, etc.)
- `checkFfprobeAvailable()` / `checkFfmpegAvailable()` - Tool detection
- `getVideoDuration()` / `hasAudio()` - Convenience functions

Key considerations:
- FFprobe availability is cached for performance
- Detects interlacing via `field_order` metadata
- Detects HDR via `color_transfer` metadata
- Parses rotation from side data or format tags

### core/frame-extractor.ts

Video frame extraction:
- `extractFrame()` - Single frame at specific time
- `extractFrameAtPercent()` - Frame at percentage position
- `extractMultipleFrames()` - Batch extraction
- `generateTimelineStrip()` - Horizontal frame concatenation
- `extractPosterFrame()` / `extractPreviewFrame()` - Configured positions

Key features:
- Safe zone calculation (skips first/last 5%)
- Black frame detection and skipping
- Automatic deinterlacing (yadif filter)
- HDR to SDR tone mapping (Hable algorithm)
- Rotation handling

### core/proxy-generator.ts

Video proxy encoding with hardware acceleration:

| HW Accel | Encoder | Platform |
|----------|---------|----------|
| VideoToolbox | `h264_videotoolbox` | macOS |
| NVENC | `h264_nvenc` | NVIDIA |
| VAAPI | `h264_vaapi` | AMD/Intel Linux |
| QSV | `h264_qsv` | Intel |
| Software | `libx264` | Any (fallback) |

Key functions:
- `detectAvailableEncoders()` - Probe FFmpeg for available encoders
- `selectEncoder(codec, hwAccel)` - Choose best encoder for config
- `generateProxy(input, output, size, config)` - Encode single proxy
- `generateProxies(input, options)` - Encode all configured sizes
- `buildFilterChain(videoInfo, height, config)` - Build FFmpeg filters

Features:
- Hardware encoder auto-detection with software fallback
- LUT color grading via `lut3d` filter
- HDR to SDR tone mapping
- Deinterlacing for interlaced sources
- Rotation handling (90°/270° dimension swap)
- NLE-friendly encoding (1-second keyframes, VFR→CFR, bt709 tagging)
- Progress tracking via FFmpeg stderr parsing
- Partial file cleanup on failure

Security:
- LUT paths validated (.cube extension, file exists)
- Path escaping for FFmpeg filter syntax

### core/decoder.ts

RAW decoding with multiple backends:

| Decoder | Method | Quality | Speed | Notes |
|---------|--------|---------|-------|-------|
| `embedded` | Extract preview | Good | Fastest | Default, no decode |
| `sharp` | libvips | Basic | Fast | Limited RAW support |
| `rawtherapee` | CLI | Best | Slow | Professional quality |
| `darktable` | CLI | Best | Slow | Alternative to RT |
| `dcraw` | CLI | Good | Medium | Legacy, basic |

Key functions:
- `decodeRawFile(path, options)` - Decode with fallback chain
- `detectAvailableDecoders()` - Check what's installed
- `selectDecoder(preferred, fallback)` - Auto-select best

Security:
- Command whitelist: `ALLOWED_COMMANDS` limits CLI decoders
- Uses `execFile` with array args - no shell injection

### core/resizer.ts

Sharp integration:
- All outputs converted to sRGB with embedded ICC profile
- Supports WebP, JPEG, PNG, AVIF output formats
- Respects EXIF orientation by default

### services/thumbnail-generator.ts

Main orchestration:
1. Check if already processed (XMP sidecar)
2. Determine file type (image, RAW, video)
3. For video: Extract frames via FFmpeg
4. For images/RAW: Extract preview or decode
5. Resize to all configured sizes
6. Update XMP sidecar with metadata

Batch processing uses `p-queue` for concurrency control with adaptive throttling.

## Security Considerations

### Input Validation
All CLI commands validate:
- Path existence and accessibility
- Numeric arguments (concurrency) within bounds
- File extensions against whitelists

### Path Safety
- `expandPath()` safely handles `~` expansion
- No path traversal vulnerabilities
- Temp files use crypto-random IDs

### Command Execution
- Decoder/metadata commands are whitelisted (`ALLOWED_DECODER_COMMANDS`)
- No shell interpolation - uses `execFile` with arrays

## SME Documentation

Comprehensive specifications are in `sme/`:

- `unified-cli-architecture.md` - Full technical spec
- `thumbnail-best-practices.md` - Research on thumbnails
- `production-hardening-plan.md` - Audit results and fixes

These documents define the expected behavior and should be consulted when implementing features.

## Questions?

- Check `techguide.md` for gotchas
- Check SME docs for specifications
- Open an issue for questions
