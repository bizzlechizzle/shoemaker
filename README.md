# Shoemaker

> A CLI that makes thumbnails from images and RAW files.

**Version:** 0.1.1 | **License:** MIT | **Node.js:** 20+

## Features

- **Fast thumbnail generation** — Extract embedded camera previews for instant thumbnails
- **RAW file support** — Sony ARW, Canon CR2/CR3, Nikon NEF, Fuji RAF, and more
- **Smart fallback** — Automatically decode RAW when previews are insufficient
- **XMP integration** — Track thumbnail state in XMP sidecars
- **Multiple presets** — Fast import, high quality, portable (CI/CD friendly)
- **Library + CLI** — Use programmatically or from the command line

## Installation

```bash
npm install -g shoemaker
```

Or use locally in a project:

```bash
npm install shoemaker
```

## Quick Start

```bash
# Generate thumbnails for a directory
shoemaker thumb /photos/import/

# Use high-quality preset (RAW decode)
shoemaker thumb /photos/import/ --preset quality

# Check what's embedded in a RAW file
shoemaker info /photos/image.arw

# See what needs processing
shoemaker status /photos/import/

# Check system dependencies
shoemaker doctor
```

## CLI Commands

### `shoemaker thumb <path>`

Generate thumbnails for files or directories.

```bash
shoemaker thumb /photos/import/           # Process directory
shoemaker thumb /photos/import/ -r        # Process recursively
shoemaker thumb /photos/image.arw         # Process single file
shoemaker thumb /photos/ --preset quality # Use high-quality preset
shoemaker thumb /photos/ --force          # Regenerate existing
shoemaker thumb /photos/ --dry-run        # Show what would be done
shoemaker thumb /photos/ --json           # Output as JSON
shoemaker thumb /photos/ -c 8             # Use 8 concurrent workers
```

**Options:**
- `-r, --recursive` — Process subdirectories
- `-p, --preset <name>` — Preset to use (fast, quality, portable)
- `-f, --force` — Regenerate even if thumbnails exist
- `--dry-run` — Show what would be done without writing files
- `-c, --concurrency <n>` — Number of files to process in parallel
- `-q, --quiet` — Suppress progress output
- `--json` — Output results as JSON

### `shoemaker info <file>`

Show embedded preview and thumbnail information.

```bash
shoemaker info /photos/image.arw
shoemaker info /photos/image.arw --json
```

**Output:**
```
image.arw
══════════════════════════════════════════════════

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
Shoemaker v0.1.0 — System Check

RAW Decoders:
  ✓ embedded           Built-in
  ✓ libraw             WASM (bundled)
  ✓ rawtherapee-cli    v5.10 (/opt/homebrew/bin/rawtherapee-cli)
  ✓ darktable-cli      v4.6.0 (/opt/homebrew/bin/darktable-cli)
  ✗ dcraw              Not found

Metadata Tools:
  ✓ exiftool           Bundled via exiftool-vendored
  ✓ exiv2              v0.28.1 (/opt/homebrew/bin/exiv2)

Image Processing:
  ✓ sharp              v8.15.1 (libvips)
  ✓ Formats: jpeg, png, webp, avif, tiff

Recommended Presets:
  • fast-import    → embedded + libraw fallback
  • high-quality   → rawtherapee-cli (best available)
  • portable       → libraw only (CI/CD safe)

✓ All core systems operational.
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

## Library Usage

```typescript
import { generateThumbnails, loadConfig, loadPreset } from 'shoemaker';

// Simple usage
const result = await generateThumbnails('/path/to/image.arw');
console.log(result.thumbnails);

// With options
const result = await generateThumbnails('/path/to/image.arw', {
  preset: 'quality',
  force: true,
  onProgress: (info) => console.log(info.status),
});

// Full control
import { generateForFile, generateForBatch, findImageFiles } from 'shoemaker';

const config = await loadConfig();
const preset = await loadPreset('fast', config);

const files = await findImageFiles('/photos/import/', config, true);
const result = await generateForBatch(files, { config, preset });
```

## Generated Thumbnails

By default, thumbnails are created in a sidecar folder next to the source:

```
/photos/
├── IMG_1234.ARW
├── IMG_1234.ARW.xmp           # XMP sidecar (updated by shoemaker)
└── IMG_1234_thumbs/           # Thumbnail folder
    ├── IMG_1234_thumb_300.webp
    ├── IMG_1234_preview_1600.webp
    └── IMG_1234_ml_2560.jpg
```

| Size | Resolution | Format | Use Case |
|------|------------|--------|----------|
| thumb | 300px | WebP | Gallery grids |
| preview | 1600px | WebP | Lightbox viewing |
| ml | 2560px | JPEG | ML inference |

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
```

Or create `.shoemaker.toml` in your project directory for project-specific settings.

## Requirements

- Node.js 20+
- libvips (Sharp will try to install automatically, but manual install may be needed)
- Optional: RawTherapee or darktable for high-quality RAW decoding

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
Sharp includes prebuilt binaries - no manual install needed.

## Error Handling

Shoemaker provides detailed error messages for common issues:

- **Path not found** - The specified file or directory doesn't exist
- **Permission denied** - Cannot read/write to the specified location
- **No embedded preview** - RAW file doesn't have extractable previews
- **File is empty** - Source file has zero bytes
- **Corrupt file** - Cannot decode the image data

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

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for contribution guidelines and architecture details.

## License

MIT
