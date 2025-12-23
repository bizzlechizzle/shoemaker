# Shoemaker: A CLI That Makes Thumbnails

> **Generated**: 2025-12-23
> **Sources current as of**: 2025-12-23
> **Scope**: Comprehensive
> **Version**: 2.3
> **Audit-Ready**: Yes

---

## Executive Summary / TLDR

**Shoemaker** is a focused CLI tool that generates thumbnails from images (including RAW files) and updates XMP sidecars. It runs **after wake-n-blake** in your workflow, consuming already-verified files and adding thumbnail metadata.

**Core Capabilities:**
1. **Tiered thumbnail generation** — 300px (grid), 1600px (lightbox), 2560px (ML)
2. **Smart extraction** — Use embedded camera previews when sufficient (fast), fallback to raw decode (slow)
3. **Color management** — Convert to sRGB with embedded ICC profile
4. **XMP integration** — Update sidecars to track thumbnail state
5. **Presets** — Fast Import (extract only) vs High Quality (full decode)

**Tech Stack:**
- **Language**: TypeScript (Node.js 20+)
- **Image Processing**: sharp (libvips) for resize/convert, libraw for RAW decode fallback
- **XMP**: fast-xml-parser or exiftool-vendored for sidecar read/write
- **CLI Framework**: Commander.js (matches wake-n-blake)

---

## Background & Context

### The Problem

When importing large photo collections (especially RAW files), you need thumbnails for:
- **Gallery grids** — Fast-loading small thumbnails (300px)
- **Lightbox viewing** — Medium previews for quick inspection (1600px)
- **ML processing** — Larger images for tagging/classification (2560px)

Current pain points:
1. RAW files are slow to decode (1-3 seconds each)
2. Embedded previews exist but vary wildly by camera
3. Color space is inconsistent (Adobe RGB, ProPhoto RGB, untagged)
4. No standard way to track what thumbnails exist

### The Workflow

```
wake-n-bake    →    wake-n-blake    →    shoemaker    →    downstream apps
  (setup)         (hash, verify,        (thumbnails,      (visual-buffet,
                   basic XMP)            XMP update)        nightfox, etc)
```

**wake-n-blake outputs:**
- `image.arw` — Original file, verified
- `image.arw.xmp` — Sidecar with BLAKE3 hash, basic metadata

**shoemaker adds:**
- `image_thumb_300.webp` — Grid thumbnail
- `image_preview_1600.webp` — Lightbox preview
- `image_ml_2560.jpg` — ML-ready image
- Updates `image.arw.xmp` with thumbnail references

---

## Embedded Preview Analysis

### What Cameras Provide

| Camera Type | Embedded Previews | Best Available |
|-------------|-------------------|----------------|
| Sony (ARW) | Thumbnail: 160x120, Preview: 1616x1080, JpgFromRaw: 4240x2832 | JpgFromRaw (full res) |
| Canon (CR2/CR3) | Thumbnail: 160x120, Preview: 5472x3648 | Preview (full res) |
| Nikon (NEF) | Preview: 640x424, OtherImage: 1620x1080, JpgFromRaw: 6016x4016 | JpgFromRaw (full res) |
| DJI (DNG) | Preview: 960x720 only | Preview (insufficient) |
| TIFF | None | Must process original |
| HEIC | Usually has preview | Preview (usually sufficient) |

### Extraction Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    For each image file:                      │
├─────────────────────────────────────────────────────────────┤
│  1. Check for JpgFromRaw (exiftool -JpgFromRaw)             │
│     └─ If exists AND >= 2560px → Extract, use as source     │
│                                                              │
│  2. Check for PreviewImage (exiftool -PreviewImage)          │
│     └─ If exists AND >= 2560px → Extract, use as source     │
│                                                              │
│  3. Check for OtherImage (exiftool -OtherImage)              │
│     └─ If exists AND >= 2560px → Extract, use as source     │
│                                                              │
│  4. FALLBACK: Decode RAW with libraw                         │
│     └─ Slow path (~1-3 seconds)                              │
│     └─ Proper demosaic, color science                        │
│                                                              │
│  5. For TIFF/JPEG/PNG: Use original directly                 │
│     └─ Already decoded, just resize                          │
└─────────────────────────────────────────────────────────────┘
```

### Color Space Handling

**Problem**: Embedded previews have inconsistent/missing ICC profiles

**Solution**: Force sRGB conversion on all outputs

```typescript
// Using sharp
await sharp(extractedPreview)
  .resize(2560, 2560, { fit: 'inside', withoutEnlargement: true })
  .toColorspace('srgb')
  .withMetadata({ icc: 'srgb' })  // Embed sRGB profile
  .jpeg({ quality: 90 })
  .toFile('image_ml_2560.jpg');
```

---

## ExifTool Commands Reference

### Analyzing Embedded Previews

```bash
# List all embedded images with sizes
exiftool -a -G1 -s -PreviewImage -JpgFromRaw -OtherImage -ThumbnailImage \
  -PreviewImageSize -JpgFromRawLength -OtherImageLength image.arw

# Get detailed info about all embedded images
exiftool -a -G1 "*Image*" "*Preview*" "*Thumb*" image.arw

# Check specific preview dimensions (returns WxH or empty)
exiftool -s3 -PreviewImageSize image.arw
exiftool -s3 -JpgFromRawWidth -JpgFromRawHeight image.arw
```

### Extracting Embedded Previews

```bash
# Extract JpgFromRaw (full-resolution JPEG from RAW) - PREFERRED
exiftool -b -JpgFromRaw image.arw > preview.jpg

# Extract PreviewImage (mid-size preview)
exiftool -b -PreviewImage image.nef > preview.jpg

# Extract OtherImage (varies by camera)
exiftool -b -OtherImage image.cr2 > preview.jpg

# Extract ThumbnailImage (tiny, ~160x120)
exiftool -b -ThumbnailImage image.arw > thumb.jpg
```

### Using exiftool-vendored (Node.js)

```typescript
import { exiftool } from 'exiftool-vendored';

// Analyze what's available
const tags = await exiftool.read(filePath);

const previews = {
  jpgFromRaw: {
    exists: !!tags.JpgFromRaw,
    width: tags.JpgFromRawWidth,
    height: tags.JpgFromRawHeight,
    length: tags.JpgFromRawLength,
  },
  previewImage: {
    exists: !!tags.PreviewImage,
    width: tags.PreviewImageWidth,
    height: tags.PreviewImageHeight,
    length: tags.PreviewImageLength,
  },
  otherImage: {
    exists: !!tags.OtherImage,
    width: tags.OtherImageWidth,
    height: tags.OtherImageHeight,
    length: tags.OtherImageLength,
  },
};

// Extract best preview to buffer
async function extractBestPreview(filePath: string): Promise<Buffer> {
  const tags = await exiftool.read(filePath);

  // Priority: JpgFromRaw > PreviewImage > OtherImage
  if (tags.JpgFromRaw) {
    return await exiftool.extractBinaryTag('JpgFromRaw', filePath);
  }
  if (tags.PreviewImage) {
    return await exiftool.extractBinaryTag('PreviewImage', filePath);
  }
  if (tags.OtherImage) {
    return await exiftool.extractBinaryTag('OtherImage', filePath);
  }

  throw new Error('No embedded preview found');
}

// Cleanup - IMPORTANT: call on app shutdown
await exiftool.end();
```

### Preview Tag by Camera Brand

| Brand | Best Tag | Typical Size | Notes |
|-------|----------|--------------|-------|
| Sony | JpgFromRaw | Full resolution | Always available |
| Canon | PreviewImage | Full resolution | CR2/CR3 |
| Nikon | JpgFromRaw | Full resolution | NEF files |
| Fujifilm | JpgFromRaw | Full resolution | RAF files |
| Panasonic | PreviewImage | ~4000px | RW2 files |
| DJI | PreviewImage | 960x720 | Insufficient for ML |
| Leica | JpgFromRaw | Full resolution | DNG files |
| Hasselblad | PreviewImage | ~8000px | 3FR files |

---

## Proposed Architecture

### Directory Structure

```
shoemaker/
├── bin/
│   └── shoemaker.js              # Entry point
├── src/
│   ├── index.ts                  # Library exports
│   ├── cli/
│   │   ├── index.ts             # CLI setup (Commander.js)
│   │   └── commands/
│   │       ├── thumb.ts         # Main thumbnail command
│   │       ├── info.ts          # Show embedded preview info
│   │       └── clean.ts         # Remove generated thumbnails
│   │
│   ├── core/
│   │   ├── extractor.ts         # Extract embedded previews (exiftool)
│   │   ├── decoder.ts           # RAW decode fallback (libraw)
│   │   ├── resizer.ts           # Resize + color convert (sharp)
│   │   └── config.ts            # Configuration loading
│   │
│   ├── services/
│   │   ├── thumbnail-generator.ts   # Orchestrates the pipeline
│   │   ├── xmp-updater.ts           # Update XMP sidecars
│   │   └── preview-analyzer.ts      # Check what's embedded
│   │
│   └── schemas/
│       └── index.ts             # Zod schemas
│
├── presets/
│   ├── fast-import.toml         # Extract-only, skip decode
│   └── high-quality.toml        # Always decode RAW properly
│
├── tests/
├── package.json
├── tsconfig.json
├── CLAUDE.md                    # From repo-depot
└── VERSION
```

### Core Pipeline

```typescript
// src/services/thumbnail-generator.ts

interface ThumbnailConfig {
  sizes: {
    thumb: number;     // 300
    preview: number;   // 1600
    ml: number;        // 2560
  };
  formats: {
    thumb: 'webp' | 'jpeg';
    preview: 'webp' | 'jpeg';
    ml: 'jpeg';  // ML models expect JPEG
  };
  quality: {
    thumb: number;     // 80
    preview: number;   // 85
    ml: number;        // 90
  };
  fallbackToRaw: boolean;  // false = fast-import preset
}

export async function generateThumbnails(
  inputPath: string,
  outputDir: string,
  config: ThumbnailConfig
): Promise<ThumbnailResult> {

  // 1. Analyze what's embedded
  const analysis = await analyzeEmbeddedPreviews(inputPath);

  // 2. Get best available source
  let sourceBuffer: Buffer;
  let sourceMethod: 'extracted' | 'decoded' | 'direct';

  if (analysis.bestPreview && analysis.bestPreview.width >= config.sizes.ml) {
    // Fast path: extract embedded preview
    sourceBuffer = await extractPreview(inputPath, analysis.bestPreview.type);
    sourceMethod = 'extracted';
  } else if (isAlreadyDecoded(inputPath)) {
    // TIFF/JPEG/PNG: read directly
    sourceBuffer = await fs.readFile(inputPath);
    sourceMethod = 'direct';
  } else if (config.fallbackToRaw) {
    // Slow path: decode RAW
    sourceBuffer = await decodeRaw(inputPath);
    sourceMethod = 'decoded';
  } else {
    // Fast-import mode: use whatever's available
    sourceBuffer = await extractLargestPreview(inputPath);
    sourceMethod = 'extracted';
  }

  // 3. Generate all sizes from source
  const results = await Promise.all([
    generateSize(sourceBuffer, config.sizes.thumb, 'thumb', config),
    generateSize(sourceBuffer, config.sizes.preview, 'preview', config),
    generateSize(sourceBuffer, config.sizes.ml, 'ml', config),
  ]);

  // 4. Update XMP sidecar
  await updateXmpSidecar(inputPath, results, sourceMethod);

  return {
    source: inputPath,
    method: sourceMethod,
    thumbnails: results,
  };
}
```

### XMP Sidecar Schema

```xml
<!-- image.arw.xmp (additions by shoemaker) -->
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description xmlns:shoemaker="http://shoemaker.local/1.0/">

      <!-- Thumbnail generation info -->
      <shoemaker:ThumbnailsGenerated>true</shoemaker:ThumbnailsGenerated>
      <shoemaker:GeneratedAt>2025-12-23T10:30:00Z</shoemaker:GeneratedAt>
      <shoemaker:SourceMethod>extracted</shoemaker:SourceMethod>
      <shoemaker:Preset>fast-import</shoemaker:Preset>

      <!-- Generated files -->
      <shoemaker:Thumbnails>
        <rdf:Bag>
          <rdf:li>
            <shoemaker:Size>thumb</shoemaker:Size>
            <shoemaker:Resolution>300</shoemaker:Resolution>
            <shoemaker:Format>webp</shoemaker:Format>
            <shoemaker:Path>image_thumb_300.webp</shoemaker:Path>
            <shoemaker:Bytes>24576</shoemaker:Bytes>
          </rdf:li>
          <rdf:li>
            <shoemaker:Size>preview</shoemaker:Size>
            <shoemaker:Resolution>1600</shoemaker:Resolution>
            <shoemaker:Format>webp</shoemaker:Format>
            <shoemaker:Path>image_preview_1600.webp</shoemaker:Path>
            <shoemaker:Bytes>163840</shoemaker:Bytes>
          </rdf:li>
          <rdf:li>
            <shoemaker:Size>ml</shoemaker:Size>
            <shoemaker:Resolution>2560</shoemaker:Resolution>
            <shoemaker:Format>jpeg</shoemaker:Format>
            <shoemaker:Path>image_ml_2560.jpg</shoemaker:Path>
            <shoemaker:Bytes>409600</shoemaker:Bytes>
          </rdf:li>
        </rdf:Bag>
      </shoemaker:Thumbnails>

    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
```

---

## CLI Commands

### Main Commands

```bash
# Generate thumbnails for file(s)
shoemaker thumb <path>                    # Single file or directory
shoemaker thumb <path> --recursive        # Process subdirectories
shoemaker thumb <path> --preset fast      # Use fast-import preset
shoemaker thumb <path> --preset quality   # Use high-quality preset

# Custom options
shoemaker thumb <path> \
  --sizes 300,1600,2560 \
  --formats webp,webp,jpeg \
  --quality 80,85,90 \
  --no-fallback                          # Don't decode RAW, use embedded only

# Analyze embedded previews (don't generate anything)
shoemaker info <path>                     # Show what's embedded
shoemaker info <path> --json              # Machine-readable output

# Clean up generated thumbnails
shoemaker clean <path>                    # Remove thumbs, update XMP
shoemaker clean <path> --dry-run          # Show what would be removed

# Regenerate (force re-process)
shoemaker thumb <path> --force            # Overwrite existing thumbnails

# Check what needs processing
shoemaker status <path>                   # Show files missing thumbnails

# Check system integrations
shoemaker doctor                          # Show available decoders/tools
```

### Example Output

```
$ shoemaker thumb /photos/import/ --preset fast

Processing 127 files...

  ✓ _DSC0008.ARW     extracted (JpgFromRaw 4240x2832) → 3 thumbs (0.2s)
  ✓ _MG_5043.CR2     extracted (Preview 5472x3648)   → 3 thumbs (0.3s)
  ✓ img012.tif       direct (4405x2773)              → 3 thumbs (0.4s)
  ⚠ DJI_0412.DNG     extracted (Preview 960x720)     → 2 thumbs (preview too small for ML)

Done: 127 files, 381 thumbnails generated
  - Extracted: 98 (77%)
  - Direct: 24 (19%)
  - Decoded: 5 (4%)

Time: 42.3s (avg 0.33s/file)
Space: 76.2 MB total (~600KB/file avg)
```

### Doctor Output

```
$ shoemaker doctor

Shoemaker v0.1.0 — System Check

RAW Decoders:
  ✓ embedded          Built-in (always available)
  ✓ libraw            WASM v0.21.0 (always available)
  ✓ rawtherapee-cli   v5.10 (/opt/homebrew/bin/rawtherapee-cli)
  ✓ darktable-cli     v4.6.0 (/opt/homebrew/bin/darktable-cli)
  ✗ dcraw             Not found

Metadata Tools:
  ✓ exiftool          v12.76 (bundled via exiftool-vendored)
  ✓ exiv2             v0.28.1 (/opt/homebrew/bin/exiv2)

Image Processing:
  ✓ sharp             v0.33.2 (libvips 8.15.1)
  ✓ vipsthumbnail     v8.15.1 (/opt/homebrew/bin/vipsthumbnail)

Format Support:
  ✓ WebP              Encode/Decode
  ✓ AVIF              Encode/Decode (libheif 1.17.6)
  ✓ HEIC              Decode only (libheif 1.17.6)
  ✗ JPEG XL           Not available (requires libvips 8.14+)

Recommended Presets:
  • fast-import    → embedded + libraw fallback
  • high-quality   → rawtherapee-cli (best available)
  • portable       → libraw only (CI/CD safe)

All systems operational.
```

---

## Presets

### fast-import.toml

```toml
# Fast import preset - extract embedded previews only, no RAW decode
# Best for: Quick imports, initial triage, when time matters more than quality

[sizes]
thumb = 300
preview = 1600
ml = 2560

[formats]
thumb = "webp"
preview = "webp"
ml = "jpeg"

[quality]
thumb = 80
preview = 85
ml = 90

[behavior]
fallback_to_raw = false           # Never decode RAW
use_largest_available = true      # If ml size not available, use what's there
skip_if_insufficient = false      # Generate smaller sizes even if source is small
```

### high-quality.toml

```toml
# High quality preset - always decode RAW for best results
# Best for: Final archives, when quality matters, smaller batches

[sizes]
thumb = 300
preview = 1600
ml = 2560

[formats]
thumb = "webp"
preview = "webp"
ml = "jpeg"

[quality]
thumb = 85
preview = 90
ml = 95

[behavior]
fallback_to_raw = true            # Always decode if embedded insufficient
use_largest_available = false     # Require full size
skip_if_insufficient = true       # Don't generate ML thumb if source too small
```

---

## Configuration File

### Location

```
~/.config/shoemaker/config.toml     # User config (primary)
./.shoemaker.toml                   # Project-local override
```

### Full Schema

```toml
# ~/.config/shoemaker/config.toml
# Shoemaker Configuration File

# Default preset to use when --preset not specified
default_preset = "fast"  # "fast" | "quality" | custom preset name

# Where to find custom presets
preset_dir = "~/.config/shoemaker/presets"

# ------------------------------------------------------------------------------
# Output Settings
# ------------------------------------------------------------------------------

[output]
# Where to put thumbnails relative to source file
# Options: "sidecar" (folder next to file), "cache" (centralized)
location = "sidecar"

# Folder name when using sidecar mode
# {stem} = filename without extension
sidecar_folder = "{stem}_thumbs"

# Cache directory when using cache mode
cache_dir = "~/.cache/shoemaker"

# Naming pattern for generated files
# Available: {stem}, {size}, {width}, {height}, {format}
naming_pattern = "{stem}_{size}_{width}.{format}"

# ------------------------------------------------------------------------------
# Processing Settings
# ------------------------------------------------------------------------------

[processing]
# Number of files to process concurrently
concurrency = 4

# Minimum preview size to consider "sufficient" for ML output
min_preview_size = 2560

# Skip files that already have thumbnails (check XMP)
skip_existing = true

# Auto-rotate based on EXIF orientation
auto_rotate = true

# Strip EXIF from thumbnails (keeps ICC profile)
strip_exif = true

# ------------------------------------------------------------------------------
# Size Definitions
# ------------------------------------------------------------------------------

[sizes.thumb]
width = 300
format = "webp"
quality = 80
# Generate even if source is smaller
allow_upscale = false

[sizes.preview]
width = 1600
format = "webp"
quality = 85
allow_upscale = false

[sizes.ml]
width = 2560
format = "jpeg"
quality = 90
allow_upscale = false

# Add custom sizes
# [sizes.social_square]
# width = 1080
# height = 1080
# format = "jpeg"
# quality = 85
# crop = "center"  # "center" | "face" | "smart"

# ------------------------------------------------------------------------------
# File Type Handling
# ------------------------------------------------------------------------------

[filetypes]
# Extensions to process (case-insensitive)
include = [
  "arw", "cr2", "cr3", "nef", "raf", "rw2", "orf", "pef", "dng",  # RAW
  "jpg", "jpeg", "png", "tif", "tiff", "heic", "heif", "webp"      # Standard
]

# Extensions to always skip
exclude = [
  "xmp", "json", "txt", "md"
]

# How to handle files with no embedded preview
no_preview_action = "decode"  # "decode" | "skip" | "warn"

# ------------------------------------------------------------------------------
# XMP Settings
# ------------------------------------------------------------------------------

[xmp]
# Update XMP sidecars with thumbnail info
update_sidecars = true

# Create XMP if it doesn't exist
create_if_missing = false

# Namespace URI for shoemaker tags
namespace = "http://shoemaker.local/1.0/"

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------

[logging]
# Log level: "error" | "warn" | "info" | "debug"
level = "info"

# Log file (empty = stderr only)
file = ""

# Show progress bar for batch operations
progress = true
```

### Zod Schema (TypeScript)

```typescript
// src/schemas/config.ts
import { z } from 'zod';

const SizeConfigSchema = z.object({
  width: z.number().int().positive(),
  height: z.number().int().positive().optional(),
  format: z.enum(['webp', 'jpeg', 'png', 'avif']),
  quality: z.number().int().min(1).max(100),
  allow_upscale: z.boolean().default(false),
  crop: z.enum(['center', 'face', 'smart']).optional(),
});

const ConfigSchema = z.object({
  default_preset: z.string().default('fast'),
  preset_dir: z.string().default('~/.config/shoemaker/presets'),

  output: z.object({
    location: z.enum(['sidecar', 'cache']).default('sidecar'),
    sidecar_folder: z.string().default('{stem}_thumbs'),
    cache_dir: z.string().default('~/.cache/shoemaker'),
    naming_pattern: z.string().default('{stem}_{size}_{width}.{format}'),
  }),

  processing: z.object({
    concurrency: z.number().int().min(1).max(32).default(4),
    min_preview_size: z.number().int().positive().default(2560),
    skip_existing: z.boolean().default(true),
    auto_rotate: z.boolean().default(true),
    strip_exif: z.boolean().default(true),
  }),

  sizes: z.record(z.string(), SizeConfigSchema).default({
    thumb: { width: 300, format: 'webp', quality: 80 },
    preview: { width: 1600, format: 'webp', quality: 85 },
    ml: { width: 2560, format: 'jpeg', quality: 90 },
  }),

  filetypes: z.object({
    include: z.array(z.string()).default([
      'arw', 'cr2', 'cr3', 'nef', 'raf', 'rw2', 'dng',
      'jpg', 'jpeg', 'png', 'tif', 'tiff', 'heic',
    ]),
    exclude: z.array(z.string()).default(['xmp', 'json']),
    no_preview_action: z.enum(['decode', 'skip', 'warn']).default('decode'),
  }),

  xmp: z.object({
    update_sidecars: z.boolean().default(true),
    create_if_missing: z.boolean().default(false),
    namespace: z.string().default('http://shoemaker.local/1.0/'),
  }),

  logging: z.object({
    level: z.enum(['error', 'warn', 'info', 'debug']).default('info'),
    file: z.string().default(''),
    progress: z.boolean().default(true),
  }),
});

export type SizeConfig = z.infer<typeof SizeConfigSchema>;
export type Config = z.infer<typeof ConfigSchema>;
```

---

## Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `sharp` | ^0.33.x | Image resize, format conversion, color management (libvips) |
| `exiftool-vendored` | ^28.x | Extract embedded previews, read/write XMP |
| `commander` | ^12.x | CLI framework (matches wake-n-blake) |
| `zod` | ^3.24.x | Schema validation |
| `ora` | ^8.x | Terminal spinners |
| `p-queue` | ^8.x | Concurrency control |

### Optional Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `@aspect-img/libraw` | ^0.3.x | RAW decode fallback (WASM-based, no native deps) |

### Why These Choices

1. **sharp** — Fastest Node.js image processor, uses libvips under the hood, excellent color management
2. **exiftool-vendored** — Bundled ExifTool binary, works on all platforms, handles all RAW formats
3. **@aspect-img/libraw** — WASM build of libraw, no native compilation needed, works in CI

---

## External Tool Integrations

### RAW Decoders

Shoemaker supports multiple RAW decode backends. Configure via `decoder` setting in presets.

| Tool | Speed | Quality | Portability | Best For |
|------|-------|---------|-------------|----------|
| **Embedded preview** | ⚡ Instant | Camera JPEG | Built-in | Fast import (default) |
| **libraw (WASM)** | 🐢 Slow | Good | npm install | CI/CD, no system deps |
| **RawTherapee CLI** | 🐢 Slow | Excellent | System install | High-quality output |
| **darktable-cli** | 🐢 Slow | Excellent | System install | Alternative to RT |
| **dcraw** | 🚀 Fast | Basic | System install | Lightweight, legacy |
| **libvips (direct)** | 🚀 Fast | Good | System install | Already have via sharp |

#### RawTherapee CLI

```bash
# Install
brew install rawtherapee        # macOS
apt install rawtherapee         # Ubuntu/Debian
choco install rawtherapee       # Windows

# Basic usage
rawtherapee-cli -o output.jpg -p neutral.pp3 -j90 -Y input.arw

# Options
# -o  output file
# -p  processing profile (.pp3)
# -j  JPEG quality (1-100)
# -Y  overwrite existing
# -b8 8-bit output (default)
# -t  TIFF output instead of JPEG
# -c  use system camera profiles
```

```typescript
// src/core/decoders/rawtherapee.ts
import { execFile } from 'child_process';
import { promisify } from 'util';

const execFileAsync = promisify(execFile);

export async function decodeWithRawTherapee(
  inputPath: string,
  outputPath: string,
  options: { profile?: string; quality?: number } = {}
): Promise<void> {
  const args = [
    '-o', outputPath,
    '-j', String(options.quality ?? 95),
    '-Y',  // overwrite
    '-c',  // use camera profiles
  ];

  if (options.profile) {
    args.push('-p', options.profile);
  }

  args.push(inputPath);

  await execFileAsync('rawtherapee-cli', args);
}
```

#### darktable-cli

```bash
# Install
brew install darktable          # macOS
apt install darktable           # Ubuntu/Debian

# Basic usage
darktable-cli input.arw output.jpg --width 2560 --height 2560

# With style/preset
darktable-cli input.arw output.jpg --style "neutral" --hq true
```

```typescript
// src/core/decoders/darktable.ts
export async function decodeWithDarktable(
  inputPath: string,
  outputPath: string,
  options: { width?: number; style?: string; hq?: boolean } = {}
): Promise<void> {
  const args = [inputPath, outputPath];

  if (options.width) {
    args.push('--width', String(options.width));
    args.push('--height', String(options.width)); // square bounding box
  }

  if (options.style) {
    args.push('--style', options.style);
  }

  if (options.hq) {
    args.push('--hq', 'true');
  }

  await execFileAsync('darktable-cli', args);
}
```

#### dcraw (Lightweight)

```bash
# Install
brew install dcraw              # macOS
apt install dcraw               # Ubuntu/Debian

# Basic usage (outputs PPM, pipe to convert)
dcraw -c -w -W input.arw | convert - output.jpg

# Options
# -c  output to stdout
# -w  use camera white balance
# -W  no auto-brighten
# -h  half-size (faster)
# -q 3  high-quality interpolation
```

### Metadata Tools

| Tool | Speed | Features | Portability |
|------|-------|----------|-------------|
| **exiftool** | Moderate | Everything | Perl (bundled via exiftool-vendored) |
| **exiv2** | Fast | Common tags | C++ binary |
| **libexif** | Fast | Basic EXIF | C library |

#### exiv2 (Fast Alternative)

```bash
# Install
brew install exiv2
apt install exiv2

# Extract preview (faster than exiftool for this)
exiv2 -ep3 image.arw           # Extract preview 3 (usually largest)
exiv2 -pp image.arw            # List all previews

# Read specific tags
exiv2 -g Exif.Image.Make image.arw
```

Use exiv2 when you only need preview extraction (faster), exiftool for XMP writing.

### Image Processing

| Tool | Speed | Via | Best For |
|------|-------|-----|----------|
| **sharp** | ⚡ Fastest | npm (libvips) | Primary processor |
| **libvips CLI** | ⚡ Fast | System | Direct vips commands |
| **ImageMagick** | 🐢 Slow | System | Complex operations |
| **GraphicsMagick** | 🚀 Moderate | System | Faster ImageMagick |

#### libvips CLI (Direct)

```bash
# Install (usually comes with sharp, but for CLI)
brew install vips
apt install libvips-tools

# Thumbnail with smart crop
vipsthumbnail input.jpg -o output_%s.webp[Q=85] --size 300x300 --smartcrop attention

# Resize preserving aspect
vipsthumbnail input.jpg -o output.webp --size 2560

# Options
# --smartcrop attention|centre|entropy|low|high
# --linear  process in linear light
# --no-rotate  ignore EXIF rotation
```

### Color Management

| Tool | Purpose | Integration |
|------|---------|-------------|
| **LittleCMS (lcms2)** | ICC profile handling | Built into sharp/libvips |
| **ArgyllCMS** | Profile creation, conversion | CLI tools |

Sharp handles sRGB conversion automatically via lcms2. No extra integration needed.

### Format Support

| Format | Library | Sharp Support | Notes |
|--------|---------|---------------|-------|
| **WebP** | libwebp | ✅ Built-in | Primary output format |
| **AVIF** | libheif/aom | ✅ Built-in | Requires libvips 8.12+ |
| **HEIC** | libheif | ⚠️ Varies | Needs libheif on Linux |
| **JPEG XL** | libjxl | ⚠️ Experimental | Future format |
| **TIFF** | libtiff | ✅ Built-in | Large file support |

#### HEIC on Linux

```bash
# Ubuntu/Debian - install libheif for HEIC support
apt install libheif-dev

# Rebuild sharp to pick up libheif
npm rebuild sharp
```

### Optional Future Integrations

| Integration | Purpose | When to Add |
|-------------|---------|-------------|
| **MediaPipe** | Face detection | Smart cropping phase |
| **ONNX Runtime** | ML model inference | Advanced features |
| **Thumbor** | Image server | Cloud deployment |
| **imgproxy** | Fast image proxy | Self-hosted CDN |

### Decoder Selection Logic

```typescript
// src/core/decoder-factory.ts
type DecoderType = 'embedded' | 'libraw' | 'rawtherapee' | 'darktable' | 'dcraw' | 'vips';

interface DecoderConfig {
  type: DecoderType;
  profile?: string;      // RT/darktable profile
  quality?: number;
  available: boolean;    // Detected at startup
}

const DECODER_PRIORITY: DecoderType[] = [
  'embedded',      // Always try first
  'rawtherapee',   // Best quality
  'darktable',     // Alternative
  'vips',          // If installed with RAW support
  'libraw',        // Portable fallback
  'dcraw',         // Last resort
];

export async function detectAvailableDecoders(): Promise<Map<DecoderType, boolean>> {
  const available = new Map<DecoderType, boolean>();

  available.set('embedded', true);  // Always available
  available.set('libraw', true);    // WASM, always available

  // Check system tools
  available.set('rawtherapee', await commandExists('rawtherapee-cli'));
  available.set('darktable', await commandExists('darktable-cli'));
  available.set('dcraw', await commandExists('dcraw'));
  available.set('vips', await commandExists('vips'));

  return available;
}

export function selectDecoder(
  preset: PresetConfig,
  available: Map<DecoderType, boolean>
): DecoderType {
  // If preset specifies decoder, use if available
  if (preset.behavior.decoder && available.get(preset.behavior.decoder)) {
    return preset.behavior.decoder;
  }

  // Otherwise use priority list
  for (const decoder of DECODER_PRIORITY) {
    if (available.get(decoder)) {
      return decoder;
    }
  }

  return 'libraw'; // Ultimate fallback
}
```

### Preset Examples with Decoders

```toml
# presets/fast-import.toml
[behavior]
decoder = "embedded"          # Just extract, never decode
fallback_decoder = "libraw"   # If no embedded preview

# presets/high-quality.toml
[behavior]
decoder = "rawtherapee"
profile = "neutral"           # ~/.config/RawTherapee/profiles/neutral.pp3
fallback_decoder = "darktable"

# presets/portable.toml (CI/CD friendly)
[behavior]
decoder = "libraw"            # WASM, no system deps
fallback_decoder = "embedded"
```

---

## Performance Characteristics

### Expected Speeds (M1 Mac)

| Operation | Time | Notes |
|-----------|------|-------|
| Extract + resize (fast path) | 0.2-0.4s | Most common case |
| Direct resize (TIFF/JPEG) | 0.3-0.5s | Already decoded |
| RAW decode + resize | 1.5-3.0s | Full demosaic |

### Concurrency

```typescript
// Process 4 files at a time (tunable)
const queue = new PQueue({ concurrency: 4 });

for (const file of files) {
  queue.add(() => generateThumbnails(file, outputDir, config));
}
```

### Memory Usage

- sharp streams data, doesn't load full image into memory
- Peak usage ~200-300MB for 50MP RAW files
- Suitable for batch processing 1000s of files

---

## Integration Points

### With wake-n-blake

Shoemaker reads XMP sidecars created by wake-n-blake:

```typescript
// Check if file was already processed by wake-n-blake
const xmp = await readXmpSidecar(imagePath);
if (!xmp.blake3Hash) {
  console.warn('File not verified by wake-n-blake, skipping');
  continue;
}
```

### With visual-buffet

Visual-buffet can use the ML thumbnails:

```bash
# Instead of processing full RAW files
visual-buffet tag /photos/import/*_ml_2560.jpg --plugin ram_plus
```

### With abandoned-archive / nightfoxfilms

Desktop apps can import shoemaker-generated thumbnails:

```typescript
// In Electron main process
const xmp = await readXmpSidecar(imagePath);
const thumbPath = xmp.shoemaker?.thumbnails?.find(t => t.size === 'thumb')?.path;
const previewPath = xmp.shoemaker?.thumbnails?.find(t => t.size === 'preview')?.path;
```

---

## Implementation Phases

### Phase 1: Core Pipeline (Week 1)

1. Set up project structure (TypeScript, Commander.js)
2. Implement `extractor.ts` — Extract embedded previews via exiftool
3. Implement `resizer.ts` — Resize + sRGB convert via sharp
4. Basic `thumb` command for single files
5. Tests for Sony/Canon/Nikon RAW files

### Phase 2: XMP Integration (Week 2)

1. Implement `xmp-updater.ts` — Read/write shoemaker namespace
2. Implement `preview-analyzer.ts` — Report embedded preview info
3. Add `info` command
4. Add `status` command (show what needs processing)

### Phase 3: Batch Processing (Week 3)

1. Directory scanning with concurrency control
2. Progress reporting (ora spinners, summary stats)
3. `--recursive` flag
4. Resume support (skip already-processed files)

### Phase 4: RAW Fallback (Week 4)

1. Integrate @aspect-img/libraw for RAW decode
2. Implement high-quality preset
3. Add `--force` regeneration
4. Add `clean` command

### Phase 5: Polish (Week 5)

1. Configuration file support (~/.config/shoemaker/config.toml)
2. Custom presets directory
3. Shell completions
4. npm publish

---

## Error Handling

### Error Categories

| Category | Example | Recovery Strategy |
|----------|---------|-------------------|
| **File Not Found** | Source file deleted | Log, skip, continue batch |
| **Permission Denied** | Read-only filesystem | Log, skip, continue batch |
| **Corrupt File** | Truncated RAW | Log, skip, continue batch |
| **No Preview** | TIFF with no embedded | Fallback to decode or skip |
| **Decode Failure** | Unsupported RAW format | Log, skip, continue batch |
| **Disk Full** | ENOSPC | Stop batch, report progress |
| **Memory Exhausted** | 100MP+ file | Reduce concurrency, retry |
| **ExifTool Crash** | Malformed metadata | Restart exiftool, retry once |

### Implementation

```typescript
// src/core/errors.ts
export class ShoemakerError extends Error {
  constructor(
    message: string,
    public readonly code: ErrorCode,
    public readonly filePath: string,
    public readonly recoverable: boolean = true
  ) {
    super(message);
    this.name = 'ShoemakerError';
  }
}

export enum ErrorCode {
  FILE_NOT_FOUND = 'FILE_NOT_FOUND',
  PERMISSION_DENIED = 'PERMISSION_DENIED',
  CORRUPT_FILE = 'CORRUPT_FILE',
  NO_PREVIEW = 'NO_PREVIEW',
  DECODE_FAILED = 'DECODE_FAILED',
  DISK_FULL = 'DISK_FULL',
  OUT_OF_MEMORY = 'OUT_OF_MEMORY',
  EXIFTOOL_ERROR = 'EXIFTOOL_ERROR',
  SHARP_ERROR = 'SHARP_ERROR',
  XMP_WRITE_FAILED = 'XMP_WRITE_FAILED',
}

// Batch processing with error handling
export async function processBatch(
  files: string[],
  config: Config,
  onProgress: (result: BatchProgress) => void
): Promise<BatchResult> {
  const queue = new PQueue({ concurrency: config.processing.concurrency });
  const results: FileResult[] = [];
  const errors: ShoemakerError[] = [];

  for (const file of files) {
    queue.add(async () => {
      try {
        const result = await generateThumbnails(file, config);
        results.push({ file, status: 'success', result });
        onProgress({ completed: results.length, total: files.length, current: file });
      } catch (err) {
        const shoemakerError = wrapError(err, file);

        if (shoemakerError.code === ErrorCode.DISK_FULL) {
          // Non-recoverable: stop entire batch
          queue.clear();
          throw shoemakerError;
        }

        if (shoemakerError.code === ErrorCode.OUT_OF_MEMORY) {
          // Reduce concurrency and retry
          queue.concurrency = Math.max(1, queue.concurrency - 1);
          console.warn(`Reduced concurrency to ${queue.concurrency} due to memory pressure`);
          // Re-queue this file
          queue.add(() => processSingleFile(file, config));
          return;
        }

        // Recoverable error: log and continue
        errors.push(shoemakerError);
        results.push({ file, status: 'error', error: shoemakerError });
        onProgress({ completed: results.length, total: files.length, current: file });
      }
    });
  }

  await queue.onIdle();

  return {
    total: files.length,
    succeeded: results.filter(r => r.status === 'success').length,
    failed: results.filter(r => r.status === 'error').length,
    errors,
  };
}

// Wrap unknown errors into ShoemakerError
function wrapError(err: unknown, filePath: string): ShoemakerError {
  if (err instanceof ShoemakerError) return err;

  const message = err instanceof Error ? err.message : String(err);

  // Detect specific error types
  if (message.includes('ENOENT')) {
    return new ShoemakerError('File not found', ErrorCode.FILE_NOT_FOUND, filePath);
  }
  if (message.includes('EACCES') || message.includes('EPERM')) {
    return new ShoemakerError('Permission denied', ErrorCode.PERMISSION_DENIED, filePath);
  }
  if (message.includes('ENOSPC')) {
    return new ShoemakerError('Disk full', ErrorCode.DISK_FULL, filePath, false);
  }
  if (message.includes('ENOMEM') || message.includes('heap')) {
    return new ShoemakerError('Out of memory', ErrorCode.OUT_OF_MEMORY, filePath);
  }
  if (message.includes('Invalid image') || message.includes('corrupt')) {
    return new ShoemakerError('Corrupt file', ErrorCode.CORRUPT_FILE, filePath);
  }

  // Generic error
  return new ShoemakerError(message, ErrorCode.DECODE_FAILED, filePath);
}
```

### CLI Error Output

```
$ shoemaker thumb /photos/import/ --preset fast

Processing 127 files...

  ✓ _DSC0008.ARW     extracted → 3 thumbs (0.2s)
  ✓ _MG_5043.CR2     extracted → 3 thumbs (0.3s)
  ✗ corrupt.arw      CORRUPT_FILE: Invalid image format
  ✓ img012.tif       direct → 3 thumbs (0.4s)
  ⚠ DJI_0412.DNG     NO_PREVIEW: Preview too small, skipped ML size

Done: 127 files
  ✓ Succeeded: 124 (97.6%)
  ✗ Failed: 2 (1.6%)
  ⚠ Warnings: 1 (0.8%)

Errors written to: /photos/import/.shoemaker-errors.json
```

### Error Log Format

```json
{
  "timestamp": "2025-12-23T10:30:00Z",
  "version": "0.1.0",
  "batch": {
    "total": 127,
    "succeeded": 124,
    "failed": 2,
    "warnings": 1
  },
  "errors": [
    {
      "file": "/photos/import/corrupt.arw",
      "code": "CORRUPT_FILE",
      "message": "Invalid image format: unexpected EOF",
      "timestamp": "2025-12-23T10:30:15Z"
    }
  ],
  "warnings": [
    {
      "file": "/photos/import/DJI_0412.DNG",
      "code": "NO_PREVIEW",
      "message": "Preview size 960x720 below minimum 2560, skipped ML size",
      "timestamp": "2025-12-23T10:30:20Z"
    }
  ]
}
```

### Resume Support

When a batch is interrupted (Ctrl+C, disk full, crash), Shoemaker can resume:

```bash
# Check what still needs processing
shoemaker status /photos/import/

# Resume (skips files with thumbnails in XMP)
shoemaker thumb /photos/import/ --resume

# Force regenerate everything
shoemaker thumb /photos/import/ --force
```

Implementation checks XMP sidecars for `shoemaker:ThumbnailsGenerated` before processing.

---

## CI/CD Setup

### GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        node: [20, 22]

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node }}
          cache: 'npm'

      # Install system dependencies (Linux)
      - name: Install system deps (Linux)
        if: runner.os == 'Linux'
        run: |
          sudo apt-get update
          sudo apt-get install -y \
            libvips-dev \
            libheif-dev \
            exiftool

      # Install system dependencies (macOS)
      - name: Install system deps (macOS)
        if: runner.os == 'macOS'
        run: |
          brew install vips libheif exiftool

      # Windows uses bundled binaries
      - name: Install dependencies
        run: npm ci

      - name: Run tests
        run: npm test

      - name: Run integration tests
        run: npm run test:integration

  test-with-rawtherapee:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: 'npm'

      - name: Install RawTherapee
        run: |
          sudo add-apt-repository ppa:dhor/myway -y
          sudo apt-get update
          sudo apt-get install -y rawtherapee libvips-dev exiftool

      - name: Verify RawTherapee
        run: rawtherapee-cli --version

      - name: Install dependencies
        run: npm ci

      - name: Run high-quality preset tests
        run: npm run test:rawtherapee

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: 'npm'
      - run: npm ci
      - run: npm run lint
      - run: npm run typecheck

  build:
    runs-on: ubuntu-latest
    needs: [test, lint]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: 'npm'
      - run: npm ci
      - run: npm run build
      - name: Upload build artifacts
        uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/
```

### Test File Caching

```yaml
# Cache test RAW files to avoid re-downloading
- name: Cache test files
  uses: actions/cache@v4
  with:
    path: tests/fixtures/raw
    key: test-raw-files-v1
    restore-keys: |
      test-raw-files-

- name: Download test files
  run: npm run test:download-fixtures
```

### Docker for Consistent Testing

```dockerfile
# Dockerfile.test
FROM node:20-slim

# Install all optional decoders for comprehensive testing
RUN apt-get update && apt-get install -y \
    libvips-dev \
    libheif-dev \
    rawtherapee \
    darktable \
    dcraw \
    exiftool \
    exiv2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .

CMD ["npm", "test"]
```

```yaml
# docker-compose.test.yml
version: '3.8'
services:
  test:
    build:
      context: .
      dockerfile: Dockerfile.test
    volumes:
      - ./tests/fixtures:/app/tests/fixtures
```

### Release Workflow

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  publish:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: 20
          registry-url: 'https://registry.npmjs.org'

      - run: npm ci
      - run: npm run build
      - run: npm test

      - name: Publish to npm
        run: npm publish --provenance --access public
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}
```

---

## Test File Sources

### RAW Sample Files

| Source | URL | Cameras | License |
|--------|-----|---------|---------|
| **raw.pixls.us** | https://raw.pixls.us/ | 1000+ cameras | CC0 |
| **rawsamples.ch** | https://www.rawsamples.ch/ | 400+ cameras | Free for testing |
| **DPReview samples** | https://www.dpreview.com/sample-galleries | Major brands | Editorial use |
| **imaging-resource** | https://www.imaging-resource.com/ | Comparometer samples | Editorial use |

### Recommended Test Suite

```typescript
// tests/fixtures/manifest.ts
export const TEST_FILES = {
  // Core formats - one per major brand
  sony_arw: {
    url: 'https://raw.pixls.us/data/Sony/ILCE-7M3/_DSC1234.ARW',
    sha256: '...',
    expectedPreviews: ['JpgFromRaw', 'PreviewImage', 'ThumbnailImage'],
    jpgFromRawSize: [4240, 2832],
  },
  canon_cr2: {
    url: 'https://raw.pixls.us/data/Canon/EOS%205D%20Mark%20IV/IMG_1234.CR2',
    sha256: '...',
    expectedPreviews: ['PreviewImage', 'ThumbnailImage'],
    previewSize: [5472, 3648],
  },
  canon_cr3: {
    url: 'https://raw.pixls.us/data/Canon/EOS%20R5/IMG_1234.CR3',
    sha256: '...',
    expectedPreviews: ['PreviewImage'],
    previewSize: [8192, 5464],
  },
  nikon_nef: {
    url: 'https://raw.pixls.us/data/Nikon/D850/DSC_1234.NEF',
    sha256: '...',
    expectedPreviews: ['JpgFromRaw', 'OtherImage', 'PreviewImage'],
    jpgFromRawSize: [8256, 5504],
  },
  fuji_raf: {
    url: 'https://raw.pixls.us/data/Fujifilm/X-T4/DSCF1234.RAF',
    sha256: '...',
    expectedPreviews: ['JpgFromRaw'],
    jpgFromRawSize: [6240, 4160],
  },
  dji_dng: {
    url: 'https://raw.pixls.us/data/DJI/FC3170/DJI_1234.DNG',
    sha256: '...',
    expectedPreviews: ['PreviewImage'],  // Small preview only!
    previewSize: [960, 720],
    needsRawDecode: true,  // Embedded preview insufficient
  },

  // Edge cases
  tiff_no_preview: {
    url: 'https://example.com/test.tif',
    sha256: '...',
    expectedPreviews: [],
    needsRawDecode: true,
  },
  heic_iphone: {
    url: 'https://example.com/IMG_1234.HEIC',
    sha256: '...',
    expectedPreviews: ['PreviewImage'],
  },

  // Problematic files
  corrupt_header: {
    url: 'https://example.com/corrupt.arw',
    sha256: '...',
    shouldFail: true,
    expectedError: 'CORRUPT_FILE',
  },
  truncated_file: {
    url: 'https://example.com/truncated.cr2',
    sha256: '...',
    shouldFail: true,
    expectedError: 'CORRUPT_FILE',
  },
};
```

### Download Script

```typescript
// scripts/download-test-fixtures.ts
import { createWriteStream } from 'fs';
import { mkdir } from 'fs/promises';
import { pipeline } from 'stream/promises';
import { createHash } from 'crypto';
import { TEST_FILES } from '../tests/fixtures/manifest';

const FIXTURES_DIR = 'tests/fixtures/raw';

async function downloadFile(name: string, config: typeof TEST_FILES[keyof typeof TEST_FILES]) {
  const filePath = `${FIXTURES_DIR}/${name}${getExtension(config.url)}`;

  // Check if already exists with correct hash
  if (await fileExists(filePath)) {
    const hash = await hashFile(filePath);
    if (hash === config.sha256) {
      console.log(`✓ ${name} (cached)`);
      return;
    }
  }

  console.log(`↓ Downloading ${name}...`);

  const response = await fetch(config.url);
  if (!response.ok) {
    throw new Error(`Failed to download ${name}: ${response.statusText}`);
  }

  await mkdir(FIXTURES_DIR, { recursive: true });
  await pipeline(response.body!, createWriteStream(filePath));

  // Verify hash
  const hash = await hashFile(filePath);
  if (hash !== config.sha256) {
    throw new Error(`Hash mismatch for ${name}: expected ${config.sha256}, got ${hash}`);
  }

  console.log(`✓ ${name}`);
}

async function main() {
  console.log('Downloading test fixtures...\n');

  for (const [name, config] of Object.entries(TEST_FILES)) {
    try {
      await downloadFile(name, config);
    } catch (err) {
      console.error(`✗ ${name}: ${err.message}`);
    }
  }

  console.log('\nDone.');
}

main();
```

### Test Examples

```typescript
// tests/extractor.test.ts
import { describe, it, expect, beforeAll } from 'vitest';
import { analyzeEmbeddedPreviews, extractBestPreview } from '../src/core/extractor';
import { TEST_FILES } from './fixtures/manifest';

describe('Embedded Preview Extraction', () => {
  beforeAll(async () => {
    // Ensure fixtures are downloaded
    await import('../scripts/download-test-fixtures');
  });

  it('should find JpgFromRaw in Sony ARW', async () => {
    const analysis = await analyzeEmbeddedPreviews('tests/fixtures/raw/sony_arw.ARW');

    expect(analysis.jpgFromRaw.exists).toBe(true);
    expect(analysis.jpgFromRaw.width).toBe(4240);
    expect(analysis.jpgFromRaw.height).toBe(2832);
  });

  it('should extract largest preview from Canon CR2', async () => {
    const buffer = await extractBestPreview('tests/fixtures/raw/canon_cr2.CR2');

    expect(buffer.length).toBeGreaterThan(1_000_000); // At least 1MB
    expect(buffer.slice(0, 2).toString('hex')).toBe('ffd8'); // JPEG magic
  });

  it('should detect insufficient preview in DJI DNG', async () => {
    const analysis = await analyzeEmbeddedPreviews('tests/fixtures/raw/dji_dng.DNG');

    expect(analysis.bestPreview?.width).toBe(960);
    expect(analysis.needsRawDecode).toBe(true);
  });

  it('should handle corrupt file gracefully', async () => {
    await expect(
      analyzeEmbeddedPreviews('tests/fixtures/raw/corrupt_header.arw')
    ).rejects.toThrow('CORRUPT_FILE');
  });
});
```

```typescript
// tests/thumbnail-generator.test.ts
import { describe, it, expect } from 'vitest';
import { generateThumbnails } from '../src/services/thumbnail-generator';
import sharp from 'sharp';

describe('Thumbnail Generation', () => {
  it('should generate all three sizes from Sony ARW', async () => {
    const result = await generateThumbnails(
      'tests/fixtures/raw/sony_arw.ARW',
      'tests/output',
      { preset: 'fast' }
    );

    expect(result.method).toBe('extracted');
    expect(result.thumbnails).toHaveLength(3);

    // Verify thumb
    const thumbMeta = await sharp('tests/output/sony_arw_thumb_300.webp').metadata();
    expect(thumbMeta.width).toBeLessThanOrEqual(300);
    expect(thumbMeta.format).toBe('webp');

    // Verify ML size
    const mlMeta = await sharp('tests/output/sony_arw_ml_2560.jpg').metadata();
    expect(mlMeta.width).toBeLessThanOrEqual(2560);
    expect(mlMeta.format).toBe('jpeg');
    expect(mlMeta.space).toBe('srgb');
  });

  it('should fall back to RAW decode for DJI DNG', async () => {
    const result = await generateThumbnails(
      'tests/fixtures/raw/dji_dng.DNG',
      'tests/output',
      { preset: 'quality', fallbackToRaw: true }
    );

    expect(result.method).toBe('decoded');
  });

  it('should skip ML size when preview too small (fast preset)', async () => {
    const result = await generateThumbnails(
      'tests/fixtures/raw/dji_dng.DNG',
      'tests/output',
      { preset: 'fast', fallbackToRaw: false }
    );

    expect(result.thumbnails).toHaveLength(2); // Only thumb and preview
    expect(result.warnings).toContain('ML size skipped: source too small');
  });
});
```

### Visual Regression Testing

```typescript
// tests/visual-regression.test.ts
import { describe, it, expect } from 'vitest';
import { compare } from 'resemblejs';
import sharp from 'sharp';

describe('Visual Regression', () => {
  it('should produce consistent output for Sony ARW', async () => {
    await generateThumbnails('tests/fixtures/raw/sony_arw.ARW', 'tests/output', {
      preset: 'fast',
    });

    const diff = await compare(
      'tests/output/sony_arw_thumb_300.webp',
      'tests/baselines/sony_arw_thumb_300.webp'
    );

    // Allow 0.1% difference for minor encoder variations
    expect(diff.rawMisMatchPercentage).toBeLessThan(0.1);
  });
});
```

### Gitignore for Test Files

```gitignore
# .gitignore
tests/fixtures/raw/          # Downloaded RAW files (large)
tests/output/                # Generated test outputs
!tests/fixtures/manifest.ts  # Keep the manifest
!tests/baselines/            # Keep visual baselines
```

---

## Limitations & Uncertainties

### What This Document Does NOT Cover

- Video thumbnail generation (separate tool, ffmpeg-based)
- Smart cropping (deferred to future version)
- Cloud storage integration
- Database storage (thumbnails go to filesystem)
- GUI interface

### Technical Uncertainties

1. **libraw WASM performance** — May be significantly slower than native; need benchmarks
2. **XMP namespace registration** — Custom namespace may cause issues with some tools
3. **HEIC support** — sharp's HEIC support varies by platform (requires libheif)

### Knowledge Gaps

- Optimal concurrency for different storage types (SSD vs NAS)
- Memory limits for very large files (100MP+ medium format)

---

## Source Appendix

| # | Source | Date | Type | Used For |
|---|--------|------|------|----------|
| 1 | [sharp documentation](https://sharp.pixelplumbing.com/) | 2025-12-23 | Primary | Image processing API |
| 2 | [exiftool-vendored](https://github.com/photostructure/exiftool-vendored.js) | 2025-12-23 | Primary | Embedded preview extraction |
| 3 | [libraw WASM](https://github.com/nicolo-ribaudo/aspect-img-libraw) | 2025-12-23 | Primary | RAW decode fallback |
| 4 | wake-n-blake codebase analysis | 2025-12-23 | Internal | XMP sidecar patterns |
| 5 | visual-buffet codebase analysis | 2025-12-23 | Internal | Thumbnail usage patterns |
| 6 | User research session | 2025-12-23 | Primary | Embedded preview analysis, color space findings |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-23 | Initial version (wrong scope - consolidated CLI) |
| 2.0 | 2025-12-23 | Rewritten for thumbnail-focused Shoemaker CLI |
| 2.1 | 2025-12-23 | Added: ExifTool commands reference, config file schema (TOML + Zod), error handling with recovery strategies |
| 2.2 | 2025-12-23 | Added: External tool integrations (RawTherapee, darktable, dcraw, exiv2, libvips CLI), decoder selection logic, `shoemaker doctor` command |
| 2.3 | 2025-12-23 | Added: CI/CD setup (GitHub Actions, Docker, release workflow), test file sources (raw.pixls.us), test fixture manifest, download script, visual regression testing |
