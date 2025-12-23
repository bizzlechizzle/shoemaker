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
```

## Project Structure

```
shoemaker/
├── src/
│   ├── index.ts              # Library entry point (exports)
│   ├── bin/
│   │   └── shoemaker.ts      # CLI entry point
│   ├── cli/
│   │   ├── index.ts          # CLI setup (Commander.js)
│   │   └── commands/         # CLI command implementations
│   ├── core/
│   │   ├── config.ts         # Configuration loading (TOML)
│   │   ├── errors.ts         # Error classes and handling
│   │   ├── extractor.ts      # ExifTool preview extraction
│   │   └── resizer.ts        # Sharp resize + color management
│   ├── services/
│   │   ├── thumbnail-generator.ts  # Main pipeline
│   │   └── xmp-updater.ts    # XMP sidecar updates
│   └── schemas/
│       └── index.ts          # Zod schemas
├── presets/                  # TOML preset files
├── tests/
│   ├── unit/                 # Unit tests
│   └── integration/          # Integration tests
├── sme/                      # SME documentation
├── package.json
├── tsconfig.json
└── vitest.config.ts
```

## Architecture

### Pipeline Flow

```
Input File
    │
    ├─ isDecodedFormat? ──→ Read directly
    │
    └─ isRawFormat? ──→ analyzeEmbeddedPreviews()
                            │
                            ├─ bestPreview >= minSize? ──→ extractBestPreview()
                            │
                            └─ fallbackToRaw? ──→ decodeRawFile()
                                                        │
                                                        └─ source buffer
                                                              │
                                                              ▼
                                                    resizeThumbnails()
                                                              │
                                                              ▼
                                                    updateXmpSidecar()
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `extractor.ts` | ExifTool wrapper for preview extraction |
| `resizer.ts` | Sharp wrapper for resize + sRGB conversion |
| `config.ts` | TOML config loading and preset management |
| `errors.ts` | Error classes with recovery strategies |
| `thumbnail-generator.ts` | Main orchestration service |
| `xmp-updater.ts` | XMP sidecar read/write |

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

## Testing

### Unit Tests

Test individual functions in isolation:

```typescript
// tests/unit/extractor.test.ts
import { isRawFormat, isDecodedFormat } from '../../src/core/extractor';

describe('extractor', () => {
  it('identifies RAW formats', () => {
    expect(isRawFormat('image.arw')).toBe(true);
    expect(isRawFormat('image.ARW')).toBe(true);
    expect(isRawFormat('image.jpg')).toBe(false);
  });
});
```

### Integration Tests

Test with real files (download fixtures first):

```bash
npm run test:download-fixtures
npm run test:integration
```

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

### Development

| Package | Purpose |
|---------|---------|
| typescript | Type checking |
| vitest | Test runner |
| eslint | Linting |
| tsx | TypeScript execution |

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
```

### "No embedded preview found"

The RAW file doesn't have embedded previews, or they're corrupted. Options:
1. Use `--preset quality` to decode the RAW file
2. Check if the file opens in other software
3. Try `exiftool -preview:all image.arw` to see what's embedded
