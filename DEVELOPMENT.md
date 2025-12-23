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
┌─────────────────────────────────────────────────────────┐
│                         CLI                              │
│  (Commander.js commands: thumb, info, clean, status)    │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                      Services                            │
│  (thumbnail-generator.ts, xmp-updater.ts)               │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                        Core                              │
│  (extractor.ts, resizer.ts, config.ts, errors.ts)       │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                    External Tools                        │
│  (ExifTool, Sharp/libvips, RawTherapee, darktable)      │
└─────────────────────────────────────────────────────────┘
```

### Module Responsibilities

| Module | File | Responsibility |
|--------|------|----------------|
| **Extractor** | `src/core/extractor.ts` | Extract embedded previews via ExifTool |
| **Resizer** | `src/core/resizer.ts` | Resize images via Sharp, convert to sRGB |
| **Config** | `src/core/config.ts` | Load TOML config and presets |
| **Errors** | `src/core/errors.ts` | Error classes and recovery logic |
| **Generator** | `src/services/thumbnail-generator.ts` | Main pipeline orchestration |
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

## Testing

### Unit Tests

```bash
npm test                 # Run all unit tests
npm run test:watch       # Watch mode
npm test -- --coverage   # With coverage
```

### Integration Tests

Integration tests require real RAW files:

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
git tag v0.1.0
git push --tags
```

GitHub Actions will publish to npm automatically.

## Debugging

### Debug ExifTool

```bash
exiftool -a -G1 "*Preview*" "*Jpg*" image.arw
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
- **Constants**: `DEFAULT_MIN_PREVIEW_SIZE`, `RAW_EXTENSIONS`, `MAX_CONCURRENCY`, etc.
- **Zod Schemas**: Validation for config, presets, and results
- **Type Exports**: All TypeScript types used across the codebase

When adding new constants or types, add them here to maintain single source of truth.

### core/errors.ts

Error handling strategy:
- `ShoemakerError` class with `ErrorCode` enum
- `wrapError()` for consistent error wrapping
- `isRecoverable()` for batch processing decisions
- `shouldStopBatch()` and `shouldReduceConcurrency()` for flow control

Error codes:
| Code | Meaning | Recoverable |
|------|---------|-------------|
| `FILE_NOT_FOUND` | File doesn't exist | No |
| `PERMISSION_DENIED` | Can't read/write | No |
| `CORRUPT_FILE` | File is damaged | Yes |
| `NO_PREVIEW` | No embedded preview | Yes |

### core/extractor.ts

ExifTool integration:
- `analyzeEmbeddedPreviews()` - Read metadata about available previews
- `extractBestPreview()` - Get the largest preview as a buffer
- `isRawFormat()` / `isDecodedFormat()` - Format detection

Key considerations:
- ExifTool runs as a persistent process - always call `shutdownExiftool()` on exit
- Preview extraction uses temp files - cleaned up in `finally` block
- Type guards used for safe value extraction from ExifTool output

### core/resizer.ts

Sharp integration:
- All outputs converted to sRGB with embedded ICC profile
- Supports WebP, JPEG, PNG, AVIF output formats
- Respects EXIF orientation by default

### services/thumbnail-generator.ts

Main orchestration:
1. Check if already processed (XMP sidecar)
2. Determine source method (direct, extracted, decoded)
3. Resize to all configured sizes
4. Update XMP sidecar with metadata

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
