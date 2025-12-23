# Shoemaker Production Hardening Plan

> **Generated**: 2025-12-23
> **Scope**: Comprehensive
> **Version**: 1.0

---

## Executive Summary

This document outlines the complete plan to bring Shoemaker from "vibe coded" prototype to production-quality application. Based on a comprehensive audit against CLAUDE.md standards, 21 issues were identified across HIGH, MEDIUM, and LOW severity levels.

**Key Findings:**
- 5 HIGH severity issues (security, input validation)
- 10 MEDIUM severity issues (magic values, error handling)
- 6 LOW severity issues (edge cases, naming)
- Current CLAUDE.md compliance: 82/100

**Target State:**
- All issues resolved
- Test coverage expanded
- Documentation complete
- CLAUDE.md compliance: 95+/100

---

## Phase 1: HIGH Severity Fixes

### Issue H1: Unsafe Environment Variable Usage
**File:** `src/services/thumbnail-generator.ts:242`

```typescript
// BEFORE
const cacheDir = config.output.cacheDir.replace('~', process.env.HOME ?? '');

// AFTER
import { expandPath } from '../core/config.js';
const cacheDir = expandPath(config.output.cacheDir);
```

### Issue H2: Command Injection Risk
**File:** `src/cli/commands/doctor.ts:105-133`

```typescript
// Add validation at module level
const ALLOWED_DECODER_COMMANDS = new Set(['rawtherapee-cli', 'darktable-cli', 'dcraw']);
const ALLOWED_METADATA_COMMANDS = new Set(['exiv2']);

// In checkDecoders() and checkMetadataTools()
// Validate before executing
```

### Issue H3: Magic Number - Preview Size
**File:** `src/core/extractor.ts:40`

```typescript
// Add constant at module level
const DEFAULT_MIN_PREVIEW_SIZE = 2560;

// Use in needsRawDecode calculation
const needsRawDecode = !bestPreview || bestPreview.width < DEFAULT_MIN_PREVIEW_SIZE;
```

### Issue H4: Missing File Path Validation
**Files:** All CLI commands

```typescript
// Add validation wrapper function
async function validatePath(inputPath: string): Promise<fs.Stats> {
  try {
    return await fs.stat(inputPath);
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
      console.error(`Error: Path not found: ${inputPath}`);
      process.exit(1);
    }
    throw wrapError(err, inputPath);
  }
}
```

### Issue H5: Unsafe Type Assertions
**File:** `src/core/extractor.ts:60-65`

```typescript
// Add type guards
function isNumber(value: unknown): value is number {
  return typeof value === 'number' && !isNaN(value);
}

// Use in extractPreviewInfo
const width = isNumber(tagsRecord[`${tagName}Width`])
  ? tagsRecord[`${tagName}Width`]
  : isNumber(tagsRecord[`${tagName}ImageWidth`])
  ? tagsRecord[`${tagName}ImageWidth`]
  : undefined;
```

---

## Phase 2: MEDIUM Severity Fixes

### Issue M6: Magic Strings - File Extensions
**File:** `src/schemas/index.ts`

```typescript
// Export constants for reuse
export const RAW_EXTENSIONS = [
  'arw', 'cr2', 'cr3', 'nef', 'raf', 'rw2', 'orf', 'pef', 'dng',
  'srw', 'x3f', 'erf', 'mrw', 'dcr', 'kdc', 'rwl', 'raw', '3fr',
  'ari', 'srf', 'sr2', 'bay', 'crw', 'iiq',
] as const;

export const DECODED_EXTENSIONS = [
  'jpg', 'jpeg', 'png', 'tif', 'tiff', 'webp', 'avif', 'heic', 'heif',
] as const;
```

### Issue M7: Magic Strings - Default Paths
**File:** `src/schemas/index.ts`

```typescript
// Add named constants with documentation
/** Sidecar folder template. Supports: {stem} = filename without extension */
export const DEFAULT_SIDECAR_FOLDER = '{stem}_thumbs';

/** Cache directory path. Supports ~ for home directory */
export const DEFAULT_CACHE_DIR = '~/.cache/shoemaker';

/** Output filename template. Supports: {stem}, {size}, {width}, {format} */
export const DEFAULT_NAMING_PATTERN = '{stem}_{size}_{width}.{format}';
```

### Issue M8: Empty File Edge Case
**File:** `src/services/thumbnail-generator.ts`

```typescript
// Add check after reading file
sourceBuffer = await fs.readFile(filePath);
if (sourceBuffer.length === 0) {
  throw new ShoemakerError(
    `File is empty: ${filePath}`,
    ErrorCode.CORRUPT_FILE,
    filePath
  );
}
```

### Issue M9-M10: Display Limits & Silent Failures
See implementation details in code fixes.

### Issue M11: Concurrency Validation
**File:** `src/cli/commands/thumb.ts`

```typescript
if (options.concurrency) {
  const concurrency = parseInt(options.concurrency, 10);
  if (isNaN(concurrency) || concurrency < 1 || concurrency > 32) {
    console.error('Error: Concurrency must be between 1 and 32');
    process.exit(1);
  }
  finalConfig.processing.concurrency = concurrency;
}
```

### Issue M12-M14: Path & Type Safety
See implementation details in code fixes.

### Issue M15: Unicode Path Normalization
```typescript
// Normalize Unicode to NFC form
const normalizedName = entry.name.normalize('NFC');
const ext = path.extname(normalizedName).slice(1).toLowerCase();
```

---

## Phase 3: LOW Severity & Improvements

- L16: Document memory considerations for large files
- L17: Evaluate boolean naming consistency
- L18: Add division-by-zero guards
- L19: Remove unused `preset` from XmpUpdateData or use it
- L20: Add crypto for temp file uniqueness
- L21-23: Refactoring for clarity

---

## Phase 4: Test Coverage Expansion

### New Unit Tests Required

1. **Input validation tests:**
   - Invalid file paths
   - Non-existent files
   - Permission denied scenarios
   - Empty files

2. **Edge case tests:**
   - Unicode filenames
   - Very long paths
   - Special characters in paths
   - Zero-byte files

3. **Concurrency tests:**
   - Invalid concurrency values
   - Boundary values (0, 1, 32, 33)

4. **Error recovery tests:**
   - Corrupt RAW files
   - Missing previews
   - XMP parse failures

### Integration Tests Required

1. **End-to-end thumbnail generation:**
   - Single file processing
   - Directory processing
   - Recursive processing
   - Force regeneration

2. **XMP integration:**
   - Create new sidecar
   - Update existing sidecar
   - Clear thumbnail info

---

## Phase 5: Documentation Updates

### README.md
- Verify all CLI commands are accurate
- Add troubleshooting section
- Add configuration examples

### DEVELOPMENT.md
- Module-by-module development guide
- Testing guide expansion
- Debugging techniques

### techguide.md
- Update gotchas with new findings
- Add security considerations

### API Documentation
- Document all exported functions
- Add usage examples
- Document error types

---

## Implementation Checklist

### For Less Experienced Developers

**Before you start:**
1. Read CLAUDE.md completely
2. Read techguide.md for project specifics
3. Ensure `npm test` passes
4. Create a new branch for changes

**For each fix:**
1. Find the file and line number from the audit
2. Read the surrounding code to understand context
3. Make the minimal change described
4. Run `npm run typecheck` after each change
5. Run `npm test` after completing related changes
6. Commit with clear message explaining what and why

**After all fixes:**
1. Run full test suite: `npm test`
2. Run build: `npm run build`
3. Run lint: `npm run lint`
4. Test CLI manually: `npm start -- doctor`
5. Verify with a real image if available

---

## Success Criteria

- [ ] All 21 audit issues resolved
- [ ] Build passes without warnings
- [ ] All tests pass
- [ ] Lint passes without errors
- [ ] CLI commands work correctly
- [ ] Documentation is accurate and complete
- [ ] CLAUDE.md compliance score: 95+/100

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-23 | Initial plan based on comprehensive audit |
