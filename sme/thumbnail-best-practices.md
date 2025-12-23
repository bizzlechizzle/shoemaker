# Thumbnail Generation Best Practices

> **Generated**: 2025-12-23
> **Sources current as of**: December 2025
> **Scope**: Comprehensive
> **Version**: 1.0
> **Audit-Ready**: Yes

---

## Executive Summary / TLDR

This document provides authoritative guidance for implementing a thumbnail generation CLI tool (Shoemaker). Key findings:

1. **Optimal Resolutions**: Use tiered thumbnails—300px (grid), 1600px (lightbox), 2560px (ML inference) [HIGH]
2. **Format Selection**: WebP offers best balance of quality, compression (26-34% smaller than JPEG), and browser support (97%) [1][HIGH]. AVIF provides 50% better compression but slightly lower support (93%) [2]
3. **Social Media Crops**: Support 5 core aspect ratios—1:1, 4:5, 9:16, 16:9, 3:4 [3][HIGH]
4. **Smart Cropping**: Combine face detection (MediaPipe/dlib) with saliency detection for subject-aware crops [4]
5. **Storage Pattern**: Use sidecar folders with size-based subdirectories (e.g., `image_thumb_300.webp`) [MEDIUM]
6. **Color Space**: Convert all outputs to sRGB with embedded ICC profile for consistency [HIGH]

---

## Background & Context

Modern image workflows require multiple thumbnail sizes for different consumption contexts:
- **Gallery grids** need small, fast-loading previews
- **Lightbox viewing** requires medium-resolution images for inspection
- **ML inference** needs larger images for accurate tagging/classification
- **Social media** requires specific aspect ratios for each platform

This guide provides research-backed specifications for a CLI tool that generates optimized thumbnails from source images (including RAW files).

---

## 1. Optimal Thumbnail Resolutions

### General Purpose Thumbnails

| Use Case | Resolution | Aspect | Rationale |
|----------|------------|--------|-----------|
| **Grid/Gallery** | 300px (long edge) | Preserve | Fast loading, ~20-30KB per image [5][HIGH] |
| **Lightbox/Preview** | 1600px (long edge) | Preserve | Detailed viewing without loading full image |
| **ML Inference** | 2560px (long edge) | Preserve | Sufficient detail for most CV models |
| **E-commerce Thumb** | 150-300px | 1:1 square | Standard product grid size [5] |

### Video Thumbnails

| Platform | Resolution | Aspect Ratio | Notes |
|----------|------------|--------------|-------|
| **YouTube** | 1280x720 (minimum) | 16:9 | Recommended: 1920x1080 for HD displays [6][HIGH] |
| **TikTok** | 1080x1920 | 9:16 | Full-screen vertical [7] |
| **Generic Video** | 1280x720 | 16:9 | Industry standard |

### ML Model Input Sizes

Common CNN input resolutions [MEDIUM]:

| Model Family | Input Size | Notes |
|--------------|------------|-------|
| ResNet/VGG | 224x224 | ImageNet standard |
| Inception/Xception | 299x299 | Higher detail |
| EfficientNet-B0 | 224x224 | Scales up to 600x600 (B7) |
| CLIP/SigLIP | 224x224 or 336x336 | Vision-language models |
| RAM++ | 384x384 | Tagging model |
| YOLO | 640x640 | Object detection |

**Recommendation**: 2560px thumbnails can be resized to any of these on-the-fly, or pre-generate 640px and 384px variants for frequent ML use.

---

## 2. Image Format Comparison

### WebP vs JPEG vs AVIF

| Metric | JPEG | WebP | AVIF |
|--------|------|------|------|
| **Compression** (baseline) | 1x | 26-34% smaller [1] | 50% smaller [2] |
| **Quality at compression** | Degrades significantly | Good | Excellent |
| **Browser Support** | 100% | ~97% [1] | ~93% [2] |
| **Encode Speed** | Fast | Fast | Slow (5-10x slower) |
| **Decode Speed** | Fast | Fast | Moderate |
| **Animation** | No | Yes | Yes |
| **Transparency** | No | Yes | Yes |
| **HDR Support** | No | No | Yes |

### Detailed Findings

**WebP Advantages** [1][2]:
- 26% smaller than PNG, up to 34% smaller than JPEG at equivalent quality
- Near-universal browser support (96.86% as of 2024)
- Fast encoding, suitable for on-the-fly generation
- Supports both lossy and lossless compression

**AVIF Advantages** [2]:
- 50% smaller than JPEG, 20-30% smaller than WebP
- Superior quality retention at high compression
- Better for photographic content with fine details
- HDR and wide color gamut support

**AVIF Disadvantages** [2]:
- Slower encoding (5-10x slower than WebP)
- Slightly lower browser support (93.16%)
- Higher memory usage during encoding
- Some edge cases where WebP/JPEG outperform

### Format Recommendations by Use Case

| Use Case | Primary Format | Fallback | Rationale |
|----------|---------------|----------|-----------|
| **Grid Thumbnails** | WebP | JPEG | Speed + compatibility |
| **Lightbox Preview** | WebP | JPEG | Balance of quality/size |
| **ML Inference** | JPEG | — | ML models expect JPEG; avoids format conversion |
| **Archival** | Original format | — | Preserve source |
| **Social Media Export** | JPEG | — | Maximum compatibility |

### Quality Settings

| Size Tier | WebP Quality | JPEG Quality | Target File Size |
|-----------|--------------|--------------|------------------|
| Thumb (300px) | 80 | 75 | ~20-30KB |
| Preview (1600px) | 85 | 80 | ~150-200KB |
| ML (2560px) | 90 | 85 | ~400-500KB |

---

## 3. Social Media Aspect Ratios & Safe Zones

### Platform Specifications (2024-2025)

#### Instagram [3][8]

| Content Type | Dimensions | Aspect Ratio |
|--------------|------------|--------------|
| **Square Post** | 1080x1080 | 1:1 |
| **Portrait Post** | 1080x1350 | 4:5 |
| **Landscape Post** | 1080x566 | 1.91:1 |
| **Stories/Reels** | 1080x1920 | 9:16 |
| **New Tall Grid** (2025) | 1080x1440 | 3:4 |

**Safe Zone**: Keep critical content within center 90% for Stories to avoid UI overlap.

#### TikTok [7]

| Content Type | Dimensions | Aspect Ratio |
|--------------|------------|--------------|
| **Video** | 1080x1920 | 9:16 |
| **Profile Picture** | 400x400 | 1:1 |
| **Image Ads** | 1200x628 or 720x1280 | ~1.91:1 or 9:16 |

**Safe Zone**: Top 150px and bottom 250px may be obscured by UI elements.

#### YouTube [6]

| Content Type | Dimensions | Aspect Ratio |
|--------------|------------|--------------|
| **Thumbnail** | 1280x720 (min) | 16:9 |
| **Thumbnail (HD)** | 1920x1080 (recommended) | 16:9 |
| **Profile Picture** | 800x800 | 1:1 |
| **Banner** | 2560x1440 | 16:9 |

**Safe Zone**: Keep text/faces in center 1546x423px for banner.

#### Facebook [3]

| Content Type | Dimensions | Aspect Ratio |
|--------------|------------|--------------|
| **Feed Image (Portrait)** | 1080x1350 | 4:5 |
| **Feed Image (Square)** | 1080x1080 | 1:1 |
| **Stories** | 1080x1920 | 9:16 |
| **Cover Photo** | 851x315 | 2.7:1 |
| **Event Cover** | 1920x1005 | ~1.91:1 |

#### X (Twitter) [3]

| Content Type | Dimensions | Aspect Ratio |
|--------------|------------|--------------|
| **In-Stream Image** | 1600x900 | 16:9 |
| **Profile Picture** | 400x400 | 1:1 |
| **Header** | 1500x500 | 3:1 |

**Safe Zone**: Header may be cropped 60px from top/bottom.

### Universal Crop Presets

Based on cross-platform analysis, implement these 5 core aspect ratios:

| Preset Name | Ratio | Primary Use |
|-------------|-------|-------------|
| `square` | 1:1 | Profile pics, Instagram square |
| `portrait` | 4:5 | Instagram/Facebook portrait |
| `story` | 9:16 | Stories, Reels, TikTok |
| `landscape` | 16:9 | YouTube, Twitter, widescreen |
| `tall` | 3:4 | Instagram 2025 tall grid |

### Safe Zone Implementation

```typescript
interface SafeZone {
  top: number;      // Percentage from top to avoid
  bottom: number;   // Percentage from bottom to avoid
  left: number;     // Percentage from left to avoid
  right: number;    // Percentage from right to avoid
}

const SAFE_ZONES: Record<string, SafeZone> = {
  'story':     { top: 0.08, bottom: 0.13, left: 0.05, right: 0.05 },
  'tiktok':    { top: 0.08, bottom: 0.15, left: 0.05, right: 0.05 },
  'youtube':   { top: 0.05, bottom: 0.10, left: 0.05, right: 0.05 },
  'default':   { top: 0.05, bottom: 0.05, left: 0.05, right: 0.05 },
};
```

---

## 4. Progressive Thumbnail Strategy

### Tiered Generation Approach

Generate thumbnails in order of priority:

```
Source Image
    │
    ├─1→ ML Size (2560px) ─── Used for all downstream sizes
    │         │
    │         ├─2→ Preview (1600px)
    │         │         │
    │         │         └─3→ Thumb (300px)
    │         │
    │         └─4→ Social Crops (on-demand)
    │
    └─── Store source path in XMP for regeneration
```

### Implementation Strategy

**On-Import (Eager)**:
1. Extract embedded preview OR decode RAW
2. Generate ML size (2560px) from best available source
3. Generate Preview (1600px) from ML size
4. Generate Thumb (300px) from Preview

**On-Demand (Lazy)**:
- Social media crops generated when requested
- Additional sizes generated as needed
- Stored alongside other thumbnails

### Responsive Image Set

For web delivery, generate a srcset-compatible set:

```html
<img
  src="image_800.webp"
  srcset="
    image_400.webp 400w,
    image_800.webp 800w,
    image_1200.webp 1200w,
    image_1600.webp 1600w,
    image_2000.webp 2000w
  "
  sizes="(max-width: 600px) 100vw, 50vw"
>
```

**Recommended breakpoints**: 400, 800, 1200, 1600, 2000px

---

## 5. Smart Cropping Algorithms

### Face-Aware Cropping

**Detection Methods** [4]:

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| **Haar Cascades** | Very fast | Moderate | Legacy, real-time |
| **dlib HOG** | Fast | Good | Desktop apps |
| **dlib CNN** | Moderate | Excellent | High accuracy needed |
| **MediaPipe** | Fast | Very good | Mobile/web, landmarks |
| **MTCNN** | Moderate | Excellent | Multiple faces |
| **RetinaFace** | Slow | Best | Professional use |

**Implementation** (from nightfoxfilms):

```typescript
interface FaceData {
  bbox: [x, y, width, height];
  confidence: number;
  landmarks?: {
    leftEye: [x, y];
    rightEye: [x, y];
    nose: [x, y];
    leftMouth: [x, y];
    rightMouth: [x, y];
  };
}

function calculateFaceWeightedCenter(faces: FaceData[]): [x, y] {
  // Weight by face size and confidence
  // Larger faces in foreground get higher weight
}
```

### Subject-Aware Cropping

**Smartcrop.js Algorithm** [4]:
1. Detect skin tones (prioritize people)
2. Edge detection (avoid cutting through objects)
3. Saturation analysis (colorful areas are interesting)
4. Face detection boost (if faces present)
5. Rule of thirds preference

**Cloudinary/ImageKit Approach** [4]:
- Deep learning saliency detection
- Object detection with 80+ named objects
- Configurable priority for specific subjects
- API-based, requires cloud service

### Cropping Priority Rules

1. **Never crop through faces** — Faces get highest preservation priority
2. **Preserve eyes** — If cropping face, keep eyes in frame
3. **Rule of thirds** — Place subjects at intersection points when possible
4. **Avoid edges** — Don't cut through high-contrast edges
5. **Respect safe zones** — Keep critical content away from platform UI areas

### Sharpness-Aware Selection

When selecting best frame/crop, use Laplacian variance [nightfoxfilms]:

```typescript
// Laplacian variance = measure of edge intensity = sharpness
function calculateSharpness(grayImage: Buffer): number {
  // Apply Laplacian kernel: [0, -1, 0], [-1, 4, -1], [0, -1, 0]
  // Calculate variance of result
  // Higher variance = sharper image
}

function normalizeSharpnessScore(variance: number): number {
  const baseline = 500; // Typical sharp image variance
  return Math.min(100, Math.max(0, (variance / baseline) * 100));
}
```

---

## 6. Storage Patterns

### Option 1: Sidecar Folders (Recommended)

```
photos/
├── IMG_1234.CR2
├── IMG_1234.CR2.xmp          # XMP sidecar (wake-n-blake)
└── IMG_1234/                 # Thumbnail folder
    ├── thumb_300.webp
    ├── preview_1600.webp
    ├── ml_2560.jpg
    └── crops/
        ├── square_1080.webp
        ├── portrait_1080x1350.webp
        └── story_1080x1920.webp
```

**Pros**: Portable, travels with source, easy backup
**Cons**: More files to manage, folder clutter

### Option 2: Centralized Cache

```
~/.cache/shoemaker/
├── a7/
│   └── a7f3b2c1d4e5f678/     # Hash-based sharding
│       ├── thumb_300.webp
│       ├── preview_1600.webp
│       └── ml_2560.jpg
```

**Pros**: Clean source folders, easy cache invalidation
**Cons**: Doesn't travel with files, requires database mapping

### Option 3: Hybrid (Recommended for Shoemaker)

- **Thumbnails**: Sidecar folder next to source (portable)
- **Temporary/ML**: Centralized cache (performance)
- **XMP tracking**: Record thumbnail paths in sidecar

### Naming Convention

```
{original_stem}_{size}_{width}[x{height}].{format}

Examples:
- IMG_1234_thumb_300.webp
- IMG_1234_preview_1600.webp
- IMG_1234_ml_2560.jpg
- IMG_1234_crop_square_1080.webp
- IMG_1234_crop_story_1080x1920.webp
```

### Database Schema (Optional)

```sql
CREATE TABLE thumbnails (
  id INTEGER PRIMARY KEY,
  source_path TEXT NOT NULL,
  source_hash TEXT NOT NULL,      -- BLAKE3 of source
  thumb_type TEXT NOT NULL,       -- 'thumb', 'preview', 'ml', 'crop'
  width INTEGER NOT NULL,
  height INTEGER NOT NULL,
  format TEXT NOT NULL,           -- 'webp', 'jpeg', 'avif'
  file_path TEXT NOT NULL,
  file_size INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  generation_method TEXT,         -- 'extracted', 'decoded', 'direct'
  UNIQUE(source_hash, thumb_type, width, height, format)
);
```

---

## 7. Color Space Handling

### The sRGB Standard

**Key Facts**:
- sRGB is the default color space for the web [HIGH]
- Browsers assume untagged images are sRGB
- Most displays are calibrated to sRGB
- ML models expect sRGB input

### Conversion Strategy

```
Source (any color space)
    │
    ├── Adobe RGB, ProPhoto RGB, Display P3
    │   └── Convert to sRGB
    │
    └── sRGB (or untagged)
        └── Use directly

All outputs → sRGB with embedded ICC profile
```

### Implementation with Sharp

```typescript
import sharp from 'sharp';

async function generateThumbnail(
  input: string,
  output: string,
  width: number
): Promise<void> {
  await sharp(input)
    .resize(width, width, {
      fit: 'inside',
      withoutEnlargement: true
    })
    .toColorspace('srgb')           // Convert to sRGB
    .withMetadata({
      icc: 'srgb'                   // Embed sRGB profile (~3KB)
    })
    .webp({ quality: 85 })
    .toFile(output);
}
```

### When to Strip vs Embed ICC Profile

| Scenario | Action | Rationale |
|----------|--------|-----------|
| **Web thumbnails** | Embed sRGB | Ensures color consistency |
| **ML inference** | Strip profile | Models don't use ICC |
| **Archive/preservation** | Preserve original | Maintain source fidelity |
| **Social media export** | Embed sRGB | Platform compatibility |

### Profile Size Impact

| Profile | Size | Recommendation |
|---------|------|----------------|
| sRGB | ~3KB | Embed (negligible overhead) |
| Adobe RGB | ~3KB | Convert to sRGB, embed sRGB |
| Display P3 | ~0.5KB | Convert to sRGB for web |
| ProPhoto RGB | ~0.5KB | Convert to sRGB for web |

---

## Analysis & Implications

### Shoemaker Implementation Recommendations

1. **Default Pipeline**: Generate 3 sizes (300, 1600, 2560) in WebP format
2. **ML Output**: Use JPEG for 2560px to avoid format conversion in ML pipelines
3. **Social Crops**: Generate on-demand, store in `crops/` subfolder
4. **Smart Cropping**: Integrate MediaPipe for face detection, fallback to center crop
5. **Color Space**: Always convert to sRGB with embedded profile
6. **Storage**: Use sidecar folders, track in XMP

### Performance Targets

| Operation | Target Time | Notes |
|-----------|-------------|-------|
| Extract preview + resize | <0.5s | Fast path (80% of files) |
| RAW decode + resize | <3s | Slow path (20% of files) |
| Smart crop calculation | <100ms | Face detection included |
| Full 3-size generation | <1s | From extracted preview |

### Storage Budget

| Size Tier | Avg File Size | 1000 Images |
|-----------|---------------|-------------|
| Thumb (300px WebP) | ~25KB | 25MB |
| Preview (1600px WebP) | ~175KB | 175MB |
| ML (2560px JPEG) | ~450KB | 450MB |
| **Total per image** | ~650KB | 650MB |

---

## Limitations & Uncertainties

### What This Document Does NOT Cover

- Video thumbnail extraction (frame selection algorithms)
- Real-time thumbnail generation (CDN/edge computing)
- Thumbnail serving/delivery optimization
- Database-backed thumbnail management at scale

### Unverified Claims

- AVIF encoding speed improvements in 2025 (codec development ongoing)
- Exact ML model input size preferences vary by model version

### Source Conflicts

- Some sources recommend 224x224 for ML, others suggest higher resolutions
- Resolution: Use 2560px as source, resize to model-specific size at inference time

### Knowledge Gaps

- Optimal AVIF quality settings for thumbnails (less documented than WebP)
- Performance benchmarks for different smart cropping algorithms

### Recency Limitations

- Social media dimensions change frequently; verify against platform docs before production use
- Browser support percentages are point-in-time (check caniuse.com for current data)

---

## Recommendations

1. **Implement tiered thumbnail generation** with 300px, 1600px, and 2560px as standard sizes [HIGH]

2. **Use WebP as primary format** with JPEG fallback for ML inference [HIGH]

3. **Support 5 core social media crops**: 1:1, 4:5, 9:16, 16:9, 3:4 [HIGH]

4. **Integrate face detection** (MediaPipe recommended) for smart cropping [MEDIUM]

5. **Store thumbnails in sidecar folders** next to source files for portability [MEDIUM]

6. **Always convert to sRGB** with embedded ICC profile for consistency [HIGH]

7. **Track thumbnail state in XMP** sidecars for integration with wake-n-blake [HIGH]

8. **Consider AVIF for future versions** once encoding speed improves [LOW]

---

## Source Appendix

| # | Source | Date | Type | Used For |
|---|--------|------|------|----------|
| 1 | [Elementor: AVIF vs WebP](https://elementor.com/blog/webp-vs-avif/) | 2024 | Secondary | Format comparison, compression stats |
| 2 | [Medium: Why AVIF over JPEG/WebP](https://medium.com/@julienetienne/why-you-should-use-avif-over-jpeg-webp-png-and-gif-in-2024-5603ac9d8781) | 2024 | Secondary | AVIF advantages, compression ratios |
| 3 | [Hootsuite: Social Media Image Sizes 2025](https://blog.hootsuite.com/social-media-image-sizes-guide/) | 2025 | Secondary | Platform dimensions, aspect ratios |
| 4 | [Cloudinary: AI Auto-Crop Algorithm](https://cloudinary.com/blog/new_ai_based_image_auto_crop_algorithm_sticks_to_the_subject) | 2024 | Secondary | Smart cropping approach |
| 5 | [Tiny-img: Website Image Size Guide 2025](https://tiny-img.com/blog/best-image-size-for-website/) | 2025 | Secondary | Thumbnail sizes, performance |
| 6 | [LOVO AI: YouTube Thumbnail Size 2024](https://lovo.ai/post/best-youtube-thumbnail-size-2024-best-practices) | 2024 | Secondary | YouTube specifications |
| 7 | [Captions: TikTok Video Dimensions](https://www.captions.ai/blog-post/tiktok-video-dimensions) | 2024 | Secondary | TikTok specifications |
| 8 | [HeyOrca: Social Media Image Sizes 2025](https://www.heyorca.com/blog/social-media-image-sizes) | 2025 | Secondary | Instagram 2025 changes |
| 9 | [GitHub: smartcrop.js](https://github.com/jwagner/smartcrop.js) | 2024 | Primary | Smart cropping algorithm |
| 10 | [ImageKit: Smart Crop](https://imagekit.io/blog/smart-crop-intelligent-image-cropping-imagekit/) | 2024 | Secondary | Face/object detection approaches |
| 11 | [ShortPixel: AVIF vs WebP](https://shortpixel.com/blog/avif-vs-webp/) | 2024 | Secondary | Detailed format comparison |
| 12 | nightfoxfilms codebase | 2025 | Internal | Sharpness scoring, face detection |
| 13 | visual-buffet codebase | 2025 | Internal | ML model input sizes |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-23 | Initial comprehensive guide |
