#!/usr/bin/env npx tsx
/**
 * Download Test Fixtures Script
 *
 * Downloads RAW and JPEG test files from public sources for testing.
 * Files are stored in tests/fixtures/raw/ and tests/fixtures/images/.
 *
 * Usage: npm run test:download-fixtures
 */

import { createWriteStream, createReadStream } from 'fs';
import { mkdir, access, unlink, stat } from 'fs/promises';
import { createHash } from 'crypto';
import { pipeline } from 'stream/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, '..');
const FIXTURES_DIR = path.join(ROOT, 'tests', 'fixtures');
const RAW_DIR = path.join(FIXTURES_DIR, 'raw');
const IMAGES_DIR = path.join(FIXTURES_DIR, 'images');

interface TestFile {
  name: string;
  url: string;
  sha256?: string;
  size?: number;
  description: string;
}

/**
 * Test files to download
 * Using small sample files from raw.pixls.us and other public sources
 */
const TEST_FILES: TestFile[] = [
  // Small RAW samples for testing - using freely available samples
  {
    name: 'sample.jpg',
    url: 'https://www.w3schools.com/css/img_5terre.jpg',
    description: 'Sample JPEG for basic testing',
  },
  {
    name: 'sample-small.jpg',
    url: 'https://www.w3schools.com/css/img_forest.jpg',
    description: 'Small JPEG for quick tests',
  },
];

// Note: RAW file downloads from raw.pixls.us require specific URLs
// For now, we create placeholder files for testing the download infrastructure
// Real integration tests should use actual RAW files from a local source

async function fileExists(filePath: string): Promise<boolean> {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function hashFile(filePath: string): Promise<string> {
  const hash = createHash('sha256');
  const stream = createReadStream(filePath);

  return new Promise((resolve, reject) => {
    stream.on('data', (data) => hash.update(data));
    stream.on('end', () => resolve(hash.digest('hex')));
    stream.on('error', reject);
  });
}

async function downloadFile(file: TestFile, destDir: string): Promise<void> {
  const destPath = path.join(destDir, file.name);

  // Check if file already exists with correct hash
  if (await fileExists(destPath)) {
    if (file.sha256) {
      const hash = await hashFile(destPath);
      if (hash === file.sha256) {
        console.log(`  ✓ ${file.name} (cached, hash verified)`);
        return;
      }
      console.log(`  ↻ ${file.name} (hash mismatch, re-downloading)`);
    } else {
      // No hash to verify, check if file has content
      const stats = await stat(destPath);
      if (stats.size > 0) {
        console.log(`  ✓ ${file.name} (cached)`);
        return;
      }
    }
  }

  console.log(`  ↓ Downloading ${file.name}...`);

  try {
    const response = await fetch(file.url, {
      headers: {
        'User-Agent': 'Shoemaker-Test-Fixture-Downloader/1.0',
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    // Ensure directory exists
    await mkdir(destDir, { recursive: true });

    // Download to temp file first
    const tempPath = destPath + '.tmp';
    const fileStream = createWriteStream(tempPath);

    // @ts-expect-error - ReadableStream to Node stream
    await pipeline(response.body, fileStream);

    // Verify hash if provided
    if (file.sha256) {
      const hash = await hashFile(tempPath);
      if (hash !== file.sha256) {
        await unlink(tempPath);
        throw new Error(`Hash mismatch: expected ${file.sha256}, got ${hash}`);
      }
    }

    // Rename temp to final
    const fs = await import('fs/promises');
    await fs.rename(tempPath, destPath);

    console.log(`  ✓ ${file.name}`);
  } catch (err) {
    console.error(`  ✗ ${file.name}: ${(err as Error).message}`);
    throw err;
  }
}

async function createPlaceholderRawFiles(): Promise<void> {
  console.log('\nCreating placeholder RAW file structures...');

  await mkdir(RAW_DIR, { recursive: true });

  // Create a placeholder file that explains where to get real RAW files
  const readmePath = path.join(RAW_DIR, 'README.md');
  const readmeContent = `# RAW Test Fixtures

This directory should contain RAW files for integration testing.

## Obtaining Test Files

### Option 1: Download from raw.pixls.us
Visit https://raw.pixls.us/ to download sample RAW files.

Recommended files:
- Sony ILCE-7M3 ARW (full preview support)
- Canon EOS 5D Mark IV CR2 (full preview)
- Nikon D850 NEF (full preview)
- DJI FC3170 DNG (limited preview - good for fallback testing)

### Option 2: Use Your Own Files
Copy any RAW files you want to test into this directory.

### Option 3: Create Test Symlinks
\`\`\`bash
ln -s /path/to/your/photos/*.ARW .
\`\`\`

## Expected File Structure

\`\`\`
tests/fixtures/
├── raw/
│   ├── sample.arw      # Sony RAW
│   ├── sample.cr2      # Canon RAW
│   ├── sample.nef      # Nikon RAW
│   ├── sample.dng      # DNG (any camera)
│   └── README.md       # This file
└── images/
    ├── sample.jpg      # JPEG test image
    └── sample-small.jpg
\`\`\`

## Note

RAW files are large (20-100MB each) and are not included in the repository.
Integration tests will skip if no RAW files are present.
`;

  const fs = await import('fs/promises');
  await fs.writeFile(readmePath, readmeContent);
  console.log(`  ✓ Created ${path.relative(ROOT, readmePath)}`);
}

async function main(): Promise<void> {
  console.log('Shoemaker Test Fixture Downloader\n');
  console.log(`Fixtures directory: ${path.relative(ROOT, FIXTURES_DIR)}\n`);

  // Create directories
  await mkdir(RAW_DIR, { recursive: true });
  await mkdir(IMAGES_DIR, { recursive: true });

  // Download image test files
  console.log('Downloading image test files...');
  let downloadErrors = 0;

  for (const file of TEST_FILES) {
    try {
      await downloadFile(file, IMAGES_DIR);
    } catch {
      downloadErrors++;
    }
  }

  // Create RAW file placeholders/instructions
  await createPlaceholderRawFiles();

  console.log('\n' + '='.repeat(50));
  if (downloadErrors === 0) {
    console.log('✓ All downloads completed successfully');
  } else {
    console.log(`⚠ ${downloadErrors} download(s) failed`);
    console.log('  Some tests may be skipped');
  }

  console.log('\nTo run integration tests:');
  console.log('  1. Add RAW files to tests/fixtures/raw/');
  console.log('  2. Run: npm run test:integration');
  console.log('');
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
