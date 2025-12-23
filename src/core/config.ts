/**
 * Configuration Loader
 *
 * Loads and validates configuration from TOML files.
 * Supports user config, project config, and presets.
 */

import fs from 'fs/promises';
import path from 'path';
import os from 'os';
import { parse as parseToml } from 'smol-toml';
import { ConfigSchema, PresetSchema, type Config, type Preset, type BehaviorConfig } from '../schemas/index.js';
import { ShoemakerError, ErrorCode } from './errors.js';

// Default configuration
const DEFAULT_CONFIG: Config = ConfigSchema.parse({});

// Built-in presets
const BUILTIN_PRESETS: Record<string, Preset> = {
  fast: {
    behavior: {
      fallbackToRaw: false,
      useLargestAvailable: true,
      skipIfInsufficient: false,
      decoder: 'embedded',
      fallbackDecoder: 'libraw',
    },
  },
  quality: {
    behavior: {
      fallbackToRaw: true,
      useLargestAvailable: false,
      skipIfInsufficient: true,
      decoder: 'rawtherapee',
      fallbackDecoder: 'darktable',
    },
  },
  portable: {
    behavior: {
      fallbackToRaw: true,
      useLargestAvailable: true,
      skipIfInsufficient: false,
      decoder: 'libraw',
      fallbackDecoder: 'embedded',
    },
  },
};

/**
 * Load configuration from all sources (merged)
 */
export async function loadConfig(): Promise<Config> {
  let config = { ...DEFAULT_CONFIG };

  // Load user config (~/.config/shoemaker/config.toml)
  const userConfigPath = path.join(os.homedir(), '.config', 'shoemaker', 'config.toml');
  const userConfig = await loadTomlFile(userConfigPath);
  if (userConfig) {
    config = mergeConfig(config, userConfig);
  }

  // Load project config (./.shoemaker.toml)
  const projectConfigPath = path.join(process.cwd(), '.shoemaker.toml');
  const projectConfig = await loadTomlFile(projectConfigPath);
  if (projectConfig) {
    config = mergeConfig(config, projectConfig);
  }

  return config;
}

/**
 * Load a preset by name
 */
export async function loadPreset(name: string, config?: Config): Promise<Preset> {
  // Check built-in presets first
  if (BUILTIN_PRESETS[name]) {
    return BUILTIN_PRESETS[name];
  }

  // Try to load from preset directory
  const presetDir = config?.presetDir ?? path.join(os.homedir(), '.config', 'shoemaker', 'presets');
  const presetPath = path.join(expandPath(presetDir), `${name}.toml`);

  const preset = await loadTomlFile(presetPath);
  if (!preset) {
    throw new ShoemakerError(
      `Preset not found: ${name}`,
      ErrorCode.PRESET_NOT_FOUND
    );
  }

  const parsed = PresetSchema.safeParse(preset);
  if (!parsed.success) {
    throw new ShoemakerError(
      `Invalid preset ${name}: ${parsed.error.message}`,
      ErrorCode.CONFIG_INVALID
    );
  }

  return parsed.data;
}

/**
 * Merge preset behavior into config
 */
export function applyPreset(config: Config, preset: Preset): Config {
  const merged = { ...config };

  // Merge sizes if provided
  if (preset.sizes) {
    merged.sizes = { ...config.sizes };
    for (const [sizeName, sizeOverrides] of Object.entries(preset.sizes)) {
      const existingSize = merged.sizes[sizeName];
      if (existingSize) {
        merged.sizes[sizeName] = {
          ...existingSize,
          ...sizeOverrides,
          width: sizeOverrides.width ?? existingSize.width,
          format: (sizeOverrides.format ?? existingSize.format) as 'webp' | 'jpeg' | 'png' | 'avif',
          quality: sizeOverrides.quality ?? existingSize.quality,
        };
      }
    }
  }

  return merged;
}

/**
 * Get effective behavior config from preset
 */
export function getBehavior(preset: Preset): BehaviorConfig {
  return preset.behavior ?? {};
}

/**
 * Load and parse a TOML file, returning null if not found
 */
async function loadTomlFile(filePath: string): Promise<Record<string, unknown> | null> {
  try {
    const content = await fs.readFile(expandPath(filePath), 'utf-8');
    return parseToml(content) as Record<string, unknown>;
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
      return null;
    }
    throw new ShoemakerError(
      `Failed to load config: ${filePath}`,
      ErrorCode.CONFIG_INVALID,
      filePath
    );
  }
}

/**
 * Merge two config objects (source overrides target)
 */
function mergeConfig(target: Config, source: Record<string, unknown>): Config {
  const merged = { ...target };

  // Simple shallow merge for now
  for (const [key, value] of Object.entries(source)) {
    if (key in merged && typeof value === 'object' && value !== null && !Array.isArray(value)) {
      (merged as Record<string, unknown>)[key] = {
        ...(merged as Record<string, unknown>)[key] as object,
        ...value as object,
      };
    } else {
      (merged as Record<string, unknown>)[key] = value;
    }
  }

  // Validate merged config
  const parsed = ConfigSchema.safeParse(merged);
  if (!parsed.success) {
    return target; // Return original if merge produces invalid config
  }

  return parsed.data;
}

/**
 * Expand ~ to home directory
 */
export function expandPath(filePath: string): string {
  if (filePath.startsWith('~')) {
    return path.join(os.homedir(), filePath.slice(1));
  }
  return filePath;
}

/**
 * Get config file paths for display
 */
export function getConfigPaths(): { user: string; project: string } {
  return {
    user: path.join(os.homedir(), '.config', 'shoemaker', 'config.toml'),
    project: path.join(process.cwd(), '.shoemaker.toml'),
  };
}
