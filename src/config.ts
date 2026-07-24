/**
 * config.ts — runtime app configuration.
 *
 * Defaults for the inference server URL and checkpoint path are read at startup
 * from `public/config.json` (served at `<base>/config.json`), so they can be
 * changed for a deployment WITHOUT rebuilding the app — just edit the JSON file
 * next to the built assets. Build-time Vite env vars (VITE_SERVER_URL /
 * VITE_CHECKPOINT_PATH) act as a fallback; the shipped `config.json` wins when
 * present.
 */

export interface AppConfig {
  /** Default inference server URL prefilled in the connect box. */
  serverUrl: string;
  /** Default checkpoint path/uri prefilled in the checkpoint box (may be ''). */
  checkpointPath: string;
}

const HARD_DEFAULTS: AppConfig = {
  serverUrl: 'http://localhost:8000',
  checkpointPath: '',
};

let cached: Promise<AppConfig> | null = null;

/**
 * Load the app config once (memoized). Precedence, highest first:
 *   1. fields present in public/config.json
 *   2. VITE_SERVER_URL / VITE_CHECKPOINT_PATH build-time env
 *   3. hard-coded defaults above
 * Never rejects — a missing/invalid config.json falls back to env/defaults.
 */
export function loadConfig(): Promise<AppConfig> {
  if (cached) return cached;
  cached = (async () => {
    const base: AppConfig = {
      serverUrl: import.meta.env.VITE_SERVER_URL || HARD_DEFAULTS.serverUrl,
      checkpointPath: import.meta.env.VITE_CHECKPOINT_PATH ?? HARD_DEFAULTS.checkpointPath,
    };
    try {
      const res = await fetch(`${import.meta.env.BASE_URL}config.json`, { cache: 'no-store' });
      if (!res.ok) return base;
      const j = (await res.json()) as Partial<AppConfig>;
      return {
        serverUrl:
          typeof j.serverUrl === 'string' && j.serverUrl.trim()
            ? j.serverUrl.trim()
            : base.serverUrl,
        checkpointPath:
          typeof j.checkpointPath === 'string' ? j.checkpointPath : base.checkpointPath,
      };
    } catch {
      return base;
    }
  })();
  return cached;
}
