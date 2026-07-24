/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Build-time default for the inference server URL (fallback for config.json). */
  readonly VITE_SERVER_URL?: string;
  /** Build-time default checkpoint path (fallback for config.json). */
  readonly VITE_CHECKPOINT_PATH?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
