export const GLCHAT_API_KEY_STORAGE_KEY = "ika:glchat:key";
export const GLCHAT_SLUG_STORAGE_KEY = "ika:glchat:slug";
export const GLCHAT_API_KEY_HEADER = "X-API-Key";
export const DEFAULT_GLCHAT_SLUG = "gik";

export function normalizeGlchatApiKey(value) {
  return String(value || "").trim();
}

export function normalizeGlchatSlug(value) {
  return String(value || "").trim() || DEFAULT_GLCHAT_SLUG;
}

export function buildGlchatAuthPayload(apiKey, slug = DEFAULT_GLCHAT_SLUG) {
  const key = normalizeGlchatApiKey(apiKey);
  const normalizedSlug = normalizeGlchatSlug(slug);
  const payload = {
    glchat_slug: normalizedSlug,
    glchat_application_slug: normalizedSlug,
  };

  if (!key) return payload;

  return {
    ...payload,
    glchat_api_key: key,
    glchat_headers: {
      [GLCHAT_API_KEY_HEADER]: key,
    },
  };
}
