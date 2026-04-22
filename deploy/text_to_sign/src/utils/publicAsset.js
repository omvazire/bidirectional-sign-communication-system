export function publicAssetPath(relativePath) {
  const base = import.meta.env.BASE_URL || '/';
  const normalizedBase = base.endsWith('/') ? base : `${base}/`;
  const normalizedPath = relativePath.replace(/^\/+/, '');
  return `${normalizedBase}${normalizedPath}`;
}
