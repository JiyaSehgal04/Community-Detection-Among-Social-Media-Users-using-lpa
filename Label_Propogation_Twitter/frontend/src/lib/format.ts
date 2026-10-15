export const num = (v: number | null | undefined, digits = 3): string =>
  v === null || v === undefined || Number.isNaN(v) ? "n/a" : v.toFixed(digits);

export const pct = (v: number | null | undefined, digits = 1): string =>
  v === null || v === undefined || Number.isNaN(v) ? "n/a" : `${(v * 100).toFixed(digits)}%`;

export const int = (v: number | null | undefined): string =>
  v === null || v === undefined ? "n/a" : v.toLocaleString("en-US");
