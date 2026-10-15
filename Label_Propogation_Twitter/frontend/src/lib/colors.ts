/** 12 distinct community colours; every community beyond the 12 largest is drawn in grey ("other"). */
export const PALETTE = [
  "#2a78d6", "#eb6834", "#1baf7a", "#8e5bd6", "#d6a62a", "#d64a8e",
  "#2aa7c9", "#7a8a2a", "#c94a2a", "#5b6bd6", "#2ac98a", "#a0522d",
];
export const OTHER = "#b8b7ad";

/** Colour for each id, given how big each id is: the 12 biggest get palette colours, the rest grey. */
export function rankColors(sizes: Map<number, number>): Map<number, string> {
  const ranked = [...sizes.entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0]);
  const colours = new Map<number, string>();
  ranked.forEach(([id], i) => colours.set(id, i < PALETTE.length ? PALETTE[i] : OTHER));
  return colours;
}

export function countBy(values: number[]): Map<number, number> {
  const counts = new Map<number, number>();
  for (const v of values) counts.set(v, (counts.get(v) ?? 0) + 1);
  return counts;
}
