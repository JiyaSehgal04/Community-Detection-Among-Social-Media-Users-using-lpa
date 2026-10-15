import cytoscape, { type Core } from "cytoscape";
import { useEffect, useRef } from "react";

interface Props {
  nodes: { id: number; degree: number }[];
  edges: { source: number; target: number }[];
  /** Colour for a node id; changing it recolours in place without redrawing the layout. */
  colorOf: (id: number) => string;
  onHover?: (id: number | null) => void;
  height?: number;
}

export function GraphView({ nodes, edges, colorOf, onHover, height = 560 }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const cy = useRef<Core | null>(null);
  const colorRef = useRef(colorOf);
  const hoverRef = useRef(onHover);
  colorRef.current = colorOf;
  hoverRef.current = onHover;

  // Build the graph (and its layout) only when the nodes or edges change.
  useEffect(() => {
    if (!host.current) return;
    const instance = cytoscape({
      container: host.current,
      elements: [
        ...nodes.map((n) => ({
          data: { id: String(n.id), size: 6 + 2 * Math.sqrt(n.degree), color: colorRef.current(n.id) },
        })),
        ...edges.map((e, i) => ({ data: { id: `e${i}`, source: String(e.source), target: String(e.target) } })),
      ],
      style: [
        { selector: "node", style: { "background-color": "data(color)", width: "data(size)", height: "data(size)" } },
        { selector: "edge", style: { width: 0.6, "line-color": "#c9c8bf", opacity: 0.55 } },
      ],
      layout: { name: edges.length ? "cose" : "grid", animate: false, randomize: true } as cytoscape.LayoutOptions,
      minZoom: 0.2,
      maxZoom: 4,
    });
    instance.on("mouseover", "node", (ev) => hoverRef.current?.(Number(ev.target.id())));
    instance.on("mouseout", "node", () => hoverRef.current?.(null));
    cy.current = instance;
    return () => {
      instance.destroy();
      cy.current = null;
    };
  }, [nodes, edges]);

  // Recolour in place whenever the colouring changes (community view, or a step of the animation).
  useEffect(() => {
    cy.current?.nodes().forEach((n) => {
      n.data("color", colorOf(Number(n.id())));
    });
  }, [colorOf, nodes, edges]);

  return <div ref={host} className="graph" style={{ height }} aria-label="Network graph" />;
}
