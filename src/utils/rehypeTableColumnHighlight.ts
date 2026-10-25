/**
 * Rehype plugin for Levitating Labs table column highlighting.
 *
 * Syntax supported in markdown table headers:
 * 1. `ROC-AUC {highlight}` or `{highlight} ROC-AUC`
 * 2. `[highlight] ROC-AUC` or `ROC-AUC [highlight]`
 * 3. `★ ROC-AUC` or `⭐ ROC-AUC`
 * 4. `!!ROC-AUC!!`
 *
 * Automatically:
 * - Detects the target column to highlight.
 * - Cleans the marker tag ({highlight}, [highlight], !!) from the rendered header text.
 * - Applies the `.highlight-col` CSS class to the <th> and all corresponding <td> cells in that column.
 * - Wraps the <table> in a `<div class="table-wrapper">` if not already wrapped.
 */

function getTextContent(node: any): string {
  if (!node) return "";
  if (node.type === "text") return node.value || "";
  if (Array.isArray(node.children)) {
    return node.children.map(getTextContent).join("");
  }
  return "";
}

function cleanTextNodes(node: any, regex: RegExp) {
  if (!node) return;
  if (node.type === "text" && typeof node.value === "string") {
    node.value = node.value.replace(regex, "").trim();
  }
  if (Array.isArray(node.children)) {
    node.children.forEach((child: any) => cleanTextNodes(child, regex));
  }
}

function addClass(node: any, className: string) {
  if (!node.properties) node.properties = {};
  const current = node.properties.className;
  let classes: string[] = [];
  if (Array.isArray(current)) {
    classes = [...current];
  } else if (typeof current === "string") {
    classes = current.split(/\s+/).filter(Boolean);
  }
  if (!classes.includes(className)) {
    classes.push(className);
  }
  node.properties.className = classes;
}

export function rehypeTableColumnHighlight() {
  return (tree: any) => {
    function visit(node: any, parent: any, index: number) {
      if (node.type === "element" && node.tagName === "table") {
        processTable(node, parent, index);
      }
      if (node.children && Array.isArray(node.children)) {
        node.children.forEach((child: any, idx: number) => visit(child, node, idx));
      }
    }

    function processTable(table: any, parent: any, index: number) {
      const highlightedIndices = new Set<number>();

      // 1. Find the header row (either in <thead> or first <tr> with <th>)
      let headerTr: any = null;
      let tbodyRows: any[] = [];

      for (const child of table.children || []) {
        if (child.type === "element" && child.tagName === "thead") {
          headerTr = child.children?.find((c: any) => c.type === "element" && c.tagName === "tr");
        } else if (child.type === "element" && child.tagName === "tbody") {
          tbodyRows = child.children?.filter((c: any) => c.type === "element" && c.tagName === "tr") || [];
        } else if (child.type === "element" && child.tagName === "tr") {
          if (!headerTr && child.children?.some((c: any) => c.type === "element" && c.tagName === "th")) {
            headerTr = child;
          } else {
            tbodyRows.push(child);
          }
        }
      }

      if (!headerTr) return;

      const thCells = headerTr.children?.filter((c: any) => c.type === "element" && (c.tagName === "th" || c.tagName === "td")) || [];

      thCells.forEach((th: any, colIdx: number) => {
        const text = getTextContent(th);
        const markerCheck = /\{highlight\}|\{:highlight\}|\{\.highlight\}|\[highlight\]|\[hl\]|\{hl\}/i;
        const exclamationCheck = /^!!.+!!$/;
        const hasMarker = markerCheck.test(text);
        const hasStar = text.includes("★") || text.includes("⭐");
        const hasExclamation = exclamationCheck.test(text.trim());

        if (hasMarker || hasStar || hasExclamation) {
          highlightedIndices.add(colIdx);
          addClass(th, "highlight-col");

          // Clean up synthetic marker tags from header text so it displays cleanly
          if (hasMarker) {
            cleanTextNodes(th, /\{highlight\}|\{:highlight\}|\{\.highlight\}|\[highlight\]|\[hl\]|\{hl\}/gi);
          } else if (hasExclamation) {
            cleanTextNodes(th, /^!!|!!$/g);
          }
        }
      });

      // 2. Highlight cells in all body rows and detect row highlights
      tbodyRows.forEach((row: any) => {
        const cells = row.children?.filter((c: any) => c.type === "element" && (c.tagName === "td" || c.tagName === "th")) || [];

        // Check if any cell in the row contains a highlight marker
        let isRowHighlighted = false;
        cells.forEach((cell: any) => {
          const text = getTextContent(cell);
          const rowMarkerCheck = /\{highlight(?:-row)?\}|\{:highlight(?:-row)?\}|\{\.highlight(?:-row)?\}|\[highlight(?:-row)?\]|\[hl\]|\{hl\}/i;
          const hasStar = text.includes("★") || text.includes("⭐");
          const hasMarker = rowMarkerCheck.test(text);

          if (hasMarker || hasStar) {
            isRowHighlighted = true;
            if (hasMarker) {
              cleanTextNodes(cell, /\{highlight(?:-row)?\}|\{:highlight(?:-row)?\}|\{\.highlight(?:-row)?\}|\[highlight(?:-row)?\]|\[hl\]|\{hl\}/gi);
            }
          }
        });

        if (isRowHighlighted) {
          addClass(row, "highlight-row");
        }

        if (highlightedIndices.size > 0) {
          highlightedIndices.forEach(colIdx => {
            if (cells[colIdx]) {
              addClass(cells[colIdx], "highlight-col");
            }
          });
        }
      });

      // 3. Wrap table in <div class="table-wrapper"> if not already wrapped
      if (parent && parent.type === "element" && parent.properties?.className) {
        const pClasses = Array.isArray(parent.properties.className)
          ? parent.properties.className
          : String(parent.properties.className).split(/\s+/);
        if (pClasses.includes("table-wrapper")) return;
      }

      if (parent && Array.isArray(parent.children) && index >= 0) {
        const wrapper = {
          type: "element",
          tagName: "div",
          properties: { className: ["table-wrapper"] },
          children: [table],
        };
        parent.children[index] = wrapper;
      }
    }

    visit(tree, null, -1);
  };
}
