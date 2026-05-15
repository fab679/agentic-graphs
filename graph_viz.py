#!/usr/bin/env python3
"""Generate a self-contained HTML graph visualisation from a JSON file.

Tip: Use FalkorDB's built-in browser at http://localhost:3000 instead.

Usage:
    python graph_viz.py graph.json              # writes graph.html
    python graph_viz.py graph.json output.html   # custom output path
"""

import json, sys, os


_HTML_TPL = """<!DOCTYPE html>
<meta charset="utf-8">
<title>Agentic Graph</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: system-ui, sans-serif; background: #0d0d1a; color: #e0e0e0; }
  #toolbar {
    padding: 10px 16px; background: #16162a; border-bottom: 1px solid #2a2a4a;
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  }
  #toolbar h1 { font-size: 16px; font-weight: 600; color: #c0c0e0; }
  #toolbar label { font-size: 13px; color: #8888aa; }
  #toolbar select, #toolbar input {
    background: #1e1e3a; color: #e0e0e0; border: 1px solid #3a3a5a;
    padding: 4px 8px; border-radius: 4px; font-size: 13px;
  }
  #toolbar select option { background: #1e1e3a; }
  #mynetwork { width: 100vw; height: calc(100vh - 48px); }
</style>
<div id="toolbar">
  <h1> Agentic Graph</h1>
  <label>Layout
    <select id="layout-select">
      <option value="barnesHut">Force-directed</option>
      <option value="hierarchicalRepulsion">Hierarchical</option>
    </select>
  </label>
  <label>Physics <input type="checkbox" id="physics-toggle" checked></label>
  <span id="node-count" style="font-size:13px;color:#8888aa;"></span>
</div>
<div id="mynetwork"></div>

<script src="https://unpkg.com/vis-network@9.1.6/dist/vis-network.min.js"></script>
<script>
const DATA = {DATA};

const TYPE_COLORS = {
  goal:     { background: '#7c3aed', border: '#9d6ff5' },
  task:     { background: '#2563eb', border: '#5b8def' },
  action:   { background: '#d97706', border: '#f59e0b' },
  synthesis:{ background: '#059669', border: '#34d399' },
};

const STATE_COLORS = {
  pending:  '#555577',
  ready:    '#3b82f6',
  active:   '#f59e0b',
  resolved: '#22c55e',
  failed:   '#ef4444',
};

function truncate(s, n) {
  return s && s.length > n ? s.slice(0, n) + '...' : s;
}

const nodes = new vis.DataSet(DATA.nodes.map(n => ({
  id: n.id,
  label: truncate(n.label, 28),
  title: 'id: ' + n.id + '\\ntype: ' + n.type + '\\nstate: ' + n.state + '\\nlabel: ' + (n.label || '') + '\\noutput: ' + (n.output || '-'),
  shape: 'box',
  color: {
    background: (TYPE_COLORS[n.type] || {}).background || '#666',
    border: (STATE_COLORS[n.state] || '#666'),
  },
  font: { color: '#fff', size: 12 },
  borderWidth: n.state === 'active' ? 3 : 1,
  chosen: false,
})));

const edges = new vis.DataSet(DATA.edges.map(e => ({
  id: e.id,
  from: e.src,
  to: e.dst,
  label: e.type,
  arrows: 'to',
  color: { color: '#555577', highlight: '#8888aa' },
  font: { color: '#8888aa', size: 10, strokeWidth: 0 },
  smooth: { type: 'curvedCW', roundness: 0.15 },
  chosen: false,
})));

const container = document.getElementById('mynetwork');
const options = {
  physics: {
    enabled: true,
    barnesHut: { gravitationalConstant: -4000, springLength: 200, springConstant: 0.04 },
  },
  layout: { improvedLayout: true },
  interaction: {
    hover: true, tooltipDelay: 100,
    navigationButtons: true, keyboard: true,
  },
  edges: { smooth: true },
  nodes: { borderWidth: 1 },
};

const network = new vis.Network(container, { nodes, edges }, options);
document.getElementById('node-count').textContent =
  nodes.length + ' nodes \u00b7 ' + edges.length + ' edges';

document.getElementById('physics-toggle').addEventListener('change', function () {
  network.setOptions({ physics: { enabled: this.checked } });
});

document.getElementById('layout-select').addEventListener('change', function () {
  const val = this.value;
  if (val === 'hierarchicalRepulsion') {
    network.setOptions({
      layout: { hierarchical: { direction: 'UD', sortMethod: 'directed' } },
      physics: { enabled: true, hierarchicalRepulsion: { nodeDistance: 150 } },
    });
  } else {
    network.setOptions({
      layout: { hierarchical: { enabled: false } },
      physics: { enabled: document.getElementById('physics-toggle').checked,
                  barnesHut: { gravitationalConstant: -4000 } },
    });
  }
});
</script>
"""


def render(data: dict, output: str = "graph.html") -> str:
    html = _HTML_TPL.replace("{DATA}", json.dumps(data, indent=2))
    with open(output, "w") as f:
        f.write(html)
    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(1)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else "graph.html"
    with open(src) as f:
        data = json.load(f)
    path = render(data, dst)
    print(f"  Written {path}")
