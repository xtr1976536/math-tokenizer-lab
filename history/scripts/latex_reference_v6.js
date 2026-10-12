/* Unmodified KaTeX 0.16.25. Input is data, never executable TeX or JS. */
const fs = require('fs');
const readline = require('readline');
const crypto = require('crypto');
const katex = require('../runtime/katex/node_modules/katex');
const settings = {output: 'mathml', throwOnError: true, strict: 'error', trust: false,
                  maxExpand: 1000, maxSize: 1000, displayMode: true};

function canonicalize(mathml) {
  return mathml.replace(/<annotation[\s\S]*?<\/annotation>/g, '')
               .replace(/\s+xmlns="[^"]*"/g, '').replace(/>[\s]+</g, '><');
}

function analyze(latex, sourceReference) {
  const mathml = katex.renderToString(latex, settings);
  const tree = katex.__parse(latex, settings);
  const offsets = new Map([[0, 0]]);
  let units = 0, bytes = 0;
  for (const char of latex) {
    units += char.length; bytes += Buffer.byteLength(char); offsets.set(units, bytes);
  }
  const intervals = new Set();
  const types = {}; const seen = new WeakSet(); let nodes = 0;
  function walk(value) {
    if (!value || typeof value !== 'object' || seen.has(value)) return null;
    seen.add(value);
    if (Array.isArray(value)) {
      const extents = value.map(walk).filter(Boolean);
      return extents.length ? [Math.min(...extents.map(v => v[0])), Math.max(...extents.map(v => v[1]))] : null;
    }
    if (value.type) {nodes++; types[value.type] = (types[value.type] || 0) + 1;}
    const location = value.loc;
    let own = null;
    if (location && offsets.has(location.start) && offsets.has(location.end) && location.end > location.start
        && (!location.lexer || location.lexer.input === latex)) {
      own = [offsets.get(location.start), offsets.get(location.end)];
      intervals.add(own.join(','));
    }
    const descendants = [];
    for (const [key, child] of Object.entries(value)) {
      if (key !== 'loc' && key !== 'lexer') {const extent = walk(child); if (extent) descendants.push(extent);}
    }
    if (descendants.length) {
      const extent = [Math.min(...descendants.map(v => v[0])), Math.max(...descendants.map(v => v[1]))];
      // Some KaTeX layout nodes have no source location. Child envelopes are
      // explicitly syntax spans, never described as semantic truth annotations.
      if (value.type && extent[1] > extent[0]) intervals.add(extent.join(','));
      if (!own) own = extent;
    }
    return own;
  }
  walk(tree);
  const spans = [...intervals].map(v => v.split(',').map(Number)).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  // The annotation embeds original TeX and must not dominate equivalence keys.
  const canonical = canonicalize(mathml);
  let sourceMacroFidelity = null;
  if (sourceReference) {
    const renderedSource = katex.renderToString(sourceReference.macro_definitions + sourceReference.unexpanded_latex, settings);
    sourceMacroFidelity = canonical === canonicalize(renderedSource);
  }
  return {valid: true, katex_version: katex.version, syntax_kind: 'independent_katex_syntax_ast',
          spans, node_types: types, nodes, source_macro_fidelity: sourceMacroFidelity,
          mathml_sha256: crypto.createHash('sha256').update(canonical).digest('hex')};
}

const input = readline.createInterface({input: process.argv[2] ? fs.createReadStream(process.argv[2]) : process.stdin,
                                      crlfDelay: Infinity});
input.on('line', line => {
  if (!line.trim()) return;
  const record = JSON.parse(line); let output;
  try {output = {...analyze(record.latex, record.source_macro_reference), id: record.id};}
  catch (error) {output = {id: record.id, valid: false, katex_version: katex.version, reason: String(error.message).slice(0, 400)};}
  process.stdout.write(JSON.stringify(output) + '\n');
});
