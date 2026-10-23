const el = id => document.getElementById(id);
const encoder = new TextEncoder();
const decoder = new TextDecoder();
const state = {size: 350, models: {}, samples: [], selected: null, level: 'all', steps: [], current: 0, timer: null};
const hex = bytes => Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('');
const bytes = value => new Uint8Array(value.match(/../g).map(part => parseInt(part, 16)));
const visible = value => value.replaceAll(' ', '␠').replaceAll('\n', '↵').replaceAll('\t', '⇥').replaceAll('\r', '↵');

function unitEnd(raw, at) {
  let end = at + 1;
  const letter = byte => (byte >= 65 && byte <= 90) || (byte >= 97 && byte <= 122);
  if (raw[at] === 92 && end < raw.length) {
    if (letter(raw[end])) while (end < raw.length && letter(raw[end])) end++;
    else end += raw[end] < 128 ? 1 : raw[end] < 224 ? 2 : raw[end] < 240 ? 3 : 4;
  } else if (raw[at] >= 128) end = at + (raw[at] < 224 ? 2 : raw[at] < 240 ? 3 : 4);
  else if (raw[at] >= 48 && raw[at] <= 57) {
    while (end < raw.length && raw[end] >= 48 && raw[end] <= 57) end++;
    if (end + 1 < raw.length && raw[end] === 46 && raw[end + 1] >= 48 && raw[end + 1] <= 57) {
      end++;
      while (end < raw.length && raw[end] >= 48 && raw[end] <= 57) end++;
    }
  }
  return Math.min(end, raw.length);
}

// 与 native/frontend_v6.c 的通用命令、花括号和上下标区间规则相同。
function bindingSpans(raw) {
  const n = raw.length, spans = [], close = Array(n + 1).fill(-1), completed = Array(n + 1).fill(-1), stack = [];
  const white = byte => byte === 32 || byte === 9 || byte === 10 || byte === 13;
  for (let at = 0; at < n;) {
    const end = unitEnd(raw, at);
    spans.push([at, end]);
    if (raw[at] === 123 && end === at + 1) stack.push(at);
    else if (raw[at] === 125 && end === at + 1 && stack.length) {
      const begin = stack.pop(); close[begin] = end; completed[end] = begin; spans.push([begin, end]);
    }
    at = end;
  }
  let previous = 0;
  for (let at = 0; at < n;) {
    const end = unitEnd(raw, at);
    if (raw[at] === 92 && at + 1 < n && /[A-Za-z]/.test(String.fromCharCode(raw[at + 1]))) {
      let p = end;
      while (p < n && white(raw[p])) p++;
      while (p < n && raw[p] === 123 && close[p] > p) {
        p = close[p]; spans.push([at, p]); completed[p] = at;
        while (p < n && white(raw[p])) p++;
      }
    }
    if ((raw[at] === 94 || raw[at] === 95) && at > 0) {
      let p = end;
      while (p < n && white(raw[p])) p++;
      if (p < n) {
        const finish = raw[p] === 123 && close[p] > p ? close[p] : unitEnd(raw, p);
        spans.push([previous, finish]); completed[finish] = previous;
      }
    }
    if (!white(raw[at])) previous = completed[end] >= 0 ? completed[end] : at;
    at = end;
  }
  if (n) spans.push([0, n]);
  const parsed = parseArithmeticSpans(raw);
  if (parsed) spans.push(...parsed);
  return [...new Map(spans.map(span => [span.join(':'), span])).values()];
}

function parseArithmeticSpans(raw) {
  const n = raw.length, spans = [];
  const isLetter = byte => (byte >= 65 && byte <= 90) || (byte >= 97 && byte <= 122);
  const isDigit = byte => byte >= 48 && byte <= 57;
  const isWhite = byte => byte === 32 || byte === 9 || byte === 10 || byte === 13;
  let position = 0, depth = 0, valid = true;
  const spaces = () => {while (position < n && isWhite(raw[position])) position++;};
  const commandIs = (start, end, name) => decoder.decode(raw.slice(start, end)) === name;
  const add = (start, end) => {if (end > start) spans.push([start, end]);};
  function atom(scripts = true) {
    spaces(); const start = position;
    if (start >= n || ++depth > 128) { valid = false; depth--; return {start, end: start}; }
    const c = raw[start]; let result = {start, end: start};
    if (c === 43 || c === 45) {position++; result.end = expression(4).end;}
    else if (c === 123 || c === 40) {
      position++; expression(0); spaces();
      if (position >= n || raw[position] !== (c === 123 ? 125 : 41)) valid = false;
      else position++;
      result.end = position;
    } else if (isLetter(c) || c >= 128 || isDigit(c)) {position = unitEnd(raw, start); result.end = position;}
    else if (c === 92) {
      const end = unitEnd(raw, start); position = end;
      const fraction = ['\\frac', '\\dfrac', '\\tfrac'].some(name => commandIs(start, end, name));
      const arity = fraction ? 2 : commandIs(start, end, '\\sqrt') ? 1 : 0;
      if (!arity) valid = false;
      for (let i = 0; i < arity && valid; i++) {
        spaces(); if (position >= n || raw[position] !== 123) {valid = false; break;}
        atom(false);
      }
      result.end = position;
    } else valid = false;
    if (valid) add(result.start, result.end);
    while (valid && scripts) {
      spaces(); if (position >= n || (raw[position] !== 94 && raw[position] !== 95)) break;
      position++; const argument = atom(false); result.end = argument.end; add(result.start, result.end);
    }
    depth--; return result;
  }
  function expression(minimum) {
    const left = atom();
    while (valid) {
      spaces(); const start = position; if (start >= n) break;
      const c = raw[start]; let precedence = 0, end = start + 1, implicit = false;
      if ([61, 60, 62].includes(c)) precedence = 1;
      else if (c === 43 || c === 45) precedence = 2;
      else if (c === 42 || c === 47) precedence = 3;
      else if (isLetter(c) || c >= 128 || isDigit(c) || c === 123 || c === 40) {precedence = 3; implicit = true;}
      else if (c === 92) {
        end = unitEnd(raw, start);
        if (['\\cdot', '\\times', '\\div'].some(name => commandIs(start, end, name))) precedence = 3;
        else if (['\\leq', '\\geq', '\\neq'].some(name => commandIs(start, end, name))) precedence = 1;
        else {precedence = 3; implicit = true;}
      }
      if (!precedence || precedence < minimum) break;
      if (!implicit) position = end;
      const right = expression(precedence + 1); left.end = right.end; if (valid) add(left.start, left.end);
    }
    return left;
  }
  if (!n) return [];
  expression(0); spaces();
  return valid && position === n ? spans : null;
}

function spanCompatible(start, end, spans) {
  return spans.every(([a, b]) => end <= a || start >= b || (start <= a && end >= b) || (start >= a && end <= b));
}

function tokenize(formula, model, spans) {
  const raw = encoder.encode(formula);
  if (raw.length > 4096) throw Error('公式超过 4096 字节，请缩短后再查看过程。');
  const allowed = new Set(model.alphabet);
  const unsupported = [...new Set(raw)].filter(byte => !allowed.has(byte));
  if (unsupported.length) throw Error(`当前冻结词表不包含字节：${unsupported.map(byte => '0x' + byte.toString(16).padStart(2, '0')).join('、')}。请使用词表支持的 LaTeX 字符。`);
  let tokens = Array.from(raw, (id, at) => ({id, start: at, end: at + 1}));
  const rules = new Map(model.merges.map(([left, right, id], rank) => [`${left}:${right}`, {rank, id}]));
  const steps = [{tokens: tokens.slice(), merge: null, prevented: 0}];
  const regions = spans || bindingSpans(raw);
  for (;;) {
    let best = null, blocked = 0;
    for (let index = 0; index < tokens.length - 1; index++) {
      const left = tokens[index], right = tokens[index + 1];
      const rule = rules.get(`${left.id}:${right.id}`);
      if (!rule) continue;
      if (!spanCompatible(left.start, right.end, regions)) { blocked++; continue; }
      if (!best || rule.rank < best.rule.rank || (rule.rank === best.rule.rank && index < best.index)) best = {index, rule, left, right};
    }
    if (!best) break;
    const merged = {id: best.rule.id, start: best.left.start, end: best.right.end, fresh: true};
    tokens = tokens.map(token => ({...token, fresh: false}));
    tokens.splice(best.index, 2, merged);
    steps.push({tokens: tokens.map(token => ({...token})), merge: best, prevented: blocked});
  }
  const compact = new Map(model.compact_ids.map((internal, id) => [internal, id + 3]));
  const finalIds = tokens.map(token => compact.get(token.id));
  if (finalIds.some(id => id === undefined)) throw Error('最终词表映射不完整。');
  return {steps, raw, regions, finalIds};
}

globalThis.astBpeVisualizer = {tokenize, bindingSpans};

function stop() { if (state.timer) clearInterval(state.timer); state.timer = null; el('play').textContent = '▶ 播放'; }
function showSamples() {
  const query = el('search').value.trim().toLowerCase();
  const matching = state.samples.filter(row => (state.level === 'all' || row.level === state.level) && (!query || row.latex.toLowerCase().includes(query) || row.id.toLowerCase().includes(query)));
  if (!query && state.selected && matching.includes(state.selected)) matching.unshift(...matching.splice(matching.indexOf(state.selected), 1));
  const shown = matching.slice(0, 120);
  const list = el('samples'); list.replaceChildren();
  for (const row of shown) {
    const button = document.createElement('button'); button.type = 'button'; button.className = `sample-item${state.selected?.id === row.id ? ' selected' : ''}`;
    const number = document.createElement('span'); number.className = 'sample-index'; number.textContent = row.level === 'simple' ? '简单' : row.level === 'moderate' ? '中等' : '复杂';
    const text = document.createElement('span'); text.className = 'sample-formula'; text.textContent = row.latex;
    button.append(number, text); button.title = row.id; button.onclick = () => {state.selected = row; el('formula').value = row.latex; showSamples(); refresh();};
    list.append(button);
  }
  if (!shown.length) list.textContent = '没有匹配的公式';
  el('sample-count').textContent = `${state.samples.length} 条`;
}

let calculation = null;
function render() {
  const result = calculation; const step = result?.steps[state.current];
  el('step-label').textContent = result ? `${state.current} / ${result.steps.length - 1}` : '0 / 0';
  el('step-range').max = Math.max(0, (result?.steps.length || 1) - 1); el('step-range').value = state.current;
  el('reset').disabled = el('previous').disabled = !result || state.current === 0;
  el('next').disabled = !result || state.current >= result.steps.length - 1;
  el('play').disabled = !result || result.steps.length <= 1;
  el('tokens-now').textContent = step ? step.tokens.length : '—';
  el('tokens-final').textContent = result ? result.finalIds.length : '—';
  el('compression').textContent = result?.finalIds.length ? (result.raw.length / result.finalIds.length).toFixed(2) + '×' : '—';
  el('stage-title').textContent = !result ? '等待公式' : state.current === 0 ? '初始字节序列' : state.current === result.steps.length - 1 ? '最终分词结果' : `第 ${state.current} 次合并`;
  el('byte-count').textContent = `${result?.raw.length || 0} 字节`;
  const track = el('token-track'); track.replaceChildren();
  if (!step) {const empty = document.createElement('span'); empty.className = 'token empty'; empty.textContent = '输入公式后将在这里显示 token'; track.append(empty); return;}
  for (const token of step.tokens) {
    const node = document.createElement('span'); node.className = `token${token.end - token.start === 1 ? ' base' : ''}${token.fresh ? ' new' : ''}`;
    const decoded = decoder.decode(result.raw.slice(token.start, token.end));
    node.textContent = visible(decoded); node.title = `内部 ID ${token.id} · 字节 ${token.start}–${token.end}`; track.append(node);
  }
  if (step.merge) {
    const left = visible(decoder.decode(result.raw.slice(step.merge.left.start, step.merge.left.end)));
    const right = visible(decoder.decode(result.raw.slice(step.merge.right.start, step.merge.right.end)));
    el('merge-detail').textContent = `${left} + ${right} → ${visible(decoder.decode(result.raw.slice(step.merge.left.start, step.merge.right.end)))}`;
    el('rank-detail').textContent = `第 ${step.merge.rule.rank + 1} 条合并规则`;
  } else { el('merge-detail').textContent = '尚未合并'; el('rank-detail').textContent = '—'; }
  el('boundary-detail').textContent = `${result.regions.length} 个结构区间${step.prevented ? ` · 本步拦截 ${step.prevented} 处跨界合并` : ''}`;
}

function refresh() {
  stop(); state.current = 0; calculation = null;
  const text = el('formula').value; const notice = el('notice'); notice.hidden = true;
  if (!text) { render(); return; }
  try {
    const row = state.selected?.latex === text ? state.selected : null;
    calculation = tokenize(text, state.models[state.size], row?.spans);
    el('status').textContent = row ? 'Benchmark 公式' : '自定义公式';
  } catch (error) { notice.textContent = error.message; notice.hidden = false; el('status').textContent = '无法分词'; }
  render();
}

async function load() {
  try {
    if (!globalThis.AST_BPE_DATA) throw Error('离线数据文件未加载');
    state.models = globalThis.AST_BPE_DATA.models;
    state.samples = globalThis.AST_BPE_DATA.formulas;
    if (state.samples.length !== 1500) throw Error('Benchmark 公式数量不匹配');
    showSamples(); state.selected = state.samples.find(row => row.level === 'simple' && row.latex.startsWith('\\partial_t U + \\partial_x F(U)=0')) || state.samples[0]; el('formula').value = state.selected.latex; refresh(); showSamples();
  } catch (error) {el('status').textContent = '数据加载失败'; el('notice').hidden = false; el('notice').textContent = `无法加载分词数据：${error.message}`;}
}
if (typeof document !== 'undefined') {
  el('formula').addEventListener('input', () => {if (state.selected?.latex !== el('formula').value) state.selected = null; refresh(); showSamples();});
  el('search').addEventListener('input', showSamples);
  document.querySelectorAll('[data-level]').forEach(button => button.onclick = () => {state.level = button.dataset.level; document.querySelectorAll('[data-level]').forEach(item => item.classList.toggle('selected', item === button)); showSamples();});
  document.querySelectorAll('[data-size]').forEach(button => button.onclick = () => {state.size = Number(button.dataset.size); document.querySelectorAll('[data-size]').forEach(item => item.classList.toggle('selected', item === button)); refresh();});
  el('reset').onclick = () => {stop(); state.current = 0; render();};
  el('previous').onclick = () => {stop(); state.current = Math.max(0, state.current - 1); render();};
  el('next').onclick = () => {stop(); state.current = Math.min(calculation.steps.length - 1, state.current + 1); render();};
  el('step-range').oninput = event => {stop(); state.current = Number(event.target.value); render();};
  el('play').onclick = () => {if (state.timer) {stop(); return;} if (state.current >= calculation.steps.length - 1) state.current = 0; el('play').textContent = '❚❚ 暂停'; state.timer = setInterval(() => {state.current++; render(); if (state.current >= calculation.steps.length - 1) stop();}, 280);};
  load();
}
