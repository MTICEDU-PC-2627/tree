'use strict';

const DATASETS = {
  golf: {
    name: 'Golfe',
    classes: ['Não', 'Sim'],
    colors: ['#ad521e', '#00786e'],
    target: 'Jogou?',
    features: [
      { name: 'Céu', values: ['Sol', 'Nublado', 'Chuva'] },
      { name: 'Temperatura', values: ['Quente', 'Amena', 'Fresca'] },
      { name: 'Humidade', values: ['Elevada', 'Normal'] },
      { name: 'Vento', values: ['Fraco', 'Forte'] }
    ],
    // Dados da tabela clássica PlayTennis, com o contexto adaptado a golfe.
    rows: [
      [0,0,0,0,0], [0,0,0,1,0], [1,0,0,0,1], [2,1,0,0,1],
      [2,2,1,0,1], [2,2,1,1,0], [1,2,1,1,1], [0,1,0,0,0],
      [0,2,1,0,1], [2,1,1,0,1], [0,1,1,1,1], [1,1,0,1,1],
      [1,0,1,0,1], [2,1,0,1,0]
    ]
  },
  iris: {
    name: 'Iris',
    classes: ['Setosa', 'Versicolor', 'Virginica'],
    colors: ['#00786e', '#ad521e', '#6154a0'],
    target: 'Espécie',
    features: [
      { name: 'Comprimento da sépala' }, { name: 'Largura da sépala' },
      { name: 'Comprimento da pétala' }, { name: 'Largura da pétala' }
    ],
    rows: IRIS
  },
  credit: {
    name: 'Crédito',
    classes: ['Recusado', 'Aprovado'],
    colors: ['#ad521e', '#00786e'],
    target: 'Decisão',
    features: [
      { name: 'Rendimento regular', question: 'Tem rendimento regular?', values: ['Não', 'Sim'] },
      { name: 'Atrasos nos últimos 12 meses', question: 'Teve atrasos nos últimos 12 meses?', values: ['Não', 'Sim'] },
      { name: 'Prestação ≥ 30% do rendimento', question: 'Prestação ≥ 30% do rendimento?', values: ['Não', 'Sim'] },
      { name: 'Montante pedido > 10 000 €', question: 'Montante pedido > 10 000 €?', values: ['Não', 'Sim'] }
    ],
    rows: [
      [0,1,0,0,1], [0,1,0,1,1], [0,1,1,0,0], [1,0,0,0,1], [1,0,0,1,1],
      [1,0,1,0,1], [1,1,0,0,1], [1,1,0,1,0], [1,1,1,0,0], [1,1,1,1,0]
    ]
  }
};

const $ = id => document.getElementById(id);
const fmt = (number, digits = 2) => number.toLocaleString('pt-PT', { maximumFractionDigits: digits });
const makeRoot = rows => ({ rows, path: '', children: null });

let datasetKey = 'golf';
let mode = 'explore';
let criterion = 'gini';
let axis = 3;
let showAll = false;
let rule;
let scored = [];
const datasetKeys = Object.keys(DATASETS);
const sortOrder = Object.fromEntries(datasetKeys.map(key => [key, null]));
let dataSource = 'node';
const samples = Object.fromEntries(Object.entries(DATASETS).map(([key, data]) => [key, { percent: 100, train: data.rows, test: [] }]));
const roots = Object.fromEntries(Object.entries(samples).map(([key, sample]) => [key, makeRoot(sample.train)]));
const evaluationOpen = Object.fromEntries(datasetKeys.map(key => [key, false]));
let node = roots.golf;

function count(rows, dataset) {
  const counts = dataset.classes.map(() => 0);
  for (const row of rows) counts[row[4]]++;
  return counts;
}

function impurity(counts, metric) {
  const n = counts.reduce((sum, value) => sum + value, 0);
  if (!n) return 0;
  const probabilities = counts.filter(Boolean).map(value => value / n);
  return metric === 'gini'
    ? Math.max(0, 1 - probabilities.reduce((sum, p) => sum + p * p, 0))
    : probabilities.reduce((sum, p) => sum - p * Math.log2(p), 0);
}

function matchesRule(row, candidate, dataset) {
  const { feature, value } = candidate;
  return dataset.features[feature].values ? row[feature] === value : row[feature] <= value;
}

function partition(rows, candidate, dataset) {
  const yes = [], no = [];
  for (const row of rows) (matchesRule(row, candidate, dataset) ? yes : no).push(row);
  return [yes, no];
}

function shuffle(items) {
  const shuffled = [...items];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

function featureGroups(dataset) {
  return dataset.features.flatMap((feature, index) => {
    if (feature.values) return feature.values.map((_, value) => dataset.rows.filter(row => row[index] === value));
    const sorted = dataset.rows.map(row => row[index]).sort((a, b) => a - b);
    const cutoffs = [0.25, 0.5, 0.75].map(q => sorted[Math.ceil(q * sorted.length) - 1]);
    const bounds = [-Infinity, ...new Set(cutoffs), Infinity];
    return bounds.slice(1).map((upper, bin) => dataset.rows.filter(row =>
      row[index] > bounds[bin] && row[index] <= upper));
  });
}

function splitSample(dataset, percent) {
  if (percent === 100) return { percent, train: dataset.rows, test: [] };
  const total = dataset.rows.length;
  const target = Math.round(total * percent / 100);
  const ratio = target / total;
  const strata = dataset.classes.map((_, index) => {
    const rows = dataset.rows.filter(row => row[4] === index);
    const quota = ratio * rows.length;
    return { rows, remainder: quota % 1, seats: Math.floor(quota) };
  });
  const unallocated = target - strata.reduce((sum, stratum) => sum + stratum.seats, 0);
  const byRemainder = shuffle(strata).sort((a, b) => b.remainder - a.remainder);
  byRemainder.slice(0, unallocated).forEach(stratum => stratum.seats++);

  // Compare random class-stratified draws by their marginal feature distributions.
  const groups = featureGroups(dataset);
  let best, bestError = Infinity;
  for (let attempt = 0; attempt < 64; attempt++) {
    const train = new Set(strata.flatMap(({ rows, seats }) => shuffle(rows).slice(0, seats)));
    const error = groups.reduce((sum, group) => {
      const observed = group.filter(row => train.has(row)).length;
      return sum + Math.abs(observed - ratio * group.length);
    }, 0);
    if (error < bestError) {
      best = train;
      bestError = error;
    }
  }
  return { percent, train: dataset.rows.filter(row => best.has(row)), test: dataset.rows.filter(row => !best.has(row)) };
}

function possibleValues(rows, index, dataset) {
  const observed = [...new Set(rows.map(row => row[index]))].sort((a, b) => a - b);
  if (observed.length < 2) return [];
  // For two categories, the second choice gives the same split with swapped branches.
  if (dataset.features[index].question) return [1];
  if (dataset.features[index].values) return observed.length === 2 ? observed.slice(0, 1) : observed;
  return observed.slice(1).map((value, i) => Number(((value + observed[i]) / 2).toFixed(4)));
}

function splitGain(rows, groups, dataset, metric) {
  const parent = impurity(count(rows, dataset), metric);
  const children = groups.reduce((sum, group) => sum + group.length * impurity(count(group, dataset), metric), 0);
  return Math.max(0, parent - children / rows.length);
}

function rank(rows, dataset, metric) {
  return dataset.features.flatMap((_, feature) =>
    possibleValues(rows, feature, dataset).map(value => {
      const candidate = { feature, value };
      const groups = partition(rows, candidate, dataset);
      return { ...candidate, gain: splitGain(rows, groups, dataset, metric), sizes: groups.map(group => group.length) };
    })
  ).sort((a, b) => b.gain - a.gain || a.feature - b.feature || a.value - b.value);
}

function growTree(rows, data, metric, path = '') {
  const item = { ...makeRoot(rows), path };
  if (count(rows, data).filter(Boolean).length === 1) return item;
  const best = rank(rows, data, metric)[0];
  if (!best) return item;
  item.split = { feature: best.feature, value: best.value };
  item.children = partition(rows, item.split, data).map((group, i) => growTree(group, data, metric, path + i));
  return item;
}

function currentData() {
  return DATASETS[datasetKey];
}

function defaultRule(index = datasetKey === 'iris' ? 2 : 0) {
  const data = currentData();
  const available = data.features.map((_, feature) => possibleValues(node.rows, feature, data));
  if (!available[index].length) index = available.findIndex(values => values.length);
  if (index < 0) return null;
  const values = available[index];
  const middle = data.features[index].values ? 0 : Math.floor(values.length / 2);
  return { feature: index, value: values[middle] };
}

function label(candidate, dataset) {
  const feature = dataset.features[candidate.feature];
  if (feature.question) return feature.question;
  return feature.values
    ? `${feature.name} = ${feature.values[candidate.value]}?`
    : `${feature.name} ≤ ${fmt(candidate.value)} cm?`;
}

function majorityClasses(counts) {
  const largest = Math.max(...counts);
  return counts.flatMap((value, index) => value === largest ? [index] : []);
}

function leafLabel(rows, dataset) {
  return majorityClasses(count(rows, dataset)).map(index => dataset.classes[index]).join(' / ');
}

function predict(row, root, dataset) {
  let leaf = root;
  while (leaf.children) leaf = leaf.children[matchesRule(row, leaf.split, dataset) ? 0 : 1];
  const classes = majorityClasses(count(leaf.rows, dataset));
  return classes.length === 1 ? classes[0] : null;
}

function chooseNode(next) {
  node = next;
  rule = next.split ?? defaultRule();
  showAll = false;
  render();
}

function distribution(counts, dataset) {
  const stack = counts.map((n, i) => `<span style="flex:${n};background:${dataset.colors[i]}"></span>`).join('');
  const chips = counts.map((n, i) => `<span class="chip" style="--class-color:${dataset.colors[i]}">${dataset.classes[i]} ${n}</span>`).join('');
  return `<span class="stack" aria-hidden="true">${stack}</span><span class="chips">${chips}</span>`;
}

function groupCard(title, rows, dataset) {
  const counts = count(rows, dataset);
  const examples = datasetKey !== 'iris' ? rows.map(row => {
    const id = dataset.rows.indexOf(row) + 1;
    const outcome = dataset.classes[row[4]];
    return `<li class="case" style="--class-color:${dataset.colors[row[4]]}"><span>#${id}</span> ${outcome}</li>`;
  }).join('') : '';
  const pure = counts.includes(rows.length);
  return `<div class="group${pure ? ' pure' : ''}"><h3>${title}<span>${rows.length} ${rows.length === 1 ? 'exemplo' : 'exemplos'}</span></h3>${distribution(counts, dataset)}
    ${examples ? `<ul class="cases" aria-label="Resultado de cada observação">${examples}</ul>` : ''}</div>`;
}

function renderTree() {
  const data = currentData();
  let leaves = 0, splits = 0;
  function branch(item, tag) {
    if (item.children) splits++;
    else leaves++;
    const selected = node === item;
    const majority = leafLabel(item.rows, data);
    const caption = item.split ? label(item.split, data) : majority;
    const counts = count(item.rows, data);
    const purity = fmt(100 * Math.max(...counts) / item.rows.length, 1);
    const pure = counts.includes(item.rows.length);
    const children = item.children ? `<ul>${branch(item.children[0], 'Sim')}${branch(item.children[1], 'Não')}</ul>` : '';
    return `<li><button type="button" class="tree-node${pure ? ' pure' : item.children ? '' : ' impure-leaf'}" data-path="${item.path}" aria-current="${selected}"><small>${tag} · ${item.rows.length} exemplos</small>${caption}
      <small>${item.split ? `Maioria: ${majority} · ` : ''}Pureza: ${purity}%</small>
      ${distribution(counts, data)}</button>${children}</li>`;
  }
  const tree = $('tree');
  tree.innerHTML = `<ul class="tree-list">${branch(roots[datasetKey], 'Raiz')}</ul>`;
  $('tree-summary').textContent = `${splits} ${splits === 1 ? 'partição' : 'partições'} · ${leaves} ${leaves === 1 ? 'folha' : 'folhas'}`;
}

function setOptions(select, entries, selected) {
  select.replaceChildren(...entries.map(([value, name]) => new Option(name, value, false, value === selected)));
}

function renderSplit() {
  const data = currentData();
  const locked = Boolean(node.children);
  const pure = count(node.rows, data).filter(Boolean).length === 1;
  $('node-path').textContent = node.path ? `Raiz / ${[...node.path].map(step => step === '0' ? 'Sim' : 'Não').join(' / ')}` : 'Raiz';
  $('node-count').textContent = `${node.rows.length} exemplos`;
  $('node-description').textContent = locked ? label(node.split, data) : 'Compara a distribuição das classes nos dois ramos.';
  $('split-form').hidden = locked || pure || !rule;
  if (rule && !locked && !pure) {
    const features = data.features.flatMap((feature, index) =>
      possibleValues(node.rows, index, data).length ? [[index, feature.name]] : []);
    const feature = data.features[rule.feature];
    const values = possibleValues(node.rows, rule.feature, data);
    const numeric = !feature.values;
    $('value-label').textContent = numeric ? 'Menor ou igual a' : 'Igual a';
    $('value-label').htmlFor = numeric ? 'threshold' : 'value';
    $('value').hidden = numeric;
    $('slider-control').hidden = !numeric;
    setOptions($('feature'), features, rule.feature);
    if (numeric) {
      const observed = node.rows.map(row => row[rule.feature]);
      const threshold = $('threshold');
      threshold.min = Math.min(...observed);
      threshold.max = (Math.max(...observed) - 0.01).toFixed(2);
      threshold.value = rule.value;
      $('threshold-value').value = `${fmt(rule.value)} cm`;
      threshold.setAttribute('aria-valuetext', `${fmt(rule.value)} centímetros`);
    } else {
      setOptions($('value'), values.map(value => [value, feature.values[value]]), rule.value);
    }
  }
  const note = $('node-note');
  note.hidden = !locked && !pure && Boolean(rule);
  if (locked) {
    const hasDescendants = node.children.some(child => child.children);
    note.innerHTML = `Partição registada. Seleciona o ramo Sim ou Não na árvore para continuar. <button type="button" class="quiet" id="prune">Desfazer partição${hasDescendants ? ' e descendentes' : ''}</button>`;
  } else {
    note.textContent = pure ? 'Todos os exemplos têm a mesma classe.' : 'As medidas disponíveis são iguais neste nó.';
  }
  const activeRule = locked ? node.split : pure ? null : rule;
  const groups = activeRule ? partition(node.rows, activeRule, data) : null;
  const question = $('preview-question');
  question.hidden = locked || !activeRule;
  question.textContent = activeRule ? label(activeRule, data) : '';
  $('groups').innerHTML = groups
    ? groupCard('Ramo Sim', groups[0], data) + groupCard('Ramo Não', groups[1], data)
    : groupCard('Distribuição no nó', node.rows, data);
}

function renderSelectedScore() {
  const candidate = node.split ?? rule;
  const metricName = criterion === 'gini' ? 'Redução da impureza de Gini' : 'Ganho de informação';
  const summary = $('metric-summary');
  summary.hidden = mode !== 'metrics' || !candidate || (!node.children && $('split-form').hidden);
  if (!summary.hidden) {
    const data = currentData();
    const groups = partition(node.rows, candidate, data);
    summary.textContent = `${metricName} desta partição: ${fmt(splitGain(node.rows, groups, data, criterion), 4)}`;
  }
}

function renderMetrics() {
  const data = currentData();
  scored = rank(node.rows, data, criterion);
  const splittable = !node.children && count(node.rows, data).filter(Boolean).length > 1;
  const limit = showAll ? scored.length : 8;
  const heading = criterion === 'gini' ? 'Redução da impureza de Gini' : 'Ganho de informação';
  $('rankings').innerHTML = scored.length ? `<table><thead><tr><th scope="col">Condição</th><th scope="col">Sim / Não</th><th scope="col">${heading}</th>${splittable ? '<th scope="col">Pré-visualizar</th>' : ''}</tr></thead><tbody>
    ${scored.slice(0, limit).map((candidate, i) => `<tr><td>${label(candidate, data)}</td><td>${candidate.sizes.join(' / ')}</td><td>${fmt(candidate.gain, 4)}</td>${splittable ? `<td><button type="button" class="quiet" data-rank="${i}">Experimentar</button></td>` : ''}</tr>`).join('')}</tbody></table>`
    : '<p class="note">Este nó não tem partições candidatas.</p>';
  const more = $('show-more');
  more.hidden = scored.length <= 8;
  more.textContent = showAll ? 'Mostrar as primeiras 8' : `Mostrar todas (${scored.length})`;
  renderSelectedScore();
}

function renderChart() {
  const data = currentData();
  const activeRule = node.split ?? ($('split-form').hidden ? null : rule);
  const x = activeRule?.feature ?? 2;
  const y = axis === x ? (x === 3 ? 2 : 3) : axis;
  setOptions($('axis'), data.features.flatMap((feature, i) => i === x ? [] : [[i, feature.name]]), y);
  const scale = { min: 0, max: 8, percent: value => 3 + 94 * value / 8 };
  const sx = scale, sy = scale;
  const dots = node.rows.map(row => `<i class="dot" style="left:${sx.percent(row[x])}%;bottom:${sy.percent(row[y])}%;--class-color:${data.colors[row[4]]}"></i>`).join('');
  const cut = activeRule ? `<span class="cut" style="left:${sx.percent(activeRule.value)}%"></span>` : '';
  const ticks = [0, .25, .5, .75, 1].map(fraction => {
    const a = sx.min + (sx.max - sx.min) * fraction;
    const b = sy.min + (sy.max - sy.min) * fraction;
    return `<span class="tick x" style="left:${sx.percent(a)}%">${fmt(a)}</span><span class="tick y" style="bottom:${sy.percent(b)}%">${fmt(b)}</span>`;
  }).join('');
  $('plot').innerHTML = dots + cut + ticks;
  $('plot').setAttribute('aria-label', `Gráfico com ${node.rows.length} exemplos. Eixo horizontal: ${data.features[x].name}. Eixo vertical: ${data.features[y].name}.${activeRule ? ' Corte em ' + fmt(activeRule.value) + ' cm.' : ''}`);
  $('plot-legend').innerHTML = `<strong>Eixo horizontal: ${data.features[x].name} (cm)</strong>` + data.classes.map((name, i) =>
    `<span class="chip" style="--class-color:${data.colors[i]}">${name}</span>`).join('');
}

function renderData() {
  const data = currentData();
  const sort = sortOrder[datasetKey];
  const columns = [...data.features.map(f => f.name + (f.values ? '' : ' (cm)')), data.target];
  const header = columns.map((name, index) => {
    const direction = sort?.column === index ? (sort.direction === 1 ? 'ascending' : 'descending') : 'none';
    const arrow = direction === 'ascending' ? '↑' : direction === 'descending' ? '↓' : '↕';
    return `<th scope="col" aria-sort="${direction}"><button type="button" class="sort-button" data-sort="${index}">${name} <span aria-hidden="true">${arrow}</span></button></th>`;
  }).join('');
  const rowsToShow = dataSource === 'node' ? node.rows : samples[datasetKey][dataSource];
  $('data-description').textContent = rowsToShow.length
    ? `${rowsToShow.length} ${rowsToShow.length === 1 ? 'exemplo' : 'exemplos'}. Clica no cabeçalho para ordenar.`
    : 'Sem exemplos de teste. Reserva uma percentagem em Treino e teste.';
  $('data-table').hidden = !rowsToShow.length;
  const sorted = [...rowsToShow];
  if (sort) {
    const names = data.features[sort.column]?.values ?? (sort.column === 4 ? data.classes : null);
    sorted.sort((a, b) => {
      const left = a[sort.column], right = b[sort.column];
      return sort.direction * (names ? names[left].localeCompare(names[right], 'pt-PT') : left - right);
    });
  }
  const rows = sorted.map(row => `<tr><th scope="row">#${data.rows.indexOf(row) + 1}</th>${data.features.map((f, i) => `<td>${f.values?.[row[i]] ?? fmt(row[i])}</td>`).join('')}<td>${data.classes[row[4]]}</td></tr>`).join('');
  $('data-table').innerHTML = `<table><thead><tr><th scope="col">Obs.</th>${header}</tr></thead><tbody>${rows}</tbody></table>`;
}

function renderSample() {
  const sample = samples[datasetKey];
  const total = sample.train.length + sample.test.length;
  $('sample-summary').textContent = `${sample.train.length} de ${total} em treino · ${sample.test.length} em teste${sample.test.length ? ' · sorteio estratificado por classe, equilibrado por característica' : ''}`;
}

function evaluateRows(rows, root, dataset) {
  return rows.map(row => ({ id: dataset.rows.indexOf(row) + 1, actual: row[4], predicted: predict(row, root, dataset) }));
}

function summarizeEvaluation(rows, root, data) {
  const results = evaluateRows(rows, root, data);
  const correct = results.filter(row => row.actual === row.predicted).length;
  const undecided = results.filter(row => row.predicted === null).length;
  return { results, correct, undecided, incorrect: rows.length - correct - undecided };
}

function evaluationCard(title, rows) {
  const data = currentData();
  if (!rows.length) return `<div class="group"><h3>${title}</h3><p>Sem exemplos. Reserva uma percentagem para teste.</p></div>`;
  const { results, correct, incorrect, undecided } = summarizeEvaluation(rows, roots[datasetKey], data);
  return `<div class="group"><h3>${title}<span>${rows.length} exemplos</span></h3>
    <p class="eval-score">${fmt(100 * correct / rows.length, 1)}% de acertos</p>
    <p>${correct} certos · ${incorrect} errados · ${undecided} sem decisão</p>
    <details><summary>Ver previsões</summary><div class="table-scroll"><table><thead><tr><th scope="col">Obs.</th><th scope="col">Real</th><th scope="col">Previsto</th></tr></thead><tbody>
      ${results.map(({ id, actual, predicted }) => `<tr><th scope="row">#${id}</th><td>${data.classes[actual]}</td><td>${data.classes[predicted] ?? 'Sem decisão'}</td></tr>`).join('')}
    </tbody></table></div></details></div>`;
}

function renderEvaluation() {
  const output = $('evaluation-results');
  output.hidden = !evaluationOpen[datasetKey];
  $('export-results').hidden = output.hidden;
  if (output.hidden) return;
  const sample = samples[datasetKey];
  output.innerHTML = evaluationCard('Treino', sample.train) + evaluationCard('Teste', sample.test)
    + '<p class="note evaluation-note">As folhas preveem a classe maioritária do treino. Empates ficam sem decisão e contam para o total avaliado.</p>';
}

function escapeXml(value) {
  return String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;' })[char]);
}

function resultsSvg(root, data, title, sample) {
  const groups = [['Treino', sample.train], ['Teste', sample.test]];
  const cards = groups.map(([name, rows], index) => {
    const total = rows.length;
    const { correct, incorrect, undecided } = summarizeEvaluation(rows, root, data);
    let offset = 14;
    const bars = total ? [[correct, '#00786e'], [incorrect, '#ad521e'], [undecided, '#788b90']].map(([n, fill]) => {
      const width = 272 * n / total;
      const segment = `<rect x="${offset}" y="69" width="${width}" height="12" fill="${fill}"/>`;
      offset += width;
      return segment;
    }).join('') : '';
    return `<g transform="translate(${20 + index * 320} 68)">
      <rect width="300" height="116" rx="8" fill="#f3f7f6" stroke="#c5d5d1"/>
      <text x="14" y="23" font-size="14">${name} · ${total} exemplos</text>
      <text x="14" y="52" font-size="21" font-weight="700">${total ? fmt(100 * correct / total, 1) + '% de acertos' : 'Sem exemplos'}</text>
      ${bars}
      <text x="14" y="103" font-size="12">${correct} certos · ${incorrect} errados · ${undecided} sem decisão</text>
    </g>`;
  });
  return `<svg xmlns="http://www.w3.org/2000/svg" width="660" height="230" viewBox="0 0 660 230" font-family="system-ui, sans-serif" fill="#192e35">
    <title>${escapeXml(title)}</title><rect width="660" height="230" fill="#fff"/>
    <text x="20" y="30" font-size="21" font-weight="700">${escapeXml(title)}</text>
    <text x="20" y="51" font-size="13">Divisão selecionada · ${sample.percent}% treino / ${100 - sample.percent}% teste</text>
    ${cards.join('')}<text x="20" y="209" font-size="12">Percentagem de acertos calculada sobre todos os exemplos de cada conjunto.</text>
  </svg>`;
}

function treeSvg(root, data, title, sample) {
  const boxes = [], lines = [];
  const margin = 20, boxWidth = 246, boxHeight = 88, stepX = 263, stepY = 148;
  let leaves = 0, maxDepth = 0;

  function wrap(caption) {
    const words = caption.split(' ');
    const lines = [''];
    for (const word of words) {
      const last = lines.length - 1;
      if (lines[last] && lines[last].length + word.length + 1 > 32) lines.push(word);
      else lines[last] = [lines[last], word].filter(Boolean).join(' ');
    }
    return lines;
  }

  function visit(item, depth, tag) {
    maxDepth = Math.max(maxDepth, depth);
    const children = item.children?.map((child, i) => visit(child, depth + 1, i === 0 ? 'Sim' : 'Não'));
    const x = children ? (children[0].x + children[1].x) / 2 : margin + boxWidth / 2 + leaves++ * stepX;
    const y = 70 + depth * stepY;
    for (const child of children ?? []) {
      const middle = (y + boxHeight + child.y) / 2;
      lines.push(`<path d="M ${x} ${y + boxHeight} V ${middle} H ${child.x} V ${child.y}" fill="none" stroke="#79918c" stroke-width="1.5"/>`);
    }
    const counts = count(item.rows, data);
    const pure = counts.includes(item.rows.length);
    const purity = fmt(100 * Math.max(...counts) / item.rows.length, 1);
    const caption = item.split ? label(item.split, data) : leafLabel(item.rows, data);
    const text = wrap(caption).map((line, i) => `<tspan x="12" dy="${i ? 16 : 0}">${escapeXml(line)}</tspan>`).join('');
    const classes = counts.map((n, i) => `${data.classes[i]} ${n}`).join(' · ');
    boxes.push(`<g transform="translate(${x - boxWidth / 2} ${y})">
      <rect width="${boxWidth}" height="${boxHeight}" rx="8" fill="${pure ? '#e1f5e7' : item.children ? '#f3f7f6' : '#fff4ce'}" stroke="${pure ? '#2d8950' : '#c5d5d1'}"/>
      <text x="12" y="18" font-size="12">${tag} · ${item.rows.length} exemplos · Pureza ${purity}%</text>
      <text x="12" y="40" font-size="13" font-weight="700">${text}</text>
      <text x="12" y="76" font-size="12">${escapeXml(classes)}</text></g>`);
    return { x, y };
  }

  visit(root, 0, 'Raiz');
  const width = Math.max(330, margin * 2 + boxWidth + (leaves - 1) * stepX);
  const height = 70 + maxDepth * stepY + boxHeight + margin;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" font-family="system-ui, sans-serif" fill="#192e35">
    <title>${escapeXml(title)}</title><rect width="100%" height="100%" fill="#fff"/>
    <text x="20" y="28" font-size="19" font-weight="700">${escapeXml(title)}</text>
    <text x="20" y="48" font-size="12">${sample.train.length} em treino · ${sample.test.length} em teste</text>
    ${lines.join('')}${boxes.join('')}</svg>`;
}

function render() {
  document.querySelectorAll('[data-mode]').forEach(button => button.setAttribute('aria-pressed', button.dataset.mode === mode));
  const working = mode === 'explore' || mode === 'metrics';
  $('workbench').hidden = !working;
  $('dataset-control').hidden = !working;
  $('reflection').hidden = mode !== 'reflect';
  $('guide').hidden = mode !== 'guide';
  document.querySelectorAll('[data-dataset]').forEach(button => {
    const selected = button.dataset.dataset === datasetKey;
    button.setAttribute('aria-selected', selected);
    button.tabIndex = selected ? 0 : -1;
  });
  $('workbench').setAttribute('aria-labelledby', `dataset-${datasetKey}`);
  if (!working) return;
  $('metrics').hidden = mode !== 'metrics';
  $('auto-confirm').hidden = true;
  $('auto-status').textContent = '';
  $('chart-panel').hidden = datasetKey !== 'iris';
  $('credit-context').hidden = datasetKey !== 'credit';
  renderTree();
  renderSample();
  renderEvaluation();
  renderSplit();
  renderData();
  if (mode === 'metrics') renderMetrics();
  else $('metric-summary').hidden = true;
  if (datasetKey === 'iris') renderChart();
}

function focusTree() {
  $('tree').querySelector('[aria-current="true"]').focus();
}

function announceRule() {
  const data = currentData();
  const [yes, no] = partition(node.rows, rule, data);
  $('announcement').textContent = `${label(rule, data)} Sim: ${yes.length}. Não: ${no.length}.`;
}

document.querySelector('.tabs').addEventListener('click', event => {
  const button = event.target.closest('[data-mode]');
  if (!button) return;
  mode = button.dataset.mode;
  render();
});
function changeDataset(key) {
  datasetKey = key;
  $('train-percent').value = samples[key].percent;
  $('test-percent').value = 100 - samples[key].percent;
  chooseNode(roots[key]);
}
$('dataset-control').addEventListener('click', event => {
  const button = event.target.closest('[data-dataset]');
  if (button) changeDataset(button.dataset.dataset);
});
$('dataset-control').addEventListener('keydown', event => {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
  event.preventDefault();
  const index = datasetKeys.indexOf(datasetKey);
  const offset = event.key === 'ArrowLeft' ? -1 : 1;
  const key = event.key === 'Home' ? datasetKeys[0] : event.key === 'End' ? datasetKeys.at(-1) : datasetKeys[(index + offset + datasetKeys.length) % datasetKeys.length];
  changeDataset(key);
  $(`dataset-${key}`).focus();
});
$('data-source').addEventListener('change', event => {
  dataSource = event.target.value;
  renderData();
});
$('data-table').addEventListener('click', event => {
  const button = event.target.closest('[data-sort]');
  if (!button) return;
  const column = Number(button.dataset.sort);
  const previous = sortOrder[datasetKey];
  sortOrder[datasetKey] = { column, direction: previous?.column === column ? -previous.direction : 1 };
  renderData();
  $('data-table').querySelector(`[data-sort="${column}"]`).focus();
});
$('tree').addEventListener('click', event => {
  const button = event.target.closest('[data-path]');
  if (!button) return;
  const next = [...button.dataset.path].reduce((item, step) => item.children[Number(step)], roots[datasetKey]);
  chooseNode(next);
  focusTree();
});
$('feature').addEventListener('change', event => {
  rule = defaultRule(Number(event.target.value));
  renderSplit();
  renderSelectedScore();
  announceRule();
  if (datasetKey === 'iris') renderChart();
});
function previewValue(value) {
  rule = { ...rule, value: Number(value) };
  renderSplit();
  renderSelectedScore();
  if (datasetKey === 'iris') renderChart();
}
$('value').addEventListener('change', event => {
  previewValue(event.target.value);
  announceRule();
});
$('threshold').addEventListener('input', event => previewValue(event.target.value));
$('threshold').addEventListener('change', announceRule);
$('axis').addEventListener('change', event => {
  axis = Number(event.target.value);
  renderChart();
});
function buildAutomatically() {
  const data = currentData();
  const train = samples[datasetKey].train;
  roots[datasetKey] = growTree(train, data, criterion);
  chooseNode(roots[datasetKey]);
  const metric = $('criterion').selectedOptions[0].textContent;
  $('auto-status').textContent = `Árvore construída com ${train.length} exemplos de treino · ${metric}.`;
  $('auto-build').focus();
}

$('auto-build').addEventListener('click', () => {
  if (!roots[datasetKey].children) {
    buildAutomatically();
    return;
  }
  $('auto-confirm').hidden = false;
  $('auto-replace').focus();
});
$('auto-replace').addEventListener('click', buildAutomatically);
$('auto-cancel').addEventListener('click', () => {
  $('auto-confirm').hidden = true;
  $('auto-build').focus();
});
$('criterion').addEventListener('change', event => {
  criterion = event.target.value;
  $('auto-confirm').hidden = true;
  $('auto-status').textContent = '';
  showAll = false;
  renderMetrics();
});
$('show-more').addEventListener('click', () => {
  showAll = !showAll;
  renderMetrics();
});
$('rankings').addEventListener('click', event => {
  const button = event.target.closest('[data-rank]');
  if (!button) return;
  const { feature, value } = scored[Number(button.dataset.rank)];
  rule = { feature, value };
  renderSplit();
  renderSelectedScore();
  announceRule();
  if (datasetKey === 'iris') renderChart();
  $(datasetKey === 'iris' ? 'threshold' : 'value').focus();
});
function registerPartition() {
  if ($('split-form').hidden || node.children || !rule) return;
  const groups = partition(node.rows, rule, currentData());
  if (groups.some(group => !group.length)) return;
  node.split = rule;
  node.children = groups.map((rows, i) => ({ rows, path: node.path + i, children: null }));
  render();
  $('announcement').textContent = 'Partição registada. Seleciona um dos novos ramos para continuar.';
  focusTree();
}
$('apply').addEventListener('click', registerPartition);
$('split-form').addEventListener('submit', event => {
  event.preventDefault();
  registerPartition();
});
$('node-note').addEventListener('click', event => {
  if (event.target.id !== 'prune') return;
  if (node.children.some(child => child.children) && !window.confirm('Remover também as partições descendentes?')) return;
  node.split = null;
  node.children = null;
  chooseNode(node);
  focusTree();
});
$('reset').addEventListener('click', () => {
  roots[datasetKey] = makeRoot(samples[datasetKey].train);
  evaluationOpen[datasetKey] = false;
  sortOrder[datasetKey] = null;
  showAll = false;
  chooseNode(roots[datasetKey]);
  $('announcement').textContent = 'Árvore reiniciada.';
  focusTree();
});

$('train-percent').addEventListener('change', event => {
  $('test-percent').value = 100 - Number(event.target.value);
});
function applySamples() {
  const percent = Number($('train-percent').value);
  if (percent < 10 || percent > 100 || percent % 10 !== 0) return;
  if (percent === 100 && samples[datasetKey].percent === 100) return;
  if (roots[datasetKey].children && !window.confirm('Sortear a divisão apaga a árvore atual deste conjunto. Continuar?')) {
    $('train-percent').value = samples[datasetKey].percent;
    $('test-percent').value = 100 - samples[datasetKey].percent;
    return;
  }
  samples[datasetKey] = splitSample(currentData(), percent);
  roots[datasetKey] = makeRoot(samples[datasetKey].train);
  evaluationOpen[datasetKey] = false;
  chooseNode(roots[datasetKey]);
}
$('apply-samples').addEventListener('click', applySamples);
$('sample-form').addEventListener('submit', event => {
  event.preventDefault();
  applySamples();
});
$('evaluate').addEventListener('click', () => {
  evaluationOpen[datasetKey] = true;
  renderEvaluation();
});
function downloadText(content, filename, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

$('export-tree').addEventListener('click', () => {
  const title = `Árvore de decisão · ${currentData().name}`;
  downloadText(treeSvg(roots[datasetKey], currentData(), title, samples[datasetKey]), `arvore-${datasetKey}.svg`, 'image/svg+xml;charset=utf-8');
});
$('export-results').addEventListener('click', () => {
  const title = `Resultados da árvore · ${currentData().name}`;
  downloadText(resultsSvg(roots[datasetKey], currentData(), title, samples[datasetKey]), `resultados-${datasetKey}.svg`, 'image/svg+xml;charset=utf-8');
});

const reflectionFields = [...document.querySelectorAll('#reflection textarea')];
const reflectionKey = 'lab02-reflection-v1';
const reflectionStatus = $('reflection-status');
try {
  const answers = JSON.parse(localStorage.getItem(reflectionKey) ?? '[]');
  reflectionFields.forEach((field, i) => {
    if (typeof answers?.[i] === 'string') field.value = answers[i];
  });
  reflectionStatus.textContent = 'As respostas ficam guardadas neste navegador.';
} catch {
  reflectionStatus.textContent = 'Descarrega as respostas antes de fechar a página.';
}
$('reflection').addEventListener('input', event => {
  if (!reflectionFields.includes(event.target)) return;
  try {
    localStorage.setItem(reflectionKey, JSON.stringify(reflectionFields.map(field => field.value)));
    reflectionStatus.textContent = 'Respostas guardadas neste navegador.';
  } catch {
    reflectionStatus.textContent = 'Descarrega as respostas antes de fechar a página.';
  }
});
$('export-reflection').addEventListener('click', () => {
  const blocks = reflectionFields.map((field, i) =>
    `${i + 1}. ${field.labels[0].textContent.trim()}\n${field.value.trim()}`);
  downloadText('Lab02 · Reflexão\n\n' + blocks.join('\n\n'), 'lab02-reflexao.txt', 'text/plain;charset=utf-8');
});

chooseNode(roots.golf);
