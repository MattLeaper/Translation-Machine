import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1';

const $ = id => document.getElementById(id);
const LANGS = { en:['English','en-US'], es:['Spanish','es-ES'], fr:['French','fr-FR'],
                de:['German','de-DE'], it:['Italian','it-IT'], ro:['Romanian','ro-RO'] };

// Theme toggle
const root = document.documentElement;
const renderTheme = () => $('themeToggle').textContent =
  root.dataset.theme === 'dark' ? 'Light mode ☀️' : 'Dark mode 🌙';
$('themeToggle').onclick = () => {
  const next = root.dataset.theme === 'dark' ? 'light' : 'dark';
  root.dataset.theme = next;
  try { localStorage.setItem('theme', next); } catch (e) {}
  renderTheme();
};
renderTheme();

// Language pickers
for (const id of ['src', 'tgt'])
  for (const [code, [name]] of Object.entries(LANGS)) $(id).add(new Option(name, code));
$('src').value = 'en';
$('tgt').value = 'es';

// Status helpers
const say = (msg, isError = false) => { $('status').textContent = msg; $('status').className = isError ? 'error' : ''; };
const busy = on => ['translate', 'define'].forEach(id => $(id).disabled = on);

// Models are fetched from the Hugging Face Hub on first use and cached by the browser
const pipes = new Map();
function getPipe(s, t) {
  const id = `Xenova/opus-mt-${s}-${t}`;
  if (!pipes.has(id)) {
    const files = {};
    const p = pipeline('translation', id, {
      progress_callback: d => {
        if (d.status !== 'progress' || !d.total) return;
        files[d.file] = d;
        const all = Object.values(files);
        const pct = 100 * all.reduce((a, f) => a + f.loaded, 0) / all.reduce((a, f) => a + f.total, 0);
        $('bar').hidden = false; $('bar').value = pct;
        say(`Downloading ${LANGS[s][0]} → ${LANGS[t][0]} model… ${Math.round(pct)}%`);
      },
    });
    p.catch(() => pipes.delete(id));
    pipes.set(id, p);
  }
  return pipes.get(id);
}

async function pair(text, s, t) {
  const tr = await getPipe(s, t);
  const out = await tr(text, { max_new_tokens: 400 });
  return out[0].translation_text;
}

// Pairs without a direct model go through English
const hop = (x, s, t) => (s === 'en' || t === 'en')
  ? pair(x, s, t) : pair(x, s, 'en').then(m => pair(m, 'en', t));

async function translate(text, s, t) {
  if (s === t) return text;
  const lines = [];
  for (const line of text.split('\n')) {
    const parts = [];
    for (const sentence of line.split(/(?<=[.!?])\s+/).filter(Boolean)) {
      say('Translating…');
      parts.push(await hop(sentence, s, t));
    }
    lines.push(parts.join(' '));
  }
  return lines.join('\n');
}

$('translate').onclick = async () => {
  const text = $('input').value.trim();
  if (!text) return say('Please enter some text.', true);
  busy(true); say('Translating…');
  try {
    $('output').value = await translate(text, $('src').value, $('tgt').value);
    say('');
  } catch (e) {
    console.error(e);
    say(`Couldn't load the ${LANGS[$('src').value][0]} → ${LANGS[$('tgt').value][0]} model. Check your connection and try again.`, true);
  } finally { busy(false); $('bar').hidden = true; }
};

$('define').onclick = async () => {
  const word = $('input').value.trim().toLowerCase();
  if ($('src').value !== 'en') return say('Definitions are only available for English words.', true);
  if (!word || /\s/.test(word)) return say('Enter a single English word to define.', true);
  busy(true); say('Looking up…');
  try {
    const r = await fetch(`https://api.dictionaryapi.dev/api/v2/entries/en/${encodeURIComponent(word)}`);
    if (!r.ok) { $('output').value = ''; return say(`No definitions found for "${word}".`, true); }
    const [entry] = await r.json();
    $('output').value = entry.meanings.map(m =>
      `${m.partOfSpeech}\n` + m.definitions.slice(0, 3).map((d, i) => `${i + 1}. ${d.definition}`).join('\n')
    ).join('\n\n');
    say('');
  } catch (e) { say("Couldn't reach the dictionary. Try again.", true); }
  finally { busy(false); }
};

// Extras
$('swap').onclick = () => {
  [$('src').value, $('tgt').value] = [$('tgt').value, $('src').value];
  [$('input').value, $('output').value] = [$('output').value, ''];
};
$('clear').onclick = () => { $('input').value = ''; $('output').value = ''; say(''); $('input').focus(); };
$('copy').onclick = async () => {
  try { await navigator.clipboard.writeText($('output').value); say('Copied.'); }
  catch (e) { say("Couldn't copy. Select the text and copy it manually.", true); }
};
$('speak').onclick = () => {
  const u = new SpeechSynthesisUtterance($('output').value);
  u.lang = LANGS[$('tgt').value][1];
  speechSynthesis.cancel(); speechSynthesis.speak(u);
};
$('mic').onclick = () => {
  const Rec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Rec) return say("Speech recognition isn't supported in this browser.", true);
  const rec = new Rec();
  rec.lang = LANGS[$('src').value][1];
  rec.onresult = e => { $('input').value = e.results[0][0].transcript; };
  rec.onerror = () => say("Couldn't hear anything. Check microphone permissions.", true);
  rec.start();
};
$('input').addEventListener('keydown', e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) $('translate').click(); });
