# Come funziona claude-search

## Panoramica

`claude-search` indicizza tutte le sessioni di Claude Code salvate localmente e permette di cercarle per contenuto, selezionare quella desiderata e riprenderla con `claude --resume`.

---

## Dove sono le sessioni

Claude Code salva ogni conversazione in un file `.jsonl` (JSON Lines) nella directory:

- **Linux / Mac**: `~/.claude/projects/<nome-progetto>/<session-id>.jsonl`
- **Windows**: `%APPDATA%\Claude\projects\<nome-progetto>\<session-id>.jsonl`

Ogni riga del file è un oggetto JSON che rappresenta un messaggio, un tool call, o un evento di sistema. Lo script estrae i messaggi di tipo `user`, ma **filtra il rumore sintetico** generato da Claude Code (vedi sotto) per indicizzare e mostrare in anteprima solo il testo realmente scritto dall'utente.

### Filtro dei messaggi sintetici

Sotto `type: "user"` Claude Code salva anche messaggi che l'utente non ha scritto. Questi vengono scartati (`claude_search/_extract.py`):

- `<local-command-caveat>` — boilerplate "Caveat: the messages below were generated…"
- `<local-command-stdout>` / `<local-command-stderr>` — output dei comandi slash (es. `/model`, `/effort`)
- `<bash-stdout>` / `<bash-stderr>` — output dei comandi bash `!`
- `<task-notification>` — notifiche di task/agent in background
- `<system-reminder>` — promemoria iniettati dall'harness
- blocchi `tool_result` (output di tool/MCP), già esclusi perché non sono blocchi `text`

Viene invece **mantenuto** ciò che l'utente ha effettivamente digitato:

- la prosa dei messaggi normali
- gli **argomenti** dei comandi slash (`<command-args>`) — es. `/deploy prod` → `deploy prod`; i comandi senza argomenti come `/model` vengono scartati perché privi di valore di ricerca
- l'input dei comandi bash `!` (`<bash-input>`) — es. `! git log` → `git log`

---

## Pipeline di ricerca

```
File JSONL  →  Estrazione testo  →  Cache su disco  →  Scoring  →  Ranking  →  UI selezione
```

### 1. Estrazione testo

Per ogni file di sessione viene estratto:
- **Testo completo**: concatenazione di tutti i messaggi utente
- **Primo messaggio**: usato come anteprima nel risultato
- **CWD (working directory)**: directory da cui riaprire la sessione

### 2. Cache su disco

Il testo estratto viene salvato in `~/.cache/claude-search/index.json`, indicizzato per path del file e `mtime` (data di modifica).

Alla seconda esecuzione, i file non modificati vengono letti dalla cache invece di essere ri-parsati: questo dimezza i tempi di avvio.

### 3. Algoritmo di scoring

Lo script supporta due algoritmi, usati in ordine di preferenza:

---

#### BM25 (Best Match 25) — preferito se `rank-bm25` è installato

BM25 è lo standard de facto per la ricerca testuale, usato da Elasticsearch e Solr.

Formula per il punteggio di un documento `d` rispetto a una query `q`:

```
score(d, q) = Σ IDF(t) · (tf(t,d) · (k1+1)) / (tf(t,d) + k1·(1 - b + b·|d|/avgdl))
```

Dove:
- `tf(t, d)` = frequenza del termine `t` nel documento `d`
- `IDF(t)` = log((N - df(t) + 0.5) / (df(t) + 0.5)) — penalizza termini comuni
- `|d|` = lunghezza del documento
- `avgdl` = lunghezza media dei documenti nel corpus
- `k1 = 1.5`, `b = 0.75` — parametri di saturazione e normalizzazione per lunghezza

**Vantaggio rispetto a TF-IDF**: normalizza automaticamente per la lunghezza del documento, evitando che sessioni molto lunghe dominino sempre il ranking.

---

#### TF-IDF con cosine similarity — fallback (solo stdlib)

TF-IDF (Term Frequency – Inverse Document Frequency) assegna a ogni termine un peso che bilancia quanto è frequente in un documento e quanto è raro nel corpus.

```
TF(t, d)  = occorrenze di t in d / totale termini in d
IDF(t)    = log((N+1) / (df(t)+1)) + 1       (smoothed)
TF-IDF(t) = TF(t, d) · IDF(t)
```

Il punteggio finale è la **cosine similarity** tra il vettore TF-IDF della query e quello di ogni sessione:

```
similarity(q, d) = (q · d) / (|q| · |d|)
```

La cosine similarity è invariante rispetto alla lunghezza del vettore: due sessioni con gli stessi termini ottengono lo stesso punteggio indipendentemente da quanto sono lunghe.

---

### 4. Tokenizzazione

Entrambi gli algoritmi usano lo stesso tokenizer:

```python
re.findall(r"[a-zA-Z0-9àèéìòùÀÈÉÌÒÙ_]+", text.lower())
```

- Tutto minuscolo
- Supporta caratteri accentati italiani
- Separa su spazi, punteggiatura, simboli

Non viene applicato stemming né stop-word removal: la semplicità è preferita per un tool locale con corpus piccolo.

---

### 5. Selezione interattiva

Se `fzf` è installato e il terminale è interattivo (TTY), viene usata l'interfaccia `fzf` con:
- Preview in tempo reale dei primi messaggi della sessione
- Fuzzy filtering aggiuntivo sul testo già rankato da BM25/TF-IDF

Altrimenti viene mostrata una lista numerata su `stderr` con prompt di selezione.

### 6. Resume

Una volta selezionata la sessione, lo script:
1. Fa `chdir` nella directory originale della sessione
2. Esegue `claude --resume <session-id>`
