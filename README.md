# claude-search

Cerca tra tutte le sessioni di Claude Code salvate localmente e riprendile con `claude --resume`.

Usa **BM25** (o TF-IDF come fallback) per rankare le sessioni per rilevanza rispetto alla query.

---

## Requisiti

- Python 3.11+
- Claude Code CLI installato

---

## Installazione

### 1. Clona o scarica il progetto

```bash
git clone <url-repo>
# oppure estrai lo zip
```

### 2. Installa il pacchetto

**Base (usa TF-IDF, nessuna dipendenza esterna):**
```bash
pip install .
```

**Con BM25 (ranking migliore, consigliato):**
```bash
pip install ".[bm25]"
```

> Su alcune distribuzioni Linux potrebbe essere necessario usare `pip3` al posto di `pip`,
> o installarlo prima con `sudo apt install python3-pip`.

### 3. Verifica che `~/.local/bin` sia nel PATH

```bash
echo $PATH | grep -q "$HOME/.local/bin" && echo "OK" || echo 'Aggiungi a ~/.bashrc: export PATH="$HOME/.local/bin:$PATH"'
```

Se non è nel PATH, aggiungilo:
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 4. (Opzionale) Installa fzf per l'UI interattiva

```bash
# Linux (Debian/Ubuntu)
sudo apt install fzf

# Mac
brew install fzf

# Windows
winget install fzf
```

Con `fzf` ottieni un'interfaccia interattiva con preview in tempo reale dei messaggi della sessione.
Senza `fzf` viene mostrata una lista numerata.

---

## Utilizzo

```bash
claude-search "<query>"
```

### Esempi

```bash
claude-search "activity report"
claude-search "location history cluster"
claude-search "blocco tastierino"
claude-search "missioni websocket"
```

### Cercare anche nelle risposte di Claude (`--all` / `-a`)

Di default si cerca solo in quello che hai scritto tu. Con `--all` vengono indicizzati anche le risposte di Claude e gli input dei tool call (path dei file scritti/modificati, comandi eseguiti, pattern di ricerca): serve per ritrovare una sessione dal **nome di un file che ha generato**.

```bash
claude-search -a Report_Vendite_Q3_v2
```

Contenuti dei file scritti (`Write`/`Edit`), blocchi di thinking e output dei tool restano esclusi.

### Elenco e ordinamento (`--list` / `--sort`)

```bash
claude-search --list                 # tutte le sessioni, dalla più aggiornata alla meno recente
claude-search --list --sort name     # in ordine alfabetico per nome
claude-search --sort date redis      # ricerca, risultati ordinati per data invece che per score
```

| Ordinamento | Significato | Default per |
|---|---|---|
| `score` | rilevanza rispetto alla query | ricerca |
| `date` | ultimo aggiornamento, dal più recente | `--list` |
| `name` | nome della sessione (`/rename`), quelle senza nome in fondo | — |

Forme accettate: `--sort date`, `--sort=date`, `-s date`. Nell'interfaccia fzf l'ordinamento si cambia al volo: `ctrl-s` score, `ctrl-d` data, `ctrl-n` nome. Mentre scrivi un filtro in fzf l'ordine scelto viene mantenuto.

### Selezione e resume

**Con fzf**: naviga con le frecce, premi `Enter` per aprire la sessione.

**Senza fzf**: inserisci il numero della sessione desiderata e premi `Enter`.

Lo script apre automaticamente Claude Code nella directory originale della sessione.

I risultati sono una tabella a colonne: score (solo in ricerca), **creata**, **aggiornata** (primo e ultimo timestamp del file, in ora locale, es. `30/09/26 13:05`), nome, cartella, primo messaggio. I testi incollati da Windows (`\r\n`), i tab e i codici colore vengono ridotti a una riga, così righe e anteprima non strabordano.

**Più account:** le sessioni vengono lette da `$CLAUDE_CONFIG_DIR/projects` se la variabile è impostata (altrimenti `~/.claude/projects`), e la sessione viene ripresa con `claude --resume`, che eredita la stessa variabile. Per cercare su un altro account basta impostarla prima di lanciare il comando (es. `CLAUDE_CONFIG_DIR=~/.claude-personal claude-search ...`).

---

## Come funziona

Vedi [ALGORITHM.md](ALGORITHM.md) per una descrizione dettagliata degli algoritmi usati.

---

## Cache

L'indice delle sessioni viene salvato in `~/.cache/claude-search/index.json` e aggiornato automaticamente solo per le sessioni nuove o modificate. La cache rende le esecuzioni successive molto più veloci.

Per forzare la ricostruzione dell'indice:
```bash
rm ~/.cache/claude-search/index.json
```

---

## Windows

Su Windows il comando `claude --resume` viene eseguito tramite `subprocess.run` invece di `os.execvp`. Tutto il resto funziona allo stesso modo.
