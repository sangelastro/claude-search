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

### Selezione e resume

**Con fzf**: naviga con le frecce, premi `Enter` per aprire la sessione.

**Senza fzf**: inserisci il numero della sessione desiderata e premi `Enter`.

Lo script apre automaticamente Claude Code nella directory originale della sessione.

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
