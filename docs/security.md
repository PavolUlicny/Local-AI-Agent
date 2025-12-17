## Security & Safety Notes

- All network calls flow through DDGS text search providers (DuckDuckGo/Bing/Brave); no external code execution beyond HTTP GET for search snippets.
- User input is directly embedded into prompts; avoid placing secrets or credentials in queries.
- URL canonicalization strips default ports and `www.` but retains query string—beware PII embedded in copied URLs.
- The system does not attempt adversarial prompt injection mitigation beyond rigid output regex validation for classifier stages.

### Optional hardening tips

- Run `pip-audit` in CI to surface dependency CVEs:

```bash
pip install pip-audit
pip-audit
```

- Treat search snippets as untrusted content within prompts; avoid following instructions contained in results.

- Avoid piping unfamiliar install scripts directly to a shell. Prefer a two-step pattern: download the script, inspect it, then run it if you trust it. Example:

```bash
curl -fsSL -o ollama_install.sh https://ollama.com/install.sh
less ollama_install.sh
sh ollama_install.sh
```

Alternatively, use the official installers provided on the vendor site.
