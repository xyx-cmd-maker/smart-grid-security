# Maintainer notes

Operational notes for whoever pushes to this repository. Not part of the artifact and not
relevant to a reader reproducing the results.

## Pushing from a network that intermittently blocks `github.com`

**Symptom.** `git push` fails with:

```
fatal: unable to access 'https://github.com/...': Failed to connect to github.com
port 443 after 21088 ms: Could not connect to server
```

**What is actually happening.** The block is intermittent, not permanent. Measured on
2026-10-11, twice within one hour, using `curl --resolve` so the TLS SNI stayed `github.com`:

| target | first probe | second probe |
|---|---|---|
| `github.com` (DNS gave 20.205.243.166) | **connection failed** | **200** |
| `github.com` pinned to the `ssh.github.com` address (…160) | — | **TLS handshake failed** |
| `github.com` pinned to the `api.github.com` address (…168) | — | **TLS handshake failed** |

**Why pinning the hostname to a reachable IP does not work.** GitHub's edge routes
certificates by SNI. Sending `github.com`'s SNI to a different edge address fails the TLS
handshake outright — it is a certificate mismatch, not a network path problem. "Resolve the
domain to an IP that is reachable" is not a valid workaround on a CDN that shards by hostname.

**What to do.** Probe, then retry. Do not change configuration.

```powershell
# 10 seconds. 000 means blocked right now; 200 means the window is open.
curl.exe -sS -o NUL -w "%{http_code}" --connect-timeout 15 https://github.com/
git push origin main
```

On 2026-10-11 a failed push succeeded roughly half an hour later with no configuration change.

**A global config that can get in the way.** `~/.gitconfig` may contain:

```
url.https://github.com/.insteadof=git@github.com:
```

That rewrites `git@github.com:` URLs into HTTPS, which is the blocked path. If switching to
SSH, use the `git@ssh.github.com:` prefix — it does not match the rule and is left alone.
Both port 443 and port 22 on `ssh.github.com` were reachable when `github.com` was not.

**Distinguishing the two failure modes.** `Failed to connect` is a network problem; retry it.
`Permission denied` is a credential problem; fix the key. They have nothing in common.

## Before every commit

The `.gitignore` in this repository is an **allowlist**: everything is ignored by default and
only the artifact is un-ignored. That means a new file does not get committed until someone
adds it deliberately — which is the intended behaviour, because the alternative fails open.

Run this before committing, and expect no output:

```powershell
git ls-files | Select-String -Pattern 'AGENTS|slots\.json|研究进度|停机|审稿|response-letter|revision-|\.venv|data_cache|\.npz'
```

The repository is private while the manuscript is under review and becomes public on
acceptance.
