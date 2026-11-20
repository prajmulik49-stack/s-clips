# Daily auto-posting: setup

Every day at 02:00 IST GitHub Actions does this:

1. **plan**: picks today's scenes for both channels from one shared list, so no scene is ever posted twice in
   either channel. Each channel gets 2–3 Shorts from the newest vlog and the rest from older vlogs.
2. **post channel_a** and **post channel_b** run in parallel: each renders its Shorts (with quiet background music),
   uploads them through the YouTube API, and schedules them through the day (08:00–21:30 IST; channel B runs 40 min later).
3. Everything posted is written to `state/` and committed back to the repo. That file is the memory that
   stops repeats.

All numbers (Shorts per day, times, how many from the latest vlog) are in `config.yaml` under `daily:`.

---

## Read this first: the limits

| Limit | What it means for you | Fix |
|---|---|---|
| **Upload quota.** Each upload costs 1,600 of the 10,000 free units a Google Cloud project gets per day. | **Max 6 uploads per day per project.** 10/day is not possible on the free quota. | Use **one Google Cloud project per channel** (6 + 6), and request more: Google Cloud → YouTube Data API v3 → Quotas → "Apply for higher quota" (audit form). Until it is approved, set `daily.per_channel: 6`. |
| **Unverified API apps upload as PRIVATE.** Google locks videos uploaded by an unaudited API project to private. | Videos upload, but nobody can see them until you make them public by hand in YouTube Studio. | Pass the **YouTube API audit** (same form as above). After approval, uploads can be public/scheduled. |
| **OAuth app in "Testing" mode.** Login tokens expire after 7 days. | Uploads stop working after a week. | OAuth consent screen → **Publish app** ("In production"). It does not need Google verification for your own accounts. |
| **YouTube blocks GitHub's servers.** yt-dlp often gets "Sign in to confirm you're not a bot" from datacenter IPs. | The plan job fails at download. | Add a `YTDLP_COOKIES` secret (below), **or** run on your own PC as a self-hosted runner (best; see the end). |
| **Copyright / reused content.** These are Sourav Joshi's videos. | Claims, strikes, or a channel that can't be monetised. Two channels posting 20 clips a day from one creator is exactly what YouTube's "reused content" and spam rules look at. | Get the creator's permission. Keep the credit + link in the description (the tool adds it). |
| **Music licence.** | Copyrighted music gets a Content ID claim on every Short. | Only royalty-free tracks (YouTube Audio Library). |

---

## 1. Google Cloud: do this once per channel

Use a **separate project for each channel** (each gets its own 6-uploads-a-day quota).

1. Go to <https://console.cloud.google.com> → create project, e.g. `shorts-channel-a`.
2. **APIs & Services → Library →** search **YouTube Data API v3 → Enable**.
3. **APIs & Services → OAuth consent screen**: user type **External**, fill in the app name and your email,
   add scopes `.../auth/youtube.upload` and `.../auth/youtube.readonly`, then **Publish app** (see the limits table).
4. **APIs & Services → Credentials → Create credentials → OAuth client ID → Application type: Desktop app.**
   Download the JSON file (e.g. `client_secret_a.json`). **Do not commit it** (`.gitignore` already blocks it).

An "API key" alone **cannot upload videos**. Uploading always needs this OAuth login.

## 2. Get each channel's refresh token (on your PC)

**First publish the OAuth app** (step 1.3). A token made while the app is in "Testing" dies after 7 days.

```powershell
python -m joshiclip auth                       # uses YOUTUBE_OAUTH_CLIENT_JSON from .env, saves YT_A_* into .env
python -m joshiclip auth client_secret_b.json --prefix YT_B    # 2nd channel, its own Cloud project
python -m joshiclip check-auth                 # any time: tests every channel's login in seconds
```

`check-auth` fails (and says why) if a token is revoked or if Google gave it an expiry date, which means the app is
still in Testing.

A browser opens. Log in with the Google account that **owns channel A** and pick channel A if it asks.
The terminal prints the channel name and three values:

```
<P>_CLIENT_ID      = ...
<P>_CLIENT_SECRET  = ...
<P>_REFRESH_TOKEN  = ...
```

Do the same for channel B with `client_secret_b.json`. Check the printed channel name each time, because
logging into the wrong channel uploads there.

## 3. GitHub repo

One repo runs both channels. A second repo isn't needed, and it would break the shared "never repeat a scene" list.

```powershell
git remote add origin https://github.com/prajmulik49-stack/s-clips.git
git push -u origin main
```

**Settings → Secrets and variables → Actions → New repository secret**, add the three per channel (only `YT_A_*` while you run one channel):

| Secret | Value |
|---|---|
| `YT_A_CLIENT_ID` | channel A client id |
| `YT_A_CLIENT_SECRET` | channel A client secret |
| `YT_A_REFRESH_TOKEN` | channel A refresh token |
| `YT_B_CLIENT_ID` | channel B client id |
| `YT_B_CLIENT_SECRET` | channel B client secret |
| `YT_B_REFRESH_TOKEN` | channel B refresh token |
| `YOUTUBE_API_KEY` | API key (Credentials → Create credentials → API key, restricted to YouTube Data API v3). Used for vlog lists, view counts and fan comments, so yt-dlp only downloads (fewer bot checks). It can't download videos: the API has no endpoint for that. |
| `YTDLP_COOKIES` *(optional)* | contents of a `cookies.txt` exported from a browser logged into a **spare** Google account (not a channel owner) |

Secrets stay hidden even in a public repo: they never show in logs, and fork pull requests can't read them.
The workflow has no pull-request trigger at all.

**Settings → Actions → General → Workflow permissions → Read and write permissions** (needed to commit `state/`).

## 4. Music

Put royalty-free tracks in `music/` (mp3/wav/m4a…). Each Short gets a random one, about 18 dB under the voice in
pauses and 26 dB under it while someone talks. Change these in `config.yaml → music:`. With an empty folder,
Shorts have no music.
For a **public** repo, check that the track licence allows re-publishing the files themselves.

## 5. First test

**Actions → daily-shorts → Run workflow → tick "Only print the plan"**. The log shows which scenes each
channel would get. Then run it once without the tick and watch the uploads in YouTube Studio.

On your PC the same thing works without GitHub:

```powershell
python -m joshiclip daily run --dry-run      # print today's plan only
python -m joshiclip daily run --no-upload    # plan + render, no upload
python -m joshiclip daily run                # everything (needs the YT_A_* / YT_B_* env vars)
```

## 6. Recommended: run on your own PC (self-hosted runner)

GitHub's free Windows machines have 4 CPUs and get bot-checked by YouTube. Your PC renders about 2× faster and
YouTube treats it like a normal home connection.

1. Make the repo **private** first. A self-hosted runner on a public repo lets strangers' pull requests run code on
   your PC. Runner time on your own PC is free for private repos.
2. **Settings → Actions → Runners → New self-hosted runner → Windows** and follow the commands it shows.
   Install it as a service so it runs while you're logged out.
3. **Settings → Secrets and variables → Actions → Variables → New variable:** `RUNNER` = `self-hosted`.
4. Leave the PC on at 02:00 IST. Both channel jobs run one after the other on it.

## Every day

- `state/posted_channel_a.jsonl` / `state/posted_channel_b.jsonl`: every upload (vlog, seconds, YouTube id, time).
- `state/plans/<date>.json`: what was planned. Re-running the same day continues that plan and doesn't duplicate.
- `state/exhausted.json`: old vlogs with no good unused scene left (skipped from then on).
- Each run's `report.md` / `upload.md` are kept as run artifacts (the videos themselves are not).
- To pause: Actions → daily-shorts → **Disable workflow**.

---

## Running for years: what can stop it, and what is already handled

| Risk | Status |
|---|---|
| OAuth token expires after 7 days (app in Testing) | **You:** publish the consent screen. `check-auth` runs before every upload and fails loudly if this happens. |
| Token unused for 6 months is revoked by Google | Handled: the daily run uses it every day. |
| Token revoked (you remove app access, change account security, or create 100+ tokens for one client) | Run `joshiclip auth` again and update the `YT_A_REFRESH_TOKEN` secret. |
| Uploads locked to **private** (API project not audited) | **You:** apply for the YouTube API audit (Quotas → "Apply for higher quota"). Until then, publish by hand in Studio. |
| GitHub disables the schedule after 60 days without commits (public repos) | Handled: daily state commits + a keepalive commit if the repo is quiet for 20 days. |
| A library update breaks the pipeline | Handled: CI installs the exact tested versions from `constraints.txt`. |
| YouTube changes break downloads | Handled as far as possible: yt-dlp updates every run, and Deno is installed for its JavaScript challenges. |
| YouTube bot check on GitHub's servers | `YTDLP_COOKIES` from a spare account (cookies go stale after weeks or months, so re-export them when downloads fail), or a self-hosted runner on a **private** repo. |
| Old GitHub Actions versions retired | Actions use the current majors (checkout v7 etc.). Every ~12 months, check the run log for deprecation warnings and bump the `@vN` numbers. |
| Python 3.11 end of life (Oct 2027) | Still installable afterwards. Move to 3.12 when you next refresh `constraints.txt`. |
| You don't notice failures | GitHub emails you when a scheduled run fails (Settings → Notifications → Actions). Leave that on. |

**Every few months (10 minutes):** `python -m joshiclip check-auth`, glance at the Actions tab, and run
`python -m joshiclip research --quick` so the "what performs" priors follow current trends.

**Never commit `.env`.** It is git-ignored. Secrets live only in `.env` (your PC) and GitHub Actions secrets.

## "Sign in to confirm you're not a bot" on your own PC

This is YouTube rate-limiting your connection after many requests. It usually clears within 1-2 hours, and the
tool now waits and retries by itself (90 s, then 5 min). If it keeps happening, give yt-dlp cookies from a
**spare** Google account (never the channel owner):

- **Easiest:** log into YouTube with that account in Firefox, then add `JOSHICLIP_YTDLP_BROWSER=firefox` to `.env`
  (Chrome and Edge lock their cookies on Windows, so yt-dlp often can't read them).
- **Or:** export `cookies.txt` with a "Get cookies.txt LOCALLY" browser extension and put it in the project folder.
  It's git-ignored. For GitHub, paste the same file's text into the `YTDLP_COOKIES` secret.
