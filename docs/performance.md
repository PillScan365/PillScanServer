# Performance baseline

Measured locally on 2026-07-17 with the same two clear Taiwan Panadol package screenshots.
Network and provider latency vary, so these samples are an engineering baseline rather than a
hard real-time guarantee.

| Pipeline | Cold/observed total | Result |
| --- | ---: | --- |
| `gpt-5.6`, high, generic catalog search | 20.46–30.57 s | Exact TFDA match |
| `gpt-5.4-mini`, high, exact-name fast path | 4.37–4.49 s | Exact TFDA match |

The optimized package path keeps `detail=high` because low detail gives the model only a 512 px
representation and can damage Chinese package text and pill-imprint accuracy. The client-side image
cap is 2048 px, matching the finite high-detail limit used by `gpt-5.4-mini`.

The TFDA exact-name query now takes about 5–10 ms. The OpenAI vision request remains the dominant
stage. A blurry `C9`/`AB` pill example returned ranked candidates in 3.56 s but did not meet the
confidence threshold for a unique identity; this safety threshold must not be relaxed merely to
claim a faster exact result.

Every successful API response includes `timings` and provider-reported `usage`. Timings are also
emitted through Loguru and the standard `Server-Timing` response header. Use `pillscan-eval` for a
resumable multi-image benchmark with p50/p95/p99, token cost, safety metrics, and confidence
intervals.
