# Branch notes — `explore/flight-level-signal`

> ⛔ **DO NOT MERGE THIS BRANCH INTO `main`.**

This branch is a **separate deployment track** from the production GNN baseline
(the spatio-temporal airport-hour graph model). It is intentionally kept
independent: no shared modules with the GNN path, no changes to it.

## Scope
Per-flight delay modeling that recovers the signal lost when flights are
aggregated to airport-hour nodes — chiefly **aircraft-rotation / late-inbound**
state and **airline** identity (see `docs/flight-level-signal-study.md`).

## Hard requirement: point-in-time correctness
Every feature must use **only data observable at the prediction time**
`t_pred = scheduled_departure − horizon`. Accumulated delay from previous flights
may come from the prior leg's *departure* delay or from *airport average delays
as-of `t_pred`* — but never from anything that happens after `t_pred`
(including the inbound leg's arrival if it hasn't landed yet, or any
post-departure field of the flight being predicted).
