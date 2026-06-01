"""Per-flight delay modeling track (SEPARATE from the GNN baseline).

⛔ This package and its branch (`explore/flight-level-signal`) are an independent
deployment track and must NOT be merged into `main`. See `BRANCH_NOTES.md`.

It recovers the signal that airport-hour aggregation discards — chiefly
aircraft-rotation / late-inbound state and airline identity — by modeling each
flight individually. Every feature is **point-in-time correct**: only data
observable at ``t_pred = scheduled_departure − horizon`` is used.
"""
