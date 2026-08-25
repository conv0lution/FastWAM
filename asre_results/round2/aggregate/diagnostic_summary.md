# ASRE Round-2 machine diagnostic

This is a 100-episode screening analysis, not a formal non-inferiority study.

## Candidate schedules

- `keep_15_29`

| Condition | Success loss | Candidate | Catastrophic task IDs |
| --- | ---: | :---: | --- |
| `keep_15_29` | +2.0 pp | yes | none |
| `keep_20_29` | +48.0 pp | no | 1, 4, 5, 7, 9 |
| `keep_25_29` | +97.0 pp | no | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| `keep_00_14` | +97.0 pp | no | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| `keep_00_19` | +85.0 pp | no | 0, 1, 2, 3, 4, 6, 7, 8, 9 |
| `keep_15_19` | +89.0 pp | no | 0, 1, 2, 3, 4, 6, 7, 8, 9 |
| `keep_15_19_25_29` | +11.0 pp | no | none |

## Pattern flags

- `A_late_half_candidate`: true
- `B_early_only_strongly_degraded`: true
- `C_critical_windows_complementary_candidate`: false
- `D_final_five_candidate`: false
- `E_all_keep_schedules_strongly_degraded`: false

## Supported interpretations

- `A_late_half_candidate`: Early direct visual retrieval is jointly dispensable in this LIBERO-Spatial screening run when later retrieval remains available.
- `B_early_only_strongly_degraded`: Visual information acquired only early in the action stream is insufficient here; later direct visual re-grounding is required.

## Round-1 / Round-2 baseline repeat

- Round-1 baseline SR: 97.0%
- Round-2 baseline SR: 97.0%
- Round-2 minus Round-1: +0.0 pp
- Aggregate rate exact match: yes
- Scope: This checks aggregate success-rate reproducibility; the Round-1 summary alone does not establish episode-level outcome identity.

A positive observed delta is reported only as no detected loss; it is not interpreted as an improvement.
The protocol stop rule applies: do not add layer schedules before joint Round-1/Round-2 interpretation.
