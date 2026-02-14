# Data Leakage Audit

## Methodology
We define "index time" as the last observation before the prediction window. Any information from `t > index_time` is considered future leakage.

## Excluded Variables (Known Leakage)
| Variable | Description | Reason |
| :--- | :--- | :--- |
| `DIAG_diabetes` | Doctor told you have diabetes | Direct target proxy |
| `MED_insulin` | Taking insulin | Treatment implies diagnosis |
| `HBA1C` | Lab result | Clinical confirmation |

## Suspect Variables (Audited)
- [ ] `weight_loss`: Could be symptom or lifestyle change. **Decision**: Kept (pre-diagnosis signal).
- [ ] `hospital_visits`: Could imply serious event. **Decision**: Kept count only.

## Automated Checks
- [ ] No future timestamps in feature window.
- [ ] No overlap between target window and feature window.
