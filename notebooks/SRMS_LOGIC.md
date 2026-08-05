# SRMS Logic Specification

## Purpose of the SRMS module

The current SRMS MVP identifies potential anomalies in SRMS-style quality-control results by extracting a candidate subset from `ResultSet.csv`, calculating QC and statistical features, applying explainable rule-based flags, applying an Isolation Forest anomaly model, and assigning a prioritised risk score and risk level.

Confirmed from the repository: the source export does not contain a dedicated SRMS subtype label. The workflow therefore treats qualifying `Standard` rows as "SRMS candidates" because Datamine clarification in the notebook says SRMS, CRM, LCS, and similar materials are assessed like Standards against target values and limits.

This document describes the current workflow as implemented in:

- `SRMS- Team32_code.ipynb`
- `tmp/jupyter-notebook/build_srms_resultset_notebook.py`
- `srms_candidate_anomaly_results.csv`
- `srms_candidate_summary.csv`

The general QC MVP in `run_mvp.py` contains related, partly duplicated anomaly logic, but it is not the SRMS-specific workflow.

### Questions for Datamine

- For the final product name, should this be described as `SRMS candidate anomaly detection`, or would `reference-material QC anomaly detection` be more accurate?
- Is the current MVP purpose correct: to prioritise reference-material/standard-style QC results for review rather than to automatically reject results?
- Should SRMS, CRM, LCS, and other reference materials remain grouped together in one workflow, or should the final product separate them into different QC material categories?

## Inputs

### Source file

The SRMS workflow reads `ResultSet.csv` from a hard-coded path:

- `/Users/linhlinh/Downloads/ResultSet.csv`

### Required source columns

The notebook declares these required input columns:

| Source column | Current SRMS output name | Purpose |
| --- | --- | --- |
| `ANALYTICAL_TYPE` | `ANALYTICAL_TYPE` | Filters source rows to `Standard`. |
| `STD_LOT_CODE` | `SRMSLotCode` | Reference-material lot code. |
| `STD_CODE` | `SRMSStandardCode` | Reference-material standard code. |
| `JOB_CODE` | `JobCode` | Job/batch identifier and part of generated `SampleID`. |
| `NUMERIC_FINAL_VALUE` | `Value` | Measured result value. |
| `ANALYSED_DATE` | `AnalysisDate` | Analysis timestamp, used for sorting and trend plots. |
| `SCHEME_CODE` | `SchemeCode` | Analytical scheme; also used in one hard-coded source correction. |
| `ANALYTE_CODE` | `Analyte` | Analyte being measured. |
| `STANDARD_STATUS` | `StandardStatus` | Source QC status, used in rule-based classification. |
| `PRECISION_STATUS` | dropped | Read as required, then excluded because the notebook says it is not applicable to Standards, Blanks, or Spikes. |
| `INTERNAL_MIN_VALUE` | `LowerLimit` | Lower acceptable limit. |
| `INTERNAL_MAX_VALUE` | `UpperLimit` | Upper acceptable limit. |
| `INTERNAL_MIN_INCLUSIVE` | `LowerLimitInclusive` | Whether the lower limit boundary is inclusive. |
| `INTERNAL_MAX_INCLUSIVE` | `UpperLimitInclusive` | Whether the upper limit boundary is inclusive. |
| `INTERNAL_MIN_WARNING_VALUE` | `WarningLower` | Lower warning threshold. |
| `INTERNAL_MAX_WARNING_VALUE` | `WarningUpper` | Upper warning threshold. |
| `INTERNAL_MIN_WARNING_INCLUSIVE` | `WarningLowerInclusive` | Whether the lower warning boundary is inclusive. |
| `INTERNAL_MAX_WARNING_INCLUSIVE` | `WarningUpperInclusive` | Whether the upper warning boundary is inclusive. |
| `INTERNAL_TARGET_VALUE` | `TargetValue` | Assigned target value. |
| `PARENT_NUMERIC_FINAL_VALUE` | `ParentValue` | Retained in outputs but not used in the SRMS feature or risk calculations. |
| `SPECIFICATION_CODE` | `SpecificationCode` | Retained as metadata. |
| `UNIT_CODE` | `UnitCode` | Unit of measurement; also used in one hard-coded source correction. |

### SRMS candidate extraction filters

A row is included in the SRMS candidate set only when all of the following are true:

- `ANALYTICAL_TYPE == "Standard"`
- `STD_CODE` is not blank, not `Sample`, and not `TSV_BLANK`
- `INTERNAL_TARGET_VALUE` is present
- both `INTERNAL_MIN_VALUE` and `INTERNAL_MAX_VALUE` are present

The generated output confirms 36,442 SRMS candidate rows.

### Added metadata fields

The workflow adds:

- `SampleID`: `JobCode` plus the original source row index, joined with `-`
- `SourceCategory`: hard-coded to `Standard`
- `SRMSAssumption`: fixed explanatory text saying reference-material standards are used as SRMS candidates because explicit SRMS subtype is absent
- `ModelVersion`: present in `SRMS- Team32_code.ipynb` as `srms-iforest-v0.1`, but not present in the generated CSV currently in the repository
- `RuleVersion`: present in `SRMS- Team32_code.ipynb` as `qc-rules-v0.1`, but not present in the generated CSV currently in the repository

### Questions for Datamine

- Is the current SRMS candidate extraction logic acceptable for the MVP?
- Should rows with `ANALYTICAL_TYPE = Standard`, valid `STD_CODE`, target value, and lower/upper limits be treated as SRMS/reference-material candidates?
- Are there any additional `STD_CODE`, `STD_LOT_CODE`, `SPECIFICATION_CODE`, or `ANALYTICAL_TYPE` values that should be included or excluded?
- Should the final merged output use the label `SRMSCandidate`, `ReferenceMaterialCandidate`, or another business-approved category name?
- Which field should be treated as the stable business sample/result identifier in the final output: `JOB_CODE`, source row index, another ResultSet field, or a generated composite ID?
- Should `PRECISION_STATUS` always be excluded from SRMS/reference-material candidate logic, or should it be retained as metadata?

## Processing steps

1. Load `ResultSet.csv`.
2. Validate required columns in the notebook version.
3. Convert `ANALYSED_DATE` to datetime with invalid dates coerced to missing.
4. Extract SRMS candidates using the filters above.
5. Rename source columns to SRMS-friendly names.
6. Generate `SampleID`, `SourceCategory`, and `SRMSAssumption`.
7. Drop `PRECISION_STATUS`.
8. Sort records by `AnalysisDate`.
9. Apply a hard-coded known source correction for one OREAS zinc target value.
10. Calculate statistical and QC-derived features.
11. Fill missing warning limits when needed.
12. Apply rule-based `RuleFlag` and `RuleFlagScore`.
13. Train and apply an Isolation Forest model.
14. Calculate `RiskScore` and `RiskLevel`.
15. Save detailed results to `srms_candidate_anomaly_results.csv`.
16. Save risk-level counts to `srms_candidate_summary.csv`.
17. Generate SRMS charts in `srms_outputs/`.

### Questions for Datamine

- Is the overall processing sequence acceptable for the final merged MVP?
- Should any source-system corrections be applied before feature calculation, and should they come from an approved correction table?
- Should records be sorted only by `AnalysisDate`, or should ordering also use job, sample/result number, laboratory sequence, or another field?
- Should the final product save only the latest output, or should it keep run history and model/rule versions for auditability?

## Statistical features currently calculated

| Feature | Calculation | Notes |
| --- | --- | --- |
| `Deviation` | `Value - TargetValue` | Signed difference from assigned target. |
| `DeviationPct` | `(Deviation / TargetValue) * 100` | If target is zero, target is replaced with missing and the final percentage is filled as `0`. |
| `RollingMean` | 5-row rolling mean of `Value` grouped by `SRMSStandardCode` and `Analyte` | Uses `min_periods=1`; depends on current sort order by `AnalysisDate`. |
| `RollingStd` | 5-row rolling standard deviation of `Value` grouped by `SRMSStandardCode` and `Analyte` | Uses `min_periods=2`; missing early values are filled as `0`. |
| `DistanceToLower` | `Value - LowerLimit` | Positive when above the lower limit; negative when below it. |
| `DistanceToUpper` | `UpperLimit - Value` | Positive when below the upper limit; negative when above it. |
| `LimitSpan` | `UpperLimit - LowerLimit` | Used for within-range position and warning-limit defaults. |
| `WithinRangePct` | `(Value - LowerLimit) / LimitSpan * 100` | Only calculated when absolute span is greater than `1e-9`; values can be less than 0 or greater than 100 outside limits. |

### Questions for Datamine

- Are these statistical features meaningful for SRMS/reference-material review?
- Is a 5-result rolling window appropriate, or should rolling statistics use a different number of records or a time-based window?
- Should rolling statistics be grouped by `SRMSStandardCode` and `Analyte` only, or should they also include lot, unit, scheme, lab, job, or specification?
- How should the system handle zero or missing target values when calculating `DeviationPct`?
- Should `WithinRangePct` values below 0 or above 100 be kept as-is, capped, or converted into a clearer breach indicator?

## Detection rules

### Warning-limit fallback

If warning limits are missing, the workflow derives them from the internal range:

- `WarningLower = LowerLimit + LimitSpan * 0.10`
- `WarningUpper = UpperLimit - LimitSpan * 0.10`

This creates a default inner warning band 10% inside each end of the allowed range.

### Boundary inclusivity

The workflow treats missing inclusivity flags as inclusive (`Y`).

Current condition names are slightly counterintuitive:

- If `LowerLimitInclusive == "Y"`, lower red breach is `Value < LowerLimit`
- If `LowerLimitInclusive != "Y"`, lower red breach is `Value <= LowerLimit`
- If `UpperLimitInclusive == "Y"`, upper red breach is `Value > UpperLimit`
- If `UpperLimitInclusive != "Y"`, upper red breach is `Value >= UpperLimit`
- Equivalent logic is applied to `WarningLowerInclusive` and `WarningUpperInclusive`

### RuleFlag classification

The rule classifier produces `Green`, `Yellow`, or `Red`.

`Red` is assigned when either condition is true:

- value breaches the lower or upper internal limit
- `StandardStatus` contains `Failure`, case-insensitive

`Yellow` is assigned when no red condition applies and either condition is true:

- value breaches the lower or upper warning limit
- `StandardStatus` contains `Warning`, case-insensitive

`Green` is assigned otherwise.

`RuleFlagScore` is mapped as:

- `Green = 0`
- `Yellow = 1`
- `Red = 2`

Confirmed from current CSV output:

- `Green`: 24,315 rows
- `Yellow`: 6,866 rows
- `Red`: 5,261 rows

### Questions for Datamine

- Should `IgnoredUpperFailure` and `IgnoredLowerFailure` still be classified as `Red`, or should ignored failures be downgraded or excluded?
- If source warning limits are missing, is it acceptable to derive warning limits using a 10% inset from the lower/upper limits?
- Should missing inclusivity flags default to inclusive (`Y`)?
- Are `Green`, `Yellow`, and `Red` the correct final rule labels, or should the final product use Datamine/CCLAS terminology such as `Pass`, `Warning`, and `Failure`?
- Should rule classification be based on calculated limit breaches, source `StandardStatus`, or both?

## Machine learning anomaly detection

The SRMS workflow trains an unsupervised Isolation Forest model.

Model settings:

- `n_estimators = 250`
- `contamination = 0.05`
- `random_state = 42`
- `n_jobs = -1`

Model input features:

- `Value`
- `TargetValue`
- `Deviation`
- `DeviationPct`
- `RollingMean`
- `RollingStd`
- `DistanceToLower`
- `DistanceToUpper`
- `LimitSpan`
- `WithinRangePct`
- `RuleFlagScore`

Missing and infinite feature values are replaced before modelling:

- infinities are converted to missing
- each feature is filled with its median
- any remaining missing value is filled with `0`

Model outputs:

- `IForestAnomaly`: `True` when Isolation Forest prediction is `-1`
- `IForestScore`: negative Isolation Forest sample score
- `IForestScoreNorm`: min-max normalised `IForestScore`, calculated as `(score - min) / (max - min + 1e-9)`

Confirmed from current CSV output:

- `IForestAnomaly == True`: 1,823 rows
- `IForestAnomaly == False`: 34,619 rows

### Questions for Datamine

- Is Isolation Forest acceptable as an MVP anomaly method for unlabelled SRMS/reference-material data?
- Is the fixed 5% expected anomaly rate appropriate, or should the contamination rate be configurable?
- Should the model use `RuleFlagScore` as an input feature, or should the ML signal remain independent from rule-based QC flags?
- Should model output be presented as a review aid only, with rule-based status remaining the authoritative QC decision?
- Are there historical reviewed anomaly examples that can be used to validate the model results?

## How RiskScore and RiskLevel are calculated

`RiskScore` is a 0-100 style score made from four components:

| Component | Formula | Maximum contribution |
| --- | --- | --- |
| Deviation component | `clip(abs(DeviationPct), 0, 50) / 50 * 35` | 35 |
| Variability component | `clip(RollingStd / std_scale, 0, 3) / 3 * 20` | 20 |
| Rule component | `20` for `Red`, `10` for `Yellow`, `0` for `Green` | 20 |
| Model component | `IForestScoreNorm * 25` | 25 |

`std_scale` is calculated as:

- `max(std(RollingStd), 1.0)`

The final score is:

- `RiskScore = round(deviation_component + variability_component + rule_component + model_component, 2)`

`RiskLevel` is assigned with fixed bins:

- `Low`: `RiskScore > -0.01` and `RiskScore <= 30`
- `Medium`: `RiskScore > 30` and `RiskScore <= 60`
- `High`: `RiskScore > 60` and `RiskScore <= 100`

Confirmed from current `srms_candidate_summary.csv`:

- `Low`: 27,617 rows, 75.78%
- `Medium`: 7,371 rows, 20.23%
- `High`: 1,454 rows, 3.99%

### Questions for Datamine

- Are `Low`, `Medium`, and `High` the correct risk levels for the final product?
- Are the current risk thresholds acceptable: Low up to 30, Medium above 30 to 60, and High above 60?
- Are the current component weights acceptable: 35 for deviation, 20 for variability, 20 for rule severity, and 25 for model score?
- Should a `Red` rule flag automatically force a minimum risk level, regardless of the numeric score?
- What business action should each risk level imply, for example no action, review recommended, or priority investigation?
- Should the final output include a `RiskReason` field explaining the main reason for the score?

## Outputs

### Detailed result CSV

`srms_candidate_anomaly_results.csv` contains source fields, renamed metadata, calculated features, rule outputs, model outputs, and risk outputs.

Current output columns:

- `ANALYTICAL_TYPE`
- `SRMSLotCode`
- `SRMSStandardCode`
- `JobCode`
- `Value`
- `AnalysisDate`
- `SchemeCode`
- `Analyte`
- `StandardStatus`
- `LowerLimit`
- `LowerLimitInclusive`
- `UpperLimit`
- `UpperLimitInclusive`
- `WarningUpper`
- `WarningUpperInclusive`
- `WarningLower`
- `WarningLowerInclusive`
- `TargetValue`
- `ParentValue`
- `UnitCode`
- `SpecificationCode`
- `SampleID`
- `SourceCategory`
- `SRMSAssumption`
- `Deviation`
- `DeviationPct`
- `RollingMean`
- `RollingStd`
- `DistanceToLower`
- `DistanceToUpper`
- `LimitSpan`
- `WithinRangePct`
- `RuleFlag`
- `RuleFlagScore`
- `IForestAnomaly`
- `IForestScore`
- `IForestScoreNorm`
- `RiskScore`
- `RiskLevel`

### Summary CSV

`srms_candidate_summary.csv` contains:

- `RiskLevel`
- `Count`
- `Percent`

### Charts

The workflow generates these chart files:

- `srms_outputs/srms_value_vs_target.png`
- `srms_outputs/srms_deviation_over_time.png`
- `srms_outputs/srms_risk_distribution.png`
- `srms_outputs/srms_top_codes_anomaly_rate.png`
- `srms_outputs/srms_deviation_histogram.png`
- `srms_outputs/srms_top_high_risk.png`

### Questions for Datamine

- What should the final merged output format be: CSV, Excel workbook, dashboard, notebook, or web app?
- Which columns are required in the final review output, and which columns should be hidden or kept only for audit?
- Should the final output include both `RuleFlag` and `IForestAnomaly`, or should these be combined into one final decision field?
- Should the output include charts, and if yes, which charts are most useful for Datamine reviewers?
- Should the final product include run metadata such as source file name, run date, rule version, and model version?

## Assumptions

Confirmed assumptions from notebook text and code:

- The source file is a real QC export named `ResultSet.csv`.
- The source data does not explicitly identify SRMS records as a separate subtype.
- `Standard` records with reference-material codes, target values, and limits are an acceptable MVP proxy for SRMS-like material records.
- SRMS-like materials can be assessed using target, warning-limit, and lower/upper-limit logic similar to Standards.
- `STD_CODE` values of blank, `Sample`, and `TSV_BLANK` are not reference-material SRMS candidates.
- Missing limit inclusivity flags should be treated as inclusive.
- Missing warning limits can be approximated using a hard-coded 10% inset from the internal limits.
- `PRECISION_STATUS` should be excluded from SRMS candidate results because it is not applicable to Standards, Blanks, or Spikes.
- Isolation Forest is appropriate for the MVP because confirmed anomaly labels are not available.

Assumptions that cannot be fully confirmed from code alone:

- That every included `Standard` reference-material row is genuinely SRMS-like rather than CRM, LCS, or another standard subtype.
- That the 5% model contamination setting reflects a real expected SRMS anomaly rate.
- That the risk component weights, especially 35/20/20/25, reflect business-approved severity priorities.
- That `RiskScore` thresholds of 30 and 60 reflect operational review thresholds.
- That the one hard-coded OREAS correction is complete and validated against source-system rules.
- That sorting only by `AnalysisDate` gives the intended order when multiple rows share the same timestamp.

### Questions for Datamine

- Can Datamine confirm that using Standard/reference-material rows as SRMS candidates is acceptable for the final MVP wording?
- Can Datamine confirm whether the hard-coded OREAS zinc correction is valid and whether there are other known source corrections?
- Can Datamine confirm whether missing warning limits, missing inclusivity flags, and missing target values have standard handling rules?
- Can Datamine confirm whether the final product should state clearly that this is not a fully labelled SRMS-only model?

## Known limitations

- No confirmed SRMS label exists in the source export, so the workflow is an SRMS candidate workflow, not a validated SRMS-only workflow.
- The source data path is hard-coded to a local user Downloads folder.
- One known source correction is hard-coded: `GE_ICP40Q12 / OREAS_351 / ZN / MG_KG` with target `46.99` is changed to `469900`.
- Warning-limit fallback uses a fixed 10% rule without documented business approval.
- Risk scoring weights and thresholds are hard-coded.
- Isolation Forest contamination is fixed at 5%.
- The model is trained on the same dataset it scores, with no separate validation set.
- `RuleFlagScore` is included as a machine-learning feature, so rule logic influences both the direct rule component and the model signal.
- `DeviationPct` is filled as `0` when target is zero or invalid after division; this may hide target-data issues.
- `RollingMean` and `RollingStd` use a 5-row rolling window, but the code does not document why 5 rows is the correct business window.
- Rolling statistics are grouped only by `SRMSStandardCode` and `Analyte`; they do not include lot, unit, scheme, or laboratory/job context.
- `ParentValue` is retained but not used in the SRMS model or risk score.
- The generated script and notebook overlap but are not identical: the notebook adds `ModelVersion` and `RuleVersion`, while the generated CSV in the repository does not contain these fields.
- Some SRMS output fields currently have missing values, including all `ParentValue` values in the current CSV.

### Questions for Datamine

- Which limitations are acceptable for the MVP, and which must be fixed before final delivery?
- Is it acceptable that the model is trained and scored on the same dataset for the MVP?
- Should missing `ParentValue`, missing analyte, or missing status values trigger data-quality warnings in the final output?
- Should hard-coded thresholds remain for the MVP, or must they be configurable before delivery?

## Duplicated, unclear, or hard-coded logic

### Duplicated logic

- The SRMS extraction, feature engineering, rule flags, Isolation Forest scoring, and risk scoring appear in both `SRMS- Team32_code.ipynb` and `tmp/jupyter-notebook/build_srms_resultset_notebook.py`.
- The general QC MVP in `run_mvp.py` implements related concepts using different names and thresholds, including range breach, warning breach, robust z-score, Isolation Forest, baseline scoring, and risk level assignment.

### Unclear logic

- The business meaning of "SRMS" cannot be confirmed directly from source fields because the dataset has no explicit SRMS subtype.
- `StandardStatus` values containing `IgnoredLowerFailure` or `IgnoredUpperFailure` are treated as red because they contain `Failure`. The code does not document whether "Ignored" failures should still raise risk.
- The workflow excludes `PRECISION_STATUS`, but the general QC MVP uses precision status in its baseline scoring. This difference should be explained in business rules.
- `ModelVersion` and `RuleVersion` are defined in the notebook but absent from the generated CSV. It is unclear which artifact is intended as the canonical implementation.
- `WithinRangePct` can go below 0 or above 100, but the business interpretation of those values is not documented.

### Hard-coded logic

- Source path: `/Users/linhlinh/Downloads/ResultSet.csv`
- Output paths under `IFn735 Project/qc_anomaly_mvp`
- SRMS proxy filter: `ANALYTICAL_TYPE == "Standard"` with non-sample `STD_CODE`
- Excluded reference codes: blank, `Sample`, `TSV_BLANK`
- Known source correction for one OREAS zinc target
- Rolling window size: 5 rows
- Warning fallback inset: 10%
- Inclusivity default: `Y`
- Isolation Forest settings: 250 trees, 5% contamination, random seed 42
- Risk weights: 35, 20, 20, 25
- Risk bins: 0-30, 30-60, 60-100

### Questions for Datamine

- Which implementation should be treated as canonical for the final product: the SRMS notebook, the generated notebook builder script, or a new merged pipeline?
- Should the general QC MVP logic and SRMS candidate logic be merged into one shared business-rule module?
- Which hard-coded rules should become configurable business settings?
- Should Datamine provide an approved rule configuration table for material type, status mapping, risk scoring, and known source corrections?

## Recommended improvements

1. Define a formal SRMS business entity and source-field mapping.
2. Add an explicit SRMS/CRM/LCS/reference-material subtype field if available from CCLAS or Datamine.
3. Move source path, output path, model settings, rule thresholds, risk weights, and risk bins into configuration.
4. Replace the hard-coded OREAS correction with a documented source-correction table.
5. Document whether `Ignored*Failure` statuses should be treated as red, yellow, green, or excluded.
6. Decide whether `PRECISION_STATUS` is always irrelevant for SRMS candidates, and document the reason.
7. Validate the 10% warning-limit fallback with domain stakeholders or avoid deriving warning limits when source warning limits are missing.
8. Review whether rolling statistics should group by lot, unit, scheme, laboratory, or job in addition to standard code and analyte.
9. Add data-quality flags for missing target, zero target, missing analyte, missing status, invalid date, and invalid limit spans.
10. Add version fields to every exported result row and ensure notebook/script outputs are consistent.
11. Separate rule-based classification from machine-learning features if a cleaner independent model signal is needed.
12. Validate risk weights and thresholds against historical reviewed incidents or stakeholder severity definitions.
13. Add tests for extraction filters, inclusivity boundary behavior, rule classification, source corrections, and risk binning.
14. Create a clearer output schema with field definitions and accepted values for every exported column.
15. Rename the module and outputs to make clear whether they are `SRMS candidate` results or confirmed `SRMS` results.

### Questions for Datamine

- Which recommended improvements are required for the final merged MVP, and which can be listed as future work?
- Should the next version prioritise business-rule clarity, model validation, dashboard usability, or output/reporting format?
- Who should approve the final rule thresholds and risk-level definitions?
- Should Datamine review a small sample of `High` and `Medium` risk rows to confirm whether the logic matches operational expectations?
