# US frequency allocation database, 30 MHz to 6 GHz: sources, parsing, validation

## Sources

| Item | Value |
|---|---|
| Primary source | FCC Online Table of Frequency Allocations, 47 C.F.R. § 2.106 (Office of Engineering and Technology) |
| File | `https://transition.fcc.gov/oet/spectrum/table/fcctable.pdf` (PDF, 180 pages) |
| Table revision | "Revised on July 1, 2022" (cover page of the PDF) |
| Retrieved | 2026-09-20T23:12:34Z |
| Secondary file | `https://transition.fcc.gov/oet/spectrum/table/fcctable.docx` (Word edition, cover page reads "Revised on February 1, 2021"); used only for a band-edge consistency check |
| Cross-check API | FCC Spectrum Dashboard `getSpectrumBands` (225-3700 MHz). Not reachable at retrieval time: the HTTPS endpoint answers 302 to `https://www.fcc.gov/what-can-we-help-you-find`, the HTTP endpoint answers an Akamai 403. The API documentation page is still published; the service itself did not answer. |
| Technology metadata | 47 CFR Parts 15, 22, 25, 27, 73, 74, 80, 87, 90, 95, 96, 97; 3GPP TS 36.101 and TS 38.101-1; IEEE 802.11-2020 Annex E; Bluetooth Core 5.4; ICAO Annex 10 Vols I, III, IV; RTCA DO-260B/DO-282B; ITU-R M.1084/M.1371/M.1463/M.1464/M.1849; GNSS interface control documents; LoRa Alliance RP002; FMH-11. Each entry in `channelization.json` names its own sources. |

The FCC landing page `https://www.fcc.gov/engineering-technology/policy-and-rules-division/general/radio-spectrum-allocation` returns an Akamai "Access Denied" page to non-browser clients, so the Word link was resolved directly at `transition.fcc.gov`. No third-party copy of the Table was used.

## Files

- `us_allocations_30MHz_6GHz.json`: `metadata` plus `bands`, one object per row, nested service lists.
- `us_allocations_30MHz_6GHz.parquet`: same rows, one per record; the three nested columns (`federal_services`, `nonfederal_services`, `international_region2_boxes`) are stored as JSON strings, all other list columns as native lists.
- `channelization.json`: `technologies` (67 entries) and `by_allocation_row` (row key to technology ids).

## Row definition

The US Table has separate Federal and Non-Federal columns whose boxes do not share edges. A row in this database is one segment of the union partition of both columns' box edges between 30 and 6000 MHz: 223 rows, built from 176 Federal boxes and 217 Non-Federal boxes. Each row records the enclosing box from each column (`federal_box`, `nonfederal_box`) so the original box structure can be recovered; segment edges are clipped to 30 and 6000 MHz, box edges are not (the last row's boxes are 5925-6700 Federal and 5925-6425 Non-Federal). 47 shared boxes (identical Federal and Non-Federal allocation printed once across both columns) are flagged `fed_nonfed_shared_box`.

Per row: services for each column with a `primary` flag (capitals in the Table, per 47 CFR 2.104(d)) and service-level footnotes; box-level footnotes; `us_footnotes` (US, NG, G series) and `intl_footnotes_in_us_table` (5.xxx series appearing in the US columns); FCC rule parts as printed in the last column (`rule_parts`, `rule_part_labels`); the ITU Region 2 boxes overlapping the row (`international_region2_boxes`, with flattened `region2_primary`/`region2_secondary`); the raw box text for audit; PDF page numbers; and `technologies` linking to `channelization.json`.

## Parsing approach

The Word edition is older than the PDF, so the PDF was parsed. Text order in the PDF is scrambled by the six-column layout; the parse therefore works from word positions and the drawn table rules (pdfplumber 0.11):

1. Each Table page ("Page 17" to "Page 43", 27.41-7145 MHz) spans two physical PDF pages (18-44 in 1-based numbering); the pair is stacked into one canvas so that boxes split by the physical page break stay whole.
2. Column boundaries are taken from the vertical rules nearest each header word ("Region 1 Table" ... "FCC Rule Part(s)") on each page, since column widths change from page to page.
3. Box boundaries in a column are the horizontal rules spanning that column. Shared Federal/Non-Federal boxes are detected from the absence of the vertical rule between the two columns over the box height; text in shared boxes runs across both columns, so both column widths are read.
4. Within a box, lines indented more than about 4 pt beyond the range line are wrapped continuations and are rejoined (with hyphenation at line breaks such as "RADIODETERMINATION-SATEL-/LITE" repaired against a service vocabulary). Trailing tokens matching `5.nnnX`, `USnnnX`, `NGnnnX`, `GnnnX` are footnotes; a line consisting only of footnotes is box-level. Text-extraction artifacts (split leading letters such as "A ERONAUTICAL", broken footnote numbers such as "5.34 1") are normalized by pattern.
5. Rule-part boxes are attached to Non-Federal boxes by vertical overlap; entries are split on the trailing "(nn)" part designator.
6. "(See previous page)" placeholders are dropped; the box they refer to is already complete on the preceding page.

## Validation

- Tiling: the Federal column (176 boxes), the Non-Federal column (217 boxes including shared boxes) and the Region 2 column (172 boxes) each cover 30-6000 MHz with no gaps and no overlaps. The 223 union rows are contiguous from 30.0 to 6000.0 MHz.
- Residue: no box produced unparsed text (every non-range line resolved to services or footnotes); no rule-part cell left unparsed text after the page-footer strings were removed.
- Every box begins with a frequency range or a "(See previous page)" marker; no cross-page row continuations remained after page-header handling was corrected.
- Spot checks of 30 bands against the printed Table (FM, aviation, marine channel 16, GMRS, 700/800/900 MHz, ISM, L1 GNSS, Wi-Fi, CBRS, C-band, U-NII-3, 6 GHz) matched.
- Word-edition edge check (substitute for the offline Dashboard API): the 2021 Word file was parsed with python-docx. Federal: 173 of 176 PDF box ranges also appear in the Word edition; Non-Federal: 176 of 217. The ranges present only in the PDF are 75.4-88 (Federal), 2483.5-2500 (Federal), 3300-3500 (Federal), 144-148 (Non-Federal) and 3300-3450/3450-3600 (Non-Federal), and the corresponding Word-only ranges are their pre-2022 splits (2483.5-2495/2495-2500, 3300-3400/3400-3500, 3500-3550/3550-3600, 144-146/146-148, 75.4-76/76-88); these are consistent with rule changes adopted between the two editions (3.45 GHz service, 2483.5-2500 MHz, amateur 3.3-3.5 GHz). Cell-text comparison between the editions was not used as a check because python-docx shifts cell indices on merged cells and places footnote rows in separate table rows.

## Known gaps

- The Spectrum Dashboard cross-check was not performed (0 of 0 bands compared); the endpoint did not respond as documented.
- The primary/secondary flag is derived from capitalization of the first word of the service name. Qualifiers printed in lower case after a primary service ("MOBILE except aeronautical mobile") are kept in the name and do not affect the flag.
- Region 1 and Region 3 columns were not parsed. Region 2 was parsed with the same method, including boxes merged across Region columns; its content was validated only for tiling, not spot-checked against the printed Table to the same extent.
- Rule-part attachment relies on vertical overlap with Non-Federal boxes. Where a rule-part cell spans several boxes, every overlapped box receives every entry of that cell, which reproduces the Table's layout but can attribute a rule part to a sub-band it does not govern.
- International footnote text (5.xxx), US footnote text (USnnn) and NG/G footnote text are recorded by number only; the footnote bodies were not extracted.
- The Table's own disclaimer applies: the Online Table may show amendments adopted but not yet in effect; the CFR is the legal source. The edition parsed is dated July 1, 2022; allocations adopted since are not reflected.
- Technology metadata: values not confirmed against the cited document are null or listed under `unverified_fields` for each entry (for example, radar PRF ranges, DECT 6.0 carrier centres, Iridium channel count, US status of 8.33 kHz aeronautical channel spacing). 3GPP channel-bandwidth lists are the commonly supported sets, not release-exact tables. Federal channel plans (225-400 MHz, NTIA) are described but not enumerated.
