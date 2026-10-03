"""Generate small, synthetic sample PDFs for the evaluation harness.

These are fictional documents (a company handbook, a product spec, a
research note) with clear, specific facts - not personal data - so the
eval suite is reproducible by anyone who clones the repo. Each document has
enough sections/distractor content to produce a realistic number of chunks,
so the retrieval benchmark in run_eval.py actually has something to
discriminate between (a 3-chunk corpus makes every strategy look perfect).

Run this once to (re)generate eval/sample_docs/*.pdf; the generated PDFs are
committed to the repo so run_eval.py works without running this first.
"""

from pathlib import Path

import fitz

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_docs"

DOCUMENTS = {
    "handbook.pdf": [
        ("title", "Northwind Retail - Employee Handbook"),
        ("body", "This handbook describes the policies and benefits that apply to all "
                 "full-time and part-time employees of Northwind Retail. It is reviewed "
                 "annually by People Operations and supersedes all prior versions."),
        ("heading", "Leave Policy"),
        ("body", "Full-time employees accrue 18 days of annual paid leave per year, "
                 "credited at the start of each calendar year in a single lump sum. "
                 "Part-time employees accrue leave on a pro-rated basis according to "
                 "their contracted weekly hours. Unused leave up to 5 days may be "
                 "carried over into the next calendar year; any balance above that is "
                 "forfeited unless local law requires otherwise."),
        ("body", "Sick leave is tracked separately from annual leave and is capped at "
                 "10 days per year. Employees returning from more than 3 consecutive "
                 "sick days must provide a note from a medical practitioner."),
        ("heading", "Remote Work Policy"),
        ("body", "Employees may work remotely up to 3 days per week, subject to manager "
                 "approval and role suitability. Fully remote arrangements, defined as "
                 "more than 3 remote days per week on an ongoing basis, require VP-level "
                 "sign-off and are reviewed every 6 months."),
        ("body", "Employees working remotely are expected to be reachable during core "
                 "hours of 10am to 4pm in their local time zone and to attend all "
                 "scheduled meetings with video enabled where practical."),
        ("heading", "Expense Reimbursement"),
        ("body", "Any expense over $50 requires an itemized receipt to be submitted "
                 "through the expense portal within 30 days of the purchase. Expenses "
                 "under $50 can be self-certified without a receipt, up to a maximum of "
                 "$200 in self-certified expenses per quarter."),
        ("body", "Historical note: prior to the 2022 policy revision, the receipt "
                 "threshold for expense reimbursement was set at $75 rather than the "
                 "current amount, and self-certification did not exist as an option."),
        ("body", "Travel booked more than 14 days in advance should use the corporate "
                 "travel portal; last-minute travel may be booked directly and expensed, "
                 "subject to the same $50 receipt threshold described above."),
        ("heading", "Data Security Policy"),
        ("body", "VPN access is mandatory for any connection to internal systems from "
                 "outside the office network. Split-tunneling is disabled by default on "
                 "all company-managed devices. Multi-factor authentication is required "
                 "for all company accounts, including email, source control, and the "
                 "expense portal."),
        ("body", "Company devices must have full-disk encryption enabled and must not "
                 "be left unattended and unlocked in public spaces. Any lost or stolen "
                 "device must be reported to IT Security within 24 hours."),
        ("heading", "Benefits"),
        ("body", "Northwind Retail matches employee 401k contributions up to 4 percent "
                 "of base salary. Health insurance premiums are covered at 80 percent "
                 "for the employee and 50 percent for dependents. Enrollment windows "
                 "open once per year in November."),
        ("heading", "Code of Conduct"),
        ("body", "Employees are expected to treat colleagues, customers, and vendors "
                 "with respect. Harassment, discrimination, and retaliation of any kind "
                 "are grounds for immediate termination, regardless of tenure or role."),
        ("heading", "Performance Reviews"),
        ("body", "Formal performance reviews are conducted twice per year, in June and "
                 "December. Managers are expected to hold informal check-ins with direct "
                 "reports at least once per month between formal review cycles."),
        ("heading", "Equipment Policy"),
        ("body", "New employees are issued a laptop and standard peripherals during "
                 "onboarding. Equipment refresh requests are evaluated every 3 years, "
                 "or sooner in the case of hardware failure verified by IT."),
    ],
    "product_spec.pdf": [
        ("title", "Aurora API Gateway - Product Specification"),
        ("body", "Aurora is a managed API gateway product used by customers to expose, "
                 "secure, and monitor internal services. This document summarizes the "
                 "current specification as of the latest release."),
        ("heading", "Rate Limiting"),
        ("body", "The gateway enforces a default rate limit of 1000 requests per minute "
                 "per API key. This limit is configurable per customer on Enterprise "
                 "plans, up to a hard ceiling of 50,000 requests per minute per key."),
        ("body", "Requests exceeding the configured limit receive an HTTP 429 response "
                 "with a Retry-After header. Burst traffic is smoothed using a token "
                 "bucket algorithm with a bucket size equal to 10 percent of the "
                 "per-minute limit."),
        ("body", "Legacy note: the original 2021 beta release used a default rate "
                 "limit of 200 requests per minute per API key before the limit was "
                 "raised in the first general-availability release."),
        ("heading", "Authentication"),
        ("body", "Aurora supports OAuth2 client-credentials flow and static API keys for "
                 "service-to-service calls. JWT bearer tokens signed with RS256 are "
                 "accepted for user-delegated requests, with a maximum token lifetime of "
                 "1 hour before re-issuance is required."),
        ("heading", "Supported Regions"),
        ("body", "Aurora is currently deployed in two regions: US-East and EU-West. "
                 "Cross-region failover is available on the Pro and Enterprise tiers, "
                 "with an expected failover time of under 30 seconds."),
        ("body", "Data residency guarantees mean that traffic routed through the "
                 "EU-West region never transits US-East infrastructure, to support "
                 "customers with regional compliance requirements."),
        ("heading", "Pricing"),
        ("body", "The Free tier includes 10,000 requests per month at no cost, with "
                 "access to US-East only and no SLA guarantee. The Pro tier costs $99 "
                 "per month and includes 1,000,000 requests, both regions, and a 99.9 "
                 "percent uptime SLA."),
        ("body", "The Enterprise tier is custom-priced and includes dedicated capacity, "
                 "configurable rate limits, a 99.99 percent uptime SLA, and a dedicated "
                 "support engineer."),
        ("heading", "Observability"),
        ("body", "Aurora exposes request-level logs, latency histograms, and error "
                 "rates via a built-in dashboard, with a 30-day retention window on "
                 "the Pro tier and a 1-year retention window on Enterprise."),
        ("heading", "Webhooks"),
        ("body", "Customers can register webhook endpoints to receive near-real-time "
                 "notifications for rate-limit breaches, certificate expiry, and "
                 "region failover events. Webhook delivery is retried up to 5 times "
                 "with exponential backoff before being marked as failed."),
    ],
    "research_note.pdf": [
        ("title", "Battery Degradation Testing - Internal Research Note"),
        ("body", "This note summarizes results from a comparative cycle-life study of "
                 "two lithium-ion electrolyte formulations conducted by the materials "
                 "team over a 4-month testing window."),
        ("heading", "Test Methodology"),
        ("body", "Lithium-ion cells were cycled through 500 full charge-discharge "
                 "cycles at a controlled ambient temperature of 25 degrees Celsius, "
                 "using a constant-current constant-voltage charging profile at a "
                 "1C charge rate and 1C discharge rate."),
        ("body", "Each formulation was tested across a batch of 24 cells to reduce the "
                 "effect of manufacturing variance. Cells were rotated between test "
                 "chambers every 100 cycles to control for chamber-specific effects."),
        ("heading", "Capacity Retention"),
        ("body", "A cell was classified as end-of-life once its capacity fell below "
                 "an 80 percent retention threshold relative to its rated capacity, "
                 "consistent with the threshold used in the team's prior generation "
                 "of cycle-life studies."),
        ("heading", "Results"),
        ("body", "Cells using the new electrolyte formulation retained an average of "
                 "87 percent capacity after 500 cycles, compared to 74 percent for the "
                 "baseline formulation under identical test conditions."),
        ("body", "Internal resistance grew by 12 percent over the test window for the "
                 "new formulation, versus 26 percent for the baseline, suggesting "
                 "meaningfully slower degradation of the electrode-electrolyte "
                 "interface over time."),
        ("heading", "Thermal Behavior"),
        ("body", "No thermal runaway events were observed in either formulation across "
                 "the full test matrix. Peak surface temperature during 1C discharge "
                 "was 34 degrees Celsius for the new formulation and 38 degrees "
                 "Celsius for the baseline."),
        ("heading", "Next Steps"),
        ("body", "The team recommends proceeding to a larger-scale pilot of 200 cells "
                 "using the new formulation, with an extended test target of 1000 "
                 "cycles, before making a recommendation for production adoption."),
    ],
}


def build_pdf(filename: str, blocks: list[tuple[str, str]]) -> None:
    doc = fitz.open()
    page = doc.new_page()
    y = 72

    for kind, text in blocks:
        fontsize = 16 if kind == "title" else 13 if kind == "heading" else 11
        # naive word-wrap so longer paragraphs don't run off the page edge
        words = text.split()
        line = ""
        wrapped_lines = []
        for word in words:
            candidate = f"{line} {word}".strip()
            if len(candidate) > 95:
                wrapped_lines.append(line)
                line = word
            else:
                line = candidate
        if line:
            wrapped_lines.append(line)

        if kind == "heading":
            y += 8

        for wline in wrapped_lines:
            if y > 760:
                page = doc.new_page()
                y = 72
            page.insert_text((72, y), wline, fontsize=fontsize)
            y += 16 if kind == "body" else 20

        if kind != "title":
            y += 4

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAMPLE_DIR / filename
    page_count = len(doc)
    doc.save(str(out_path))
    doc.close()
    print(f"Wrote {out_path} ({page_count} page(s))")


def main() -> None:
    for filename, blocks in DOCUMENTS.items():
        build_pdf(filename, blocks)


if __name__ == "__main__":
    main()
