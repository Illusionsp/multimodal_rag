"""
generate_sample_pdf.py
-----------------------
Builds a synthetic "quarterly business report" PDF containing:
  - narrative text
  - a multi-level-header financial table (merged header cells)
  - a bar chart (quarterly revenue)
  - a pie chart (regional market share)

Why this exists: for a freelance portfolio, the win condition is a client
being able to see the system work in under a minute WITHOUT supplying their
own document. This script produces exactly the kind of document real
finance/ops clients pay to have parsed - so `sample_documents/` ships with
a ready-made demo PDF and this script if you want to regenerate/customize it.

Run: python scripts/generate_sample_pdf.py
"""
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "sample_documents" / "demo_quarterly_business_report.pdf"
TMP_DIR = Path(__file__).resolve().parent.parent / "data" / "images" / "_sample_charts"
TMP_DIR.mkdir(parents=True, exist_ok=True)


def make_bar_chart() -> str:
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    revenue = [82.4, 91.7, 104.3, 118.9]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(quarters, revenue, color="#2563eb")
    ax.set_title("Quarterly Revenue, FY2025 ($M)")
    ax.set_ylabel("Revenue ($M)")
    for b, v in zip(bars, revenue):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"${v}M", ha="center", fontsize=9)
    fig.tight_layout()
    path = TMP_DIR / "bar_chart.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def make_pie_chart() -> str:
    regions = ["North America", "EMEA", "APAC", "LATAM"]
    share = [45, 28, 19, 8]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(share, labels=regions, autopct="%1.0f%%", startangle=90,
           colors=["#2563eb", "#60a5fa", "#93c5fd", "#bfdbfe"])
    ax.set_title("FY2025 Revenue by Region")
    fig.tight_layout()
    path = TMP_DIR / "pie_chart.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def build_pdf():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(OUTPUT_PATH), pagesize=letter,
                             topMargin=0.7 * inch, bottomMargin=0.7 * inch)
    story = []

    story.append(Paragraph("Acme Holdings — FY2025 Quarterly Business Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "This report summarizes Acme Holdings' financial performance for fiscal year 2025. "
        "Total revenue grew 44% year-over-year, driven primarily by expansion in the North "
        "America and EMEA segments. Operating margin improved from 14.2% to 17.8% following "
        "the Q3 restructuring of the logistics division.",
        styles["BodyText"],
    ))
    story.append(Spacer(1, 18))

    story.append(Paragraph("Table 1: Regional Revenue Breakdown by Quarter ($M)", styles["Heading3"]))
    story.append(Spacer(1, 6))

    # Multi-level header table: top row spans 2 cols per region ("Revenue","Growth %"),
    # this is the merged-cell / nested-header structure the parser is built to handle.
    table_data = [
        ["Region", "Q1", "", "Q2", "", "Q3", "", "Q4", ""],
        ["", "Rev", "YoY%", "Rev", "YoY%", "Rev", "YoY%", "Rev", "YoY%"],
        ["North America", "38.1", "+22%", "41.9", "+25%", "47.8", "+31%", "53.2", "+29%"],
        ["EMEA", "21.6", "+18%", "24.2", "+20%", "27.1", "+24%", "31.5", "+27%"],
        ["APAC", "15.3", "+9%", "16.8", "+11%", "18.9", "+14%", "21.4", "+16%"],
        ["LATAM", "7.4", "+5%", "8.8", "+7%", "10.5", "+12%", "12.8", "+15%"],
        ["Total", "82.4", "+16%", "91.7", "+18%", "104.3", "+23%", "118.9", "+24%"],
    ]
    table = Table(table_data, colWidths=[1.1 * inch] + [0.55 * inch] * 8)
    table.setStyle(TableStyle([
        ("SPAN", (1, 0), (2, 0)), ("SPAN", (3, 0), (4, 0)),
        ("SPAN", (5, 0), (6, 0)), ("SPAN", (7, 0), (8, 0)),
        ("SPAN", (0, 0), (0, 1)),
        ("BACKGROUND", (0, 0), (-1, 1), colors.HexColor("#1e3a8a")),
        ("TEXTCOLOR", (0, 0), (-1, 1), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#dbeafe")),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
    ]))
    story.append(table)
    story.append(Spacer(1, 24))

    story.append(Paragraph("Figure 1: Quarterly Revenue Trend", styles["Heading3"]))
    story.append(Image(make_bar_chart(), width=5.5 * inch, height=3.2 * inch))
    story.append(Spacer(1, 18))

    story.append(Paragraph("Figure 2: Revenue Distribution by Region", styles["Heading3"]))
    story.append(Image(make_pie_chart(), width=4.5 * inch, height=3.6 * inch))
    story.append(Spacer(1, 18))

    story.append(Paragraph(
        "Outlook: management expects continued double-digit growth in North America and "
        "EMEA through FY2026, with APAC investment increasing following the Singapore "
        "office opening announced in Q4.",
        styles["BodyText"],
    ))

    doc.build(story)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
