# ─────────────────────────────────────────────────────────────────────────────
# reports/report_generator.py
#
# Generates a professional PDF session report using ReportLab.
# The report includes:
#   - Cover page with title, date, and summary statistics
#   - All three analytics graphs
#   - Detailed event table (first 200 rows)
#   - Conclusions section
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def generate_pdf_report(
    event_log: List[Dict],
    entries: int,
    exits: int,
    peak_occupancy: int,
    graphs_dir: Path | str = "graphs",
    out_path: Path | str = "reports/session_report.pdf",
    title: str = "Occupancy Monitoring – Session Report",
    logo_path: Optional[Path] = None,
) -> Path:
    """
    Build and save the PDF report.

    Parameters
    ----------
    event_log      : Full list of event dicts from OccupancyCounter.
    entries        : Total entries for the session.
    exits          : Total exits for the session.
    peak_occupancy : Highest occupancy value observed.
    graphs_dir     : Directory containing the generated graph PNGs.
    out_path       : Destination PDF path.
    title          : Report title shown on the cover.
    logo_path      : Optional logo PNG; skipped if not found.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, mm
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
        HRFlowable, PageBreak,
    )

    out_path   = Path(out_path)
    graphs_dir = Path(graphs_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title=title,
        author="Occupancy Monitoring System v1.0",
    )

    styles = getSampleStyleSheet()

    # ── Custom styles ─────────────────────────────────────────────────────────
    cover_title  = ParagraphStyle("CoverTitle",  parent=styles["Title"],
                                  fontSize=24, spaceAfter=12, alignment=TA_CENTER,
                                  textColor=colors.HexColor("#1A73E8"))
    cover_sub    = ParagraphStyle("CoverSub",    parent=styles["Normal"],
                                  fontSize=12, spaceAfter=6, alignment=TA_CENTER,
                                  textColor=colors.HexColor("#555555"))
    section_hdr  = ParagraphStyle("SectionHdr",  parent=styles["Heading1"],
                                  fontSize=14, spaceBefore=14, spaceAfter=6,
                                  textColor=colors.HexColor("#1A73E8"),
                                  borderPad=4)
    body_text    = ParagraphStyle("BodyText",    parent=styles["Normal"],
                                  fontSize=10, leading=16, alignment=TA_JUSTIFY)
    caption_styl = ParagraphStyle("Caption",     parent=styles["Normal"],
                                  fontSize=9, alignment=TA_CENTER,
                                  textColor=colors.grey)

    story = []
    now   = datetime.now()

    # ── Cover page ─────────────────────────────────────────────────────────────
    # Logo (optional)
    if logo_path and Path(logo_path).exists():
        try:
            story.append(Image(str(logo_path), width=4*cm, height=4*cm))
            story.append(Spacer(1, 0.4*cm))
        except Exception:
            pass

    story.append(Spacer(1, 2*cm))
    story.append(Paragraph(title, cover_title))
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="100%", thickness=1.5,
                             color=colors.HexColor("#1A73E8")))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(f"Generated: {now.strftime('%A, %d %B %Y  %H:%M:%S')}",
                            cover_sub))
    story.append(Spacer(1, 1.5*cm))

    # Summary statistics box
    summary_data = [
        ["Metric",         "Value"],
        ["Total Entries",  str(entries)],
        ["Total Exits",    str(exits)],
        ["Peak Occupancy", str(peak_occupancy)],
        ["Events Logged",  str(len(event_log))],
        ["Report Date",    now.strftime("%Y-%m-%d")],
    ]
    summary_table = Table(summary_data, colWidths=[7*cm, 7*cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#1A73E8")),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0), 11),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#F0F4FF"), colors.white]),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 1), (-1, -1), 10),
        ("ROWHEIGHT",   (0, 0), (-1, -1), 20),
    ]))
    story.append(summary_table)
    story.append(PageBreak())

    # ── Section 1 – Analytics Graphs ─────────────────────────────────────────
    story.append(Paragraph("1. Analytics Graphs", section_hdr))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#CCCCCC")))
    story.append(Spacer(1, 0.3*cm))

    graph_specs = [
        ("entries_per_hour.png",    "Figure 1 – Entries per Hour"),
        ("exits_per_hour.png",      "Figure 2 – Exits per Hour"),
        ("occupancy_over_time.png", "Figure 3 – Occupancy Over Time"),
        ("dashboard.png",           "Figure 4 – Combined Analytics Dashboard"),
    ]

    for fname, caption in graph_specs:
        gpath = graphs_dir / fname
        if gpath.exists():
            try:
                story.append(Image(str(gpath), width=16*cm, height=8*cm,
                                   kind="proportional"))
                story.append(Paragraph(caption, caption_styl))
                story.append(Spacer(1, 0.6*cm))
            except Exception as exc:
                logger.warning("Could not embed graph %s: %s", fname, exc)
        else:
            story.append(Paragraph(
                f"<i>[Graph not available: {fname}]</i>", body_text))
            story.append(Spacer(1, 0.3*cm))

    story.append(PageBreak())

    # ── Section 2 – Event Log (first 200 rows) ────────────────────────────────
    story.append(Paragraph("2. Event Log (first 200 records)", section_hdr))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#CCCCCC")))
    story.append(Spacer(1, 0.3*cm))

    if event_log:
        table_data = [["#", "Timestamp", "Event", "Track ID", "Occupancy"]]
        for i, row in enumerate(event_log[:200], 1):
            table_data.append([
                str(i),
                str(row.get("timestamp", "")),
                str(row.get("event", "")),
                str(row.get("track_id", "")),
                str(row.get("occupancy", "")),
            ])

        col_w = [1.2*cm, 5.5*cm, 2.8*cm, 2.8*cm, 3*cm]
        ev_table = Table(table_data, colWidths=col_w, repeatRows=1)
        ev_table.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#1A73E8")),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 8),
            ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#F9F9F9"), colors.white]),
            ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#DDDDDD")),
            ("ROWHEIGHT",   (0, 0), (-1, -1), 14),
        ]))
        story.append(ev_table)
    else:
        story.append(Paragraph("No events were recorded in this session.",
                               body_text))

    story.append(PageBreak())

    # ── Section 3 – Conclusions ───────────────────────────────────────────────
    story.append(Paragraph("3. Conclusions", section_hdr))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#CCCCCC")))
    story.append(Spacer(1, 0.3*cm))

    net_flow = entries - exits
    if peak_occupancy > 0:
        conclusion_text = (
            f"The monitoring session recorded a total of <b>{entries}</b> entries and "
            f"<b>{exits}</b> exits, resulting in a net flow of "
            f"<b>{net_flow:+d}</b> people. "
            f"Peak occupancy reached <b>{peak_occupancy}</b> concurrent occupants. "
            f"A total of <b>{len(event_log)}</b> events were logged and saved to the "
            f"CSV file for further processing. "
            f"The system successfully tracked individuals across the virtual counting "
            f"line using YOLOv8 detection combined with Deep SORT re-identification, "
            f"providing reliable occupancy figures without manual counting. "
        )
    else:
        conclusion_text = (
            "No occupancy data was recorded in this session. "
            "Please ensure a video source with people crossing the counting line "
            "is provided, and that detection confidence thresholds are correctly "
            "configured in <i>config/settings.py</i>."
        )

    story.append(Paragraph(conclusion_text, body_text))
    story.append(Spacer(1, 0.5*cm))

    recommendations = [
        "• Adjust <i>LINE_POSITION_RATIO</i> in settings.py if the counting line "
          "does not align with the entrance.",
        "• Increase <i>YOLO_CONF_THRESH</i> to reduce false positives in crowded scenes.",
        "• Use a GPU-enabled device string (<i>\"cuda\"</i>) for real-time performance.",
        "• Review the event CSV log for anomalies and re-run analytics as needed.",
    ]
    for rec in recommendations:
        story.append(Paragraph(rec, body_text))

    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#1A73E8")))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "Real-Time Occupancy Monitoring System v1.0  –  Powered by YOLOv8 + Deep SORT",
        ParagraphStyle("Footer", parent=styles["Normal"],
                       fontSize=8, alignment=TA_CENTER,
                       textColor=colors.grey)
    ))

    # ── Build ─────────────────────────────────────────────────────────────────
    try:
        doc.build(story)
        logger.info("PDF report saved → %s", out_path)
    except Exception as exc:
        logger.exception("PDF build failed: %s", exc)
        raise

    return out_path