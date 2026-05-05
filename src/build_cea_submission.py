#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
import textwrap
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

from build_plant_methods_submission_ready import metrics


ROOT = Path(r"D:\cornTrain\DINOV3")
OUT = ROOT / "cea_submission"
FIG_SRC = ROOT / "paper_figures"
FIG_OUT = OUT / "figures"
TABLE_OUT = OUT / "tables"
REPRO_OUT = OUT / "reproducibility_json"
DATA325_IMAGE_DIR = ROOT / "images"
DATA325_TTA8_JSON = ROOT / "data325_zero_shot_attn_aug_tta8" / "data325_zero_shot_comparison_attn_aug_tta8.json"
SOURCE_SAMPLE_DIR = ROOT / "hand_bbox_visualization" / "samples"

BLUE = "4472C4"      # matches common.py COLOR_SOURCE
GREEN = "1D9E75"     # matches common.py COLOR_TARGET
ORANGE = "D85A30"    # matches common.py COLOR_WARN
GRAY = "888780"      # matches common.py COLOR_NEUTRAL
BLACK = "202124"
HEADER = "EAF1F7"
LIGHT_GREEN = "E7F3EE"
LIGHT_ORANGE = "F8EEE9"

TITLE = (
    "Attention-guided DINOv3-DiffCorn-Fusion for zero-shot "
    "cross-greenhouse maize height estimation"
)
AUTHOR_NAME = "Hong Wu"
AFFILIATION = "China Agricultural University"
CORRESPONDING_EMAIL = "jchen@cau.edu.cn"
OPEN_RELEASE_ARCHIVE = "dinov3_dcf_maize_height_open_release_v0_1_0.zip"
REPOSITORY_URL = "https://github.com/sanyueuy/dinov3-dcf-maize-height"

FIGURES = [
    (
        "fig_real_protocol",
        "Figure_1_Real_image_protocol",
        "Fig. 1. Real-image protocol for zero-shot cross-greenhouse maize height estimation. (a) Source-domain hand-box examples with measured heights. (b) Independent DATA325 target greenhouse images with manual bounding-box annotations. (c) Extracted ROI crops and per-box evaluation records including ground-truth height, DINOv3-DCF prediction, absolute error, camera height, and TTA standard deviation.",
        6.15,
    ),
    (
        "fig3",
        "Figure_2_Dataset_statistics",
        "Fig. 2. Source and DATA325 dataset statistics. The source-domain hand-box set spans taller plants, whereas DATA325 contains more early-stage plants and a different target greenhouse distribution.",
        4.95,
    ),
    (
        "fig1",
        "Figure_3_DINOv3_DCF_workflow",
        "Fig. 3. DINOv3-DiffCorn-Fusion (DINOv3-DCF) workflow for ROI-level maize height estimation. A frozen DINOv3 ViT-L backbone extracts plant-region features, the DCF head maps visual and camera-height inputs to phytomer parameters, and height is obtained from internode-related outputs.",
        6.15,
    ),
    (
        "fig2",
        "Figure_4_Attention_weighted_pooling",
        "Fig. 4. Attention-weighted patch aggregation. Final-layer CLS-to-patch attention weights are averaged across heads, normalized across patch positions, and used to pool patch tokens before DCF regression.",
        6.15,
    ),
    (
        "fig4",
        "Figure_5_Domain_shift_tsne",
        "Fig. 5. DINOv3 feature-space domain shift between source ROIs and DATA325 target ROIs. The t-SNE visualization uses 20 sampled source and 20 sampled target ROI features.",
        4.95,
    ),
    (
        "fig5",
        "Figure_6_Ablation_results",
        "Fig. 6. Zero-shot DATA325 ablation results. Replacing CLS pooling with attention-weighted patch aggregation produced the largest single reduction in external MAE.",
        6.15,
    ),
    (
        "fig6",
        "Figure_7_Height_bin_errors",
        "Fig. 7. DATA325 error distribution by ground-truth plant-height bin for the best Attn+aug+TTA8 model. Early-stage plants below 80 cm remain the dominant source of relative error.",
        6.15,
    ),
    (
        "fig_real_stage_error_gallery",
        "Figure_8_DATA325_stage_error_gallery",
        "Fig. 8. DATA325 real-image stage and error gallery. Full target-greenhouse images with manual boxes and ROI crops show that early plants and cluttered backgrounds are visually overrepresented among the high-error cases.",
        6.15,
    ),
    (
        "fig7",
        "Figure_9_Diagnostic_experiments",
        "Fig. 9. Diagnostic experiments for alternative explanations of the domain gap, including geometric priors, feature-statistic alignment, and domain-adversarial training.",
        6.15,
    ),
    (
        "fig8",
        "Figure_10_Qualitative_DATA325_examples",
        "Fig. 10. Real DATA325 ROI examples and DINOv3 attention overlays. Low-stage examples have less plant-structure evidence and show larger relative errors than taller plants.",
        6.15,
    ),
]
FIG_NAME_BY_STEM = {source_stem: upload_stem for source_stem, upload_stem, _, _ in FIGURES}

REPRO_FILES = [
    ROOT / "data325_zero_shot_attn_aug_tta8" / "data325_zero_shot_comparison_attn_aug_tta8.json",
    ROOT / "data325_zero_shot_corrected_camheight" / "data325_zero_shot_comparison_corrected_camheight.json",
    ROOT / "data325_zero_shot_attn_aug" / "data325_zero_shot_comparison_attn_aug.json",
    ROOT / "data325_zero_shot_attn_aug_featurealign" / "data325_zero_shot_comparison_attn_aug_featurealign.json",
    ROOT / "data325_zero_shot_dann" / "data325_zero_shot_comparison_dann.json",
    ROOT / "tsne_source_vs_data325.json",
    ROOT / "data325_zero_shot_attn_aug_featurealign" / "data325_feature_stats_attn_aug_featurealign.json",
    ROOT / "bbox_geometry_prior_phaseA.json",
    ROOT / "attention_geometry_prior_phaseA.json",
    ROOT / "checkpoints" / "diffcorn_fusion_hand_dann_history.json",
]


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def fmt(value: float, ndigits: int = 2) -> str:
    return f"{value:.{ndigits}f}"


def setup_doc() -> Document:
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.15)
    sec.bottom_margin = Cm(2.15)
    sec.left_margin = Cm(2.0)
    sec.right_margin = Cm(2.0)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Arial"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
    normal.font.size = Pt(10.5)

    for name, size, color, bold in [
        ("Title", 16.5, BLUE, True),
        ("Subtitle", 10.5, GRAY, False),
        ("Heading 1", 13.5, BLUE, True),
        ("Heading 2", 11.5, GREEN, True),
        ("Heading 3", 10.5, BLACK, True),
    ]:
        style = styles[name]
        style.font.name = "Arial"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string(color)
        style.font.bold = bold

    if "CaptionClean" not in styles:
        caption = styles.add_style("CaptionClean", 1)
    else:
        caption = styles["CaptionClean"]
    caption.font.name = "Arial"
    caption._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
    caption.font.size = Pt(8.2)
    caption.font.italic = True
    caption.font.color.rgb = RGBColor.from_string(GRAY)
    return doc


def add_p(doc: Document, text: str = "", style: str | None = None) -> None:
    p = doc.add_paragraph(style=style)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.12
    p.add_run(text)


def add_caption(doc: Document, text: str) -> None:
    p = doc.add_paragraph(text, style="CaptionClean")
    p.paragraph_format.space_after = Pt(8)


def add_figure(doc: Document, stem: str, caption: str, width_in: float) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(FIG_OUT / f"{FIG_NAME_BY_STEM.get(stem, stem)}.png"), width=Inches(width_in))
    add_caption(doc, caption)


def add_named_figure(doc: Document, stem: str) -> None:
    for source_stem, _, caption, width_in in FIGURES:
        if source_stem == stem:
            add_figure(doc, stem, caption, width_in)
            return
    raise KeyError(f"Unknown figure stem: {stem}")


def shade_cell(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_width(cell, width_cm: float) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:type"), "dxa")
    tc_w.set(qn("w:w"), str(int(width_cm * 567)))


def set_table_borders(table, color: str = "C8D0D8", size: str = "4") -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = borders.find(qn(f"w:{edge}"))
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def fill_cell(cell, text: str, bold: bool = False, align: int = WD_ALIGN_PARAGRAPH.CENTER, size: float = 8.2) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    run = p.add_run(text)
    run.bold = bold
    run.font.name = "Arial"
    run.font.size = Pt(size)
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths_cm: list[float]) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    tbl_layout = table._tbl.tblPr.find(qn("w:tblLayout"))
    if tbl_layout is None:
        tbl_layout = OxmlElement("w:tblLayout")
        table._tbl.tblPr.append(tbl_layout)
    tbl_layout.set(qn("w:type"), "fixed")
    set_table_borders(table)

    for idx, text in enumerate(headers):
        cell = table.rows[0].cells[idx]
        shade_cell(cell, HEADER)
        set_cell_width(cell, widths_cm[idx])
        fill_cell(cell, text, bold=True, size=8.0)
    for row_idx, row_values in enumerate(rows, start=1):
        cells = table.add_row().cells
        for col_idx, text in enumerate(row_values):
            cell = cells[col_idx]
            set_cell_width(cell, widths_cm[col_idx])
            if row_idx % 2 == 0:
                shade_cell(cell, "F7F9FB")
            align = WD_ALIGN_PARAGRAPH.LEFT if col_idx in (0, 1, len(headers) - 1) else WD_ALIGN_PARAGRAPH.CENTER
            fill_cell(cell, text, align=align, size=7.8)
    doc.add_paragraph()


def pil_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: tuple[int, int, int], max_width: int, line_gap: int = 4) -> int:
    x, y = xy
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    line_h = int(font.size * 1.15) if hasattr(font, "size") else 16
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_h + line_gap
    return y


def paste_fit(canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int], bg: tuple[int, int, int] = (255, 255, 255)) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    target = Image.new("RGB", (x1 - x0, y1 - y0), bg)
    im = image.convert("RGB")
    im.thumbnail((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
    px = (target.width - im.width) // 2
    py = (target.height - im.height) // 2
    target.paste(im, (px, py))
    canvas.paste(target, (x0, y0))
    return (x0 + px, y0 + py, x0 + px + im.width, y0 + py + im.height)


def draw_bbox_on_image(path: Path, bbox: list[float], color: tuple[int, int, int] = (0, 158, 115)) -> Image.Image:
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    x, y, w, h = [int(round(v)) for v in bbox]
    width = max(5, image.width // 260)
    draw.rectangle((x, y, x + w, y + h), outline=color, width=width)
    return image


def crop_roi(path: Path, bbox: list[float], pad: int = 8) -> Image.Image:
    image = Image.open(path).convert("RGB")
    x, y, w, h = [int(round(v)) for v in bbox]
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(image.width, x + w + pad)
    y1 = min(image.height, y + h + pad)
    return image.crop((x0, y0, x1, y1))


def save_png_pdf(image: Image.Image, upload_stem: str) -> None:
    png_path = FIG_OUT / f"{upload_stem}.png"
    pdf_path = FIG_OUT / f"{upload_stem}.pdf"
    image.save(png_path)
    image.convert("RGB").save(pdf_path, "PDF", resolution=300.0)


def patch_embedded_figure_titles() -> None:
    fixes = [
        ("Figure_5_Domain_shift_tsne", "t-SNE visualization of source and target domain feature distributions", 82, 38),
    ]
    for upload_stem, title, cover_h, font_size in fixes:
        png_path = FIG_OUT / f"{upload_stem}.png"
        if not png_path.exists():
            continue
        image = Image.open(png_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, cover_h), fill=(255, 255, 255))
        font = pil_font(font_size, True)
        text_w = draw.textlength(title, font=font)
        draw.text(((image.width - text_w) / 2, max(6, (cover_h - font_size) // 2)), title, font=font, fill=(32, 33, 36))
        image.save(png_path)
        image.save(FIG_OUT / f"{upload_stem}.pdf", "PDF", resolution=300.0)


def load_data325_records() -> list[dict]:
    data = json.loads(DATA325_TTA8_JSON.read_text(encoding="utf-8"))
    records = data["model_results"][0]["box_results"]
    out = []
    seen = set()
    for record in records:
        image_path = DATA325_IMAGE_DIR / record["file_name"]
        key = (record["file_name"], record["box_id"])
        if image_path.exists() and key not in seen:
            seen.add(key)
            out.append(record)
    return out


def choose_record(records: list[dict], predicate, rank: str = "low") -> dict:
    subset = [r for r in records if predicate(r)]
    if not subset:
        raise RuntimeError("No DATA325 records matched the requested figure filter.")
    subset = sorted(subset, key=lambda r: r["abs_error_cm"])
    if rank == "high":
        return subset[-1]
    if rank == "mid":
        return subset[len(subset) // 2]
    return subset[0]


def source_sample(pattern: str) -> Path:
    matches = sorted(SOURCE_SAMPLE_DIR.glob(pattern))
    if not matches:
        raise RuntimeError(f"Missing source-domain sample for pattern {pattern}")
    return matches[0]


def build_real_image_protocol_figure(records: list[dict]) -> None:
    upload_stem = FIG_NAME_BY_STEM["fig_real_protocol"]
    canvas = Image.new("RGB", (2400, 1810), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = pil_font(52, True)
    head_font = pil_font(31, True)
    label_font = pil_font(26, True)
    body_font = pil_font(24)
    small_font = pil_font(21)
    tag_font = pil_font(23, True)
    blue = (68, 114, 196)     # #4472C4
    green = (29, 158, 117)    # #1D9E75
    orange = (216, 90, 48)    # #D85A30
    gray = (136, 135, 128)    # #888780
    black = (32, 33, 36)
    light_blue = (234, 241, 247)
    light_green = (231, 243, 238)
    light_orange = (248, 238, 233)

    # Title
    draw.text((70, 38), "Real-image zero-shot protocol for DINOv3-DCF", font=title_font, fill=black)
    draw.text((70, 105), "Source-domain hand boxes and independent DATA325 target greenhouse ROIs", font=body_font, fill=gray)

    # Column headers with (a)(b)(c) labels
    cols = [(70, 230, 730, 1545), (870, 230, 1530, 1545), (1670, 230, 2330, 1545)]
    headers = [
        ("(a)", "Source hand-box image", blue, light_blue),
        ("(b)", "DATA325 target image + bbox", green, light_green),
        ("(c)", "ROI crop + prediction record", orange, light_orange),
    ]
    for (x0, y0, x1, _), (tag, header, col_color, bg_color) in zip(cols, headers):
        draw.rectangle((x0, 175, x1, 225), fill=bg_color, outline=col_color, width=2)
        draw.text((x0 + 15, 184), tag, font=tag_font, fill=col_color)
        draw.text((x0 + 55, 184), header, font=head_font, fill=col_color)

    # Row definitions with height range labels
    rows = [
        ("Early-stage (0-80 cm)", "106-64-*_bbox.jpg", choose_record(records, lambda r: r["true_height_cm"] < 80, "low"), green),
        ("Mid-stage (80-120 cm)", "140-121-*_bbox.jpg", choose_record(records, lambda r: 80 <= r["true_height_cm"] < 120, "mid"), orange),
        ("Tall (120+ cm)", "140-270-*_bbox.jpg", choose_record(records, lambda r: r["true_height_cm"] >= 120, "low"), blue),
    ]
    y_starts = [255, 695, 1135]
    for row_idx, ((label, source_pattern, record, color), y0) in enumerate(zip(rows, y_starts)):
        y1 = y0 + 385
        # Row label on the left
        draw.text((72, y0 - 32), label, font=label_font, fill=color)

        # Column cell outlines
        for x0, _, x1, _ in cols:
            draw.rectangle((x0, y0, x1, y1), outline=(218, 224, 230), width=2)

        # (a) Source image
        source = Image.open(source_sample(source_pattern)).convert("RGB")
        paste_fit(canvas, source, (95, y0 + 25, 705, y1 - 58), (250, 250, 250))
        draw.text((105, y1 - 43), f"Source example ({source_pattern.split('-')[1]} cm class)", font=small_font, fill=gray)

        # Arrow from (a) to (b) - zero-shot transfer indicator
        arrow_y = y0 + 190
        draw.line((740, arrow_y, 860, arrow_y), fill=gray, width=3)
        draw.polygon([(855, arrow_y - 8), (870, arrow_y), (855, arrow_y + 8)], fill=gray)
        if row_idx == 1:
            draw.text((745, arrow_y - 28), "Zero-shot", font=small_font, fill=gray)

        # (b) Target image with bbox
        target_path = DATA325_IMAGE_DIR / record["file_name"]
        target = draw_bbox_on_image(target_path, record["bbox"], color)
        paste_fit(canvas, target, (895, y0 + 25, 1505, y1 - 58), (250, 250, 250))
        draw.text((905, y1 - 43), f"Manual bbox: {record['box_id']}", font=small_font, fill=gray)

        # Arrow from (b) to (c)
        draw.line((1515, arrow_y, 1635, arrow_y), fill=gray, width=3)
        draw.polygon([(1630, arrow_y - 8), (1645, arrow_y), (1630, arrow_y + 8)], fill=gray)

        # (c) ROI crop + prediction card
        roi = crop_roi(target_path, record["bbox"])
        paste_fit(canvas, roi, (1695, y0 + 25, 1965, y1 - 72), (250, 250, 250))
        card_x = 1990
        card_y = y0 + 28
        draw.rectangle((card_x, card_y, 2308, y1 - 68), fill=(247, 249, 251), outline=color, width=3)
        draw.text((card_x + 20, card_y + 18), "Evaluation record", font=label_font, fill=color)
        lines = [
            f"GT height: {record['true_height_cm']:.1f} cm",
            f"Prediction: {record['pred_height_cm']:.1f} cm",
            f"Abs. error: {record['abs_error_cm']:.1f} cm",
            f"Camera height: {record['camera_height_cm']:.0f} cm",
            f"TTA std.: {record['pred_std_cm']:.2f} cm",
        ]
        yy = card_y + 72
        for line in lines:
            draw.text((card_x + 20, yy), line, font=body_font, fill=black)
            yy += 45

    # Bottom workflow bar
    bar_y = 1620
    bar_h = 50
    bar_steps = [
        ("Source images", blue),
        ("Manual bbox", green),
        ("ROI crop", green),
        ("DINOv3-DCF", blue),
        ("Height prediction", orange),
    ]
    step_w = 400
    total_w = step_w * len(bar_steps) + 40 * (len(bar_steps) - 1)
    bar_x0 = (2400 - total_w) // 2
    for i, (step_label, step_color) in enumerate(bar_steps):
        sx = bar_x0 + i * (step_w + 40)
        draw.rectangle((sx, bar_y, sx + step_w, bar_y + bar_h), fill=step_color, outline=step_color, width=2)
        text_w = draw.textlength(step_label, font=label_font)
        draw.text((sx + (step_w - text_w) // 2, bar_y + 10), step_label, font=label_font, fill=(255, 255, 255))
        if i < len(bar_steps) - 1:
            ax = sx + step_w + 5
            draw.line((ax, bar_y + bar_h // 2, ax + 28, bar_y + bar_h // 2), fill=gray, width=3)
            draw.polygon([(ax + 23, bar_y + bar_h // 2 - 7), (ax + 33, bar_y + bar_h // 2), (ax + 23, bar_y + bar_h // 2 + 7)], fill=gray)

    draw.text((70, 1720), "All images are real captured greenhouse photographs; boxes, ROI crops, and prediction values are from the DATA325 evaluation JSON.", font=small_font, fill=gray)
    save_png_pdf(canvas, upload_stem)


def build_real_stage_error_gallery(records: list[dict]) -> None:
    upload_stem = FIG_NAME_BY_STEM["fig_real_stage_error_gallery"]
    canvas = Image.new("RGB", (2400, 2020), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = pil_font(50, True)
    head_font = pil_font(29, True)
    label_font = pil_font(24, True)
    small_font = pil_font(20)
    black = (32, 33, 36)
    gray = (136, 135, 128)    # #888780
    colors = [(29, 158, 117), (216, 147, 36), (216, 90, 48)]  # green, amber, orange

    draw.text((70, 35), "DATA325 stage and error gallery from real target-greenhouse images", font=title_font, fill=black)
    draw.text((70, 100), "Each cell shows a full image with manual bbox, the extracted ROI crop, and the Attn+aug+TTA8 height record.", font=small_font, fill=gray)

    col_labels = ["Low error", "Median error", "High error"]
    rank_names = ["low", "mid", "high"]
    bins = [
        ("0-80 cm early stage", lambda r: r["true_height_cm"] < 80),
        ("80-120 cm jointing stage", lambda r: 80 <= r["true_height_cm"] < 120),
        ("120+ cm tall stage", lambda r: r["true_height_cm"] >= 120),
    ]
    cell_w = 680
    cell_h = 520
    x0s = [260, 955, 1650]
    y0s = [245, 790, 1335]
    for x, label, color in zip(x0s, col_labels, colors):
        draw.text((x + 18, 190), label, font=head_font, fill=color)
    for row_idx, (bin_label, predicate) in enumerate(bins):
        y0 = y0s[row_idx]
        for col_idx, rank in enumerate(rank_names):
            x0 = x0s[col_idx]
            record = choose_record(records, predicate, rank)
            color = colors[col_idx]
            draw.rectangle((x0, y0, x0 + cell_w, y0 + cell_h), fill=(255, 255, 255), outline=(218, 224, 230), width=2)
            target_path = DATA325_IMAGE_DIR / record["file_name"]
            target = draw_bbox_on_image(target_path, record["bbox"], color)
            roi = crop_roi(target_path, record["bbox"])
            paste_fit(canvas, target, (x0 + 20, y0 + 24, x0 + 365, y0 + 315), (248, 249, 250))
            paste_fit(canvas, roi, (x0 + 390, y0 + 24, x0 + 660, y0 + 315), (248, 249, 250))
            draw.rectangle((x0 + 20, y0 + 340, x0 + 660, y0 + 492), fill=(247, 249, 251), outline=color, width=2)
            text = (
                f"GT {record['true_height_cm']:.1f} cm | Pred {record['pred_height_cm']:.1f} cm | "
                f"Err {record['abs_error_cm']:.1f} cm | TTA std. {record['pred_std_cm']:.2f} cm"
            )
            draw_wrapped(draw, (x0 + 42, y0 + 365), text, label_font, black, 580, 5)
            draw.text((x0 + 42, y0 + 447), f"{record['file_name']} / {record['box_id']}", font=small_font, fill=gray)
        draw_wrapped(draw, (58, y0 + 18), bin_label, head_font, (68, 114, 196), 170, 6)

    draw.text((70, 1945), "The gallery is deterministic: examples are selected by ground-truth height bin and absolute-error rank from the same DATA325 result file used for quantitative evaluation.", font=small_font, fill=gray)
    save_png_pdf(canvas, upload_stem)


def build_real_image_figures() -> None:
    records = load_data325_records()
    build_real_image_protocol_figure(records)
    build_real_stage_error_gallery(records)


def build_diagnostic_figure() -> None:
    bbox = json.loads((ROOT / "bbox_geometry_prior_phaseA.json").read_text(encoding="utf-8"))
    attn_aug = json.loads((ROOT / "data325_zero_shot_attn_aug" / "data325_zero_shot_comparison_attn_aug.json").read_text(encoding="utf-8"))
    feature_align = json.loads((ROOT / "data325_zero_shot_attn_aug_featurealign" / "data325_zero_shot_comparison_attn_aug_featurealign.json").read_text(encoding="utf-8"))
    dann = json.loads((ROOT / "checkpoints" / "diffcorn_fusion_hand_dann_history.json").read_text(encoding="utf-8"))

    gt = [s["true_height_cm"] for s in bbox["samples"]]
    est = [s["estimated_height_cm"] for s in bbox["samples"]]
    attn_mae = attn_aug["model_results"][0]["summary"]["mae_cm"]
    align_mae = feature_align["model_results"][0]["summary"]["mae_cm"]
    epochs = dann["history"]["epoch"]
    domain_acc = dann["history"]["domain_acc"]

    # Unified color palette matching common.py
    c_source = "#4472C4"
    c_target = "#1D9E75"
    c_warn = "#D85A30"
    c_neutral = "#888780"

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4), dpi=300)
    fig.patch.set_facecolor("white")

    # (a) Bbox geometric prior scatter
    ax = axes[0]
    ax.scatter(gt, est, s=40, color=c_neutral, alpha=0.70, edgecolor="white", linewidth=0.3)
    lo = min(min(gt), min(est)) - 5
    hi = max(max(gt), max(est)) + 5
    ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.2, label="Ideal (r=1.0)")
    coef = np.polyfit(gt, est, 1)
    xfit = [min(gt), max(gt)]
    yfit = [coef[0] * x + coef[1] for x in xfit]
    ax.plot(xfit, yfit, color=c_source, linewidth=2.0)
    ax.text(0.97, 0.13, f"r = {bbox['pearson_corr']:.3f}\nMAE = {bbox['mae_cm']:.2f} cm", transform=ax.transAxes, ha="right", va="bottom")
    ax.set_title("(a) Bbox geometric prior", weight="bold")
    ax.set_xlabel("Ground truth height (cm)")
    ax.set_ylabel("Geometric estimate (cm)")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, color="#D0D0D0", alpha=0.35)

    # (b) Feature alignment bar chart
    ax = axes[1]
    bars = ax.bar(["Attn+aug", "Feature-aligned"], [attn_mae, align_mae], color=[c_target, c_warn], width=0.56)
    ax.axhline(attn_mae, color=c_neutral, linestyle="--", linewidth=1.3)
    degradation = (align_mae - attn_mae) / attn_mae * 100.0
    ax.text(1, align_mae + 1.4, f"+{degradation:.1f}% degradation", color=c_warn, ha="center")
    ax.set_title("(b) Feature alignment fails", weight="bold")
    ax.set_ylabel("DATA325 MAE (cm)")
    ax.set_ylim(0, max(align_mae, attn_mae) + 10)
    ax.grid(True, axis="y", color="#D0D0D0", alpha=0.35)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"{bar.get_height():.2f}", ha="center", fontsize=9)

    # (c) DANN training curves
    ax = axes[2]
    ax.axhspan(0.60, 0.70, color="#F4A3A3", alpha=0.40)
    ax.plot(epochs, domain_acc, color=c_source, linewidth=2.2)
    ax.text(0.08, 0.65, "Target range\n60-70%", transform=ax.transAxes, color=c_warn)
    ax.text(0.95, 0.96, "~100%", transform=ax.transAxes, color=c_warn, ha="right", va="top")
    ax.set_title("(c) DANN instability", weight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Domain classifier accuracy")
    ax.set_ylim(0.45, 1.05)
    ax.grid(True, color="#D0D0D0", alpha=0.35)

    fig.tight_layout()
    upload_stem = FIG_NAME_BY_STEM["fig7"]
    fig.savefig(FIG_OUT / f"{upload_stem}.png", bbox_inches="tight", pad_inches=0.08, dpi=300)
    fig.savefig(FIG_OUT / f"{upload_stem}.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def copy_assets() -> None:
    reset_dir(OUT)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    TABLE_OUT.mkdir(parents=True, exist_ok=True)
    REPRO_OUT.mkdir(parents=True, exist_ok=True)

    for source_stem, upload_stem, _, _ in FIGURES:
        for ext in ("png", "pdf"):
            src = FIG_SRC / f"{source_stem}.{ext}"
            if src.exists():
                shutil.copy2(src, FIG_OUT / f"{upload_stem}.{ext}")

    for src in REPRO_FILES:
        if src.exists():
            shutil.copy2(src, REPRO_OUT / src.name)


def write_csvs(m: dict) -> None:
    with (TABLE_OUT / "table1_ablation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Variant", "Feature/intervention", "Source MAE (cm)", "DATA325 MAE (cm)", "DATA325 RMSE (cm)", "DATA325 MAPE (%)", "Note"])
        for row in m["ablation"]:
            writer.writerow(
                [
                    row["method"],
                    row["feature"],
                    fmt(row["source_mae"]),
                    fmt(row["data325_mae"]),
                    fmt(row["data325_rmse"]),
                    fmt(row["data325_mape"]),
                    row["note"],
                ]
            )
    with (TABLE_OUT / "table2_height_bins.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Height bin (cm)", "n", "MAE (cm)", "MAPE (%)"])
        for row in m["height_bins"]:
            writer.writerow([row["bucket"], row["n"], fmt(row["mae"]), fmt(row["mape"])])


def build_graphical_abstract(m: dict) -> Path:
    def c(hex_color: str) -> str:
        return f"#{hex_color}"

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig = plt.figure(figsize=(13.28, 5.31), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    boxes = [
        (0.04, 0.56, 0.19, 0.28, "Source greenhouse", "156 hand-box ROIs\n64-270 cm"),
        (0.29, 0.56, 0.19, 0.28, "Frozen DINOv3", "ViT-L features\n224 x 224 ROI"),
        (0.54, 0.56, 0.19, 0.28, "Attention pooling", "CLS-to-patch weights\n1024D descriptor"),
        (0.79, 0.56, 0.17, 0.28, "DCF head", "camera height + features\nplant height"),
    ]
    for idx, (x, y, w, h, title, body) in enumerate(boxes):
        color = c([BLUE, GREEN, ORANGE, BLUE][idx])
        ax.add_patch(plt.Rectangle((x, y), w, h, fc="#F7F9FB", ec=color, lw=2))
        ax.text(x + w / 2, y + h - 0.075, title, ha="center", va="center", weight="bold", color=color, fontsize=13)
        ax.text(x + w / 2, y + 0.105, body, ha="center", va="center", color=c(BLACK), fontsize=11, linespacing=1.35)
        if idx < len(boxes) - 1:
            ax.annotate("", xy=(x + w + 0.045, y + h / 2), xytext=(x + w + 0.01, y + h / 2), arrowprops=dict(arrowstyle="->", lw=2, color=c(GRAY)))

    ax.text(0.04, 0.91, "Zero-shot cross-greenhouse maize height estimation", fontsize=21, weight="bold", color=c(BLACK))
    ax.text(0.04, 0.09, "Non-AI graphical abstract generated from code, real evaluation metrics, and schematic vector elements.", fontsize=9.5, color=c(GRAY))

    # Compact result panel
    ax.add_patch(plt.Rectangle((0.05, 0.22), 0.42, 0.22, fc=c(LIGHT_GREEN), ec=c(GREEN), lw=1.5))
    ax.text(0.07, 0.39, "Main gain", weight="bold", color=c(GREEN), fontsize=13)
    ax.text(0.07, 0.31, "CLS baseline: 41.75 cm MAE", color=c(BLACK), fontsize=12)
    ax.text(0.07, 0.25, "Attention pooling: 30.37 cm MAE", color=c(BLACK), fontsize=12)
    ax.text(0.28, 0.31, "Best Attn+aug+TTA8: 29.57 cm MAE", color=c(BLACK), fontsize=12)

    ax.add_patch(plt.Rectangle((0.53, 0.22), 0.42, 0.22, fc=c(LIGHT_ORANGE), ec=c(ORANGE), lw=1.5))
    ax.text(0.55, 0.39, "Remaining bottleneck", weight="bold", color=c(ORANGE), fontsize=13)
    ax.text(0.55, 0.31, f"Feature-domain centroid distance: {fmt(m['centroid_distance'])}", color=c(BLACK), fontsize=12)
    ax.text(0.55, 0.25, "Plants below 80 cm: 55.11% MAPE", color=c(BLACK), fontsize=12)

    out_png = OUT / "graphical_abstract_non_ai.png"
    out_pdf = OUT / "graphical_abstract_non_ai.pdf"
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return out_png


def add_title_page(doc: Document) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(TITLE)
    run.bold = True
    run.font.size = Pt(17)
    run.font.color.rgb = RGBColor.from_string(BLUE)
    add_p(doc, "Article type: Original research paper", "Subtitle")
    add_p(doc, "Target journal: Computers and Electronics in Agriculture")
    add_p(doc, f"Author: {AUTHOR_NAME}")
    add_p(doc, f"Affiliation: {AFFILIATION}")
    add_p(doc, f"Corresponding author: {AUTHOR_NAME} ({CORRESPONDING_EMAIL})")
    add_p(doc, "Highlights file: included separately. Graphical abstract: optional non-AI code-generated file included.")
    add_p(doc, "Figure files: 10. Editable tables: 2. Supplementary material: 1.")
    add_p(doc, f"Code and data repository: {REPOSITORY_URL}")
    add_p(doc, "Optional archive DOI, optional ORCID ID, and any required CEA submission-system fields must be completed before online submission.")
    doc.add_page_break()


def add_ablation_table(doc: Document, m: dict) -> None:
    rows = []
    note_map = {
        "": "",
        "best": "best tested",
        "failed": "diagnostic negative",
    }
    for row in m["ablation"]:
        rows.append(
            [
                row["method"],
                row["feature"],
                fmt(row["source_mae"]),
                fmt(row["data325_mae"]),
                fmt(row["data325_mape"]),
                note_map.get(row["note"], row["note"]),
            ]
        )
    add_caption(doc, "Table 1. Ablation study under the DATA325 zero-shot cross-greenhouse protocol.")
    add_table(
        doc,
        ["Variant", "Feature/intervention", "Source MAE", "DATA325 MAE", "DATA325 MAPE", "Interpretation"],
        rows,
        [2.25, 4.15, 1.7, 1.85, 1.95, 3.2],
    )


def add_height_bin_table(doc: Document, m: dict) -> None:
    rows = [[row["bucket"], str(row["n"]), fmt(row["mae"]), fmt(row["mape"])] for row in m["height_bins"]]
    add_caption(doc, "Table 2. Height-bin error profile for the best Attn+aug+TTA8 model on DATA325.")
    add_table(doc, ["Height bin (cm)", "n", "MAE (cm)", "MAPE (%)"], rows, [4.0, 2.0, 3.0, 3.0])


def build_manuscript(m: dict) -> Path:
    doc = setup_doc()
    add_title_page(doc)

    doc.add_heading("Abstract", level=1)
    abstract = (
        "Real greenhouse images contain maize plants together with pots, benches, labels, irrigation hardware, shadows, and facility-specific backgrounds. "
        "These factors make cross-greenhouse plant-height estimation difficult when a model is trained in one imaging environment and deployed in another. "
        "We evaluated DINOv3-DiffCorn-Fusion (DINOv3-DCF), an ROI-level maize height pipeline that connects frozen DINOv3 visual features to a DiffCorn-Fusion regression head, under a strict zero-shot protocol in which the independent DATA325 greenhouse was not used for training, tuning, or model selection. "
        "The study emphasizes real captured images, manual ROI annotation, crop extraction, attention-based feature pooling, and external error analysis. "
        "Replacing CLS-token pooling with attention-weighted patch aggregation reduced DATA325 MAE from 41.75 to 30.37 cm. "
        "Adding visual augmentation and eight-sample test-time augmentation gave the best tested result, with 29.57 cm MAE, 38.98 cm RMSE, and 36.14% MAPE. "
        "Feature analysis confirmed domain separation, with a t-SNE centroid distance of 36.89 and a 36.72% relative mean shift. "
        "Residual error concentrated in plants below 80 cm, where MAPE reached 55.11%. "
        "Per-image camera-height correction, simple geometric priors, feature-statistic alignment, and the tested DANN configuration did not close the gap. "
        "These results show that attention-guided foundation-model feature aggregation improves zero-shot maize height transfer, while early-stage plants require stage-aware and plant-focused adaptation."
    )
    add_p(doc, abstract)
    add_p(doc, "Keywords: maize phenotyping; plant height; greenhouse imaging; DINOv3; foundation model; zero-shot transfer; domain shift; attention pooling; DiffCorn-Fusion")

    doc.add_heading("1. Introduction", level=1)
    for para in [
        "High-throughput crop phenotyping is a core enabling technology for breeding, controlled-environment experimentation, and digital crop management. Plant height is frequently measured because it reflects developmental stage, biomass accumulation, lodging risk, irrigation response, and management status. Image-based crop-height systems have been developed with UAS structure-from-motion products, crop height models, stereo vision, and protected-facility machine vision, showing that automated height measurement can reduce manual labor while preserving biologically useful growth information (Chang et al., 2017; Xie et al., 2021; Kim et al., 2021; Jayasuriya et al., 2024).",
        "Agricultural computer vision has also moved from hand-engineered descriptors toward deep learning, object detection, and phenotyping-specific visual pipelines (Patricio and Rieder, 2018; Kamilaris and Prenafeta-Boldu, 2018; Li et al., 2020; Ariza-Sentis et al., 2024). For maize and other crops, vision systems have been used for crop counting, crop evaluation, growth-stage estimation, and decision-support inputs (Li et al., 2019; Veramendi and Cruvinel, 2024; Che et al., 2024). These studies demonstrate the value of visual phenotyping, but they also show that imaging geometry, field or greenhouse environment, and data-processing assumptions strongly affect downstream trait estimates.",
        "Cross-domain generalization is therefore a persistent obstacle in agricultural computer vision. Multi-environment phenotyping studies explicitly account for spatial heterogeneity and environmental variation, while broader machine-learning benchmarks show that distribution shift can cause large performance drops when training and deployment data differ (Che et al., 2024; Koh et al., 2021; Gulrajani and Lopez-Paz, 2021). Greenhouse images contain plant tissue together with pots, benches, labels, substrate, shadows, and equipment. These context cues vary between facilities and can be strongly encoded by deep representations even when they are unrelated to plant height.",
        "ROI quality and plant-structure representation are additional bottlenecks. Precision-farming detection systems must separate crop targets from cluttered agricultural scenes, and robot-localization studies report failures caused by lighting variation, complex backgrounds, and small target regions (Ariza-Sentis et al., 2024; Xing et al., 2023). Segmentation foundation models such as Segment Anything provide a promising route to plant-mask normalization, but they do not by themselves solve trait regression or cross-greenhouse representation shift (Kirillov et al., 2023). Three-dimensional and hyperspectral phenotyping reviews and datasets further emphasize that structural traits require high-quality annotated data and explicit geometric or part-level reasoning (Liu et al., 2020; Reena et al., 2025).",
        "Recent self-supervised vision transformers provide robust general image features and have become attractive backbones for phenotyping pipelines (Dosovitskiy et al., 2021; Caron et al., 2021; Oquab et al., 2024; Simeoni et al., 2025). However, the standard CLS token is a global summary representation. In a plant ROI, it can integrate background and acquisition context together with plant structure. For plant-height estimation, patch-level aggregation may be more appropriate because plant tissue occupies only part of the crop, especially at early growth stages.",
        "This study evaluates DINOv3-DiffCorn-Fusion (DINOv3-DCF), a pipeline that connects frozen DINOv3 features to a DCF head that predicts a phytomer-parameter representation and plant height. The central question is whether attention-weighted patch aggregation improves zero-shot cross-greenhouse maize height estimation compared with CLS pooling and patch-mean pooling. We also test several plausible explanations for the remaining error, including camera-height metadata, simple geometry, feature-statistic alignment, and domain-adversarial learning (Sun and Saenko, 2016; Ganin et al., 2016).",
        "The contribution is an original cross-greenhouse evaluation of foundation-model feature aggregation for maize height phenotyping. Unlike UAS/SfM, stereo-vision, or protected-facility geometry pipelines, this work asks whether frozen foundation-model features can transfer across greenhouse domains when only source-domain labels are available. We report both successful and unsuccessful interventions because negative diagnostic results are important for identifying the actual bottleneck. The CEA-oriented interpretation is not that the method fully solves external maize height estimation, but that attention-weighted pooling is a strong representation improvement and that early-stage plants remain the key domain-transfer target.",
    ]:
        add_p(doc, para)

    doc.add_heading("2. Materials and methods", level=1)
    doc.add_heading("2.1. Study design and datasets", level=2)
    for para in [
        "Model development used a source-domain hand-bounding-box dataset and external testing used the independent DATA325 target greenhouse. Target-domain annotations were restricted to final evaluation and diagnostic visualization. They were not used for training, hyperparameter tuning, model selection, or early stopping. The reported DATA325 results therefore represent zero-shot cross-greenhouse transfer.",
        "All compared DATA325 evaluations used the same 82 manually annotated plant boxes, the same plant-height labels, and the same image files. Manual ROIs were used because object detection, tracking, and segmentation can introduce their own errors under complex agricultural backgrounds (Ariza-Sentis et al., 2024; Xing et al., 2023; Kirillov et al., 2023). This design isolates the height-regression and representation-transfer problem from automatic detector quality. Per-image camera height was resolved from the capture-height mapping file rather than from a single global default. The correction changed DATA325 MAE by only about 0.03 cm, indicating that camera-height metadata mismatch was not the main source of error.",
        "The source-domain hand-box dataset contains 156 samples. Plant heights range from 64.00 to 270.00 cm, with mean 140.50 cm and standard deviation 66.30 cm. Source camera heights include 106 and 140 cm. DATA325 contains 75 original images; 25 completed images currently provide 82 evaluated boxes. DATA325 height labels range from 30.00 to 178.00 cm, with median 97.00 cm and mean 102.79 cm. DATA325 contains a larger fraction of plants below 80 cm than the source set.",
    ]:
        add_p(doc, para)
    add_p(doc, "Figure 1 makes the evaluation protocol auditable by showing source-domain hand-box images, independent DATA325 target images with manual boxes, the extracted ROI crops, and the exact height-prediction records used in the external evaluation.")
    add_named_figure(doc, "fig_real_protocol")
    add_named_figure(doc, "fig3")

    doc.add_heading("2.2. DINOv3-DCF architecture", level=2)
    for para in [
        "The pipeline receives a plant image and a region of interest (ROI). DATA325 ROIs are manually drawn in the current benchmark to isolate the height-regression problem from detector errors. In a deployed greenhouse system, the same ROI-level pipeline can be connected to an automatic detector or segmentation model, but detector quality is outside the scope of this controlled zero-shot evaluation.",
        "Each ROI is resized to 224 x 224 pixels and passed through a frozen DINOv3 ViT-L backbone. The visual output has a 1024-dimensional hidden representation. The DCF head concatenates this visual descriptor with one scalar camera-height input, forming a 1025-dimensional input vector. The head outputs a 64-dimensional phytomer representation, interpreted as 16 phytomers with four parameters each. Plant height is computed from internode-length-related components of this representation.",
        "The DINOv3 backbone remains frozen in all main experiments. This isolates the contribution of feature aggregation and DCF training, reduces computational cost, and improves reproducibility. The DCF head is the trainable module for height estimation, except in the DANN experiment where an additional domain classifier is trained for adversarial domain discrimination.",
    ]:
        add_p(doc, para)
    add_named_figure(doc, "fig1")

    doc.add_heading("2.3. Feature aggregation modes", level=2)
    for para in [
        "Three feature aggregation modes were evaluated. The CLS baseline uses the first token of the final hidden state. Patch-mean aggregation averages all patch tokens and removes the explicit CLS bottleneck. Attention-weighted aggregation uses the final-layer CLS-to-patch attention map: attention weights from the CLS token to patch tokens are averaged across heads, normalized over patches, and used to compute a weighted sum of final-layer patch tokens.",
        "Attention-weighted aggregation keeps the feature dimension unchanged at 1024, so the DCF input interface remains compatible with the original 1025-dimensional design. The change is localized to feature extraction. This experimental control means that differences between CLS, patch-mean, and attention-weighted models can be attributed primarily to feature aggregation rather than to a redesigned regressor.",
        "The biological motivation is that a maize ROI contains plant tissue as well as pot edges, bench structures, substrate, labels, and background objects. The CLS token can summarize all of these signals, while attention-weighted patch aggregation reduces the contribution of less salient patches. It is not a segmentation mask, but it is a lightweight mechanism for suppressing background-dominated regions without training a new visual encoder.",
    ]:
        add_p(doc, para)
    add_named_figure(doc, "fig2")

    doc.add_heading("2.4. Training variants and inference", level=2)
    for para in [
        "The old baseline used the original CLS-only feature bundle and the original DCF checkpoint. To separate representation effects from training artifacts, CLS retrain, patch-mean retrain, and attention-weighted retrain models were built from newly extracted bundles using the same sample identities, bounding boxes, camera heights, and height labels. Only the visual feature values changed across these three bundles.",
        "The attention-augmentation model used attention-weighted features and training-time visual perturbations. The augmentation strategy combined background and color-domain perturbations to reduce sensitivity to greenhouse-specific appearance. Background augmentation perturbed peripheral ROI regions, while color augmentation sampled brightness, contrast, saturation, and hue changes. The DINOv3 backbone remained frozen.",
        "Test-time augmentation (TTA8) was applied only at inference. For each ROI, one prediction used the original crop and seven predictions used independent color perturbations. The reported prediction is the mean of the eight outputs. Camera height was held constant across the TTA samples.",
    ]:
        add_p(doc, para)

    doc.add_heading("2.5. Domain-shift diagnostics", level=2)
    for para in [
        "We tested several alternative explanations for the external error. First, DATA325 camera-height inputs were corrected using per-image capture-height mapping. Second, two geometry routes were evaluated: attention-derived top-bottom localization and the ratio of bounding-box pixel height to image height multiplied by camera height. Third, source and target feature statistics were computed for attention-weighted features, and target features were aligned to source mean and standard deviation by per-dimension standardization, following the general motivation of correlation-alignment methods (Sun and Saenko, 2016). Fourth, a DANN-style domain-adversarial model was trained with source labels and unlabeled target images (Ganin et al., 2016).",
        "These experiments were designed as diagnostic controls rather than as exhaustive optimization of each adaptation family. They identify whether the observed error is primarily metadata-driven, geometry-driven, marginal feature-distribution-driven, or reducible through the tested adversarial setup. This framing follows the broader distribution-shift literature, where source-domain accuracy alone is not sufficient evidence of deployment robustness (Koh et al., 2021; Gulrajani and Lopez-Paz, 2021).",
    ]:
        add_p(doc, para)

    doc.add_heading("2.6. Evaluation metrics", level=2)
    add_p(doc, "Evaluation uses mean absolute error (MAE), root mean squared error (RMSE), and mean absolute percentage error (MAPE). MAPE is computed at the sample level as |prediction - ground truth| / ground truth x 100% and then averaged. DATA325 results are grouped into height bins of 0-80, 80-100, 100-120, 120-140, 140-160, and 160-180 cm to locate stage-dependent error.")

    doc.add_heading("3. Results", level=1)
    doc.add_heading("3.1. DATA325 differs from the source domain in height distribution and feature space", level=2)
    add_p(doc, "DATA325 differs from the source domain in both greenhouse appearance and plant-stage composition. The target set has a larger fraction of plants below 80 cm, whereas the source-domain training set extends to taller plants. This creates a difficult evaluation condition because the model must transfer across both visual background and stage distribution.")
    add_p(doc, f"Feature analysis confirmed a measurable domain gap. A t-SNE visualization using 20 source and 20 DATA325 ROI features showed a centroid distance of {fmt(m['centroid_distance'])}. In high-dimensional feature statistics, the relative mean shift was {fmt(m['mean_shift'])}% and the relative standard-deviation shift was {fmt(m['std_shift'])}%. These results support the interpretation that source and target domains are separated in DINOv3 feature space.")
    add_named_figure(doc, "fig4")

    doc.add_heading("3.2. Attention-weighted aggregation is the most effective single change", level=2)
    add_p(doc, "The original CLS baseline produced 41.75 cm MAE on DATA325. Retraining a CLS model did not improve external performance; it increased MAE to 45.85 cm. Patch-mean aggregation reduced MAE to 33.44 cm, indicating that moving away from the CLS token is beneficial. Attention-weighted patch aggregation produced the largest single improvement, reducing MAE to 30.37 cm while maintaining low source-domain MAE.")
    add_p(doc, "Visual augmentation and TTA8 provided additional but smaller gains. Attention plus augmentation reached 29.89 cm MAE, and attention plus augmentation plus TTA8 reached 29.57 cm MAE, 38.98 cm RMSE, and 36.14% MAPE. The improvement pattern shows a strong initial gain from representation change followed by diminishing returns from augmentation and test-time averaging.")
    add_named_figure(doc, "fig5")
    add_ablation_table(doc, m)

    doc.add_heading("3.3. Remaining error concentrates in early-stage plants", level=2)
    add_p(doc, "Errors were not uniformly distributed across height ranges. In the 0-80 cm bin, MAPE was 55.11%, much higher than in taller bins. For the 120-140, 140-160, and 160-180 cm bins, relative errors were lower, suggesting that larger plants provide more stable ROI structure and more reliable attention patterns.")
    add_p(doc, "This pattern is consistent with visual inspection. Low-stage ROIs often contain a high fraction of pot, substrate, bench, labels, and background relative to plant tissue. Even when the attention map highlights plant-like areas, the representation contains less structural evidence for height. Taller plants fill more of the ROI, making the visual descriptor more stable under cross-domain changes.")
    add_named_figure(doc, "fig6")
    add_height_bin_table(doc, m)
    add_p(doc, "To connect the quantitative height-bin result to the actual imaging conditions, Figure 8 groups real DATA325 photographs by growth stage and absolute-error rank. The high-error examples show that early plants and cluttered backgrounds are not abstract statistical labels; they are visible target-domain image conditions.")
    add_named_figure(doc, "fig_real_stage_error_gallery")

    doc.add_heading("3.4. Diagnostic experiments reject several simple explanations", level=2)
    add_p(doc, f"The geometry experiments did not meet the threshold for a usable prior. The bounding-box-ratio prior reached Pearson r={fmt(m['bbox_r'], 3)} and MAE {fmt(m['bbox_mae'])} cm. Attention-map thresholding and vertical projection were weaker, with r={fmt(m['attn_threshold_r'], 3)} and r={fmt(m['attn_projection_r'], 3)}, respectively. These results show that the available camera-height definition and ROI geometry do not form a closed physical height model.")
    add_p(doc, "Feature-statistic alignment also failed. Although the source and target domains had measurable mean and variance shifts, mapping target features to source mean and standard deviation worsened DATA325 MAE to 45.98 cm. This indicates that the domain difference is structural rather than a purely marginal channel-wise shift.")
    add_p(doc, f"The tested DANN configuration did not improve target-domain performance. Its source-domain MAE remained acceptable at 2.45 cm, but DATA325 MAE was 34.20 cm. The domain classifier converged to approximately {fmt(m['dann_final_acc'] * 100, 0)}% accuracy, meaning source and target representations remained easily separable. In this setting, adversarial training did not produce the intended domain confusion.")
    add_named_figure(doc, "fig7")

    doc.add_heading("3.5. Qualitative DATA325 evidence supports the stage-dependent error pattern", level=2)
    add_p(doc, "Real DATA325 examples show the same pattern as the height-bin analysis. Low-stage plants often have sparse plant pixels and dispersed attention, and their prediction errors can exceed 40 cm. High-stage examples fill more of the ROI and show more concentrated attention, with lower relative errors. The qualitative evidence does not replace quantitative evaluation, but it explains why the 0-80 cm group remains the main bottleneck.")
    add_named_figure(doc, "fig8")

    doc.add_heading("4. Discussion", level=1)
    for para in [
        "The main result is that feature aggregation in a frozen foundation-model backbone strongly affects cross-greenhouse phenotyping transfer. CLS-only features are convenient, but they are a global image summary and can encode greenhouse context. Attention-weighted patch aggregation is a small post-processing change, yet it produced the largest reduction in DATA325 MAE. This supports the view that plant-focused pooling is more important than simply retraining the same CLS-based regressor.",
        "From a Computers and Electronics in Agriculture perspective, the result is useful because it identifies a practical representation-level intervention for agricultural computer vision systems that must operate across facilities. Earlier CEA crop-height studies often recover height through UAS/SfM crop-height models, stereo geometry, or facility-specific machine vision (Chang et al., 2017; Xie et al., 2021; Kim et al., 2021; Jayasuriya et al., 2024). The present work does not try to replace those geometry-rich systems. Instead, it tests whether frozen foundation-model features can support zero-shot transfer from one greenhouse to another when only ROI images and camera-height metadata are available.",
        "This distinction is important for deployment. Geometry-based systems can be accurate when the imaging platform, terrain or bench reference, and calibration assumptions are well controlled. ROI-level greenhouse images are cheaper to process, but they expose the model to background, pot, and growth-stage shifts. The current results show that attention-weighted patch pooling reduces this representation problem, but they also show that frozen features alone do not fully recover a physically stable height signal in early-stage plants.",
        "The DATA325 benchmark is small, but it has a specific diagnostic role. Multi-environment phenotyping work has shown that environmental and spatial variation can distort growth-stage estimates and cultivar comparisons (Che et al., 2024). Similarly, the present target set reveals a cross-greenhouse failure mode that source-domain validation would miss. The value of the dataset is therefore not only its sample size, but its ability to expose a realistic deployment shift with fixed images, fixed manual ROIs, and reproducible evaluation JSON files.",
        "The current architecture nevertheless has a clear performance ceiling on DATA325. Even the best variant remains near 30 cm MAE, which is insufficient for precise agronomic measurement in early growth stages. The main contribution is therefore a controlled cross-domain benchmark and an empirically validated representation improvement, not a claim of solved greenhouse height estimation.",
        "The diagnostic experiments clarify the method boundary. Camera-height correction ruled out a metadata-driven explanation. Attention and bounding-box geometry ruled out a simple height-ratio solution. Feature-statistic alignment ruled out a purely marginal distribution shift. The tested DANN result showed that this adversarial setup was insufficient to align domains. Together, these findings support a structural, stage-dependent interpretation of the remaining shift.",
        "Recent CEA work on multi-environment crop monitoring provides useful context. Che et al. (2024) showed that UAV-based soybean growth-stage estimation requires explicit spatial-heterogeneity and climate corrections across environments. Jayasuriya et al. (2024) reported that protected-facility height estimation is sensitive to camera angle, plant architecture, and background complexity. The present results are consistent with these findings: cross-greenhouse transfer is not only a sensor-calibration problem but also a representation problem that depends on how visual features aggregate plant and background information.",
        "Future work should focus on the low-stage regime, where the current best model still reaches 55.11% MAPE. A concrete target is reducing 0-80 cm MAPE below 30% while maintaining performance in taller bins. Promising directions include stage-aware models, segmentation-guided ROI normalization, explicit plant-mask pooling, target-domain self-supervised adaptation, and multi-greenhouse pretraining. Segment Anything-style masks and precision-farming detection pipelines could help reduce pot and background contamination before feature extraction (Kirillov et al., 2023; Ariza-Sentis et al., 2024). Plant-mask pooling and 3D structural priors should also be evaluated, because plant phenotyping studies increasingly emphasize part-level annotation and structural data for reliable trait extraction (Liu et al., 2020; Reena et al., 2025). A better DANN design may also be possible, but the present result suggests that source ROI features and target images must be matched more carefully before adversarial alignment can be useful.",
    ]:
        add_p(doc, para)

    doc.add_heading("4.1. Limitations", level=2)
    for para in [
        "The DATA325 evaluation set contains 82 annotated boxes from 25 completed images. This is sufficient to reveal a strong cross-domain failure mode, but larger target-domain collections are needed to estimate performance across cultivars, developmental stages, camera configurations, and greenhouse layouts.",
        "The current benchmark evaluates manually drawn DATA325 boxes to isolate DCF regression from detector errors. A complete greenhouse deployment will require robust automatic ROI detection and quality control.",
        "DINOv3 remains frozen. This improves reproducibility and controls the experiment, but it limits the ability to learn plant-specific invariances. Future studies should compare frozen features with carefully regularized fine-tuning or adapter-based tuning across multiple greenhouses.",
    ]:
        add_p(doc, para)

    doc.add_heading("5. Conclusions", level=1)
    add_p(doc, "We evaluated DINOv3-DCF for maize plant-height phenotyping under a strict zero-shot cross-greenhouse protocol. Attention-weighted patch aggregation reduced DATA325 MAE from 41.75 to 30.37 cm, and the best Attn+aug+TTA8 variant reached 29.57 cm. The remaining error is dominated by plants below 80 cm. Camera-height correction, simple geometry, feature-statistic alignment, and the tested DANN configuration did not resolve the gap. These results define a practical representation improvement and a clear next target: stage-aware, plant-focused adaptation for early maize growth.")

    doc.add_heading("CRediT authorship contribution statement", level=1)
    add_p(doc, f"{AUTHOR_NAME}: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing - original draft, Writing - review & editing, and Visualization.")
    doc.add_heading("Declaration of competing interest", level=1)
    add_p(doc, "The author declares no competing interests.")
    doc.add_heading("Funding", level=1)
    add_p(doc, "No external funding was reported for this work.")
    doc.add_heading("Acknowledgements", level=1)
    add_p(doc, "None.")
    doc.add_heading("Data availability", level=1)
    add_p(doc, f"DATA325 raw images, manual bounding boxes, plant-height labels, camera-height mapping, evaluation outputs, selected DCF checkpoints, figure-generation scripts, and reproducibility notes have been organized for public release in the accompanying open-source repository: {REPOSITORY_URL}. The archive file prepared for release is {OPEN_RELEASE_ARCHIVE}.")
    doc.add_heading("Declaration of generative AI and AI-assisted technologies in the writing process", level=1)
    add_p(doc, "During preparation of this work the author used AI-assisted writing and formatting tools to help draft and organize text. After using these tools, the author reviewed and edited the content as needed and takes full responsibility for the content of the publication. No generative AI was used to create or modify the submitted scientific figures, graphical abstract, or result images.")

    doc.add_heading("References", level=1)
    refs = [
        "Araus, J.L., Cairns, J.E., 2014. Field high-throughput phenotyping: the new crop breeding frontier. Trends in Plant Science 19, 52-61.",
        "Ariza-Sentis, M., Velez, S., Martinez-Pena, R., Baja, H., Valente, J., 2024. Object detection and tracking in Precision Farming: a systematic review. Computers and Electronics in Agriculture 219, 108757. doi:10.1016/j.compag.2024.108757.",
        "Caron, M., Touvron, H., Misra, I., Jegou, H., Mairal, J., Bojanowski, P., Joulin, A., 2021. Emerging properties in self-supervised vision transformers. Proceedings of the IEEE/CVF International Conference on Computer Vision, 9650-9660.",
        "Chang, A., Jung, J., Maeda, M.M., Landivar, J., 2017. Crop height monitoring with digital imagery from Unmanned Aerial System (UAS). Computers and Electronics in Agriculture 141, 232-237. doi:10.1016/j.compag.2017.07.008.",
        "Che, Y., Gu, Y., Bai, D., Li, D., Li, J., Zhao, C., Wang, Q., Qiu, H., et al., 2024. Accurately estimate soybean growth stages from UAV imagery by accounting for spatial heterogeneity and climate factors across multiple environments. Computers and Electronics in Agriculture 225, 109313. doi:10.1016/j.compag.2024.109313.",
        "Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al., 2021. An image is worth 16x16 words: transformers for image recognition at scale. International Conference on Learning Representations.",
        "Fahlgren, N., Gehan, M.A., Baxter, I., 2015. Lights, camera, action: high-throughput plant phenotyping is ready for a close-up. Current Opinion in Plant Biology 24, 93-99.",
        "Fiorani, F., Schurr, U., 2013. Future scenarios for plant phenotyping. Annual Review of Plant Biology 64, 267-291.",
        "Ganin, Y., Ustinova, E., Ajakan, H., et al., 2016. Domain-adversarial training of neural networks. Journal of Machine Learning Research 17, 1-35.",
        "Gulrajani, I., Lopez-Paz, D., 2021. In search of lost domain generalization. International Conference on Learning Representations.",
        "Jayasuriya, N., Guo, Y., Hu, W., Ghannoum, O., 2024. Machine vision based plant height estimation for protected crop facilities. Computers and Electronics in Agriculture 218, 108669. doi:10.1016/j.compag.2024.108669.",
        "Kamilaris, A., Prenafeta-Boldu, F.X., 2018. Deep learning in agriculture: a survey. Computers and Electronics in Agriculture 147, 70-90. doi:10.1016/j.compag.2018.02.016.",
        "Kim, W., Lee, D., Kim, Y., Kim, T., Lee, W., Choi, C., 2021. Stereo-vision-based crop height estimation for agricultural robots. Computers and Electronics in Agriculture 181, 105937. doi:10.1016/j.compag.2020.105937.",
        "Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., et al., 2023. Segment anything. arXiv:2304.02643.",
        "Koh, P.W., Sagawa, S., Marklund, H., Xie, S.M., Zhang, M., Balsubramani, A., Hu, W., Yasunaga, M., et al., 2021. WILDS: a benchmark of in-the-wild distribution shifts. Proceedings of the 38th International Conference on Machine Learning 139, 5637-5664.",
        "Li, L., Zhang, Q., Huang, D., 2014. A review of imaging techniques for plant phenotyping. Sensors 14, 20078-20111.",
        "Li, M., Sui, R., Meng, Y., Yan, H., 2019. A real-time fuzzy decision support system for alfalfa irrigation. Computers and Electronics in Agriculture 163, 104870. doi:10.1016/j.compag.2019.104870.",
        "Li, Z., Guo, R., Li, M., Chen, Y., Li, G., 2020. A review of computer vision technologies for plant phenotyping. Computers and Electronics in Agriculture 176, 105672. doi:10.1016/j.compag.2020.105672.",
        "Liu, H., Bruning, B., Garnett, T., Berger, B., 2020. Hyperspectral imaging and 3D technologies for plant phenotyping: from satellite to close-range sensing. Computers and Electronics in Agriculture 175, 105621. doi:10.1016/j.compag.2020.105621.",
        "Oquab, M., Darcet, T., Moutakanni, T., et al., 2024. DINOv2: learning robust visual features without supervision. Transactions on Machine Learning Research.",
        "Patricio, D.I., Rieder, R., 2018. Computer vision and artificial intelligence in precision agriculture for grain crops: a systematic review. Computers and Electronics in Agriculture 153, 69-81. doi:10.1016/j.compag.2018.08.001.",
        "Reena, Doonan, J.H., Williams, K., Corke, F.M.K., Zhang, H., Batke, S., Liu, Y., 2025. Wheat3D PartNet: annotated dataset for 3D wheat part segmentation. Computers and Electronics in Agriculture 238, 110697. doi:10.1016/j.compag.2025.110697.",
        "Shorten, C., Khoshgoftaar, T.M., 2019. A survey on image data augmentation for deep learning. Journal of Big Data 6, 60.",
        "Simeoni, O., Vo, H.V., Seitzer, M., et al., 2025. DINOv3. arXiv:2508.10104. doi:10.48550/arXiv.2508.10104. Official source repository: https://github.com/facebookresearch/dinov3. Accessed 28 Apr 2026.",
        "Sun, B., Saenko, K., 2016. Deep CORAL: correlation alignment for deep domain adaptation. European Conference on Computer Vision Workshops, 443-450. doi:10.1007/978-3-319-49409-8_35.",
        "Ubbens, J.R., Stavness, I., 2017. Deep Plant Phenomics: a deep learning platform for complex plant phenotyping tasks. Frontiers in Plant Science 8, 1190.",
        "Veramendi, W., Cruvinel, P., 2024. Method for maize plants counting and crop evaluation based on multispectral images analysis. Computers and Electronics in Agriculture 216, 108470. doi:10.1016/j.compag.2023.108470.",
        "Wang, D., Shelhamer, E., Liu, S., Olshausen, B., Darrell, T., 2021. Tent: fully test-time adaptation by entropy minimization. International Conference on Learning Representations.",
        "Xie, T., Li, J., Yang, C., Jiang, Z., Chen, Y., Guo, L., Zhang, J., 2021. Crop height estimation based on UAV images: methods, errors, and strategies. Computers and Electronics in Agriculture 185, 106155. doi:10.1016/j.compag.2021.106155.",
        "Xing, Z., Zhang, Z., Shi, R., Guo, Q., Zeng, C., 2023. Filament-necking localization method via combining improved PSO with rotated rectangle algorithm for safflower-picking robots. Computers and Electronics in Agriculture 215, 108464. doi:10.1016/j.compag.2023.108464.",
    ]
    for ref in refs:
        add_p(doc, ref)

    out = OUT / "manuscript_cea.docx"
    doc.save(out)
    return out


def build_highlights() -> Path:
    highlights = [
        "Real greenhouse ROIs test zero-shot maize height transfer.",
        "Attention pooling reduced DATA325 MAE from 41.75 to 30.37 cm.",
        "The best Attn+aug+TTA8 variant reached 29.57 cm MAE on DATA325.",
        "Errors concentrated below 80 cm, where MAPE reached 55.11%.",
        "Geometry, feature-statistic alignment, and DANN did not close the gap.",
    ]
    for item in highlights:
        if len(item) > 85:
            raise ValueError(f"Highlight exceeds Elsevier 85-character guidance: {item}")
    doc = setup_doc()
    doc.add_heading("Highlights", level=1)
    for item in highlights:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(6)
        p.add_run(f"- {item}")
    out = OUT / "highlights.docx"
    doc.save(out)
    (OUT / "highlights.txt").write_text("\n".join(f"- {item}" for item in highlights) + "\n", encoding="utf-8")
    return out


def build_cover_letter() -> Path:
    doc = setup_doc()
    doc.add_heading("Cover letter", level=1)
    paragraphs = [
        "Dear Editors,",
        f"I submit the manuscript entitled \"{TITLE}\" for consideration as an Original research paper in Computers and Electronics in Agriculture.",
        "The manuscript addresses a practical agricultural computer vision problem: maize height estimation that transfers from one greenhouse to another without target-domain training labels. It is positioned against recent Computers and Electronics in Agriculture work on crop-height estimation, protected-crop machine vision, plant phenotyping, object detection, and multi-environment visual phenotyping. The study combines frozen DINOv3 visual features with a DiffCorn-Fusion regression head and uses real greenhouse images, manual ROI crops, attention visualization, and external error galleries to show that attention-weighted patch aggregation substantially improves zero-shot external performance.",
        "The main result is that attention-weighted pooling reduced DATA325 MAE from 41.75 to 30.37 cm, and the best attention plus augmentation plus TTA8 variant reached 29.57 cm MAE. The manuscript also reports diagnostic negative results: camera-height correction, simple geometry, feature-statistic alignment, and the tested DANN setup did not close the remaining domain gap. These controls identify early-stage plants below 80 cm as the primary bottleneck.",
        "The work fits the scope of Computers and Electronics in Agriculture because it evaluates a computational imaging method for crop phenotyping, quantifies cross-greenhouse domain shift, and provides reproducible evaluation assets for future agricultural computer vision research.",
        "The revised figure set contains ten main figures, including newly generated real-image protocol and DATA325 stage-error galleries. All figures and the optional graphical abstract in this CEA package are generated from real images, measured results, or deterministic Python/vector plotting. Generative AI was not used to create or modify submitted scientific figures, graphical abstract, or result images.",
        f"DATA325 raw images, bounding boxes, height labels, evaluation outputs, selected DCF checkpoints, and figure-generation scripts have been organized for public release in the accompanying open-source repository: {REPOSITORY_URL}. The archive file prepared for release is {OPEN_RELEASE_ARCHIVE}. All quantitative results reported in the manuscript can be independently reproduced from the released evaluation JSON files and scripts.",
        "This manuscript is original and is not under consideration elsewhere.",
        "Sincerely,",
        f"{AUTHOR_NAME}",
        f"{AFFILIATION}",
        CORRESPONDING_EMAIL,
    ]
    for para in paragraphs:
        add_p(doc, para)
    out = OUT / "cover_letter_cea.docx"
    doc.save(out)
    return out


def build_supplement(m: dict) -> Path:
    doc = setup_doc()
    doc.add_heading("Supplementary material", level=1)
    add_p(doc, f"Supplementary material for: {TITLE}")
    doc.add_heading("S1. Released and reproducible assets", level=2)
    for item in [
        "DATA325 raw greenhouse images and manually annotated boxes.",
        "Plant-height labels and per-image camera-height mapping used for corrected evaluation.",
        "Evaluation JSON files for CLS, patch-mean, attention-weighted, augmented, TTA8, feature-alignment, and DANN variants.",
        "Figure-generation scripts and non-AI figure assets.",
        "Main tables as editable manuscript tables and CSV sidecars.",
    ]:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        p.add_run(f"- {item}")

    doc.add_heading("S2. Feature extraction modes", level=2)
    add_p(doc, "CLS mode uses the first token from the final hidden state. Patch-mean mode averages all patch tokens. Attention-weighted mode averages final-layer CLS-to-patch attention across heads, normalizes over patch positions, and uses the resulting weights to aggregate patch tokens.")

    doc.add_heading("S3. Diagnostic negative results", level=2)
    for item in [
        f"Bounding-box-ratio geometry reached Pearson r={fmt(m['bbox_r'], 3)} and MAE={fmt(m['bbox_mae'])} cm.",
        f"Attention-map thresholding and vertical projection reached r={fmt(m['attn_threshold_r'], 3)} and r={fmt(m['attn_projection_r'], 3)}, respectively.",
        "Feature-statistic alignment worsened DATA325 performance to 45.98 cm MAE.",
        f"The DANN domain classifier finished near {fmt(m['dann_final_acc'] * 100, 0)}% accuracy, indicating unsuccessful domain confusion.",
    ]:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        p.add_run(f"- {item}")

    doc.add_heading("S4. Literature-positioning notes", level=2)
    for item in [
        "The revised manuscript cites CEA crop-height studies based on UAS/SfM, stereo vision, and protected-facility machine vision to clarify that DINOv3-DCF targets ROI-level zero-shot representation transfer rather than explicit 3D reconstruction.",
        "The revised manuscript cites CEA review and detection papers to motivate agricultural computer vision, ROI quality, and background clutter as deployment constraints.",
        "The revised manuscript cites distribution-shift and domain-generalization work to frame DATA325 as an external diagnostic benchmark rather than a source-domain validation split.",
    ]:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        p.add_run(f"- {item}")

    doc.add_heading("S5. Additional DATA325 visual evidence", level=2)
    add_p(doc, "Two additional real-data visual figures from the Plant Methods preparation are retained outside the main CEA manuscript as optional supplementary evidence. They show the DATA325 annotation protocol and stage-wise error gallery. They are not generated or modified by generative AI.")
    extra = ROOT / "plant_methods_submission_final_public_tone" / "figures_extra"
    for src_name, dst_name, caption in [
        ("fig9_data325_protocol.png", "figS1_data325_protocol.png", "Supplementary Fig. S1. DATA325 annotation and external evaluation protocol."),
        ("fig10_error_pattern_gallery.png", "figS2_error_pattern_gallery.png", "Supplementary Fig. S2. Error-pattern gallery across growth stages."),
    ]:
        src = extra / src_name
        if src.exists():
            dst = OUT / "supplementary_figures"
            dst.mkdir(exist_ok=True)
            shutil.copy2(src, dst / dst_name)
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run().add_picture(str(dst / dst_name), width=Inches(5.9))
            add_caption(doc, caption)

    out = OUT / "supplementary_material.docx"
    doc.save(out)
    return out


def write_sidecars(m: dict) -> None:
    (OUT / "data_availability_statement.md").write_text(
        textwrap.dedent(
            """\
            # Data availability statement

            DATA325 raw images, manual bounding boxes, plant-height labels, camera-height mapping,
            evaluation JSON files, selected DCF checkpoints, figure-generation scripts, and
            reproducibility notes have been organized for public release in the accompanying
            open-source repository: https://github.com/sanyueuy/dinov3-dcf-maize-height.
            The archive file prepared for release is dinov3_dcf_maize_height_open_release_v0_1_0.zip.

            If a Zenodo DOI is minted later, the DOI can be added before journal submission.
            The DINOv3 foundation-model weights are not redistributed and must be obtained under
            their upstream license.
            """
        ),
        encoding="utf-8",
    )
    (OUT / "submission_checklist_cea.md").write_text(
        textwrap.dedent(
            """\
            # Computers and Electronics in Agriculture submission checklist

            ## Included in this package
            - manuscript_cea.docx
            - highlights.docx and highlights.txt
            - cover_letter_cea.docx
            - supplementary_material.docx
            - graphical_abstract_non_ai.png/pdf (optional upload)
            - figures/Figure_1...Figure_10 as PNG/PDF
            - tables/table1_ablation.csv and table2_height_bins.csv
            - reproducibility_json/*.json
            - data_availability_statement.md
            - submission_asset_manifest.md

            ## Still required before online submission
            - Optional ORCID ID for Hong Wu
            - Optional archive DOI if a Zenodo release is minted
            - Author approval of AI-assisted writing disclosure
            - Final check in Elsevier submission system for article-type-specific fields
            - Confirm whether "No external funding was reported" is the final funding statement

            ## CEA-specific handling
            - Article type is prepared as Original research paper.
            - Abstract is non-structured and below 250 words.
            - Highlights are provided as a separate file.
            - Figures are non-AI assets generated from real images, measured results, or deterministic plotting.
            - Main figures include real-image DATA325 annotation, ROI, and error-gallery panels.
            - Tables are editable in the manuscript and also provided as CSV sidecars.
            """
        ),
        encoding="utf-8",
    )

    manifest_lines = [
        "# CEA submission asset manifest",
        "",
        "## Main files",
        "- manuscript_cea.docx",
        "- highlights.docx",
        "- highlights.txt",
        "- cover_letter_cea.docx",
        "- supplementary_material.docx",
        "- graphical_abstract_non_ai.png",
        "- graphical_abstract_non_ai.pdf",
        "",
        "## Figures",
    ]
    for _, upload_stem, caption, _ in FIGURES:
        manifest_lines.append(f"- figures/{upload_stem}.png and .pdf: {caption}")
    manifest_lines.extend(
        [
            "",
            "## Real-image figure generation",
            "- figures/Figure_1_Real_image_protocol.* is generated from DATA325 captured greenhouse photographs, manual bounding boxes, ROI crops, and Attn+aug+TTA8 prediction records.",
            "- figures/Figure_8_DATA325_stage_error_gallery.* is generated from DATA325 photographs grouped by ground-truth height bin and absolute-error rank.",
            "- figures/Figure_9_Diagnostic_experiments.* is redrawn deterministically from bbox geometry, feature-alignment, and DANN-history JSON files.",
            "",
            "## Open-source release package",
            f"- {OPEN_RELEASE_ARCHIVE}: local public-release archive containing DATA325 images, cleaned annotations, sanitized evaluation JSON, selected DCF checkpoints, reproducibility scripts, licenses, citation metadata, and checksums. Public repository: {REPOSITORY_URL}.",
            "",
            "## Tables",
            "- tables/table1_ablation.csv",
            "- tables/table2_height_bins.csv",
            "",
            "## Newly added literature sources",
            "- CEA crop-height and protected-facility machine vision: Chang et al. 2017; Xie et al. 2021; Kim et al. 2021; Jayasuriya et al. 2024.",
            "- CEA agricultural computer vision and phenotyping reviews: Patricio and Rieder 2018; Kamilaris and Prenafeta-Boldu 2018; Li et al. 2020; Liu et al. 2020; Ariza-Sentis et al. 2024.",
            "- CEA crop-specific and multi-environment examples: Li et al. 2019; Xing et al. 2023; Che et al. 2024; Veramendi and Cruvinel 2024; Reena et al. 2025.",
            "- Domain-shift and foundation-model background: Sun and Saenko 2016; Koh et al. 2021; Gulrajani and Lopez-Paz 2021; Caron et al. 2021; Kirillov et al. 2023.",
            "",
            "## Reproducibility JSON",
        ]
    )
    for p in sorted(REPRO_OUT.glob("*.json")):
        manifest_lines.append(f"- reproducibility_json/{p.name}")
    (OUT / "submission_asset_manifest.md").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")


def zip_package() -> Path:
    zip_path = OUT / "cea_submission_package.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob("*"):
            if not p.is_file() or p == zip_path:
                continue
            if "qa_render" in p.parts:
                continue
            z.write(p, p.relative_to(OUT))
    return zip_path


def main() -> None:
    copy_assets()
    build_real_image_figures()
    patch_embedded_figure_titles()
    build_diagnostic_figure()
    m = metrics()
    write_csvs(m)
    build_graphical_abstract(m)
    manuscript = build_manuscript(m)
    highlights = build_highlights()
    cover = build_cover_letter()
    supplement = build_supplement(m)
    write_sidecars(m)
    package = zip_package()
    print("CEA_DIR", OUT)
    print("MANUSCRIPT", manuscript)
    print("HIGHLIGHTS", highlights)
    print("COVER", cover)
    print("SUPPLEMENT", supplement)
    print("ZIP", package)


if __name__ == "__main__":
    main()
