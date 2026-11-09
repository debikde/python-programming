"""Render the reviewed Russian EDA narrative as a portable HTML with inline plots."""

import base64
import csv
import html
import json
import os
import re
from pathlib import Path

from markdown_it import MarkdownIt


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "docs/eda_report.md"
    output = root / "artifacts/reports/eda_ntnu/report.html"
    summary = json.loads((output.parent / "summary.json").read_text(encoding="utf-8"))
    narrative = source.read_text(encoding="utf-8")
    source_hashes = [item["sha256"] for item in summary.get("source_archives", [])]
    source_hashes = source_hashes or [summary["source_archive_sha256"]]
    if not all(digest in narrative for digest in source_hashes):
        raise ValueError("The reviewed narrative describes a different source archive")
    body = MarkdownIt("commonmark").enable("table").render(narrative)

    def embed(match: re.Match[str]) -> str:
        path = (source.parent / html.unescape(match.group(1))).resolve()
        if not path.is_relative_to(output.parent.resolve()):
            raise ValueError("Report image must be an EDA output")
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'src="data:image/png;base64,{data}"'

    body = re.sub(r'src="([^"]+\.png)"', embed, body)

    def local_link(match: re.Match[str]) -> str:
        link = html.unescape(match.group(1))
        if re.match(r"(?:https?://|#|data:)", link):
            return match.group(0)
        path = (source.parent / link).resolve()
        relative = os.path.relpath(path, output.parent).replace("\\", "/")
        return f'href="{html.escape(relative, quote=True)}"'

    body = re.sub(r'href="([^"]+)"', local_link, body)
    body = body.replace("<table>", '<div class="table-scroll"><table>')
    body = body.replace("</table>", "</table></div>")
    downloads = []
    for path in sorted((output.parent / "tables").glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as stream:
            count = sum(1 for _ in csv.reader(stream)) - 1
        relative = os.path.relpath(path, output.parent).replace("\\", "/")
        downloads.append(
            f'<li><a href="{relative}">{html.escape(path.stem)}</a>'
            f" — {count:,} строк</li>"
        )
    document = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Assistive Gait Research — EDA NTNU</title>
<style>
body{margin:0;background:#f3f6f8;color:#172c39;font:17px/1.65 system-ui,sans-serif}
main{max-width:1080px;margin:30px auto;padding:40px 50px;background:white}
h1,h2,h3{line-height:1.25;color:#123e50}h1{font-size:36px}
h2{margin-top:48px;border-top:1px solid #dce6eb;padding-top:22px}
h3{margin-top:32px}a{color:#006d83}img{max-width:100%;height:auto}
table{border-collapse:collapse;width:100%;font-size:15px}th,td{padding:10px 12px;
border-bottom:1px solid #dce6eb;text-align:left}th{background:#eaf2f5}
.table-scroll{overflow-x:auto}pre{background:#142b39;color:#edf6fa;padding:20px;
overflow-x:auto;border-radius:8px}code{font-size:.9em;overflow-wrap:anywhere}
p code,li code{background:#eef3f6;padding:2px 4px}li{margin:6px 0}
.badge{color:#12647b;letter-spacing:.06em;font-size:13px;font-weight:700}
@media(max-width:700px){main{margin:0;padding:22px}h1{font-size:28px}}
@media print{body{background:white}main{margin:0;padding:0;max-width:none}
h2,h3{break-after:avoid}img,table{break-inside:avoid}a{color:inherit}}
</style></head><body><main><div class="badge">ASSISTIVE GAIT RESEARCH / EDA</div>
"""
    document += body
    document += "<h2>Подробные таблицы</h2><p>Графики встроены в HTML. Для ссылок "
    document += "на CSV сохраните рядом папку tables.</p><ul>"
    document += "\n".join(downloads) + "</ul></main></body></html>\n"
    output.write_text(document, encoding="utf-8", newline="\n")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
