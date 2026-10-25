# scripts/build_blog.py
"""Compila blog/reporte.md en un archivo web autocontenido blog/index.html
con soporte para ecuaciones LaTeX (MathJax), tablas estilizadas e imágenes Base64.
"""

import base64
import os
import re
from pathlib import Path
import markdown


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte Técnico — Riesgo Crediticio con Redes Neuronales (UNAL)</title>
    <!-- MathJax para renderizar ecuaciones LaTeX ($...$ y $$...$$) -->
    <script>
        MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                processEscapes: true
            }}
        }};
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        :root {{
            --primary: #1a365d;
            --accent: #2b6cb0;
            --bg: #f8fafc;
            --card: #ffffff;
            --text: #1e293b;
            --border: #e2e8f0;
        }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            line-height: 1.7;
            color: var(--text);
            background-color: var(--bg);
            margin: 0;
            padding: 2.5rem 1rem;
        }}
        .container {{
            max-width: 980px;
            margin: 0 auto;
            background: var(--card);
            padding: 3.5rem 4.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 25px rgba(0, 0, 0, 0.07);
        }}
        h1, h2, h3, h4 {{
            color: var(--primary);
            font-weight: 700;
        }}
        h1 {{
            font-size: 2.3rem;
            border-bottom: 4px solid var(--accent);
            padding-bottom: 0.8rem;
            margin-top: 0;
        }}
        h2 {{
            font-size: 1.6rem;
            border-bottom: 1.5px solid var(--border);
            padding-bottom: 0.4rem;
            margin-top: 2.5rem;
        }}
        h3 {{
            font-size: 1.25rem;
            margin-top: 1.8rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1.8rem 0;
            font-size: 0.93rem;
        }}
        th, td {{
            padding: 0.8rem 1rem;
            border: 1px solid var(--border);
            text-align: left;
        }}
        th {{
            background-color: #f1f5f9;
            color: var(--primary);
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background-color: #f8fafc;
        }}
        tr:hover {{
            background-color: #f1f5f9;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 2rem auto;
            border-radius: 8px;
            box-shadow: 0 3px 12px rgba(0,0,0,0.09);
        }}
        code {{
            background: #f1f5f9;
            padding: 0.2rem 0.45rem;
            border-radius: 4px;
            font-size: 0.9em;
            color: #b91c1c;
            font-family: Consolas, Monaco, monospace;
        }}
        pre code {{
            display: block;
            padding: 1.2rem;
            overflow-x: auto;
            color: #334155;
            background: #f8fafc;
            border: 1px solid var(--border);
        }}
        blockquote {{
            border-left: 5px solid var(--accent);
            margin: 1.8rem 0;
            padding: 0.8rem 1.4rem;
            background: #f0f7ff;
            border-radius: 0 8px 8px 0;
            color: #1e3a8a;
        }}
        hr {{
            border: 0;
            height: 1px;
            background: var(--border);
            margin: 2.5rem 0;
        }}
    </style>
</head>
<body>
    <main class="container">
        {content}
    </main>
</body>
</html>
"""


def embed_images_as_base64(md_text: str, blog_dir: Path) -> str:
    """Reemplaza rutas de imágenes Markdown por data URIs en Base64
    para que el HTML sea 100% portable e independiente de carpetas.
    """
    pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

    def replacer(match):
        alt_text = match.group(1)
        img_rel_path = match.group(2).strip()

        if img_rel_path.startswith("http"):
            return match.group(0)

        # Rutas candidatas: relativa desde blog/ o desde la raíz del repo
        candidate_paths = [
            (blog_dir / img_rel_path).resolve(),
            (blog_dir.parent / img_rel_path).resolve(),
        ]

        # También prueba eliminando prefijos '../'
        clean_rel = img_rel_path.lstrip("../")
        candidate_paths.append((blog_dir.parent / clean_rel).resolve())

        for p in candidate_paths:
            if p.exists() and p.is_file():
                b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
                ext = p.suffix.lower().lstrip(".") or "png"
                return f'![{alt_text}](data:image/{ext};base64,{b64})'

        print(f"[WARN] No se encontró la imagen: {img_rel_path}")
        return match.group(0)

    return pattern.sub(replacer, md_text)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    md_path = repo_root / "blog" / "reporte.md"
    html_path = repo_root / "blog" / "index.html"

    if not md_path.exists():
        print(f"[ERROR] No existe el archivo: {md_path}")
        return

    raw_md = md_path.read_text(encoding="utf-8")
    md_with_b64 = embed_images_as_base64(raw_md, md_path.parent)

    html_body = markdown.markdown(
        md_with_b64,
        extensions=["tables", "fenced_code", "toc"],
    )

    final_html = HTML_TEMPLATE.format(content=html_body)
    html_path.write_text(final_html, encoding="utf-8")
    print(f"[OK] Reporte web compilado exitosamente en: {html_path}")


if __name__ == "__main__":
    main()