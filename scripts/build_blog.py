# scripts/build_blog.py
"""Compila blog/reporte.md en un archivo web autocontenido blog/index.html
con estética ejecutiva financiera, segmentación en 5 pestañas interactivas,
soporte para ecuaciones LaTeX (MathJax), tablas estilizadas e imágenes Base64.
Conserva el 100% de la información original de blog/reporte.md.
"""

import base64
import re
from pathlib import Path
import markdown


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte Técnico — Riesgo Crediticio con Redes Neuronales (UNAL)</title>
    <!-- MathJax para renderizar ecuaciones LaTeX ($...$ y $$...$$) -->
    <script>
        MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']],
                processEscapes: true
            }
        };
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        :root {
            --navy-900: #0A192F;
            --navy-800: #112240;
            --primary: #1a365d;
            --accent: #2b6cb0;
            --gold: #D97706;
            --emerald: #059669;
            --bg: #f8fafc;
            --card: #ffffff;
            --text: #1e293b;
            --slate: #475569;
            --border: #e2e8f0;
        }
        * {
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            line-height: 1.7;
            color: var(--text);
            background-color: var(--bg);
            margin: 0;
            padding: 0;
        }
        /* Encabezado Financiero Institucional */
        .exec-header {
            background: linear-gradient(135deg, var(--navy-900) 0%, var(--navy-800) 100%);
            color: #ffffff;
            padding: 2.8rem 2rem 2.2rem;
            border-bottom: 4px solid var(--gold);
        }
        .header-inner {
            max-width: 1140px;
            margin: 0 auto;
        }
        .exec-header h1 {
            color: #ffffff;
            font-size: 2.2rem;
            margin: 0 0 0.8rem 0;
            border: none;
            padding: 0;
            line-height: 1.3;
        }
        .exec-header p {
            color: #cbd5e1;
            margin: 0.3rem 0;
            font-size: 1rem;
        }
        .exec-header a {
            color: #fde68a;
            text-decoration: none;
            font-weight: 600;
            padding: 0.2rem 0.6rem;
            background: rgba(217, 119, 6, 0.25);
            border: 1px solid #f59e0b;
            border-radius: 6px;
            margin-right: 0.4rem;
            display: inline-block;
            transition: all 0.2s;
        }
        .exec-header a:hover {
            background: #f59e0b;
            color: #0A192F;
        }
        /* Cinta de KPIs Reales del Dataset */
        .kpi-ribbon {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 1rem;
            margin-top: 1.6rem;
        }
        .kpi-card {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 8px;
            padding: 0.9rem 1.2rem;
        }
        .kpi-card .kpi-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
            display: block;
        }
        .kpi-card .kpi-val {
            font-family: Consolas, Monaco, monospace;
            font-size: 1.45rem;
            font-weight: 700;
            color: #ffffff;
            margin: 0.15rem 0;
        }
        .kpi-card .kpi-sub {
            font-size: 0.8rem;
            color: #34d399;
        }
        /* Barra de Pestañas */
        .tabs-bar {
            background: #ffffff;
            border-bottom: 1px solid var(--border);
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
        }
        .tabs-nav {
            max-width: 1140px;
            margin: 0 auto;
            display: flex;
            overflow-x: auto;
            padding: 0 1rem;
            gap: 0.25rem;
        }
        .tab-btn {
            background: transparent;
            border: none;
            border-bottom: 3px solid transparent;
            padding: 1rem 1.3rem;
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--slate);
            cursor: pointer;
            white-space: nowrap;
            transition: all 0.2s ease;
        }
        .tab-btn:hover {
            color: var(--navy-900);
            background: #f1f5f9;
        }
        .tab-btn.active {
            color: var(--navy-900);
            border-bottom-color: var(--gold);
            background: #fffbeb;
        }
        /* Contenedor Principal */
        .container {
            max-width: 1140px;
            margin: 2rem auto 4rem;
            background: var(--card);
            padding: 3rem 4rem;
            border-radius: 12px;
            box-shadow: 0 4px 25px rgba(0, 0, 0, 0.06);
            border: 1px solid var(--border);
        }
        .tab-panel {
            display: none;
        }
        .tab-panel.active {
            display: block;
        }
        h1, h2, h3, h4 {
            color: var(--primary);
            font-weight: 700;
        }
        h2 {
            font-size: 1.55rem;
            border-left: 4px solid var(--gold);
            padding-left: 0.8rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.4rem;
            margin-top: 2.2rem;
        }
        h3 {
            font-size: 1.25rem;
            margin-top: 1.8rem;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1.8rem 0;
            font-size: 0.92rem;
        }
        th, td {
            padding: 0.75rem 1rem;
            border: 1px solid var(--border);
            text-align: left;
        }
        th {
            background-color: var(--navy-900);
            color: #ffffff;
            font-weight: 600;
        }
        tr:nth-child(even) {
            background-color: #f8fafc;
        }
        tr:hover {
            background-color: #f1f5f9;
        }
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 2rem auto;
            border-radius: 8px;
            box-shadow: 0 3px 12px rgba(0,0,0,0.09);
        }
        code {
            background: #f1f5f9;
            padding: 0.2rem 0.45rem;
            border-radius: 4px;
            font-size: 0.9em;
            color: #b91c1c;
            font-family: Consolas, Monaco, monospace;
        }
        pre code {
            display: block;
            padding: 1.2rem;
            overflow-x: auto;
            color: #334155;
            background: #f8fafc;
            border: 1px solid var(--border);
        }
        blockquote {
            border-left: 5px solid var(--accent);
            margin: 1.8rem 0;
            padding: 0.8rem 1.4rem;
            background: #f0f7ff;
            border-radius: 0 8px 8px 0;
            color: #1e3a8a;
        }
        hr {
            border: 0;
            height: 1px;
            background: var(--border);
            margin: 2.5rem 0;
        }
        footer {
            text-align: center;
            padding: 2rem;
            color: var(--slate);
            font-size: 0.88rem;
            border-top: 1px solid var(--border);
            background: #ffffff;
        }
        @media (max-width: 768px) {
            .container {
                padding: 1.5rem;
            }
        }
    </style>
</head>
<body>
    <header class="exec-header">
        <div class="header-inner">
            __HEADER_HTML__
            <div class="kpi-ribbon">
                <div class="kpi-card">
                    <span class="kpi-label">Registros Válidos (PD 0/1)</span>
                    <div class="kpi-val">268,530</div>
                    <span class="kpi-sub">Raw: 887,379 filas x 74 cols | Tasa PD: 21.90%</span>
                </div>
                <div class="kpi-card">
                    <span class="kpi-label">Partición Temporal (Out-of-Time)</span>
                    <div class="kpi-val">202,707 Train</div>
                    <span class="kpi-sub">Val: 34,988 (27.12%) | Test: 30,835 (25.46%)</span>
                </div>
                <div class="kpi-card">
                    <span class="kpi-label">Desempeño MLP [64, 32] en Test</span>
                    <div class="kpi-val">AUC 0.6983</div>
                    <span class="kpi-sub">Gini: 0.3966 | KS: 0.2923 | Brier: 0.1731</span>
                </div>
                <div class="kpi-card">
                    <span class="kpi-label">Estabilidad (30 Semillas)</span>
                    <div class="kpi-val">0.6970 ± 0.0014</div>
                    <span class="kpi-sub">38 Features One-Hot | 4,801 Parámetros</span>
                </div>
            </div>
        </div>
    </header>

    <div class="tabs-bar">
        <nav class="tabs-nav" role="tablist">
            <button class="tab-btn active" onclick="openTab(event, 'tab-pd')" role="tab">📈 Parte 1: PD y Scorecard</button>
            <button class="tab-btn" onclick="openTab(event, 'tab-lgd')" role="tab">🏦 Parte 2: LGD y Pérdida Esperada</button>
            <button class="tab-btn" onclick="openTab(event, 'tab-math')" role="tab">📐 Fundamentación Matemática</button>
            <button class="tab-btn" onclick="openTab(event, 'tab-ia')" role="tab">🤖 Uso de IA</button>
            <button class="tab-btn" onclick="openTab(event, 'tab-audit')" role="tab">🔍 Cacería de Alucinaciones</button>
        </nav>
    </div>

    <main class="container">
        <section id="tab-pd" class="tab-panel active">
            __TAB_PD__
        </section>
        <section id="tab-lgd" class="tab-panel">
            __TAB_LGD__
        </section>
        <section id="tab-math" class="tab-panel">
            __TAB_MATH__
        </section>
        <section id="tab-ia" class="tab-panel">
            __TAB_IA__
        </section>
        <section id="tab-audit" class="tab-panel">
            __TAB_AUDIT__
        </section>
    </main>

    <footer>
        Curso: Redes Neuronales y Algoritmos Bioinspirados — Universidad Nacional de Colombia, Sede Medellín (Facultad de Minas) · Semestre 2026-02
    </footer>

    <script>
        function openTab(evt, tabId) {
            var panels = document.querySelectorAll('.tab-panel');
            panels.forEach(function(p) { p.classList.remove('active'); });

            var buttons = document.querySelectorAll('.tab-btn');
            buttons.forEach(function(b) { b.classList.remove('active'); });

            document.getElementById(tabId).classList.add('active');
            evt.currentTarget.classList.add('active');

            if (window.MathJax && window.MathJax.typesetPromise) {
                window.MathJax.typesetPromise();
            }
        }
    </script>
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

        if img_rel_path.startswith("http") or img_rel_path.startswith("data:"):
            return match.group(0)

        candidate_paths = [
            (blog_dir / img_rel_path).resolve(),
            (blog_dir.parent / img_rel_path).resolve(),
        ]

        clean_rel = img_rel_path.lstrip("./")
        candidate_paths.append((blog_dir.parent / clean_rel).resolve())

        for p in candidate_paths:
            if p.exists() and p.is_file():
                b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
                ext = p.suffix.lower().lstrip(".") or "png"
                return f"![{alt_text}](data:image/{ext};base64,{b64})"

        print(f"[WARN] No se encontró la imagen: {img_rel_path}")
        return match.group(0)

    return pattern.sub(replacer, md_text)


def md_to_html_safe_math(md_text: str) -> str:
    """Convierte Markdown a HTML protegiendo bloques LaTeX ($$...$$ y $...$)
    para evitar que python-markdown altere los guiones bajos (_) o barras (\\).
    """
    math_blocks = []

    def save_math(match):
        math_blocks.append(match.group(0))
        return f"@@MATH_BLOCK_{len(math_blocks) - 1}@@"

    # Proteger primero bloques $$...$$ y luego inline $...$
    protected = re.sub(r"\$\$[\s\S]*?\$\$", save_math, md_text)
    protected = re.sub(r"\$[^\$\n]+\$", save_math, protected)

    html = markdown.markdown(
        protected,
        extensions=["tables", "fenced_code", "toc"],
    )

    for idx, block in enumerate(math_blocks):
        html = html.replace(f"@@MATH_BLOCK_{idx}@@", block)

    return html


def split_markdown_sections(raw_md: str):
    """Divide reporte.md en encabezado y secciones H2 (## ...) sin perder texto."""
    parts = re.split(r"(?m)^(?=##\s+)", raw_md)
    header_md = parts[0].strip()
    # Limpiar línea horizontal final del header si existe
    header_md = re.sub(r"\n---\s*$", "", header_md)

    sections = {}
    for part in parts[1:]:
        first_line = part.splitlines()[0]
        sections[first_line] = part.strip()

    return header_md, sections


def main():
    repo_root = Path(__file__).resolve().parents[1]
    md_path = repo_root / "blog" / "reporte.md"
    html_path = repo_root / "blog" / "index.html"

    if not md_path.exists():
        print(f"[ERROR] No existe el archivo: {md_path}")
        return

    raw_md = md_path.read_text(encoding="utf-8")
    md_with_b64 = embed_images_as_base64(raw_md, md_path.parent)

    header_md, sections = split_markdown_sections(md_with_b64)

    tab_pd_md = []
    tab_lgd_md = []
    tab_math_md = []
    tab_ia_md = []
    tab_audit_md = []

    for title, content in sections.items():
        if "4." in title or "Fundamentación Matemática" in title:
            tab_math_md.append(content)
        elif "6." in title or "Parte 2:" in title:
            tab_lgd_md.append(content)
        elif "8." in title or "Uso de IA" in title:
            tab_ia_md.append(content)
            tab_audit_md.append(content)
        else:
            # Secciones 1, 2, 3, 5, 7 y 9 van completas en Parte 1 (PD)
            tab_pd_md.append(content)

    # Si existen docs/uso_ia.md o docs/caceria_alucinacion.md, agregarlos a sus pestañas respectivas
    uso_ia_doc = repo_root / "docs" / "uso_ia.md"
    if uso_ia_doc.exists():
        tab_ia_md.append("---\n\n" + embed_images_as_base64(uso_ia_doc.read_text(encoding="utf-8"), md_path.parent))

    caceria_doc = repo_root / "docs" / "caceria_alucinacion.md"
    if caceria_doc.exists():
        tab_audit_md.append("---\n\n" + embed_images_as_base64(caceria_doc.read_text(encoding="utf-8"), md_path.parent))

    final_html = (
        HTML_TEMPLATE
        .replace("__HEADER_HTML__", md_to_html_safe_math(header_md))
        .replace("__TAB_PD__", md_to_html_safe_math("\n\n---\n\n".join(tab_pd_md)))
        .replace("__TAB_LGD__", md_to_html_safe_math("\n\n---\n\n".join(tab_lgd_md)))
        .replace("__TAB_MATH__", md_to_html_safe_math("\n\n---\n\n".join(tab_math_md)))
        .replace("__TAB_IA__", md_to_html_safe_math("\n\n---\n\n".join(tab_ia_md)))
        .replace("__TAB_AUDIT__", md_to_html_safe_math("\n\n---\n\n".join(tab_audit_md)))
    )

    html_path.write_text(final_html, encoding="utf-8")
    print(f"[OK] Reporte web compilado exitosamente con 5 pestañas en: {html_path}")


if __name__ == "__main__":
    main()