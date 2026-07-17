from pathlib import Path
from pypdf import PdfReader

root = Path(r"99_Attachments (图片、PDF、附件)") / "论文"
out = Path("tmp") / "paper_extracts"
out.mkdir(parents=True, exist_ok=True)

for index, pdf in enumerate(sorted(root.glob("*.pdf")), 1):
    try:
        reader = PdfReader(str(pdf))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception as exc:
                pages.append(f"\n[Text extraction error: {exc}]\n")
        target = out / f"{index:02d}.txt"
        target.write_text("\n\n===== PAGE BREAK =====\n\n".join(pages), encoding="utf-8")
        print(f"{index:02d}\t{pdf.name}\t{len(reader.pages)}\t{sum(map(len,pages))}\t{target}")
    except Exception as exc:
        print(f"{index:02d}\t{pdf.name}\tERROR\t{exc}")
