from src.ingestion.pdf_extractor import process_pdf


pdf_path = r"C:\Users\anura\OneDrive\Documents\study plan.pdf"

with open(pdf_path, "rb") as file:
    pdf_bytes = file.read()

document = process_pdf(
    file_name="study plan.pdf",
    file_bytes=pdf_bytes
)

print("Name:", document.name)
print("Pages:", document.num_pages)
print("Extractable pages:", document.extractable_pages)
print("Hash:", document.content_hash[:16] + "...")
print("Has text:", document.has_text)

print("\nFirst page:")
print(document.pages[0].text[:1000])