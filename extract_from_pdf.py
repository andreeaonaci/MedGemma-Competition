import pymupdf  # PyMuPDF
import json
import re
import glob
import os

doc = pymupdf.open("/mnt/7ac346ab-63e9-4291-b1d0-6ac9ad09954a/work/medical_work_from_sgementation/OCTraining_EN.pdf")
#doc = pymupdf.open("/home/justdoit/small2.pdf")

case_structure_empty = {"raw_image":None,
                  "annotated_image":None,
                  "fundus_image":None,

                  "Interpretation":None,
                  "Age":None,
                  "Location":None,
                  "Case":None,
                  "Scan Quality":None,
                  "IR Image":None}

data = []
current_area = None


def process_chunks(case_data, field, items, i, item_type="space" ):
    if item_type=="space":
        case_data[field]=items[i].split()[1].strip()
        increment = 0
    else:
        case_data[field]=items[i+1]
        increment=1
    return increment

cases = []

for page_index, page in enumerate(doc):
    if page_index<55 :#or page_index >60:
        continue
    print(f"Extract page {page_index}")
    images = page.get_images(full=True)
    images = sorted(images, key=lambda x: x[2])
    #print(images)
    for img_index, img in enumerate(images):
        xref = img[0]
        #print(img)
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]

        with open(f"data/case_images/image_{page_index}_{img_index}.png", "wb") as f:
            f.write(image_bytes)


    text = page.get_text("text")
    blocks = page.get_text("blocks")
    text3 = page.get_text("rawdict")["blocks"]

    lines = text.split("\n")


    i=0

    if "IR Image:" in lines:
        print("first page", lines)
        case_structure = case_structure_empty.copy()
        case_structure["raw_image"] = f"image_{page_index}_1.png"
        case_structure["fundus_image"]=f"image_{page_index}_0.png"
        while i<len(blocks):
            if blocks[i][4].startswith("Age"):
                parts = blocks[i][4].split("\n")
                case_structure["Age"] = parts[0].split()[1].strip()
                if "Case" in blocks[i][4]:
                    case_structure["Case"] = parts[1].split()[1].strip()
            if blocks[i][4].startswith("IR Image"):
                case_structure["IR Image"] = " ".join(blocks[i][4].split("\n")[1:])
            if blocks[i][4].startswith("Scan Quality"):
                case_structure["Scan Quality"] = " ".join(blocks[i][4].split("\n")[1:])
            if blocks[i][4].startswith("Location"):
                case_structure["Location"] = " ".join(blocks[i][4].split("\n")[1:])
            i=i+1

    if "Interpretation" in lines: # findings page
        case_structure["annotated_image"] = f"image_{page_index}_0.png"
        i=1

        findings = []
        while i<len(blocks):
            if blocks[i][4].startswith("Area"):
                for j in range(i, len(blocks)):
                    if blocks[j][4].startswith("Interpretation"):
                        break
                case_structure["Interpretation"] = blocks[j-1][4].replace("\n", " ")
                i=i+1

                finding={}
                while i<j-1:
                    parts = blocks[i][4].strip().split("\n")


                    if len(parts)==3:
                        if finding!={}:
                            findings.append(finding)
                        finding = {"area":parts[0]}
                        finding["content"]=[{"alteration": parts[1], "description":parts[2]}]

                    elif len(parts)==2:
                        finding["content"].append({"alteration": parts[0], "description":parts[1]})
                    i=i+1
                findings.append(finding)
                i=len(blocks)
            i=i+1

        case_structure["findings"] = findings
        cases.append(case_structure)


with open('data/cases.json', 'w', encoding='utf-8') as f:
    json.dump(cases, f, ensure_ascii=False, indent=4)


#clean the irelevant images
for name in glob.glob("data/case_images/*"):
    file_name = os.path.basename(name)

    fparts = os.path.splitext(file_name)

    if fparts[1]==".png":
        _,number, number2 = fparts[0].split("_")
        number=int(number)
        if (number %2 ==0 and number2=="1"):

            os.remove(name)
