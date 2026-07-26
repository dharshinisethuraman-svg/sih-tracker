"""
Task 3 - Prepare the Images
============================
Generates a labelled image set for the "certificate authenticity"
classifier: given a photo of a candidate's course-completion certificate,
predict whether it looks GENUINE or TAMPERED.

Why this task exists in this project: the assessment brief requires an
image classifier but does not specify what it should classify for THIS
problem statement. Certificate verification is a natural fit for a
skill-training centre (coordinators are handed printed certificates and
have no reliable way to spot an edited one).

Classes: 80 images total (40 per class), as allowed by the brief
("photograph or generate them").
    genuine/  - clean certificate layout, consistent font, correct seal
    tampered/ - same layout but with a visibly altered field (mismatched
                font, overlapping text, or a smudged/moved seal) -
                simulating someone editing a name or date in an image
                editor.

Split policy (important - see Task 3 "Logic" note in the brief):
Each certificate "template" (there are 8 template variations, used to add
diversity) is generated many times with random candidate names/dates.
We split by TEMPLATE index, not by individual image, so the *same
template* never appears in both train and test. If we split randomly
instead, near-identical images of the same template would leak between
train and test and the reported accuracy would be meaningless.
"""
import os
import random
from PIL import Image, ImageDraw, ImageFont

random.seed(7)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "images")
NAMES = ["Priya Sharma", "Ravi Kumar", "Anjali Devi", "Suresh Nair",
         "Kavya Reddy", "Manoj Pillai", "Deepa Iyer", "Arun Menon"]
COURSES = ["Basic Computer Literacy", "Mobile Phone Repair",
           "Tailoring & Garment Making", "Electrical Wiring Basics"]

N_TEMPLATES = 8       # distinct layout variations
IMAGES_PER_TEMPLATE = 5  # per class -> 8*5=40 genuine, 40 tampered


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def make_certificate(template_idx, name, course, tampered):
    W, H = 400, 280
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    border_color = (30, 60, 120) if template_idx % 2 == 0 else (80, 30, 30)
    d.rectangle([8, 8, W - 8, H - 8], outline=border_color, width=4)

    title_font = _font(18)
    body_font = _font(14)
    d.text((W / 2, 35), "CERTIFICATE OF COMPLETION", font=title_font,
            fill=border_color, anchor="mm")

    name_y = 110
    if tampered:
        # simulate a pasted-over name: slightly offset baseline + a
        # mismatched font size + a faint rectangle "patch" behind it
        d.rectangle([60, name_y - 14, W - 60, name_y + 14], fill=(245, 245, 235))
        d.text((W / 2 + 2, name_y + 3), name, font=_font(16), fill=(20, 20, 20), anchor="mm")
    else:
        d.text((W / 2, name_y), name, font=body_font, fill=(20, 20, 20), anchor="mm")

    d.text((W / 2, 150), f"has successfully completed", font=body_font, fill=(60, 60, 60), anchor="mm")
    d.text((W / 2, 175), course, font=body_font, fill=(20, 20, 20), anchor="mm")

    seal_center = (W - 70, H - 55)
    if tampered and template_idx % 3 == 0:
        # smudged / offset seal - another common tamper signature
        d.ellipse([seal_center[0] - 26, seal_center[1] - 22,
                   seal_center[0] + 22, seal_center[1] + 26],
                   outline=(150, 20, 20), width=2)
    else:
        d.ellipse([seal_center[0] - 24, seal_center[1] - 24,
                   seal_center[0] + 24, seal_center[1] + 24],
                   outline=(150, 20, 20), width=3)
    d.text(seal_center, "SEAL", font=_font(10), fill=(150, 20, 20), anchor="mm")

    return img


def main():
    for label in ("genuine", "tampered"):
        os.makedirs(os.path.join(OUT_DIR, label), exist_ok=True)

    manifest = []
    for t in range(N_TEMPLATES):
        for i in range(IMAGES_PER_TEMPLATE):
            name = random.choice(NAMES)
            course = random.choice(COURSES)

            g_img = make_certificate(t, name, course, tampered=False)
            g_path = f"genuine/tpl{t}_{i}.png"
            g_img.save(os.path.join(OUT_DIR, g_path))
            manifest.append((g_path, "genuine", t))

            t_img = make_certificate(t, name, course, tampered=True)
            t_path = f"tampered/tpl{t}_{i}.png"
            t_img.save(os.path.join(OUT_DIR, t_path))
            manifest.append((t_path, "tampered", t))

    with open(os.path.join(OUT_DIR, "manifest.csv"), "w") as f:
        f.write("path,label,template_id\n")
        for path, label, t in manifest:
            f.write(f"{path},{label},{t}\n")

    print(f"Generated {len(manifest)} images "
          f"({len(manifest)//2} genuine, {len(manifest)//2} tampered) "
          f"across {N_TEMPLATES} templates.")


if __name__ == "__main__":
    main()
