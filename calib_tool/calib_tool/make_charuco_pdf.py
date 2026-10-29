"""Print the cube ChArUco boards at real size (14 cm) on A4 pages, with a cut line and a 10 cm ruler.

ros2 run calib_tool make_charuco_pdf <charuco_boards dir (charuco_board_*.bmp)> <out.pdf>
"""

import sys
import glob
import os
from PIL import Image, ImageDraw, ImageFont


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: make_charuco_pdf <charuco_boards dir> <out.pdf>")
    src, out = sys.argv[1], sys.argv[2]
    BOARD_CM = 14.0
    files = sorted(glob.glob(os.path.join(src, "charuco_board_*.bmp")))
    w0 = Image.open(files[0]).width
    dpi = w0 / (BOARD_CM / 2.54)  # board pixels -> exactly 14 cm
    px_cm = dpi / 2.54
    W, H = round(21.0 * px_cm), round(29.7 * px_cm)  # A4 portrait
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", round(0.5 * px_cm))
    except OSError:
        font = ImageFont.load_default()
    pages = []
    for i, f in enumerate(files):
        page = Image.new("L", (W, H), 255)
        b = Image.open(f).convert("L")
        x, y = (W - b.width) // 2, round(3 * px_cm)
        page.paste(b, (x, y))
        d = ImageDraw.Draw(page)
        # thin cut line drawn just outside the board so the board itself stays 14 cm
        lw = 3  # ~0.2 mm
        d.rectangle(
            [x - lw, y - lw, x + b.width + lw - 1, y + b.height + lw - 1],
            outline=0,
            width=lw,
        )
        # 10 cm control ruler with 1 cm ticks
        ry = y + b.height + round(2 * px_cm)
        x0 = (W - round(10 * px_cm)) // 2
        d.line([(x0, ry), (x0 + round(10 * px_cm), ry)], fill=0, width=4)
        for k in range(11):
            xk = x0 + round(k * px_cm)
            d.line(
                [
                    (xk, ry - round(0.4 * px_cm if k % 5 == 0 else 0.2 * px_cm)),
                    (xk, ry),
                ],
                fill=0,
                width=3,
            )
        d.text(
            (x0, ry + round(0.4 * px_cm)), "         10 cm         ", fill=0, font=font
        )
        d.text(
            (x, round(1.5 * px_cm)),
            f"Board {i:03d} - 14 x 14 cm - print at actual size (100%)",
            fill=0,
            font=font,
        )
        pages.append(page)
    pages[0].save(out, save_all=True, append_images=pages[1:], resolution=dpi)
    print(f"{len(pages)} pages, dpi={dpi:.3f} -> {out}")


if __name__ == "__main__":
    main()
