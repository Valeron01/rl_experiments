import random

import cv2
import numpy as np
from collections import deque


# ------------------------------------------------------------------
# Helper: colour mapping for the 1‑9 numbers (BGR order – OpenCV!)
NUM_COLORS = {
    1: (255,   0,   0),   # blue      – Minesweeper classic
    2: (  0, 255,   0),   # green
    3: (  0,   0, 255),   # red
    4: (255,   0, 255),   # magenta
    5: (255, 255,   0),   # cyan
    6: (  0, 255, 255),   # yellow
    7: (128, 128, 128),   # gray
    8: (  0,   0,   0)    # black
}
# ------------------------------------------------------------------


def draw_minesweeper(field: np.ndarray,
                     opened: np.ndarray,
                     cell_size: int = 40,
                     grid_color=(50, 50, 50),
                     closed_color=(100, 100, 100),
                     open_color=(200, 200, 200)) -> np.ndarray:
    """
    Render a Minesweeper board.

    Parameters
    ----------
    field : (H,W) int array
        -1 = bomb, 0–9 = number of bombs around the cell.
    opened : (H,W) bool/uint8 array
        1 → cell is already revealed; 0 → still hidden.
    cell_size : int
        Size in pixels of a single cell (square).
    grid_color : BGR tuple
        Colour of the grid lines.
    closed_color : BGR tuple
        Background colour for hidden cells.
    open_color : BGR tuple
        Background colour for revealed cells.

    Returns
    -------
    img : (H*cell_size, W*cell_size, 3) uint8 image ready to show with cv2.imshow()
    """
    h, w = field.shape

    # Create a blank canvas
    img = np.zeros((h * cell_size, w * cell_size, 3), dtype=np.uint8)

    # ---------- 1. Paint all cells ----------
    # Hidden cells: dark gray
    img[:] = closed_color

    # Reveal only the open cells – we will paint them later
    # (the default background is already set to closed_color)

    # ---------- 2. Draw each cell individually ----------
    for y in range(h):
        for x in range(w):
            top_left   = (x * cell_size,     y * cell_size)
            bottom_right = ((x + 1) * cell_size - 1,
                            (y + 1) * cell_size - 1)

            # If the cell is open …
            if opened[y, x]:
                # a) set background colour for open cells
                cv2.rectangle(img, top_left, bottom_right, open_color, thickness=-1)

                val = field[y, x]
                center_x = x * cell_size + cell_size // 2
                center_y = y * cell_size + cell_size // 2

                # b) bomb ?
                if val == -1:
                    # simple red circle for a bomb – you could load an image instead
                    radius = cell_size // 4
                    cv2.circle(img, (center_x, center_y), radius,
                               (0, 0, 255), thickness=-1)

                # c) number ?
                elif val > 0:
                    txt_color = NUM_COLORS.get(val, (0, 0, 0))
                    # Put the number in the centre of the cell
                    text = str(val)
                    font   = cv2.FONT_HERSHEY_SIMPLEX
                    scale  = 0.8
                    thickness = 2

                    # Get text size to centre it properly
                    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
                    txt_x = center_x - text_w // 2
                    txt_y = center_y + text_h // 2

                    cv2.putText(img, text,
                                (txt_x, txt_y),
                                font, scale, txt_color, thickness)

    # ---------- 3. Draw the grid ----------
    for i in range(h + 1):
        y = i * cell_size
        cv2.line(img, (0, y), (w * cell_size, y), grid_color, 1)
    for j in range(w + 1):
        x = j * cell_size
        cv2.line(img, (x, 0), (x, h * cell_size), grid_color, 1)

    return img



def create_minesweeper_field(height: int,
                            width: int,
                            num_bombs: int,
                            seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive integers")
    if not (0 <= num_bombs <= height * width):
        raise ValueError(f"num_bombs must be in [0,{height*width}]")

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Place bombs
    # ------------------------------------------------------------------
    field = np.zeros((height, width), dtype=int)
    bomb_indices = rng.choice(height * width, size=num_bombs, replace=False)
    field.flat[bomb_indices] = -1          # mark bombs with –1

    # ------------------------------------------------------------------
    # 2. Count neighbours for every non‑bomb cell
    # ------------------------------------------------------------------
    bomb_mask = (field == -1).astype(int)   # 1 where a bomb sits, else 0

    # Pad the mask so we can sum over all 8 directions without wrap‑around.
    pad = np.pad(bomb_mask, ((1, 1), (1, 1)), mode='constant')

    neighbour_counts = (
        pad[:-2, :-2] + pad[1:-1, :-2] + pad[2:, :-2] +
        pad[:-2, 1:-1]                + pad[2:, 1:-1] +
        pad[:-2, 2:]   + pad[1:-1, 2:] + pad[2:, 2:]
    )

    # Assign the counts to cells that are not bombs.
    field[field != -1] = neighbour_counts[field != -1]

    # ------------------------------------------------------------------
    # 3. Prepare an “opened” mask (all zeros for a new game)
    # ------------------------------------------------------------------
    opened = np.zeros_like(field, dtype=np.uint8)

    return field, opened

def click(field: np.ndarray,
          opened: np.ndarray,
          y: int,
          x: int) -> tuple[np.ndarray, bool]:
    """
    Process a single user click on a Minesweeper board.

    Parameters
    ----------
    field : np.ndarray[int]
        Shape `(h,w)` containing -1 for bombs and 0‑8 for adjacent‑bomb counts.
    opened : np.ndarray[bool | uint8]
        Same shape as *field*.  `True` (or non‑zero) means the cell is already open.
    y, x : int
        Row and column indices of the clicked cell (0‑based).

    Returns
    -------
    new_opened : np.ndarray[uint8]
        Updated mask with the newly opened cells.
    lost : bool
        `True` if the click hit a bomb, otherwise `False`.

    Notes
    -----
    * If the user clicks on an already open cell nothing changes – the function just
      returns the original state.
    * Clicking a bomb sets `lost=True`.  The clicked bomb is revealed in the returned
      mask; you may optionally reveal all bombs by uncommenting the line at the end of
      the function.
    * If the user clicks on an empty cell (`field[y,x]==0`) a flood‑fill (BFS) opens
      all connected zero cells and their bordering numbered neighbours – exactly how
      classic Minesweeper behaves.

    Example
    -------
    Lost: False
    """
    h, w = field.shape

    if not (0 <= y < h and 0 <= x < w):
        raise ValueError(f"Click coordinates ({y},{x}) are out of bounds for board {h}×{w}")

    # If the cell is already open, nothing changes.
    if opened[y, x]:
        return opened.copy(), False

    new_opened = opened.copy()
    lost = False

    # --- Bomb click ---------------------------------------------------
    if field[y, x] == -1:
        lost = True
        new_opened[y, x] = 1          # reveal the bomb that was clicked
        # Uncomment the following line to show *all* bombs when you lose.
        # new_opened[field == -1] = 1
        return new_opened, lost

    # --- Safe click ----------------------------------------------------
    # Open the clicked cell
    new_opened[y, x] = 1

    # If it is an empty (0) cell, flood‑fill all connected zeros.
    if field[y, x] == 0:
        q = deque([(y, x)])
        while q:
            cy, cx = q.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        # Skip bombs and already opened cells
                        if field[ny, nx] == -1 or new_opened[ny, nx]:
                            continue
                        new_opened[ny, nx] = 1
                        # Only keep expanding from zeros
                        if field[ny, nx] == 0:
                            q.append((ny, nx))

    return new_opened, lost


def main():
    field, opened = create_minesweeper_field(16, 32, 100)
    for i in range(10000):
        opened, lost = click(field, opened, 10, 10)
        img = draw_minesweeper(field, opened, cell_size=25)

        if lost:
            field, opened = create_minesweeper_field(16, 32, 100)

        cv2.imshow("Minesweeper", img)
        cv2.waitKey(1)
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
