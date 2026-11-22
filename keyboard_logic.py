"""Pure Python gesture and pad/character logic; no webcam dependencies."""
from dataclasses import dataclass
import math
from typing import Optional

# Pad numbers 1...12; each tuple is (T, O, Y).
# Pad 9 includes a SPACE for the 9-pad fallback; pad 10 has controls.
LETTER_GROUPS = (
    ("A", "B", "C"), ("D", "E", "F"), ("G", "H", "I"),
    ("J", "K", "L"), ("M", "N", "O"), ("P", "Q", "R"),
    ("S", "T", "U"), ("V", "W", "X"), ("Y", "Z", "SPACE"),
    ("SPACE", "BACKSPACE", "ENTER"),
    (".", ",", "?"), ("!", "'", "-"),
)
GESTURE_SLOTS = {'DOWN': 0, 'T': 1, 'UP': 2}
GESTURE_ACTIONS = {'O': 'MODE', 'X': 'DELETE', 'Y': 'ACCEPT'}
GESTURE_HOLDS = {'DOWN': .40, 'T': .40, 'UP': .40,
                 'O': .50, 'X': 1.0, 'Y': .65}


def mapped_key(mode: str, slot: Optional[int], pad: int, pad_count: int) -> Optional[str]:
    """pad is zero-indexed. In NUMBERS, slot is ignored with >=10 pads."""
    if not 8 <= pad_count <= 12 or not 0 <= pad < pad_count:
        return None
    if mode not in ('LETTERS', 'NUMBERS'):
        return None
    # In the 12-pad layout these controls never depend on the pose or mode.
    if pad_count == 12:
        if pad == 10:
            return 'SPACE'
        if pad == 11:
            return 'BACKSPACE'
        if mode == 'LETTERS' and pad == 9:
            return ('.', ',', 'ENTER')[slot] if slot in (0, 1, 2) else None
    if mode == "LETTERS":
        if slot not in (0, 1, 2):
            return None
        return LETTER_GROUPS[pad][slot]
    if mode != "NUMBERS":
        return None
    if pad_count >= 10:
        return ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
                "SPACE", "BACKSPACE")[pad]
    # Fallback for fewer than 10 pads: T has numbers 1..8/9; O has remaining.
    # Number mode then requires selecting T/O via CV, just like letters.
    if slot is None:
        return None
    if slot == 0:
        return str(pad+1)
    if slot == 1:
        extras = ("9", "0", "SPACE", "BACKSPACE") if pad_count == 8 else (
            "0", "SPACE", "BACKSPACE", "ENTER")
        return extras[pad] if pad < len(extras) else None
    return None


def apply_key(text: str, key: str) -> str:
    if key == "SPACE":
        return text + " "
    if key == "BACKSPACE":
        return text[:-1]
    if key == "ENTER":
        return text + "\n"
    return text + key


def _visible(point, min_visibility: float = 0.5) -> bool:
    score = getattr(point, "visibility", None)
    return score is None or score >= min_visibility


def accept_tracking_ready(points):
    """Require visible, in-frame arms; don't accept inferred off-screen hands."""
    return bool(points and len(points)>16 and all(
        _visible(points[i], .65) and math.isfinite(points[i].x) and
        math.isfinite(points[i].y) and .005 < points[i].x < .995 and
        .005 < points[i].y < .995 for i in (11,12,13,14,15,16)))


def arms_lowered(points):
    """A real release, as opposed to a tracking failure or hidden hands."""
    if not points or len(points)<17 or not all(
            _visible(points[i],.65) for i in (11,12,15,16)):
        return False
    span = abs(points[11].x-points[12].x)
    return (span>.065 and points[15].y>points[11].y+.25*span and
            points[16].y>points[12].y+.25*span)


def classify_gesture(points, width: int, height: int) -> Optional[str]:
    """Recognize T/O/Y/X plus straight-up and diagonal-down actions.

    Basic geometric heuristics; tune thresholds for room/camera/participants.
    Camera should see both arms, ideally from waist up, front facing.
    None means no recognized gesture (not a command).
    """
    if not points or len(points) < 25:
        return None
    # Official MediaPipe indices: 0 nose; 11/12 shoulders; 13/14 elbows;
    # 15/16 wrists; 23/24 hips.
    ids = (0, 11, 12, 13, 14, 15, 16)
    if not all(_visible(points[i]) for i in ids):
        return None

    def xy(i):
        return (points[i].x * width, points[i].y * height)

    # Assign each arm based on the *image* left/right ordering. This works
    # for mirrored webcams too and does not depend on MediaPipe 'left' labels.
    arms = [(xy(11), xy(13), xy(15)), (xy(12), xy(14), xy(16))]
    arms.sort(key=lambda arm: arm[0][0])
    left, right = arms
    left_sh, left_el, left_wr = left
    right_sh, right_el, right_wr = right
    sw = right_sh[0] - left_sh[0]
    if sw < width * 0.065:  # too small / sideways / too far away
        return None
    mid_x = (left_sh[0] + right_sh[0]) / 2.0
    sh_y = (left_sh[1] + right_sh[1]) / 2.0
    hip_y = (xy(23)[1] + xy(24)[1]) / 2.0
    nose_y = xy(0)[1]

    # Inverted Y: both arms extended outward and down (roughly 30-60 degrees).
    # Require extended forearms to distinguish hands on hips and relaxed arms.
    def diagonal_down(arm, outward):
        sh, el, wr = arm
        dx, dy = outward*(wr[0]-sh[0]), wr[1]-sh[1]
        return (dx > .65*sw and dy > .45*sw and .55 < dy/dx < 1.8 and
                outward*(el[0]-sh[0]) > .20*sw and
                el[1] > sh[1]+.15*sw and
                outward*(wr[0]-el[0]) > .20*sw and
                wr[1] > el[1]+.15*sw)
    if diagonal_down(left,-1) and diagonal_down(right,1):
        return 'DOWN'

    def straightness(arm):
        sh,el,wr = arm
        a = (el[0]-sh[0],el[1]-sh[1])
        b = (wr[0]-el[0],wr[1]-el[1])
        length = math.hypot(*a)*math.hypot(*b)
        return (a[0]*b[0]+a[1]*b[1])/length if length else -1

    # Accept: narrow, straight arms pointing up. O keeps bent, rounded arms.
    upright = (left_wr[1] < nose_y-.55*sw and
               right_wr[1] < nose_y-.55*sw and
               left_el[1] < left_sh[1]-.35*sw and
               right_el[1] < right_sh[1]-.35*sw and
               left_wr[1] < left_el[1]-.25*sw and
               right_wr[1] < right_el[1]-.25*sw and
               math.dist(left_wr,right_wr) < .65*sw and
               abs((left_wr[0]+right_wr[0])/2-mid_x) < .40*sw)
    arm_straightness = (straightness(left),straightness(right))
    if upright and min(arm_straightness) > .90 and accept_tracking_ready(points):
        return 'UP'

    # X: wrists cross the body's center, across the upper chest.
    chest_top = sh_y - 0.40*sw
    chest_bottom = max(sh_y + 0.45*sw, hip_y + 0.12*sw)
    crossed = (left_wr[0] > mid_x + 0.07*sw and
               right_wr[0] < mid_x - 0.07*sw and
               chest_top <= left_wr[1] <= chest_bottom and
               chest_top <= right_wr[1] <= chest_bottom and
               left_el[0] < mid_x and right_el[0] > mid_x)
    if crossed:
        return "X"

    # O: both wrists meet over the head; elbows may be spread outward.
    gap = math.dist(left_wr, right_wr)
    wrists_above_head = (left_wr[1] < nose_y - 0.55*sw and
                         right_wr[1] < nose_y - 0.55*sw)
    centered = abs((left_wr[0] + right_wr[0])/2.0 - mid_x) < 0.65*sw
    if (wrists_above_head and gap < 0.72*sw and centered
            and max(arm_straightness) < .80):
        return "O"

    # Y: both wrists high and widely separated, slanting outward.
    left_up = (left_wr[0] < left_sh[0] - 0.25*sw and
               left_wr[1] < left_sh[1] - 0.45*sw and
               left_el[1] < left_sh[1] + 0.20*sw)
    right_up = (right_wr[0] > right_sh[0] + 0.25*sw and
                right_wr[1] < right_sh[1] - 0.45*sw and
                right_el[1] < right_sh[1] + 0.20*sw)
    if left_up and right_up and gap > 1.35*sw and accept_tracking_ready(points):
        return "Y"

    # T: arms horizontally to sides; wrists and elbows close to shoulders' Y.
    def t_arm(arm, outward):
        sh, el, wr = arm
        return (outward*(el[0]-sh[0]) > 0.22*sw and
                outward*(wr[0]-sh[0]) > 0.68*sw and
                abs(el[1]-sh[1]) < 0.34*sw and
                abs(wr[1]-sh[1]) < 0.40*sw)
    if t_arm(left, -1) and t_arm(right, 1):
        return "T"
    return None


@dataclass
class GestureHold:
    """Only fire a gesture once after stable continuous observation."""
    candidate: Optional[str] = None
    since: float = 0.0
    fired: bool = False
    missing_since: Optional[float] = None
    dropout_grace: float = 0.12
    action_locked: bool = False
    neutral_since: Optional[float] = None

    def update(self, gesture: Optional[str], now: float, allowed: bool = True,
               release_observed: Optional[bool] = None) -> Optional[str]:
        released = gesture is None if release_observed is None else release_observed
        if allowed and gesture is None and released:
            if self.neutral_since is None:
                self.neutral_since = now
            if now-self.neutral_since >= .25:
                self.action_locked = False
        else:
            self.neutral_since = None
        if not allowed:  # Freeze recognition while a foot is touching a pad.
            # A gesture must be observed afresh after stepping off a pad.
            self.missing_since = None
            self.since = now
            # Preserve X's one-shot latch until a release or different pose.
            return None
        if gesture is None and self.candidate is not None:
            if self.missing_since is None:
                self.missing_since = now
            if now-self.missing_since <= self.dropout_grace:
                return None
        if self.missing_since is not None:
            if gesture == self.candidate and now-self.missing_since <= self.dropout_grace:
                self.since += now-self.missing_since
            else:
                self.candidate = None
            self.missing_since = None
        if gesture != self.candidate:
            self.candidate = gesture
            self.since = now
            self.fired = False
        if gesture is None or self.fired:
            return None
        if gesture in GESTURE_ACTIONS and self.action_locked:
            return None
        required = GESTURE_HOLDS.get(gesture, .40)
        if now - self.since >= required:
            self.fired = True
            if gesture in GESTURE_ACTIONS:
                self.action_locked = True
            return gesture
        return None


class PadPresses:
    """Receive debounced Arduino DOWN/UP events, output one press per step."""
    def __init__(self, count: int, dwell: float = 0.12):
        self.count = count
        self.dwell = dwell
        self.down_at = {}  # pad -> seconds
        self.done = set()
        self.blocked_until_clear = False

    def on_event(self, event: str, pad: int, now: float) -> None:
        if not 0 <= pad < self.count:
            return
        if event == "DOWN":
            if pad in self.down_at:
                return
            self.down_at[pad] = now
            if len(self.down_at) > 1:
                self.blocked_until_clear = True
        elif event == "UP":
            self.down_at.pop(pad, None)
            self.done.discard(pad)
            if not self.down_at:
                self.blocked_until_clear = False

    def ready(self, now: float) -> Optional[int]:
        if self.blocked_until_clear or len(self.down_at) != 1:
            return None
        pad, began = next(iter(self.down_at.items()))
        if pad in self.done or now - began < self.dwell:
            return None
        self.done.add(pad)
        return pad
