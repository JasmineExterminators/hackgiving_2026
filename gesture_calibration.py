"""User-taught arm pose templates; only landmarks are saved, never images."""
from collections import deque
import json
import math
from pathlib import Path
import statistics

STEPS = ('O', 'UP', 'DOWN')
LABELS = {'O':'O: round arms above head, elbows BENT and out',
          'UP':'THIRD position: arms STRAIGHT UP, hands together',
          'DOWN':'FIRST position: both arms diagonally DOWN and out'}


def arm_features(points, width, height):
    if not points or len(points)<17:
        return None
    ids = (11,12,13,14,15,16)
    if any(getattr(points[i],'visibility',None) is not None and
           points[i].visibility < .45 for i in ids):
        return None
    def xy(i):
        return points[i].x*width, points[i].y*height
    arms = sorted(((xy(11),xy(13),xy(15)), (xy(12),xy(14),xy(16))))
    sw = arms[1][0][0]-arms[0][0][0]
    if sw < width*.065:
        return None
    cx = (arms[0][0][0]+arms[1][0][0])/2
    cy = (arms[0][0][1]+arms[1][0][1])/2
    values = tuple(v for arm in arms for x,y in arm[1:]
                   for v in ((x-cx)/sw,(y-cy)/sw))
    return values if all(math.isfinite(v) for v in values) else None


def distance(a,b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/len(a))


def straight_up_features(features):
    """Keep personalized Accept distinct from rounded O or a raised hand."""
    if features is None:
        return False
    le,lw,re,rw = [features[i:i+2] for i in (0,2,4,6)]
    if not (lw[1]<-.8 and rw[1]<-.8 and le[1]<-.3 and re[1]<-.3
            and lw[1]<le[1]-.2 and rw[1]<re[1]-.2
            and math.dist(lw,rw)<.8):
        return False
    for shoulder,elbow,wrist in (((-.5,0),le,lw),((.5,0),re,rw)):
        a = (elbow[0]-shoulder[0],elbow[1]-shoulder[1])
        b = (wrist[0]-elbow[0],wrist[1]-elbow[1])
        norm = math.hypot(*a)*math.hypot(*b)
        if not norm or (a[0]*b[0]+a[1]*b[1])/norm <= .80:
            return False
    return True


class GestureCalibration:
    def __init__(self, path, legacy_path=None):
        self.path = Path(path)
        self.templates = {}
        self.pending = {}
        self.step = None
        self.samples = deque(maxlen=60)
        self.message = ''
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
            if (set(data)==set(STEPS) and all(isinstance(data[k],list) and len(data[k])==8
                and all(isinstance(x,(int,float)) and math.isfinite(x) for x in data[k])
                for k in STEPS)):
                self.templates = data
        except (OSError, ValueError, TypeError):
            pass
        if not self.templates and legacy_path:
            # Preserve only the working Delete template from the old pairing.
            try:
                old_data = json.loads(Path(legacy_path).read_text(encoding='utf-8'))
                old = old_data['DELETE']
                if isinstance(old,list) and len(old)==8 and all(
                        isinstance(x,(int,float)) and math.isfinite(x) for x in old):
                    self.templates['DOWN'] = old
                for old_key,new_key in (('ACCEPT','UP'),('O','O')):
                    value = old_data.get(old_key)
                    if isinstance(value,list) and len(value)==8 and all(
                            isinstance(x,(int,float)) and math.isfinite(x) for x in value):
                        self.templates[new_key] = value
            except (OSError, ValueError, TypeError, KeyError):
                pass

    @property
    def active(self):
        return self.step is not None

    @property
    def hint(self):
        if self.active:
            return LABELS[STEPS[self.step]]+'; hold still, then press C'
        return self.message

    def observe(self, features, now):
        if features is None:
            self.samples.clear()
        else:
            self.samples.append((now,features))

    def capture(self, now):
        if not self.active:
            self.pending = {}
            self.step = 0
            self.samples.clear()
            self.message = ''
            return
        recent = [f for t,f in self.samples if now-t <= .8]
        times = [t for t,f in self.samples if now-t <= .8]
        if len(recent)<8 or times[-1]-times[0]<.4:
            self.message = 'Keep both shoulders, elbows and wrists visible; hold still longer.'
            return
        template = [statistics.median(v) for v in zip(*recent)]
        if STEPS[self.step] == 'UP' and not straight_up_features(template):
            self.message = 'For THIRD position, straighten BOTH elbows upward and bring hands together.'
            return
        if max(distance(template,f) for f in recent) > .10:
            self.message = 'Pose is moving. Hold still for a second, then press C.'
            return
        if any(distance(template,other)<.12 for other in self.pending.values()):
            self.message = 'Too similar to an earlier pose; make this pose more distinct and retry C.'
            return
        self.pending[STEPS[self.step]] = template
        self.step += 1
        self.samples.clear()
        self.message = ''
        if self.step == len(STEPS):
            self.templates = self.pending.copy()
            self.step = None
            try:
                self.path.write_text(json.dumps(self.templates),encoding='utf-8')
                self.message = 'Calibration saved. Lower arms, then try the new gestures.'
            except OSError:
                self.message = 'Calibration active for this session; could not save it.'

    def classify(self, features, fallback):
        if not self.templates or features is None:
            return fallback
        scores = sorted((distance(features,template),key) for key,template in self.templates.items())
        margin = scores[1][0]-scores[0][0] if len(scores)>1 else float('inf')
        if scores[0][0]<.20 and margin>.08:
            if scores[0][1]=='UP' and not straight_up_features(features):
                return None
            return scores[0][1]
        # Do not let the generic O/action rules override a calibrated mismatch.
        return fallback if fallback in ('T','Y','X') or fallback not in self.templates else None
