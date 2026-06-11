import os, re
from collections import defaultdict

ROOT = r"C:\Users\nicesixman\Desktop\VSCode_ClaudeCode and Cursor\qai\data\eval_set"
SCOPES = ["normal", "bug"]
VARIANT_ORDER = ["female_black", "female_white", "male_black", "male_white"]
ANGLE_MAP = {"A0": "front", "A90": "left", "A180": "back", "A270": "right"}

fname_re = re.compile(r"^(?P<item>\d+)_A(?P<angle>\d+)_(?P<date>\d{8})_(?P<time>\d{6})\.png$", re.I)

def parse(fn):
    m = fname_re.match(fn)
    if not m:
        return None
    return {"item": m.group("item"), "anglecode": "A" + m.group("angle"),
            "dt": m.group("date") + m.group("time")}

renames = []  # (vpath, old, new)

for scope in SCOPES:
    scope_root = os.path.join(ROOT, scope)
    for pubg in sorted(os.listdir(scope_root)):
        pubg_path = os.path.join(scope_root, pubg)
        if not os.path.isdir(pubg_path):
            continue
        for itemfolder in sorted(os.listdir(pubg_path)):
            if_path = os.path.join(pubg_path, itemfolder)
            if not os.path.isdir(if_path):
                continue
            case = 0
            for variant in VARIANT_ORDER:
                vpath = os.path.join(if_path, variant)
                if not os.path.isdir(vpath):
                    continue
                parsed = []
                for f in os.listdir(vpath):
                    if not f.lower().endswith(".png"):
                        continue
                    p = parse(f)
                    if p:
                        p["old"] = f
                        parsed.append(p)
                sessions = defaultdict(list)
                for p in parsed:
                    sessions[p["item"]].append(p)
                ordered_items = sorted(sessions, key=lambda it: min(x["dt"] for x in sessions[it]))
                for item in ordered_items:
                    case += 1
                    cstr = f"case{case:02d}"
                    for p in sorted(sessions[item], key=lambda x: x["anglecode"]):
                        angle = ANGLE_MAP[p["anglecode"]]
                        new = f"{cstr}_{angle},{scope}.png"
                        renames.append((vpath, p["old"], new))

# safety: detect duplicate target names within a folder
targets = defaultdict(list)
for vpath, old, new in renames:
    targets[(vpath, new)].append(old)
dups = {k: v for k, v in targets.items() if len(v) > 1}
if dups:
    print("ABORT: duplicate target names detected:")
    for (vp, nm), olds in dups.items():
        print("  ", vp, nm, olds)
    raise SystemExit(1)

# two-phase rename to avoid any collision with existing files
TMP = "__tmp__."
for vpath, old, new in renames:
    os.rename(os.path.join(vpath, old), os.path.join(vpath, TMP + new))
for vpath, old, new in renames:
    os.rename(os.path.join(vpath, TMP + new), os.path.join(vpath, new))

print(f"Renamed {len(renames)} files ({sum(1 for r in renames if r)} total).")
print("Done.")
