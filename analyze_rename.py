import os, re
from collections import defaultdict

ROOT = r"C:\Users\nicesixman\Desktop\VSCode_ClaudeCode and Cursor\qai\data\eval_set"
SCOPES = ["normal", "bug"]
VARIANT_ORDER = ["female_black", "female_white", "male_black", "male_white"]

# angle code -> name (left/right assumption flagged for confirmation)
ANGLE_MAP = {"A0": "front", "A90": "right", "A180": "back", "A270": "left"}

fname_re = re.compile(r"^(?P<item>\d+)_A(?P<angle>\d+)_(?P<date>\d{8})_(?P<time>\d{6})\.png$", re.I)

def parse(fn):
    m = fname_re.match(fn)
    if not m:
        return None
    return {
        "item": m.group("item"),
        "anglecode": "A" + m.group("angle"),
        "dt": m.group("date") + m.group("time"),
    }

# plan[scope][itemfolder] = list of (variant, case, item, anglecode, oldname, newname)
plan = {s: defaultdict(list) for s in SCOPES}
# mapping[scope][itemfolder][(variant, case)] = item  (for consistency check)
mapping = {s: {} for s in SCOPES}

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
            key = f"{pubg}/{itemfolder}"
            mapping[scope].setdefault(key, {})
            case = 0
            for variant in VARIANT_ORDER:
                vpath = os.path.join(if_path, variant)
                if not os.path.isdir(vpath):
                    continue
                files = [f for f in os.listdir(vpath) if f.lower().endswith(".png")]
                parsed = []
                for f in files:
                    p = parse(f)
                    if p:
                        p["old"] = f
                        parsed.append(p)
                # group into sessions by itemcode
                sessions = defaultdict(list)
                for p in parsed:
                    sessions[p["item"]].append(p)
                # order sessions by earliest timestamp
                ordered_items = sorted(sessions, key=lambda it: min(x["dt"] for x in sessions[it]))
                for item in ordered_items:
                    case += 1
                    cstr = f"case{case:02d}"
                    mapping[scope][key][(variant, cstr)] = item
                    for p in sorted(sessions[item], key=lambda x: x["anglecode"]):
                        angle = ANGLE_MAP.get(p["anglecode"], p["anglecode"])
                        newname = f"{cstr}_{angle},{scope}.png"
                        plan[scope][key].append(
                            (variant, cstr, item, p["anglecode"], angle, p["old"], newname)
                        )

# ---- Print preview per item folder ----
for key in sorted(plan["normal"]):
    print("=" * 90)
    print("ITEM FOLDER:", key)
    nmap = mapping["normal"].get(key, {})
    bmap = mapping["bug"].get(key, {})
    allcases = sorted(set(nmap) | set(bmap))
    print(f"  {'variant':14} {'case':7} {'normal item':12} {'bug item':12} match?")
    for (variant, cstr) in allcases:
        ni = nmap.get((variant, cstr), "-")
        bi = bmap.get((variant, cstr), "-")
        ok = "OK" if ni == bi and ni != "-" else "*** MISMATCH ***"
        print(f"  {variant:14} {cstr:7} {ni:12} {bi:12} {ok}")

print("=" * 90)
# sample rename listing for first folder
firstkey = sorted(plan["normal"])[0]
print("\nSAMPLE renames (normal,", firstkey, "):")
for row in plan["normal"][firstkey]:
    variant, cstr, item, ac, angle, old, new = row
    print(f"  {variant}/{old}  ->  {new}")

# totals
for scope in SCOPES:
    total = sum(len(v) for v in plan[scope].values())
    print(f"\nTOTAL files to rename in '{scope}': {total}")

print("\nANGLE MAP USED:", ANGLE_MAP)
