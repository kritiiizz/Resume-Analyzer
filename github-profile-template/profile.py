from pathlib import Path

ascii_text=Path("ascii-art.txt").read_text(encoding="utf-8")
template=Path("light_mode.svg").read_text(encoding="utf-8")
out=template.replace("PASTE CONTENTS OF ascii-art.txt HERE (or generate automatically with profile.py)", ascii_text)
Path("generated_light_mode.svg").write_text(out,encoding="utf-8")
print("generated_light_mode.svg created")
