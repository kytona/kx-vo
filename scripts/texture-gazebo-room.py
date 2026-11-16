"""Add reproducible visual features to the Gazebo camera diagnostic room."""

import random
import sys
from xml.etree import ElementTree as ET

source, output = sys.argv[1:]
tree = ET.parse(source)
world = tree.getroot().find("world")
texture = ET.SubElement(world, "model", name="texture")
ET.SubElement(texture, "static").text = "true"
link = ET.SubElement(texture, "link", name="tiles")
rng = random.Random(47)


def tile(name, pose, size):
    visual = ET.SubElement(link, "visual", name=name)
    ET.SubElement(visual, "pose").text = " ".join(map(str, (*pose, 0, 0, 0)))
    box = ET.SubElement(ET.SubElement(visual, "geometry"), "box")
    ET.SubElement(box, "size").text = " ".join(map(str, size))
    material = ET.SubElement(visual, "material")
    color = " ".join(f"{v:.2f}" for v in [*(rng.uniform(0.12, 0.88) for _ in range(3)), 1.0])
    for channel in ("ambient", "diffuse"):
        ET.SubElement(material, channel).text = color


for x in range(-8, 9):
    for y in range(-8, 9):
        tile(f"floor_{x}_{y}", (x * 0.4 + rng.uniform(-0.04, 0.04),
                              y * 0.4 + rng.uniform(-0.04, 0.04), 0.005),
             (rng.uniform(0.22, 0.30), rng.uniform(0.22, 0.30), 0.01))
for x in range(-9, 10):
    for z in range(9):
        tile(f"wall_{x}_{z}", (x * 0.35, 3.8, 0.2 + z * 0.35), (0.34, 0.03, 0.34))
tree.write(output, encoding="unicode")
