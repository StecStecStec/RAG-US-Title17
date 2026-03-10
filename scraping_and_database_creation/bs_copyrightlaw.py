import requests
from bs4 import BeautifulSoup
import os
import re

URL = "https://www.govinfo.gov/content/pkg/USCODE-2023-title17/html/USCODE-2023-title17.htm#_1_target"
resp = requests.get(URL)
soup = BeautifulSoup(resp.text, "html.parser")

chapter_directory = "../data/title17_chapters_v2"

current_section = ""
previous_section = ""
sections = []

os.makedirs(chapter_directory, exist_ok=True)

for header in soup.find_all("h3", class_="chapter-head"):
    chapter = re.sub(r'[\\/*?:"<>|,;.-]', "_", header.get_text().strip()).replace("—", "_").replace(" ", "_").replace("__", "_")

    section_directory = os.path.join(chapter_directory, chapter)
    os.makedirs(section_directory, exist_ok=True)

    content_parts = []

    for sibling in header.find_next_siblings():
        if sibling.name == "h3" and sibling.get("class"):
            if any(c == "chapter-head" for c in sibling.get("class")):
                break

            elif any(c == "section-head" for c in sibling.get("class")):
                current_section = sibling.get_text().strip().replace("§", "section_")
                current_section = re.sub(r'[\\/*?:"<>|,;.-]', "_", current_section).replace("—", "_").replace(" ", "_").replace("__", "_")

                #with open(os.path.join(section_directory, current_section + '.txt'), "a", encoding="utf-8") as f:
                #    f.write(current_section + "\n")

                previous_section = current_section

        elif sibling.name == "p" and sibling.get("class"):
            if any(c.startswith("statutory-body") for c in sibling.get("class")):
                with open(os.path.join(section_directory, previous_section + '.txt'), "a", encoding="utf-8") as f:
                    f.write(sibling.get_text() + "\n")