import requests
from collections import defaultdict
import os
from development.docs.write_openapi_spec import DOCS_ROOT_DIR
from jinja2 import Environment, FileSystemLoader

template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
jinja_env = Environment(loader=FileSystemLoader(template_dir))

def render_template(template_name, **kwargs):
    template = jinja_env.get_template(template_name)
    return template.render(**kwargs)

TUTORIAL_URL = "https://roboflow.ghost.io/ghost/api/content/posts/?key=" + os.getenv("GHOST_API_KEY", "")

if not TUTORIAL_URL:
    raise ValueError("GHOST_API_KEY environment variable is not set. Please set it to access the blog API.")

TEMPLATE = """
- **{title}**

    <a href="{url}" style="color: black !important;">
    <img src="{image_url}">

    <p>{description}</p>
    </a>
"""

CUSTOM = [
    {"title": "Measure Object Sizes with Computer Vision [video]", "url": "https://www.youtube.com/watch?v=FQY7TSHfZeI", "feature_image": "https://media.roboflow.com/inference/cover-images/size-measurement.jpg", "blocks": ["Size Measurement"], "description": "Learn how to measure the size of objects in images using computer vision techniques."},
    {"title": "Use YOLOv12 with Roboflow and Workflows [video]", "url": "https://www.youtube.com/watch?v=fksJmIMIfXo", "feature_image": "https://media.roboflow.com/inference/cover-images/yolov12.jpg", "blocks": ["Object Detection Model"], "description": "Learn how to use YOLOv12 for object detection in your computer vision projects."},
    {"title": "Use Florence-2 with Roboflow and Workflows [video]", "url": "https://www.youtube.com/watch?v=_u53TxShLsk", "feature_image": "https://media.roboflow.com/inference/cover-images/florence-2.jpg", "blocks": ["Florence-2 Model"], "description": "Learn how to use the Florence-2 model for advanced computer vision tasks."},
    {"title": "Use Depth Anything 2 with Workflows [video]", "url": "https://www.youtube.com/watch?v=lqPf3198wjw", "feature_image": "https://media.roboflow.com/inference/cover-images/depth-anything.jpg", "blocks": ["Depth Estimation"], "description": "Learn how to use Depth Anything 2 for depth estimation in images."},
    {"title": "Use Qwen2.5-VL with Workflows [video]", "url": "https://www.youtube.com/watch?v=xEfh0IR8Fvo", "feature_image": "https://media.roboflow.com/inference/cover-images/qwen.jpg", "blocks": ["Qwen2.5-VL"], "description": "Learn how to use the Qwen2.5-VL model for vision-language tasks."},
]

def get_tutorials():
    next = ""
    results = defaultdict(list)
    results_indexed_by_used_workflow_block = defaultdict(list)
    
    for custom in CUSTOM:
        title = custom["title"]
        url = custom["url"]
        image_url = custom["feature_image"]
        description = custom.get("description", "")
        blocks = custom.get("blocks", [])
        
        if not blocks:
            continue
        
        for block in blocks:
            results_indexed_by_used_workflow_block[block].append(
                TEMPLATE.format(
                    title=title,
                    url=url,
                    image_url=image_url,
                    description=description or "",
                )
            )

    while next is not None:
        try:
            response = requests.get(f"{TUTORIAL_URL}{next}&include=tags,authors&limit=all")
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching tutorials: {e}")
            break

        data = response.json()

        for template in data.get("posts", []):
            title = template["title"]
            url = template["url"]
            image_url = "https://media.roboflow.com/inference/cover-images/" + template["slug"] + ".png"
            description = template["custom_excerpt"].split(". ")[0] if template.get("custom_excerpt") else ""
            tags = template.get("tags", [])
            tags.append({"name": "#Workflows_None"})

            tags = [i["name"].replace("#Workflows_", "").replace("_", " ") for i in tags if i["name"].startswith("#Workflows_")]
            main_tag = tags[0] if tags else "None"

            # if none in tag, skip
            if main_tag == "None":
                continue

            # if any tag ends in _Block, add to results_indexed_by_used_workflow_block
            for tag in tags:
                if tag.endswith(" Block"):
                    results_indexed_by_used_workflow_block[tag.replace(" Block", "")].append(
                        TEMPLATE.format(
                            title=title,
                            url=url,
                            image_url=image_url,
                            description=description or "",
                        )
                    )
                else:
                    results[tag.replace(" Block", "")].append(
                        TEMPLATE.format(
                            title=title,
                            url=url,
                            image_url=image_url,
                            description=description or "",
                        )
                    )
            # # save meta image as img/post-slug.png
            # with open(f"cover-images/{template['slug']}.png", "wb") as img_file:
            #     img = Image.open(requests.get(image_url, stream=True).raw)
            #     img = img.convert("RGB")
            #     img.thumbnail((800, 800)) #, Image.ANTIALIAS)
            #     img.save(img_file, format="PNG", optimize=True, quality=85)

        next = data.get("meta", {}).get("pagination", {}).get("next", None)
        next = None
        if next:
            next = f"&page={next}"

        if next is None:
            break

    # delete "None" key
    if "None" in results:
        del results["None"]

    return results, results_indexed_by_used_workflow_block

if __name__ == "__main__":
    WRITTEN_TUTORIALS_FILE = os.path.join(DOCS_ROOT_DIR, "guides", "written.md")

    results, results_indexed_by_used_workflow_block = get_tutorials()

    text = ""
    
    for tag, tutorials in results.items():
        text += f"## {tag}\n\n"
        text += "<div class=\"grid cards\" markdown>\n"
        for tutorial in tutorials:
            text += tutorial + "\n\n"
        text += "</div>\n"

    template = render_template("tutorials.md", text=text)

    with open(WRITTEN_TUTORIALS_FILE, "w+") as f:
        f.write(template)