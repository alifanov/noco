import sys
sys.path.append('..')

import random
import numpy as np
from dataset_generator.html_renderer import HTMLRenderer, HTML2VECConverter


def generate_dataset():
    renderer = HTMLRenderer()
    converter = HTML2VECConverter()
    for i in range(1000):
        item = [converter.html_int_map['<body>']]
        tags = list(HTML2VECConverter.html_int_map.values())
        while True:
            tag = random.choice(tags)
            if tag == converter.html_int_map['</body>'] and tag in item:
                continue
            item.append(tag)
            if tag == converter.html_int_map['</body>']:
                break
        html = converter.convert(item, direction=converter.VEC2HTML_DIRECTION)
        # TODO: add fill text in HTML
        print(html) #TODO: filter "button<input<textarea"
        if html.count('</') != html.count('<')/2:
            continue
        if html.count('<') != html.count('>'):
            continue
        if html == '<body></body>':
            continue
        image_data = renderer.render_html(converter.convert(item, direction=converter.VEC2HTML_DIRECTION))
        np.save('images/{}'.format('-'.join([str(i) for i in item])), image_data)

if __name__ == "__main__":
    generate_dataset()