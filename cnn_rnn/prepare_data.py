import sys
sys.path.append('..')

import random
import numpy as np
from scipy.misc import toimage
from dataset_generator.html_renderer import HTMLRenderer, HTML2VECConverter, HTMLGame


def generate_dataset():
    renderer = HTMLRenderer()
    converter = HTML2VECConverter()
    for i in range(10000000):
        item = [converter.html_int_map['<body>']]
        tags = list(HTML2VECConverter.html_int_map.values())
        while True:
            tag = random.choice(tags)
            if tag == converter.html_int_map['</body>'] and tag in item:
                continue
            item.append(tag)
            if tag == converter.html_int_map['</body>']:
                break
        if len(item) < 7:
            continue
        html = converter.convert(item, direction=converter.VEC2HTML_DIRECTION)
        html = HTMLGame.fill_text_for_html(html)
        bad_words = [
            '>>',
            '<p<',
            '<div<',
            '<a<',
            '<body<',
        ]
        passed = True
        for w in bad_words:
            if w in html:
                passed = False
                break
        for tag in [
            ('<a>', '</a>'),
            ('<body>', '</body>'),
            ('<div>', '</div>'),
            ('<p>', '</p>'),
        ]:
            if html.count(tag[0]) != html.count(tag[1]):
                passed = False
                break
            if tag[0] in html and tag[1] in html:
                # print(html)
                # print(tag, html.index(tag[0]), html.index(tag[1]))
                if html.index(tag[1]) < html.index(tag[0]):
                    passed = False
                    break
            if tag[0] in html and tag[1] not in html:
                passed = False
                break
            if tag[0] not in html and tag[1] in html:
                passed = False
                break
        if not passed:
            continue
        if html.count('</') != html.count('<')/2:
            continue
        if html.count('<') != html.count('>'):
            continue
        if html == '<body></body>':
            continue
        print(html)
        image_data = renderer.render_html(html)
        np.save('images/{}'.format('-'.join([str(i) for i in item])), image_data)

if __name__ == "__main__":
    generate_dataset()