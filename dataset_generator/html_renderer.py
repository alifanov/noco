import random
import sys, os
sys.path.append('..')

from scipy.spatial import distance
from html2vec.converter import HTML2VECConverter
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWebKitWidgets import *
from PyQt5.QtCore import *
from PIL import Image


class HTMLRenderer(QWebView):
    """
    Renderer for HTML to numpy data
    """
    HTML_WRAPPER = """
<html>
<style>
body {{
    background: #eee;
    color: #449EF3;
}}
div {{
    background: #50C878;
    padding: 10px;
    margin: 10px;
    border: 2px solid #F4A460;
    border-radius: 5px;
}}
p {{
    border: 1px solid #00b3f4;
    border-radius: 5px;
    padding: 10px;
}}
a {{
    text-decoration: underline;
    color: #2A52BE;
}}
</style>
{}
</html>
"""

    def __init__(self):
        self.app = QApplication([])
        QWebView.__init__(self)
        self.settings().setUserStyleSheetUrl(QUrl.fromLocalFile('style.css'))

    def render(self, html):
        """
        Render HTML to numpy array
        :param html: 
        :return: 
        """
        self.setHtml(HTMLRenderer.HTML_WRAPPER.format(html))
        frame = self.page().mainFrame()
        self.page().setViewportSize(frame.contentsSize())
        # render image
        image = QImage(self.page().viewportSize(), QImage.Format_RGB888)
        painter = QPainter(image)
        frame.render(painter)
        painter.end()
        image = image.convertToFormat(QImage.Format_RGB888)
        bytes = image.bits().asstring(image.byteCount())

        mode = "RGB"
        pilimg = Image.frombuffer(mode, (image.width(), image.height()), bytes, 'raw', mode, 0, 1)
        # pilimg.show()

        pilimg.save('test_render.png')
        return np.array(pilimg)


class HTMLGame:
    """
    Environment for build HTML and return state for each step
    """
    TEXT_CONTENT_MAP = {
        'button': 'ButtonText',
        'div': 'DivText',
        'p': 'PText',
        'td': 'TableCellText',
        'li': 'ListItemText',
        'a': 'LinkText',
    }

    def __init__(self, result_image):
        self.result_image = np.array(result_image)
        self.renderer = HTMLRenderer()
        self.html_covr = HTML2VECConverter()
        self.html_vec = []

    def reset(self):
        self.__init__(self.result_image)
        return self.step(self.html_vec)

    def fill_text_for_html(self, html):
        for k,v in HTMLGame.TEXT_CONTENT_MAP.items():
            tag = '<{tag}></{tag}>'.format(tag=k)
            tag_text = '<{tag}>{text}</{tag}>'.format(text=v, tag=k)
            html = html.replace(tag, tag_text)
        return html

    def action_sample(self):
        return random.choice(HTML2VECConverter.html_int_map.values())

    def step(self, action):
        """
        Render HTML and return state, reward, done for each step
        :param action: 
        :return: 
        """
        self.html_vec.append(action)
        html = self.html_covr.convert(self.html_vec, direction=HTML2VECConverter.VEC2HTML_DIRECTION)
        html = self.fill_text_for_html(html)
        state = self.renderer.render(html)
        reward = 1.0 - distance.braycurtis(self.result_image.flatten(), state.flatten())
        done = False
        if reward == 1.0:
            done = True
        return state, reward, done


# s = HTMLRenderer()
# data = s.render('<html><body><p>PText</p></body></html>')
