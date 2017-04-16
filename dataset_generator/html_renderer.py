import sys, os
sys.path.append('..')

from scipy.spatial import distance
from html2vec.converter import HTML2VECConverter
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWebKitWidgets import *
from PIL import Image


class HTMLRenderer(QWebView):
    """
    Renderer for HTML to numpy data
    """
    def __init__(self):
        self.app = QApplication([])
        QWebView.__init__(self)

    def render(self, html):
        """
        Render HTML to numpy array
        :param html: 
        :return: 
        """
        self.setHtml(html)
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

        # pilimg.save('test_render.png')
        return np.array(pilimg)


class HTMLGame:
    """
    Environment for build HTML and return state for each step
    """
    TEXT_CONTENT_MAP = {
        'button': 'Button',
        'div': 'Lore ipsum Lore ipsum',
        'p': 'PText',
        'td': 'table row',
        'li': 'list items',
        'a': 'This is the link',
    }

    def __init__(self, result_image):
        self.result_image = np.array(result_image)
        self.renderer = HTMLRenderer()
        self.html_covr = HTML2VECConverter()

    def fill_text_for_html(self, html):
        for k,v in HTMLGame.TEXT_CONTENT_MAP.items():
            tag = '<{tag}></{tag}>'.format(tag=k)
            tag_text = '<{tag}>{text}</{tag}>'.format(text=v, tag=k)
            html = html.replace(tag, tag_text)
        return html

    def step(self, html_vec):
        """
        Render HTML and return state, reward, done for each step
        :param html_vec: 
        :return: 
        """
        html = self.html_covr.convert(html_vec, direction=HTML2VECConverter.VEC2HTML_DIRECTION)
        html = self.fill_text_for_html(html)
        state = self.renderer.render(html)
        reward = 1.0 - distance.braycurtis(self.result_image.flatten(), state.flatten())
        done = False
        if reward == 1.0:
            done = True
        return state, reward, done


# s = HTMLRenderer()
# data = s.render('<html><body><p>abcdefghijklmnopqrstuvwxyz</p></body></html>')
