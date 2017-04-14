from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.QtWebKit import *


class HTMLRenderer(QWebView):
    def __init__(self):
        self.app = QApplication([])
        QWebView.__init__(self)

    def render(self, html):
        self.setHtml(html)
        frame = self.page().mainFrame()
        self.page().setViewportSize(frame.contentsSize())
        # render image
        image = QImage(self.page().viewportSize(), QImage.Format_ARGB32)
        painter = QPainter(image)
        frame.render(painter)
        painter.end()
        image.save('output.png')

s = HTMLRenderer()
s.render('<p>123</p>')
