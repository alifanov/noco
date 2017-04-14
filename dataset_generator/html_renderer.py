from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.QtWebKit import *
from PIL import Image


class HTMLRenderer(QWebView):
    def __init__(self):
        self.app = QApplication([])
        QWebView.__init__(self)

    def render(self, html):
        self.setHtml(html)
        frame = self.page().mainFrame()
        self.page().setViewportSize(frame.contentsSize())
        # render image
        image = QImage(self.page().viewportSize(), QImage.Format_RGB888)
        painter = QPainter(image)
        frame.render(painter)
        painter.end()
        image = image.convertToFormat(QImage.Format_RGB888)
        bytes = image.bits().asstring(image.numBytes())

        mode = "RGB"
        pilimg = Image.frombuffer(mode, (image.width(), image.height()), bytes, 'raw', mode, 0, 1)
        pilimg.show()
        image.save('output.png')

s = HTMLRenderer()
s.render('<html style="background: yellow; color: green;"><body><p>123</p></body></html>')
